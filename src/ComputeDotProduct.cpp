/*
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Copyright (C) 2019, Arm Limited and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif
#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#include <cassert>
#ifdef HPCG_USE_DDOT_ARMPL
#include "armpl.h"
#endif
#ifdef HPCG_USE_SVE
#include "arm_sve.h"
#endif

inline double ComputeDotProduct_unrolled(const local_int_t n, const double*xv, const double *yv);

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

	assert(x.localLength >= n);
	assert(y.localLength >= n);

	double *xv = x.values;
	double *yv = y.values;
	double local_result = 0.0;

#if defined HPCG_USE_SVE
	if ( xv == yv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += svcntd()) {
			svbool_t pg = svwhilelt_b64(i, n);
			svfloat64_t svx = svld1_f64(pg, &xv[i]);

            svfloat64_t svlr = svmul_f64_z(pg, svx, svx);

            local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += svcntd()) {
			svbool_t pg = svwhilelt_b64_u64(i, n);
			svfloat64_t svx = svld1_f64(pg, &xv[i]);
			svfloat64_t svy = svld1_f64(pg, &yv[i]);

            svfloat64_t svlr = svmul_f64_z(pg, svx, svy);
            
            local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	}
#elif defined HPCG_USE_DDOT_ARMPL
	local_result = cblas_ddot(n, xv, 1, yv, 1);
#elif defined HPCG_MAN_OPT_DDOT
	local_result = ComputeDotProduct_unrolled(n, xv, yv);
/*
	double local_result0 = 0.0, local_result1=0.0, local_result2=0.0, local_result3=0.0;
	if (yv == xv) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=2 ) {
            local_result0 += xv[i+0] * xv[i+0];
            local_result1 += xv[i+1] * xv[i+1];
        }
		local_result += local_result0+local_result1;
	}
	else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=2 ) {
			local_result0 += xv[i] * yv[i];
			local_result1 += xv[i+1] * yv[i+1];
		}
		local_result += local_result0+local_result1;
	}
*/
#else //HPCG_USE_DDOT_ARMPL
	if ( yv == xv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result)
//#pragma clang loop vectorize_width(8)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
            local_result += xv[i] * xv[i];
        }
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result)
//#pragma clang loop vectorize_width(8)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i++ ) {
			local_result += xv[i] * yv[i];
		}
	}
#endif //HPCG_USE_DDOT_ARMPL

#ifndef HPCG_NO_MPI
	// Use MPI's reduce function to collect all partial sums
	double t0 = mytimer();
	double global_result = 0.0;
	MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	result = global_result;
	time_allreduce += mytimer() - t0;
#else //HPCG_NO_MPI
	time_allreduce += 0.0;
	result = local_result;
#endif //HPCG_NO_MPI

	return 0;
}

//2,4,6-way unrolling using intrinsics
#ifdef HPCG_MAN_OPT_DDOT

#ifdef DDOT_INTRINSICS
	#include "arm_sve.h"

	#ifdef DDOT_2_UNROLL
		#define ComputeDotProduct_2_intrinsics_unrolling ComputeDotProduct_unrolled
	#elif defined DDOT_4_UNROLL
		#define ComputeDotProduct_4_intrinsics_unrolling ComputeDotProduct_unrolled
	#elif defined DDOT_6_UNROLL
		#define ComputeDotProduct_6_intrinsics_unrolling ComputeDotProduct_unrolled
	#else
		No valid (1)
	#endif
#else
	#ifdef DDOT_2_UNROLL
	#define ComputeDotProduct_2_unrolling ComputeDotProduct_unrolled
	#elif defined DDOT_4_UNROLL
		#define ComputeDotProduct_4_unrolling ComputeDotProduct_unrolled
	#elif defined DDOT_6_UNROLL
		#define ComputeDotProduct_6_unrolling ComputeDotProduct_unrolled
	#else
		no valid (2)
	#endif
#endif

#ifdef DDOT_INTRINSICS
inline double ComputeDotProduct_6_intrinsics_unrolling(const local_int_t n, const double*xv, const double *yv) {
	double local_result=0;

	if ( xv == yv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 6*svcntd()) {
			//local_int_t ij = i+svcntd();
			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);

			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);

            svx0 = svmul_f64_z(pg0, svx0, svx0);
            svx1 = svmul_f64_z(pg1, svx1, svx1);

			svbool_t pg2 = svwhilelt_b64_u64(i+2*svcntd(), n);
			svbool_t pg3 = svwhilelt_b64_u64(i+3*svcntd(), n);			

			svx1 = svadd_f64_z(svptrue_b64(), svx0, svx1);

			svfloat64_t svx2 = svld1_f64(pg2, &xv[i+2*svcntd()]);
			svfloat64_t svx3 = svld1_f64(pg3, &xv[i+3*svcntd()]);

			svbool_t pg4 = svwhilelt_b64_u64(i+4*svcntd(), n);
			svbool_t pg5 = svwhilelt_b64_u64(i+5*svcntd(), n);			

            svx2 = svmul_f64_z(pg2, svx2, svx2);
            svx3 = svmul_f64_z(pg3, svx3, svx3);
            //svx2 = svmla_f64_m(pg2, svx0, svx2, svx2);
            //svx3 = svmla_f64_m(pg3, svx1, svx3, svx3);

			svfloat64_t svx4 = svld1_f64(pg4, &xv[i+4*svcntd()]);
			svfloat64_t svx5 = svld1_f64(pg5, &xv[i+5*svcntd()]);

			svx3 = svadd_f64_z(svptrue_b64(), svx2, svx3);
			svx3 = svadd_f64_z(svptrue_b64(), svx1, svx3);

            svx4 = svmul_f64_z(pg4, svx4, svx4);
            svx5 = svmul_f64_z(pg5, svx5, svx5);

			svx5 = svadd_f64_z(svptrue_b64(), svx4, svx5);
			svfloat64_t svlr = svadd_f64_z(svptrue_b64(), svx3, svx5);

            local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 6*svcntd()) {
			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);
			
			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);
			svfloat64_t svy0 = svld1_f64(pg0, &yv[i]);
			svfloat64_t svy1 = svld1_f64(pg1, &yv[i+svcntd()]);

			svbool_t pg2 = svwhilelt_b64_u64(i+2*svcntd(), n);
			svbool_t pg3 = svwhilelt_b64_u64(i+3*svcntd(), n);			

            svx0 = svmul_f64_z(pg0, svx0, svy0);
            svx1 = svmul_f64_z(pg1, svx1, svy1);
			svx1 = svadd_f64_z(svptrue_b64(), svx0, svx1);

			svfloat64_t svx2 = svld1_f64(pg2, &xv[i+2*svcntd()]);
			svfloat64_t svx3 = svld1_f64(pg3, &xv[i+3*svcntd()]);
			svfloat64_t svy2 = svld1_f64(pg2, &yv[i+2*svcntd()]);
			svfloat64_t svy3 = svld1_f64(pg3, &yv[i+3*svcntd()]);

			svbool_t pg4 = svwhilelt_b64_u64(i+4*svcntd(), n);
			svbool_t pg5 = svwhilelt_b64_u64(i+5*svcntd(), n);

            svx2 = svmul_f64_z(pg2, svx2, svy2);
            svx3 = svmul_f64_z(pg3, svx3, svy3);
			svx3 = svadd_f64_z(svptrue_b64(), svx2, svx3);

			svx3= svadd_f64_z(svptrue_b64(), svx1, svx3);

			svfloat64_t svx4 = svld1_f64(pg4, &xv[i+4*svcntd()]);
			svfloat64_t svx5 = svld1_f64(pg5, &xv[i+5*svcntd()]);
			svfloat64_t svy4 = svld1_f64(pg4, &yv[i+4*svcntd()]);
			svfloat64_t svy5 = svld1_f64(pg5, &yv[i+5*svcntd()]);
			
            svx4 = svmul_f64_z(pg4, svx4, svy4);
            svx5 = svmul_f64_z(pg5, svx5, svy5);
			svx5 = svadd_f64_z(svptrue_b64(), svx4, svx5);

			svfloat64_t svlr = svadd_f64_z(svptrue_b64(), svx3, svx5);

	        local_result += svaddv_f64(svptrue_b64(), svlr);
		}
	}

	return local_result;
}

inline double ComputeDotProduct_4_intrinsics_unrolling(const local_int_t n, const double*xv, const double *yv) {
	double local_result=0;

	if ( xv == yv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 4*svcntd()) {
			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);

			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);

            svx0 = svmul_f64_z(pg0, svx0, svx0);
            svx1 = svmul_f64_z(pg1, svx1, svx1);
			//svx1 = svmla_f64_m(pg1, svx0, svx1, svx1);

			svbool_t pg2 = svwhilelt_b64_u64(i+2*svcntd(), n);
			svbool_t pg3 = svwhilelt_b64_u64(i+3*svcntd(), n);			

			//svx1 = svadd_f64_z(svptrue_b64(), svx0, svx1);

			svfloat64_t svx2 = svld1_f64(pg2, &xv[i+2*svcntd()]);
			svfloat64_t svx3 = svld1_f64(pg3, &xv[i+3*svcntd()]);
			
            //svx2 = svmul_f64_z(pg2, svx2, svx2);
            //svx3 = svmul_f64_z(pg3, svx3, svx3);
			svx2 = svmla_f64_m(pg2, svx0, svx2, svx2);
			svx3 = svmla_f64_m(pg3, svx1, svx3, svx3);

			//svx3 = svadd_f64_z(svptrue_b64(), svx2, svx3);
			//svx1 = svadd_f64_z(svptrue_b64(), svx1, svx3);
			svfloat64_t svl = svadd_f64_z(svptrue_b64(), svx2, svx3);

            local_result += svaddv_f64(svptrue_b64(), svl);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 4*svcntd()) {
			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);
			
			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);
			svfloat64_t svy0 = svld1_f64(pg0, &yv[i]);
			svfloat64_t svy1 = svld1_f64(pg1, &yv[i+svcntd()]);

			svbool_t pg2 = svwhilelt_b64_u64(i+2*svcntd(), n);
			svbool_t pg3 = svwhilelt_b64_u64(i+3*svcntd(), n);			

            svx0 = svmul_f64_z(pg0, svx0, svy0);
            svx1 = svmul_f64_z(pg1, svx1, svy1);
			//svfloat64_t svl1 = svadd_f64_z(svptrue_b64(), svx0, svx1);

			svfloat64_t svx2 = svld1_f64(pg2, &xv[i+2*svcntd()]);
			svfloat64_t svx3 = svld1_f64(pg3, &xv[i+3*svcntd()]);
			svfloat64_t svy2 = svld1_f64(pg2, &yv[i+2*svcntd()]);
			svfloat64_t svy3 = svld1_f64(pg3, &yv[i+3*svcntd()]);

            //svx2 = svmul_f64_z(pg2, svx2, svy2);
            //svx3 = svmul_f64_z(pg3, svx3, svy3);
			//svfloat64_t svl2 = svadd_f64_z(svptrue_b64(), svx2, svx3);
			svx2 = svmla_f64_m(pg2, svx0, svx2, svy2);
			svx3 = svmla_f64_m(pg3, svx1, svx3, svy3);

			//svfloat64_t svl = svadd_f64_z(svptrue_b64(), svl1, svl2);
			svfloat64_t svl = svadd_f64_z(svptrue_b64(), svx3, svx2);

	        local_result += svaddv_f64(svptrue_b64(), svl);
		}
	}

	return local_result;
}

inline double ComputeDotProduct_2_intrinsics_unrolling(const local_int_t n, const double*xv, const double *yv) {

	double local_result=0;

	if ( xv == yv ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 2*svcntd()) {
			//svfloat64_t svsum0 = svdup_f64(0.0);

			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);

			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);

            svx0 = svmul_f64_z(pg0, svx0, svx0);
            //svx1 = svmul_f64_z(pg1, svx1, svx1);

			//svx1 = svadd_f64_z(svptrue_b64(), svx0, svx1);
			svx1 = svmla_f64_m(pg1, svx0, svx1, svx1);

            local_result += svaddv_f64(svptrue_b64(), svx1);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+:local_result)
#endif
		for ( local_int_t i = 0; i < n; i += 2*svcntd()) {
			svbool_t pg0 = svwhilelt_b64_u64(i, n);
			svbool_t pg1 = svwhilelt_b64_u64(i+svcntd(), n);
			
			svfloat64_t svx0 = svld1_f64(pg0, &xv[i]);
			svfloat64_t svx1 = svld1_f64(pg1, &xv[i+svcntd()]);
			svfloat64_t svy0 = svld1_f64(pg0, &yv[i]);
			svfloat64_t svy1 = svld1_f64(pg1, &yv[i+svcntd()]);

            svx0 = svmul_f64_z(pg0, svx0, svy0);
            //svx1 = svmul_f64_z(pg1, svx1, svy1);
			//svfloat64_t svl = svadd_f64_z(svptrue_b64(), svx0, svx1);
			svx0 = svmla_f64_m(pg1, svx0, svx1, svy1);

	        local_result += svaddv_f64(svptrue_b64(), svx0);
		}
	}

	return local_result;
}
#endif //DDOT_INTRINSICS

inline double ComputeDotProduct_4_unrolling(const local_int_t n, const double*xv, const double *yv) {

	double local_result = 0.0;
	double local_result0 = 0.0, local_result1=0.0, local_result2=0.0, local_result3=0.0;

	if (yv == xv) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1,local_result2,local_result3)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=4 ) {
            local_result0 += xv[i+0] * xv[i+0];
            local_result1 += xv[i+1] * xv[i+1];
            local_result2 += xv[i+2] * xv[i+2];
            local_result3 += xv[i+3] * xv[i+3];
        }
		local_result += local_result0+local_result1+local_result2+local_result3;
	}
	else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1,local_result2,local_result3)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=4 ) {
			local_result0 += xv[i] * yv[i];
			local_result1 += xv[i+1] * yv[i+1];
			local_result2 += xv[i+2] * yv[i+2];
			local_result3 += xv[i+3] * yv[i+3];
		}
		local_result += local_result0+local_result1+local_result2+local_result3;
	}
	return local_result;
}

inline double ComputeDotProduct_2_unrolling(const local_int_t n, const double*xv, const double *yv) {

	double local_result = 0.0;
	double local_result0 = 0.0, local_result1=0.0;

	if (yv == xv) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=4 ) {
            local_result0 += xv[i+0] * xv[i+0];
            local_result1 += xv[i+1] * xv[i+1];
        }
		local_result += local_result0+local_result1;
	}
	else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction (+:local_result0,local_result1)
#endif //HPCG_NO_OPENMP
		for ( local_int_t i = 0; i < n; i+=2 ) {
			local_result0 += xv[i] * yv[i];
			local_result1 += xv[i+1] * yv[i+1];
		}
		local_result += local_result0+local_result1;
	}
	return local_result;
}
#endif //HPCG_MAN_OPT_DDOT
