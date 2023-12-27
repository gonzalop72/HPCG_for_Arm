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
  @file ComputeSPMV.cpp

  HPCG routine
  */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"
#include <cassert>
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#ifdef HPCG_USE_NEON
#include "arm_neon.h"
#endif
#ifdef HPCG_USE_SVE
#include "arm_sve.h"
#endif
#ifdef HPCG_USE_ARMPL_SPMV
#include "armpl_sparse.h"
#endif

/*!
  Routine to compute sparse matrix vector product y = Ax where:
Precondition: First call exchange_externals to get off-processor values of x

This routine calls the reference SpMV implementation by default, but
can be replaced by a custom, optimized routine suited for
the target system.

@param[in]  A the known system matrix
@param[in]  x the known vector
@param[out] y the On exit contains the result: Ax.

@return returns 0 upon success and non-zero otherwise

@see ComputeSPMV_ref
*/

/*#ifdef HPCG_MAN_OPT_SCHEDULE_ON*/
	//#define SCHEDULE(T)	schedule(T)
/*#elif defined(HPCG_MAN_SPVM_SCHEDULE)
	#define SCHEDULE(T) schedule(static,720)
/*#else
	#define SCHEDULE(T)
#endif*/
#ifdef HPCG_MAN_SPVM_SCHEDULE_720
	#define SCHEDULE(T) schedule(static,720)
#elif defined(HPCG_MAN_SPVM_SCHEDULE_528)
	#define SCHEDULE(T) schedule(static,720)
#else
	#define SCHEDULE(T)
#endif

#ifdef SPMV_2_UNROLL
#define ComputeSPMV_unroll2 ComputeSPMV_manual
#elif defined SPMV_4_UNROLL
#define ComputeSPMV_unroll4 ComputeSPMV_manual
#endif
inline void ComputeSPMV_manual(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow);

#ifdef TEST_SPMV_AS_TDG
//inline void test_tdg(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow);
#define test_tdg ComputeSPMV_manual
#endif
#ifdef TEST_SPMV_AS_TDG_REF
//inline void test_ref(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow);
#define test_ref ComputeSPMV_manual
#endif


int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

	assert(x.localLength >= A.localNumberOfColumns);
	assert(y.localLength >= A.localNumberOfRows);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif
	const double * const xv = x.values;
	double * const yv = y.values;
	const local_int_t nrow = A.localNumberOfRows;

#if defined(HPCG_MAN_OPT_SPMV_UNROLL)
	ComputeSPMV_manual(A, xv, yv, nrow);

#elif defined(HPCG_USE_NEON) && !defined(HPCG_USE_ARMPL_SPMV)
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
	for ( local_int_t i = 0; i < nrow-1; i+=2 ) {
		float64x2_t sum0 = vdupq_n_f64(0.0);
		float64x2_t sum1 = vdupq_n_f64(0.0);
		if ( A.nonzerosInRow[i] == A.nonzerosInRow[i+1] ) {
			local_int_t j = 0;
			for ( j = 0; j < A.nonzerosInRow[i]-1; j+=2 ) {
				float64x2_t values0 = vld1q_f64(&A.matrixValues[i  ][j]);
				float64x2_t values1 = vld1q_f64(&A.matrixValues[i+1][j]);

				float64x2_t xvValues0 = { xv[A.mtxIndL[i  ][j]], xv[A.mtxIndL[i  ][j+1]] };
				float64x2_t xvValues1 = { xv[A.mtxIndL[i+1][j]], xv[A.mtxIndL[i+1][j+1]] };

				sum0 = vfmaq_f64(sum0, values0, xvValues0);
				sum1 = vfmaq_f64(sum1, values1, xvValues1);

			}
			double s0 = vgetq_lane_f64(sum0, 0) + vgetq_lane_f64(sum0, 1);
			double s1 = vgetq_lane_f64(sum1, 0) + vgetq_lane_f64(sum1, 1);

			if ( j < A.nonzerosInRow[i] ) {
				s0 += A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
				s1 += A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
			}
			yv[i  ] = s0;
			yv[i+1] = s1;
		} else if ( A.nonzerosInRow[i] > A.nonzerosInRow[i+1] ) {
			local_int_t j = 0;
			for ( j = 0; j < A.nonzerosInRow[i+1]-1; j+=2 ) {
				float64x2_t values0 = vld1q_f64(&A.matrixValues[i  ][j]);
				float64x2_t values1 = vld1q_f64(&A.matrixValues[i+1][j]);

				float64x2_t xvValues0 = { xv[A.mtxIndL[i  ][j]], xv[A.mtxIndL[i  ][j+1]] };
				float64x2_t xvValues1 = { xv[A.mtxIndL[i+1][j]], xv[A.mtxIndL[i+1][j+1]] };

				sum0 = vfmaq_f64(sum0, values0, xvValues0);
				sum1 = vfmaq_f64(sum1, values1, xvValues1);

			}
			double s1 = vgetq_lane_f64(sum1, 0) + vgetq_lane_f64(sum1, 1);
			if ( j < A.nonzerosInRow[i+1] ) {
				s1 += A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
			}

			for ( ; j < A.nonzerosInRow[i]-1; j+=2 ) {
				float64x2_t values0 = vld1q_f64(&A.matrixValues[i  ][j]);

				float64x2_t xvValues0 = { xv[A.mtxIndL[i  ][j]], xv[A.mtxIndL[i  ][j+1]] };

				sum0 = vfmaq_f64(sum0, values0, xvValues0);
			}
			double s0 = vgetq_lane_f64(sum0, 0) + vgetq_lane_f64(sum0, 1);
			if ( j < A.nonzerosInRow[i] ) {
				s0 += A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
			}
			yv[i  ] = s0;
			yv[i+1] = s1;
		} else { // A.nonzerosInRow[i] < A.nonzerosInRow[i+1]
			local_int_t j = 0;
			for ( j = 0; j < A.nonzerosInRow[i]-1; j+=2 ) {
				float64x2_t values0 = vld1q_f64(&A.matrixValues[i  ][j]);
				float64x2_t values1 = vld1q_f64(&A.matrixValues[i+1][j]);

				float64x2_t xvValues0 = { xv[A.mtxIndL[i  ][j]], xv[A.mtxIndL[i  ][j+1]] };
				float64x2_t xvValues1 = { xv[A.mtxIndL[i+1][j]], xv[A.mtxIndL[i+1][j+1]] };

				sum0 = vfmaq_f64(sum0, values0, xvValues0);
				sum1 = vfmaq_f64(sum1, values1, xvValues1);

			}
			double s0 = vgetq_lane_f64(sum0, 0) + vgetq_lane_f64(sum0, 1);
			if ( j < A.nonzerosInRow[i] ) {
				s0 += A.matrixValues[i][j] * xv[A.mtxIndL[i][j]];
			}

			for ( ; j < A.nonzerosInRow[i+1]-1; j+=2 ) {
				float64x2_t values1 = vld1q_f64(&A.matrixValues[i+1][j]);

				float64x2_t xvValues1 = { xv[A.mtxIndL[i+1][j]], xv[A.mtxIndL[i+1][j+1]] };

				sum1 = vfmaq_f64(sum1, values1, xvValues1);
			}
			double s1 = vgetq_lane_f64(sum1, 0) + vgetq_lane_f64(sum1, 1);
			if ( j < A.nonzerosInRow[i+1] ) {
				s1 += A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
			}
			yv[i  ] = s0;
			yv[i+1] = s1;
		}
	}
#elif defined(HPCG_USE_SVE) && !defined(HPCG_USE_ARMPL_SPMV)

	if ( nrow % 4 == 0 ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < nrow-3; i+=4 ) {
			local_int_t maxnnz01 = A.nonzerosInRow[i  ] > A.nonzerosInRow[i+1] ? A.nonzerosInRow[i  ] : A.nonzerosInRow[i+1];
			local_int_t maxnnz23 = A.nonzerosInRow[i+2] > A.nonzerosInRow[i+3] ? A.nonzerosInRow[i+2] : A.nonzerosInRow[i+3];
			local_int_t maxnnz = maxnnz01 > maxnnz23 ? maxnnz01 : maxnnz23;
			svfloat64_t svsum0 = svdup_f64(0.0);
			svfloat64_t svsum1 = svdup_f64(0.0);
			svfloat64_t svsum2 = svdup_f64(0.0);
			svfloat64_t svsum3 = svdup_f64(0.0);
			for ( local_int_t j = 0; j < maxnnz; j += svcntd() ) {
				svbool_t pg0 = svwhilelt_b64(j, A.nonzerosInRow[i  ]);
				svbool_t pg1 = svwhilelt_b64(j, A.nonzerosInRow[i+1]);
				svbool_t pg2 = svwhilelt_b64(j, A.nonzerosInRow[i+2]);
				svbool_t pg3 = svwhilelt_b64(j, A.nonzerosInRow[i+3]);
				svfloat64_t values0 = svld1_f64(pg0, &A.matrixValues[i  ][j]);
				svfloat64_t values1 = svld1_f64(pg1, &A.matrixValues[i+1][j]);
				svfloat64_t values2 = svld1_f64(pg2, &A.matrixValues[i+2][j]);
				svfloat64_t values3 = svld1_f64(pg3, &A.matrixValues[i+3][j]);
				svuint64_t indices0 = svld1sw_u64(pg0, &A.mtxIndL[i  ][j]);
				svuint64_t indices1 = svld1sw_u64(pg1, &A.mtxIndL[i+1][j]);
				svuint64_t indices2 = svld1sw_u64(pg2, &A.mtxIndL[i+2][j]);
				svuint64_t indices3 = svld1sw_u64(pg3, &A.mtxIndL[i+3][j]);

				svfloat64_t svxv0 = svld1_gather_u64index_f64(pg0, xv, indices0);
				svfloat64_t svxv1 = svld1_gather_u64index_f64(pg1, xv, indices1);
				svfloat64_t svxv2 = svld1_gather_u64index_f64(pg2, xv, indices2);
				svfloat64_t svxv3 = svld1_gather_u64index_f64(pg3, xv, indices3);
				svsum0 = svmla_f64_m(pg0, svsum0, values0, svxv0);
				svsum1 = svmla_f64_m(pg1, svsum1, values1, svxv1);
				svsum2 = svmla_f64_m(pg2, svsum2, values2, svxv2);
				svsum3 = svmla_f64_m(pg3, svsum3, values3, svxv3);
			}
			yv[i  ] = svaddv(svptrue_b64(), svsum0);
			yv[i+1] = svaddv(svptrue_b64(), svsum1);
			yv[i+2] = svaddv(svptrue_b64(), svsum2);
			yv[i+3] = svaddv(svptrue_b64(), svsum3);
		}
	} else if ( nrow % 2 == 0 ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < nrow-1; i+=2 ) {
			local_int_t maxnnz = A.nonzerosInRow[i] > A.nonzerosInRow[i+1] ? A.nonzerosInRow[i] : A.nonzerosInRow[i+1];
			svfloat64_t svsum0 = svdup_f64(0.0);
			svfloat64_t svsum1 = svdup_f64(0.0);
			for ( local_int_t j = 0; j < maxnnz; j += svcntd() ) {
				svbool_t pg0 = svwhilelt_b64(j, A.nonzerosInRow[i]);
				svbool_t pg1 = svwhilelt_b64(j, A.nonzerosInRow[i+1]);
				svfloat64_t values0 = svld1_f64(pg0, &A.matrixValues[i][j]);
				svfloat64_t values1 = svld1_f64(pg1, &A.matrixValues[i+1][j]);
				svuint64_t indices0 = svld1sw_u64(pg0, &A.mtxIndL[i][j]);
				svuint64_t indices1 = svld1sw_u64(pg1, &A.mtxIndL[i+1][j]);

				svfloat64_t svxv0 = svld1_gather_u64index_f64(pg0, xv, indices0);
				svfloat64_t svxv1 = svld1_gather_u64index_f64(pg1, xv, indices1);
				svsum0 = svmla_f64_m(pg0, svsum0, values0, svxv0);
				svsum1 = svmla_f64_m(pg1, svsum1, values1, svxv1);
			}
			yv[i] = svaddv(svptrue_b64(), svsum0);
			yv[i+1] = svaddv(svptrue_b64(), svsum1);
		}
	} else {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < nrow; i++ ) {
			local_int_t maxnnz = A.nonzerosInRow[i];
			svfloat64_t svsum = svdup_f64(0.0);
			for ( local_int_t j = 0; j < maxnnz; j += svcntd() ) {
				svbool_t pg = svwhilelt_b64(j, maxnnz);
				svfloat64_t values = svld1_f64(pg, &A.matrixValues[i][j]);
				svuint64_t indices = svld1sw_u64(pg, &A.mtxIndL[i][j]);

				svfloat64_t svxv = svld1_gather_u64index_f64(pg, xv, indices);
				svsum = svmla_f64_m(pg, svsum, values, svxv);
			}
			yv[i] = svaddv(svptrue_b64(), svsum);
		}
	}
#elif defined(HPCG_USE_ARMPL_SPMV)
	double alpha = 1.0;
	double beta = 0.0;

	armpl_spmv_exec_d(ARMPL_SPARSE_OPERATION_NOTRANS, alpha, A.armpl_mat, xv, beta, yv);

#else
//#pragma statement scache_isolate_way L2=10
//#pragma statement scache_isolate_assign xv

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
//#pragma loop nounroll
	for ( local_int_t i = 0; i < nrow; i++ ) {
		double sum = 0.0;
		//#pragma fj loop zfill
	//#pragma loop nounroll
		for ( local_int_t j = 0; j < A.nonzerosInRow[i]; j++ ) {
			local_int_t curCol = A.mtxIndL[i][j];
			sum += A.matrixValues[i][j] * xv[curCol];
		}
		yv[i] = sum;
	}
	//#pragma statement end_scache_isolate_assign
	//#pragma statement end_scache_isolate_way
#endif

	return 0;
}

#ifdef HPCG_MAN_OPT_SPMV_UNROLL
inline void ComputeSPMV_unroll2(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow) {
//#pragma fj loop zfill
//#pragma statement scache_isolate_way L2=10
//#pragma statement scache_isolate_assign xv

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
//#pragma loop nounroll_and_jam
	for ( local_int_t i = 0; i < nrow-1; i+=2 ) {
		double sum0 = 0.0;
		double sum1 = 0.0;
		if (A.nonzerosInRow[i] == A.nonzerosInRow[i+1]) {
			for ( local_int_t j = 0; j < A.nonzerosInRow[i]; ++j ) {
				local_int_t curCol0 = A.mtxIndL[i][j];
				local_int_t curCol1 = A.mtxIndL[i+1][j];

				sum0 += A.matrixValues[i][j] * xv[curCol0];
				sum1 += A.matrixValues[i+1][j] * xv[curCol1];
			}
		}
		else {
			local_int_t min0 = A.nonzerosInRow[i] < A.nonzerosInRow[i+1] ? A.nonzerosInRow[i] : A.nonzerosInRow[i+1];
			for ( local_int_t j = 0; j < min0; ++j ) {
				local_int_t curCol0 = A.mtxIndL[i][j];
				local_int_t curCol1 = A.mtxIndL[i+1][j];

				sum0 += A.matrixValues[i][j] * xv[curCol0];
				sum1 += A.matrixValues[i+1][j] * xv[curCol1];
			}

			for ( local_int_t j = min0; j < A.nonzerosInRow[i]; ++j ) {
				local_int_t curCol = A.mtxIndL[i][j];
				sum0 += A.matrixValues[i][j] * xv[curCol];
			}
			for ( local_int_t j = min0; j < A.nonzerosInRow[i+1]; ++j ) {
				local_int_t curCol = A.mtxIndL[i+1][j];
				sum1 += A.matrixValues[i+1][j] * xv[curCol];
			}
		}
		yv[i] = sum0;
		yv[i+1] = sum1;
	}
	//#pragma statement end_scache_isolate_assign
	//#pragma statement end_scache_isolate_way
}

inline void ComputeSPMV_unroll4(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow) {

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
	//#pragma statement scache_isolate_way L2=10
	//#pragma statement scache_isolate_assign xv
	#pragma loop nounroll_and_jam
	for ( local_int_t i = 0; i < nrow-3; i+=4) {
		double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;

		if ((A.nonzerosInRow[i] == A.nonzerosInRow[i+1]) && (A.nonzerosInRow[i] == A.nonzerosInRow[i+2]) && (A.nonzerosInRow[i] == A.nonzerosInRow[i+3]) ){
			#pragma fj loop zfill
			#pragma loop nounroll
			for ( local_int_t j = 0; j < A.nonzerosInRow[i]; ++j ) {
				local_int_t curCol0 = A.mtxIndL[i][j];
				local_int_t curCol1 = A.mtxIndL[i+1][j];
				local_int_t curCol2 = A.mtxIndL[i+2][j];
				local_int_t curCol3 = A.mtxIndL[i+3][j];

				sum0 += A.matrixValues[i][j] * xv[curCol0];
				sum1 += A.matrixValues[i+1][j] * xv[curCol1];
				sum2 += A.matrixValues[i+2][j] * xv[curCol2];
				sum3 += A.matrixValues[i+3][j] * xv[curCol3];
			}

			yv[i] = sum0;
			yv[i+1] = sum1;
			yv[i+2] = sum2;
			yv[i+3] = sum3;
		}
		else {
			#pragma fj loop zfill			
			for ( local_int_t ix = 0; ix < 4; ++ix) {
				double sum = 0.0;
				for ( local_int_t j = 0; j < A.nonzerosInRow[i+ix]; ++j ) {
					local_int_t curCol = A.mtxIndL[i+ix][j];
					sum += A.matrixValues[i+ix][j] * xv[curCol];
				}
				yv[i+ix] = sum;
			}
		}
	}
	//#pragma statement end_scache_isolate_assign
    //#pragma statement end_scache_isolate_way	
}
#endif //HPCG_MAN_OPT_SPMV_UNROLL

inline void test_ref(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow) {

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
	//#pragma loop nounroll
	for ( local_int_t i = 0; i < nrow; i++ ) {
		double sum = 0.0;
		//#pragma fj loop zfill
		//#pragma loop nounroll
		for ( local_int_t j = 0; j < A.nonzerosInRow[i]; j++ ) {
			local_int_t curCol = A.mtxIndL[i][j];
			sum += A.matrixValues[i][j] * xv[curCol];
		}
		yv[i] = sum;
	}
}

inline void test_tdg(const SparseMatrix & A, const double * const xv, double * const yv,	const local_int_t nrow) {

#ifndef HPCG_NO_OPENMP
#pragma omp parallel
{
#endif
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp for SCHEDULE(runtime)
#endif
		//#pragma loop nounroll
		for ( local_int_t ix = 0; ix < A.tdg[l].size(); ix++ ) {
			local_int_t i = A.tdg[l][ix];
			double sum = 0.0;

			//#pragma fj loop zfill
			//#pragma loop nounroll
			for ( local_int_t j = 0; j < A.nonzerosInRow[i]; j++ ) {
				local_int_t curCol = A.mtxIndL[i][j];
				sum += A.matrixValues[i][j] * xv[curCol];
			}
			yv[i] = sum;
		}
	}
#ifndef HPCG_NO_OPENMP
}
#endif	
}
