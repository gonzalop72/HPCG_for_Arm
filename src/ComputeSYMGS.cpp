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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"
#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif

#include "likwid_instrumentation.hpp"

#ifdef HPCG_MAN_OPT_SCHEDULE_ON
	#define SCHEDULE(T)	schedule(T)
#else
	#define SCHEDULE(T)
#endif

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/* SVE IMPLEMENTATIONS                                                                            */
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/

#include "arm_sve.h"
#ifdef HPCG_USE_SVE
#include "arm_sve.h"

inline void SYMGS_VERSION_1(const SparseMatrix& A, double * const& xv, const double * const& rv);	//UNROLL-2
inline void SYMGS_VERSION_2(const SparseMatrix& A, double * const& xv, const double * const& rv);	//UNROLL-2 V2
inline void SYMGS_VERSION_3(const SparseMatrix& A, double * const& xv, const double * const& rv);	//UNROLL-4 - OPTIMUM
inline void SYMGS_VERSION_4(const SparseMatrix& A, double * const& xv, const double * const& rv);	//UNROLL-6

/*
 * TDG VERSION
 */
int ComputeSYMGS_TDG_SVE(const SparseMatrix & A, const Vector & r, Vector & x, TraceData &trace) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

LIKWID_START(trace.enabled, "symgs_tdg");

#ifndef TEST_XX
SYMGS_VERSION_3(A, xv, rv);
#else

//#pragma statement scache_isolate_way L2=10
//#pragma statement scache_isolate_assign xv
	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {

		local_int_t totalSize = A.tdg[l].size();
		local_int_t size1 = 2*(totalSize/2);
		//#pragma loop nounroll
		//#pragma loop nounroll_and_jam
		//if((A.tdg[l].size()%2) == 0) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel
{
#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < size1; i+=2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i+1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);

			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_2 = svdup_f64(0.0);
			
			const int maxNumberOfNonzeros = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);

			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);

				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double sum_1 = rv[row_1] - totalContribution_1;

			sum_1 += xv[row_1] * currentDiagonal_1;
			xv[row_1] = sum_1 / currentDiagonal_1;

			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_2] = sum_2 / currentDiagonal_2;
		}
		//}
		//else
		//{
#ifndef HPCG_NO_OPENMP
//#pragma omp parallel for SCHEDULE(runtime)
#pragma omp single 
{
#endif
		if (size1 < totalSize) {
			local_int_t i = size1;
		//for ( local_int_t i = size1; i < totalSize; i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		//}
		}
#ifndef HPCG_NO_OPENMP
}
}
#endif
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}

/*#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = size1-1; i >= 0; i-= 2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i-1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);
			svfloat64_t contribs_2 = svdup_f64(0.0);

			//#pragma loop nounroll
			//#pragma loop nounroll_and_jam
			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}*/
	}
//#pragma statement end_scache_isolate_assign
//#pragma statement end_scache_isolate_way

#endif //TEST_XX

LIKWID_STOP(trace.enabled, "symgs_tdg");

	return 0;
}
/*
 * END OF TDG VERSION
 */

/*
 * TDG FUSED SYMGS-SPMV VERSION
 */
int ComputeFusedSYMGS_SPMV_SVE(const SparseMatrix & A, const Vector & r, Vector & x, Vector & y, TraceData& trace) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;
	double * const yv = y.values;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = A.tdg[l].size(); i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			totalContribution -= xv[row] * currentDiagonal; // remove diagonal contribution
			double sum = rv[row] - totalContribution; // substract contributions from RHS
			xv[row] = sum / currentDiagonal; // update row

			// SPMV part
			totalContribution += xv[row] * currentDiagonal; // add updated diagonal contribution
			yv[row] = totalContribution; // update SPMV output vector
			
		}
	}

	return 0;
}
/*
 * END OF TDG FUSED SYMGS-SPMV VERSION
 */

/*
 * BLOCK COLORED VERSION
 */
int ComputeSYMGS_BLOCK_SVE(const SparseMatrix & A, const Vector & r, Vector & x, TraceData& trace ) {
	assert(x.localLength >= A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;
	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];

LIKWID_START(trace.enabled, "symgs_bc");		

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) { // for each color
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) { // for each superblock with the same color
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize * A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) { // for each chunk of this super block
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;
				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues3 = A.matrixValues[i+3];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal2 = matrixDiagonal[i+2][0];
					const double currentDiagonal3 = matrixDiagonal[i+3][0];

					svfloat64_t contribs0 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs2 = svdup_f64(0.0);
					svfloat64_t contribs3 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values2 = svld1_f64(pg, &currentValues2[j]);
						svfloat64_t values3 = svld1_f64(pg, &currentValues3[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices2 = svld1sw_u64(pg, &currentColIndices2[j]);
						svuint64_t indices3 = svld1sw_u64(pg, &currentColIndices3[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv2 = svld1_gather_u64index_f64(pg, xv, indices2);
						svfloat64_t xvv3 = svld1_gather_u64index_f64(pg, xv, indices3);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1);
						contribs2 = svmla_f64_m(pg, contribs2, xvv2, values2);
						contribs3 = svmla_f64_m(pg, contribs3, xvv3, values3);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution2 = svaddv_f64(svptrue_b64(), contribs2);
					double totalContribution3 = svaddv_f64(svptrue_b64(), contribs3);

					double sum0 = rv[i  ] - totalContribution0;
					double sum1 = rv[i+1] - totalContribution1;
					double sum2 = rv[i+2] - totalContribution2;
					double sum3 = rv[i+3] - totalContribution3;

					sum0 += xv[i  ] * currentDiagonal0;
					sum1 += xv[i+1] * currentDiagonal1;
					sum2 += xv[i+2] * currentDiagonal2;
					sum3 += xv[i+3] * currentDiagonal3;

					xv[i  ] = sum0 / currentDiagonal0;
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i+2] = sum2 / currentDiagonal2;
					xv[i+3] = sum3 / currentDiagonal3;
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];

					svfloat64_t contribs0 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);

					double sum0 = rv[i  ] - totalContribution0;
					double sum1 = rv[i+1] - totalContribution1;

					sum0 += xv[i  ] * currentDiagonal0;
					sum1 += xv[i+1] * currentDiagonal1;

					xv[i  ] = sum0 / currentDiagonal0;
					xv[i+1] = sum1 / currentDiagonal1;
				} else { //A.chunkSize == 1
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0);
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum0 = rv[i  ] - totalContribution0;

					sum0 += xv[i  ] * currentDiagonal0;

					xv[i  ] = sum0 / currentDiagonal0;
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) {
			local_int_t firstRow = ((block+1) * A.blockSize) - 1;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow - A.blockSize * A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues3 = A.matrixValues[i+3];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal3 = matrixDiagonal[i+3][0];
					const double currentDiagonal2 = matrixDiagonal[i+2][0];
					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs3 = svdup_f64(0.0);
					svfloat64_t contribs2 = svdup_f64(0.0);
					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values3 = svld1_f64(pg, &currentValues3[j]);
						svfloat64_t values2 = svld1_f64(pg, &currentValues2[j]);
						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices3 = svld1sw_u64(pg, &currentColIndices3[j]);
						svuint64_t indices2 = svld1sw_u64(pg, &currentColIndices2[j]);
						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv3 = svld1_gather_u64index_f64(pg, xv, indices3);
						svfloat64_t xvv2 = svld1_gather_u64index_f64(pg, xv, indices2);
						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs3 = svmla_f64_m(pg, contribs3, xvv3, values3 );
						contribs2 = svmla_f64_m(pg, contribs2, xvv2, values2 );
						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1 );
						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution3 = svaddv_f64(svptrue_b64(), contribs3);
					double totalContribution2 = svaddv_f64(svptrue_b64(), contribs2);
					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum3 = rv[i+3] - totalContribution3;
					double sum2 = rv[i+2] - totalContribution2;
					double sum1 = rv[i+1] - totalContribution1;
					double sum0 = rv[i  ] - totalContribution0;

					sum3 += xv[i+3] * currentDiagonal3;
					sum2 += xv[i+2] * currentDiagonal2;
					sum1 += xv[i+1] * currentDiagonal1;
					sum0 += xv[i  ] * currentDiagonal0;
					
					xv[i+3] = sum3 / currentDiagonal3;
					xv[i+2] = sum2 / currentDiagonal2;
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i  ] = sum0 / currentDiagonal0;
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal1 = matrixDiagonal[i+1][0];
					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs1 = svdup_f64(0.0);
					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values1 = svld1_f64(pg, &currentValues1[j]);
						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices1 = svld1sw_u64(pg, &currentColIndices1[j]);
						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv1 = svld1_gather_u64index_f64(pg, xv, indices1);
						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs1 = svmla_f64_m(pg, contribs1, xvv1, values1 );
						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution1 = svaddv_f64(svptrue_b64(), contribs1);
					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum1 = rv[i+1] - totalContribution1;
					double sum0 = rv[i  ] - totalContribution0;

					sum1 += xv[i+1] * currentDiagonal1;
					sum0 += xv[i  ] * currentDiagonal0;
					
					xv[i+1] = sum1 / currentDiagonal1;
					xv[i  ] = sum0 / currentDiagonal0;
				} else { // A.chunkSize == 1
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal0 = matrixDiagonal[i  ][0];

					svfloat64_t contribs0 = svdup_f64(0.0);

					for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd() ) {
						svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);

						svfloat64_t values0 = svld1_f64(pg, &currentValues0[j]);

						svuint64_t indices0 = svld1sw_u64(pg, &currentColIndices0[j]);

						svfloat64_t xvv0 = svld1_gather_u64index_f64(pg, xv, indices0);

						contribs0 = svmla_f64_m(pg, contribs0, xvv0, values0 );
					}

					double totalContribution0 = svaddv_f64(svptrue_b64(), contribs0);

					double sum0 = rv[i  ] - totalContribution0;

					sum0 += xv[i  ] * currentDiagonal0;
					
				}
			}
		}
	}
LIKWID_STOP(trace.enabled, "symgs_bc");			

	return 0;
}
/*
 * END OF BLOCK COLORED VERSION
 */
#elif defined(HPCG_USE_NEON)

/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/* NEON IMPLEMENTATIONS                                                                           */
/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/

#include "arm_neon.h"

/*
 * TDG VERSION
 */
int ComputeSYMGS_TDG_NEON(const SparseMatrix & A, const Vector & r, Vector & x) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	/*
	 * BACKWARD
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	return 0;
}
/*
 *
 */
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
 * TDG FUSED VERSION
 */
int ComputeFusedSYMGS_SPMV_NEON(const SparseMatrix & A, const Vector & r, Vector & x, Vector & y) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double * const yv = y.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			double sum = rv[row] - totalContribution;
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				sum -= currentValues[j] * xv[currentColIndices[j]];
			}
			sum += xv[row] * currentDiagonal; // remove diagonal contribution
			xv[row] = sum / currentDiagonal; // update row
		}
	}

	/*
	 * BACKWARD (fusing SYMGS and SPMV)
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			float64x2_t contribs = vdupq_n_f64(0.0);

			local_int_t j = 0;
			for ( j = 0; j < currentNumberOfNonzeros-1; j+=2 ) {
				// Load the needed j values
				float64x2_t mtxValues = vld1q_f64(&currentValues[j]);
				// Load the needed x values
				double aux[] = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };
				float64x2_t xvv = vld1q_f64(aux);
				// Add the contribution
				contribs = vfmaq_f64(contribs, mtxValues, xvv);
			}
			// reduce contributions
			double totalContribution = vgetq_lane_f64(contribs, 0) + vgetq_lane_f64(contribs, 1);
			// Add missing values from last loop
			if ( j < currentNumberOfNonzeros ) {
				totalContribution += currentValues[j] * xv[currentColIndices[j]];
			}
			totalContribution -= xv[row] * currentDiagonal; // remove diagonal contribution
			double sum = rv[row] - totalContribution; // substract contributions from RHS
			xv[row] = sum / currentDiagonal; // update row
			// Fusion part
			totalContribution += xv[row] * currentDiagonal; // add updated diagonal contribution
			yv[row] = totalContribution; // update SPMV output vector
		}
	}

	return 0;
}
/*
 *
 */
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/*
 * BLOCK COLORED VERSION
 */
int ComputeSYMGS_BLOCK_NEON(const SparseMatrix & A, const Vector & r, Vector & x) {

	assert(x.localLength >= A.localNumberOfColumns);
	
#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;

	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) { // for each color
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) { // for each super block with the same color
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize*A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) { // for each chunk of this super block
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues3 = A.matrixValues[i+3];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];

					const double currentDiagonal[4] = { matrixDiagonal[i  ][0],\
														matrixDiagonal[i+1][0],\
														matrixDiagonal[i+2][0],\
														matrixDiagonal[i+3][0]};
					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);
					float64x2_t diagonal23 = vld1q_f64(&currentDiagonal[2]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);
					float64x2_t contribs2 = vdupq_n_f64(0.0);
					float64x2_t contribs3 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i]);
					float64x2_t vrv23 = vld1q_f64(&rv[i+2]);

					float64x2_t vxv01 = vld1q_f64(&xv[i]);
					float64x2_t vxv23 = vld1q_f64(&xv[i+2]);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);
						float64x2_t values2 = vld1q_f64(&currentValues2[j]);
						float64x2_t values3 = vld1q_f64(&currentValues3[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };
						float64x2_t vxv2 = { xv[currentColIndices2[j]], xv[currentColIndices2[j+1]] };
						float64x2_t vxv3 = { xv[currentColIndices3[j]], xv[currentColIndices3[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
						contribs2 = vfmaq_f64(contribs2, values2, vxv2);
						contribs3 = vfmaq_f64(contribs3, values3, vxv3);
					}
					// Reduce contribution
					// First for i and i+1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Then for i+2 and i+3
					float64x2_t totalContribution23;
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs2), totalContribution23, 0);
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs3), totalContribution23, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);
					float64x2_t sum23 = vsubq_f64(vrv23, totalContribution23);

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j], currentValues1[j] };
						float64x2_t values23 = { currentValues2[j], currentValues3[j] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j]], xv[currentColIndices1[j]] };
						float64x2_t vx23 = { xv[currentColIndices2[j]], xv[currentColIndices3[j]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
						sum23 = vfmsq_f64(sum23, values23, vx23);
					}

					// Remove diagonal contribution and update rows i and i+1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];

					// Remove diagonal contribution and update rows i+2 and i+3
					sum23 = vfmaq_f64(sum23, vxv23, diagonal23);
					xv[i+2] = vgetq_lane_f64(sum23, 0) / currentDiagonal[2];
					xv[i+3] = vgetq_lane_f64(sum23, 1) / currentDiagonal[3];
				} else if ( A.chunkSize == 2 ) {
					const double * const currentValues0 = A.matrixValues[i  ];
					const double * const currentValues1 = A.matrixValues[i+1];

					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];

					const double currentDiagonal[2] = { matrixDiagonal[i  ][0],\
														matrixDiagonal[i+1][0]};
					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i]);

					float64x2_t vxv01 = vld1q_f64(&xv[i]);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
					}
					// Reduce contribution
					// First for i and i+1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j], currentValues1[j] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j]], xv[currentColIndices1[j]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i and i+1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
				} else { // A.chunkSize == 1
					const double * const currentValues = A.matrixValues[i];
					const local_int_t * const currentColIndices = A.mtxIndL[i];
					const double currentDiagonal = matrixDiagonal[i][0];
					float64x2_t contribs = vdupq_n_f64(0.0);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values = vld1q_f64(&currentValues[j]);

						// Load x
						float64x2_t vxv = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };

						// Add contribution
						contribs = vfmaq_f64(contribs, values, vxv);
					}
					// Reduce contribution
					// First for i and i+1
					double totalContribution;
					totalContribution = vaddvq_f64(contribs);

					// Substract contributions from RHS
					double sum = rv[i] - totalContribution;

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						sum -= currentValues[j] * xv[currentColIndices[j]];
					}

					// Remove diagonal contribution and update rows i and i+1
					sum += xv[i] * currentDiagonal;
					xv[i] = sum / currentDiagonal;
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) { // we skip a whole superblock on each iteration
			local_int_t firstRow = ((block+1) * A.blockSize) - 1; // this is the last row of the last block (i.e., next block first row - 1)
			local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
			local_int_t lastChunk = (firstRow - A.blockSize*A.chunkSize) / A.chunkSize; 

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				const int currentNumberOfNonzeros = A.nonzerosInChunk[chunk];
				if ( A.chunkSize == 4 ) {
					local_int_t i = last-1-3;

					const double * const currentValues3 = A.matrixValues[i+3];
					const double * const currentValues2 = A.matrixValues[i+2];
					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices3 = A.mtxIndL[i+3];
					const local_int_t * const currentColIndices2 = A.mtxIndL[i+2];
					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal[4] = {\
							matrixDiagonal[i  ][0],\
							matrixDiagonal[i+1][0],\
							matrixDiagonal[i+2][0],\
							matrixDiagonal[i+3][0]};

					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);
					float64x2_t diagonal23 = vld1q_f64(&currentDiagonal[2]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);
					float64x2_t contribs2 = vdupq_n_f64(0.0);
					float64x2_t contribs3 = vdupq_n_f64(0.0);

					float64x2_t vrv23 = vld1q_f64(&rv[i+2]);
					float64x2_t vrv01 = vld1q_f64(&rv[i  ]);

					float64x2_t vxv23 = vld1q_f64(&xv[i+2]);
					float64x2_t vxv01 = vld1q_f64(&xv[i  ]);

					local_int_t j = 0;
					for ( j = currentNumberOfNonzeros-2; j >= 0; j -= 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);
						float64x2_t values2 = vld1q_f64(&currentValues2[j]);
						float64x2_t values3 = vld1q_f64(&currentValues3[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };
						float64x2_t vxv2 = { xv[currentColIndices2[j]], xv[currentColIndices2[j+1]] };
						float64x2_t vxv3 = { xv[currentColIndices3[j]], xv[currentColIndices3[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
						contribs2 = vfmaq_f64(contribs2, values2, vxv2);
						contribs3 = vfmaq_f64(contribs3, values3, vxv3);
					}
					// Reduce contribution
					// First for i and i-1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Then for i-2 and i-3
					float64x2_t totalContribution23;
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs2), totalContribution23, 0);
					totalContribution23 = vsetq_lane_f64(vaddvq_f64(contribs3), totalContribution23, 1);

					// Substract contributions from RHS
					float64x2_t sum23 = vsubq_f64(vrv23, totalContribution23);
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j == -1 ) {
						// Load current values
						float64x2_t values23 = { currentValues2[j+1], currentValues3[j+1] };
						float64x2_t values01 = { currentValues0[j+1], currentValues1[j+1] };

						// Load x
						float64x2_t vx23 = { xv[currentColIndices2[j+1]], xv[currentColIndices3[j+1]] };
						float64x2_t vx01 = { xv[currentColIndices0[j+1]], xv[currentColIndices1[j+1]] };

						// Add contributions
						sum23 = vfmsq_f64(sum23, values23, vx23);
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i-2 and i-3
					sum23 = vfmaq_f64(sum23, vxv23, diagonal23);
					xv[i+3] = vgetq_lane_f64(sum23, 1) / currentDiagonal[3];
					xv[i+2] = vgetq_lane_f64(sum23, 0) / currentDiagonal[2];

					// Remove diagonal contribution and update rows i and i-1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
				} else if ( A.chunkSize == 2 ) {
					local_int_t i = last-1-1;

					const double * const currentValues1 = A.matrixValues[i+1];
					const double * const currentValues0 = A.matrixValues[i  ];

					const local_int_t * const currentColIndices1 = A.mtxIndL[i+1];
					const local_int_t * const currentColIndices0 = A.mtxIndL[i  ];

					const double currentDiagonal[2] = {\
							matrixDiagonal[i  ][0],\
							matrixDiagonal[i+1][0]};

					float64x2_t diagonal01 = vld1q_f64(&currentDiagonal[0]);

					float64x2_t contribs0 = vdupq_n_f64(0.0);
					float64x2_t contribs1 = vdupq_n_f64(0.0);

					float64x2_t vrv01 = vld1q_f64(&rv[i  ]);

					float64x2_t vxv01 = vld1q_f64(&xv[i  ]);

					local_int_t j = 0;
					for ( j = currentNumberOfNonzeros-2; j >= 0; j -= 2 ) {
						// Load values
						float64x2_t values0 = vld1q_f64(&currentValues0[j]);
						float64x2_t values1 = vld1q_f64(&currentValues1[j]);

						// Load x
						float64x2_t vxv0 = { xv[currentColIndices0[j]], xv[currentColIndices0[j+1]] };
						float64x2_t vxv1 = { xv[currentColIndices1[j]], xv[currentColIndices1[j+1]] };

						// Add contribution
						contribs0 = vfmaq_f64(contribs0, values0, vxv0);
						contribs1 = vfmaq_f64(contribs1, values1, vxv1);
					}
					// Reduce contribution
					// First for i and i-1
					float64x2_t totalContribution01;
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs0), totalContribution01, 0);
					totalContribution01 = vsetq_lane_f64(vaddvq_f64(contribs1), totalContribution01, 1);

					// Substract contributions from RHS
					float64x2_t sum01 = vsubq_f64(vrv01, totalContribution01);

					// Add contributions from missing elements (if any)
					if ( j == -1 ) {
						// Load current values
						float64x2_t values01 = { currentValues0[j+1], currentValues1[j+1] };

						// Load x
						float64x2_t vx01 = { xv[currentColIndices0[j+1]], xv[currentColIndices1[j+1]] };

						// Add contributions
						sum01 = vfmsq_f64(sum01, values01, vx01);
					}

					// Remove diagonal contribution and update rows i and i-1
					sum01 = vfmaq_f64(sum01, vxv01, diagonal01);
					xv[i+1] = vgetq_lane_f64(sum01, 1) / currentDiagonal[1];
					xv[i  ] = vgetq_lane_f64(sum01, 0) / currentDiagonal[0];
				} else { // A.chunkSize == 1
					local_int_t i = last - 1; // == first
					const double * const currentValues = A.matrixValues[i];
					const local_int_t * const currentColIndices = A.mtxIndL[i];
					const double currentDiagonal = matrixDiagonal[i][0];

					float64x2_t contribs = vdupq_n_f64(0.0);

					local_int_t j = 0;
					for ( j = 0; j < currentNumberOfNonzeros-1; j += 2 ) {
						// Load values
						float64x2_t values = vld1q_f64(&currentValues[j]);

						// Load x
						float64x2_t vxv = { xv[currentColIndices[j]], xv[currentColIndices[j+1]] };

						// Add contribution
						contribs = vfmaq_f64(contribs, values, vxv);
					}
					// Reduce contribution
					double totalContribution = vaddvq_f64(contribs);

					// Substract contribution from RHS
					double sum = rv[i] - totalContribution;

					// Add contributions from missing elements (if any)
					if ( j < currentNumberOfNonzeros ) {
						sum -= currentValues[j] * xv[currentColIndices[j]];
					}

					// Remove diagonal contribution and updated row i
					sum += xv[i] * currentDiagonal;
					xv[i] = sum / currentDiagonal;
				}
			}
		}
	}

	return 0;
}
/*
 *
 */
#endif
//#else // !HPCG_USE_SVE ! HPCG_USE_NEON

int ComputeFusedSYMGS_SPMV ( const SparseMatrix & A, const Vector & r, Vector & x, Vector & y ) {
	assert(x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double * const yv = y.values;
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD (fusing SYMGS and SPMV)
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = 0.0;

			for ( local_int_t j = currentNumberOfNonzeros-1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum += currentValues[j] * xv[curCol];
			}
			sum -= xv[row] * currentDiagonal;
			xv[row] = (rv[row] - sum) / currentDiagonal;
			sum += xv[row] * currentDiagonal;
			yv[row] = sum;
		}
	}

	return 0;
}

int ComputeSYMGS_TDG ( const SparseMatrix & A, const Vector & r, Vector & x, TraceData& trace ) {

	assert( x.localLength == A.localNumberOfColumns);

#ifndef HPCG_NO_MPI
	ExchangeHalo(A,x);
#endif

	const double * const rv = r.values;
	double * const xv = x.values;
	double **matrixDiagonal = A.matrixDiagonal;

/*#ifndef HPCG_NO_OPENMP
#pragma omp parallel SCHEDULE(runtime)
{
#endif
*/
#pragma statement scache_isolate_way L2=10
#pragma statement scache_isolate_assign xv

	/*
	 * FORWARD
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		#pragma loop unroll_and_jam(4)
		for ( local_int_t i = 0; i < A.tdg[l].size(); i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j++ ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	/*
	 * BACKWARD
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		#pragma loop unroll_and_jam(4)
		for ( local_int_t i = A.tdg[l].size()-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			double sum = rv[row];

			for ( local_int_t j = currentNumberOfNonzeros-1; j >= 0; j-- ) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
	}

	#pragma statement end_scache_isolate_assign
	#pragma statement end_scache_isolate_way
/*#ifndef HPCG_NO_OPENMP
}
#endif*/

	return 0;
}

int ComputeSYMGS_BLOCK( const SparseMatrix & A, const Vector & r, Vector & x, TraceData& trace ) {

	assert(x.localLength >= A.localNumberOfColumns);
	
#ifndef HPCG_NO_MPI
	ExchangeHalo(A, x);
#endif

	const local_int_t nrow = A.localNumberOfRows;
	double **matrixDiagonal = A.matrixDiagonal;
	const double * const rv = r.values;
	double * const xv = x.values;

	local_int_t firstBlock = 0;
	local_int_t lastBlock = firstBlock + A.numberOfBlocksInColor[0];
	/*
	 * FORWARD
	 */
	for ( local_int_t color = 0; color < A.numberOfColors; color++ ) {
		if ( color > 0 ) {
			firstBlock += A.numberOfBlocksInColor[color-1];
			lastBlock = firstBlock + A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block < lastBlock; block += A.chunkSize ) {
			local_int_t firstRow = block * A.blockSize;
			local_int_t firstChunk = firstRow / A.chunkSize;
			local_int_t lastChunk = (firstRow + A.blockSize*A.chunkSize) / A.chunkSize;

			for ( local_int_t chunk = firstChunk; chunk < lastChunk; chunk++ ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = first; i < last; i+= (A.chunkSize/2)) {
				local_int_t i = first;
				if ( A.chunkSize == 4 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];
					double sum2 = rv[i+2];
					double sum3 = rv[i+3];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
						sum2 -= A.matrixValues[i+2][j] * xv[A.mtxIndL[i+2][j]];
						sum3 -= A.matrixValues[i+3][j] * xv[A.mtxIndL[i+3][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
					sum2 += matrixDiagonal[i+2][0] * xv[i+2];
					xv[i+2] = sum2 / matrixDiagonal[i+2][0];
					sum3 += matrixDiagonal[i+3][0] * xv[i+3];
					xv[i+3] = sum3 / matrixDiagonal[i+3][0];
				} else if ( A.chunkSize == 2 ) {
					double sum0 = rv[i+0];
					double sum1 = rv[i+1];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
						sum1 -= A.matrixValues[i+1][j] * xv[A.mtxIndL[i+1][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
					sum1 += matrixDiagonal[i+1][0] * xv[i+1];
					xv[i+1] = sum1 / matrixDiagonal[i+1][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i+0];

					for ( local_int_t j = 0; j < A.nonzerosInChunk[chunk]; j++ ) {
						sum0 -= A.matrixValues[i+0][j] * xv[A.mtxIndL[i+0][j]];
					}
					sum0 += matrixDiagonal[i+0][0] * xv[i+0];
					xv[i+0] = sum0 / matrixDiagonal[i+0][0];
				}
			}
		}
	}

	firstBlock = A.numberOfBlocks-1;
	lastBlock = firstBlock - A.numberOfBlocksInColor[A.numberOfColors-1];
	/*
	 * BACKWARD
	 */
	for ( local_int_t color = A.numberOfColors-1; color >= 0; color-- ) {
		if ( color < A.numberOfColors-1 ) {
			firstBlock -= A.numberOfBlocksInColor[color+1];
			lastBlock = firstBlock - A.numberOfBlocksInColor[color];
		}
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t block = firstBlock; block > lastBlock; block -= A.chunkSize ) {
			local_int_t firstRow = ((block+1) * A.blockSize) - 1; // this is the last row of the last block
			local_int_t firstChunk = firstRow / A.chunkSize; // this is the  chunk of the row above
			local_int_t lastChunk = (firstRow - A.blockSize*A.chunkSize) / A.chunkSize; 

			for ( local_int_t chunk = firstChunk; chunk > lastChunk; chunk-- ) {
				local_int_t first = A.chunkSize * chunk;
				local_int_t last = first + A.chunkSize;

				//for ( local_int_t i = last-1; i >= first; i -= (A.chunkSize/2)) {
				local_int_t i = last-1;
				if ( A.chunkSize == 4 ) {
					double sum3 = rv[i-3];
					double sum2 = rv[i-2];
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum3 -= A.matrixValues[i-3][j] * xv[A.mtxIndL[i-3][j]];
						sum2 -= A.matrixValues[i-2][j] * xv[A.mtxIndL[i-2][j]];
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}
					sum3 += matrixDiagonal[i-3][0] * xv[i-3];
					xv[i-3] = sum3 / matrixDiagonal[i-3][0];

					sum2 += matrixDiagonal[i-2][0] * xv[i-2];
					xv[i-2] = sum2 / matrixDiagonal[i-2][0];

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else if ( A.chunkSize == 2 ) {
					double sum1 = rv[i-1];
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum1 -= A.matrixValues[i-1][j] * xv[A.mtxIndL[i-1][j]];
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum1 += matrixDiagonal[i-1][0] * xv[i-1];
					xv[i-1] = sum1 / matrixDiagonal[i-1][0];

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				} else { // A.chunkSize == 1
					double sum0 = rv[i  ];

					for ( local_int_t j = A.nonzerosInChunk[chunk]-1; j >= 0; j-- ) {
						sum0 -= A.matrixValues[i  ][j] * xv[A.mtxIndL[i  ][j]];
					}

					sum0 += matrixDiagonal[i  ][0] * xv[i  ];
					xv[i  ] = sum0 / matrixDiagonal[i  ][0];
				}
			}
		}
	}

	return 0;
}
//#endif



/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x, TraceData& trace) {

	// This function is just a stub right now which decides which implementation of the SYMGS will be executed (TDG or block coloring)
	if ( A.TDG ) {
#ifdef HPCG_USE_NEON
		return ComputeSYMGS_TDG_NEON(A, r, x);
#elif defined HPCG_USE_SVE
		return ComputeSYMGS_TDG_SVE(A, r, x, trace);
#else
		return ComputeSYMGS_TDG(A, r, x, trace);
#endif
	}
#ifdef HPCG_USE_NEON
	return ComputeSYMGS_BLOCK_NEON(A, r, x);
#elif defined HPCG_USE_SVE
	return ComputeSYMGS_BLOCK_SVE(A, r, x, trace);
#else
	return ComputeSYMGS_BLOCK(A, r, x, trace);
#endif
}

inline void SYMGS_VERSION_1(const SparseMatrix& A, double * const& xv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		if((tdgLevelSize%2) == 0) {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < tdgLevelSize; i+=2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i+1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);

			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_2 = svdup_f64(0.0);
			
			const int maxNumberOfNonzeros = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);

			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);

				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double sum_1 = rv[row_1] - totalContribution_1;

			sum_1 += xv[row_1] * currentDiagonal_1;
			xv[row_1] = sum_1 / currentDiagonal_1;

			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_2] = sum_2 / currentDiagonal_2;
		}
		}
		else
		{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < tdgLevelSize; i++ ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
		}
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		if((tdgLevelSize%2) == 0) {		
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = tdgLevelSize-1; i >= 0; i-= 2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i-1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);
			svfloat64_t contribs_2 = svdup_f64(0.0);

			const int maxNumberOfNonzeros = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);				
							
			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);
				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_1 = rv[row_1] - totalContribution_1;
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_1 += xv[row_1] * currentDiagonal_1;
			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_1] = sum_1 / currentDiagonal_1;
			xv[row_2] = sum_2 / currentDiagonal_2;
		}
		}
		else
		{
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for SCHEDULE(runtime)
#endif
		for ( local_int_t i = tdgLevelSize-1; i >= 0; i-- ) {
			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
		}
	}
}

inline void SYMGS_VERSION_2(const SparseMatrix& A, double * const& xv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 2*(tdgLevelSize / 2);

#ifndef HPCG_NO_OPENMP
	#pragma omp parallel
	{
	#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < maxLevelSize; i+=2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i+1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);

			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_2 = svdup_f64(0.0);
			
			const int maxNumberOfNonzeros = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);

			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);

				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double sum_1 = rv[row_1] - totalContribution_1;

			sum_1 += xv[row_1] * currentDiagonal_1;
			xv[row_1] = sum_1 / currentDiagonal_1;

			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_2] = sum_2 / currentDiagonal_2;
		}

		#pragma omp single 
		if (maxLevelSize < tdgLevelSize) {
			local_int_t i = maxLevelSize;

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
#ifndef HPCG_NO_OPENMP
	}
#endif
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 2*(tdgLevelSize / 2);

#ifndef HPCG_NO_OPENMP
#pragma omp parallel 
	{
		#pragma omp single nowait 
		{
#endif
		if (tdgLevelSize > maxLevelSize) {
			local_int_t i = maxLevelSize-1;

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
#ifndef HPCG_NO_OPENMP
		}
#pragma omp for SCHEDULE(runtime)
#endif
		for ( local_int_t i = maxLevelSize-1; i >= 0; i-= 2 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i-1];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);
			svfloat64_t contribs_2 = svdup_f64(0.0);

			const int maxNumberOfNonzeros = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);				
							
			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);
				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_1 = rv[row_1] - totalContribution_1;
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_1 += xv[row_1] * currentDiagonal_1;
			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_1] = sum_1 / currentDiagonal_1;
			xv[row_2] = sum_2 / currentDiagonal_2;
		}
#ifndef HPCG_NO_OPENMP
	}
#endif
	}
}

inline void SYMGS_VERSION_3(const SparseMatrix& A, double * const& prxv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;
	double *xv = prxv;

//#pragma statement scache_isolate_way L2=10
//#pragma statement scache_isolate_assign xv
#ifndef HPCG_NO_OPENMP
	#pragma omp parallel
	{
#endif
	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 4*(tdgLevelSize / 4);

#ifndef HPCG_NO_OPENMP
	//#pragma loop nounroll
	#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < maxLevelSize; i+=4 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i+1];
			local_int_t row_3 = A.tdg[l][i+2];
			local_int_t row_4 = A.tdg[l][i+3];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);

			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_2 = svdup_f64(0.0);

			const double * const currentValues_3 = A.matrixValues[row_3];
			const local_int_t * const currentColIndices_3 = A.mtxIndL[row_3];
			const int currentNumberOfNonzeros_3 = A.nonzerosInRow[row_3];
			const double currentDiagonal_3 = matrixDiagonal[row_3][0];
			svfloat64_t contribs_3 = svdup_f64(0.0);

			const double * const currentValues_4 = A.matrixValues[row_4];
			const local_int_t * const currentColIndices_4 = A.mtxIndL[row_4];
			const int currentNumberOfNonzeros_4 = A.nonzerosInRow[row_4];
			const double currentDiagonal_4 = matrixDiagonal[row_4][0];
			svfloat64_t contribs_4 = svdup_f64(0.0);

			const int maxNumberOfNonzeros1 = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);
			const int maxNumberOfNonzeros2 = std::max(currentNumberOfNonzeros_3, currentNumberOfNonzeros_4);
			const int maxNumberOfNonzeros = std::max(maxNumberOfNonzeros1, maxNumberOfNonzeros2);

			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);

				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);

				svbool_t pg_3 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_3);
				svfloat64_t mtxValues_3 = svld1_f64(pg_3, &currentValues_3[j]);
				svuint64_t indices_3 = svld1sw_u64(pg_3, &currentColIndices_3[j]);
				svfloat64_t xvv_3 = svld1_gather_u64index_f64(pg_3, xv, indices_3);

				contribs_3 = svmla_f64_m(pg_3, contribs_3, xvv_3, mtxValues_3);

				svbool_t pg_4 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_4);
				svfloat64_t mtxValues_4 = svld1_f64(pg_4, &currentValues_4[j]);
				svuint64_t indices_4 = svld1sw_u64(pg_4, &currentColIndices_4[j]);
				svfloat64_t xvv_4 = svld1_gather_u64index_f64(pg_4, xv, indices_4);

				contribs_4 = svmla_f64_m(pg_4, contribs_4, xvv_4, mtxValues_4);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double sum_1 = rv[row_1] - totalContribution_1;

			sum_1 += xv[row_1] * currentDiagonal_1;
			xv[row_1] = sum_1 / currentDiagonal_1;

			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_2] = sum_2 / currentDiagonal_2;

			double totalContribution_3 = svaddv_f64(svptrue_b64(), contribs_3);
			double sum_3 = rv[row_3] - totalContribution_3;

			sum_3 += xv[row_3] * currentDiagonal_3;
			xv[row_3] = sum_3 / currentDiagonal_3;

			double totalContribution_4 = svaddv_f64(svptrue_b64(), contribs_4);
			double sum_4 = rv[row_4] - totalContribution_4;

			sum_4 += xv[row_4] * currentDiagonal_4;
			xv[row_4] = sum_4 / currentDiagonal_4;

		}

//#pragma omp single
		if (maxLevelSize < tdgLevelSize) {
/************
#ifndef HPCG_NO_OPENMP
//#pragma loop nounroll
#pragma omp for SCHEDULE(runtime)
#endif
		for ( local_int_t i = maxLevelSize; i < tdgLevelSize; i++ ) {

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
*******/
		#pragma omp sections nowait
		{
			#pragma omp section 
			{
				local_int_t i = maxLevelSize;
				local_int_t row = A.tdg[l][i];
				const double * const currentValues = A.matrixValues[row];
				const local_int_t * const currentColIndices = A.mtxIndL[row];
				const int currentNumberOfNonzeros = A.nonzerosInRow[row];
				const double currentDiagonal = matrixDiagonal[row][0];
				svfloat64_t contribs = svdup_f64(0.0);

				for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
					svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
					
					svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
					svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
					svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

					contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
				}

				double totalContribution = svaddv_f64(svptrue_b64(), contribs);
				double sum = rv[row] - totalContribution;

				sum += xv[row] * currentDiagonal;
				xv[row] = sum / currentDiagonal;
			}
			#pragma omp section 
			{
				local_int_t i = maxLevelSize + 1;
				if (i < tdgLevelSize) {
				local_int_t row = A.tdg[l][i];
				const double * const currentValues = A.matrixValues[row];
				const local_int_t * const currentColIndices = A.mtxIndL[row];
				const int currentNumberOfNonzeros = A.nonzerosInRow[row];
				const double currentDiagonal = matrixDiagonal[row][0];
				svfloat64_t contribs = svdup_f64(0.0);

				for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
					svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
					
					svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
					svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
					svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

					contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
				}

				double totalContribution = svaddv_f64(svptrue_b64(), contribs);
				double sum = rv[row] - totalContribution;

				sum += xv[row] * currentDiagonal;
				xv[row] = sum / currentDiagonal;
				}
			}
			#pragma omp section 
			{
				local_int_t i = maxLevelSize + 2;
				if (i < tdgLevelSize) {
				local_int_t row = A.tdg[l][i];
				const double * const currentValues = A.matrixValues[row];
				const local_int_t * const currentColIndices = A.mtxIndL[row];
				const int currentNumberOfNonzeros = A.nonzerosInRow[row];
				const double currentDiagonal = matrixDiagonal[row][0];
				svfloat64_t contribs = svdup_f64(0.0);

				for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
					svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
					
					svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
					svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
					svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

					contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
				}

				double totalContribution = svaddv_f64(svptrue_b64(), contribs);
				double sum = rv[row] - totalContribution;

				sum += xv[row] * currentDiagonal;
				xv[row] = sum / currentDiagonal;
				}
			}
		}

/***********/
#ifndef HPCG_NO_OPENMP
	}
	#pragma omp barrier
#endif
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 4*(tdgLevelSize / 4);

#ifndef HPCG_NO_OPENMP
		//#pragma omp single nowait 
		//{
		//#pragma loop nounroll
		#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = tdgLevelSize-1; i >= maxLevelSize; i-- ) {

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
#ifndef HPCG_NO_OPENMP
		//}
//#pragma loop nounroll
#pragma omp for SCHEDULE(runtime)
#endif
		for ( local_int_t i = maxLevelSize-1; i >= 0; i-= 4 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i-1];
			local_int_t row_3 = A.tdg[l][i-2];
			local_int_t row_4 = A.tdg[l][i-3];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const double * const currentValues_2 = A.matrixValues[row_2];
			const double * const currentValues_3 = A.matrixValues[row_3];
			const double * const currentValues_4 = A.matrixValues[row_4];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const local_int_t * const currentColIndices_3 = A.mtxIndL[row_3];
			const local_int_t * const currentColIndices_4 = A.mtxIndL[row_4];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const int currentNumberOfNonzeros_3 = A.nonzerosInRow[row_3];
			const int currentNumberOfNonzeros_4 = A.nonzerosInRow[row_4];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			const double currentDiagonal_3 = matrixDiagonal[row_3][0];
			const double currentDiagonal_4 = matrixDiagonal[row_4][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);
			svfloat64_t contribs_2 = svdup_f64(0.0);
			svfloat64_t contribs_3 = svdup_f64(0.0);
			svfloat64_t contribs_4 = svdup_f64(0.0);

			const int maxNumberOfNonzeros1 = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);				
			const int maxNumberOfNonzeros2 = std::max(currentNumberOfNonzeros_3, currentNumberOfNonzeros_4);				
			const int maxNumberOfNonzeros = std::max(maxNumberOfNonzeros1, maxNumberOfNonzeros2);				
							
			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svbool_t pg_3 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_3);
				svbool_t pg_4 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_4);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svfloat64_t mtxValues_3 = svld1_f64(pg_3, &currentValues_3[j]);
				svfloat64_t mtxValues_4 = svld1_f64(pg_4, &currentValues_4[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svuint64_t indices_3 = svld1sw_u64(pg_3, &currentColIndices_3[j]);
				svuint64_t indices_4 = svld1sw_u64(pg_4, &currentColIndices_4[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);
				svfloat64_t xvv_3 = svld1_gather_u64index_f64(pg_3, xv, indices_3);
				svfloat64_t xvv_4 = svld1_gather_u64index_f64(pg_4, xv, indices_4);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);
				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
				contribs_3 = svmla_f64_m(pg_3, contribs_3, xvv_3, mtxValues_3);
				contribs_4 = svmla_f64_m(pg_4, contribs_4, xvv_4, mtxValues_4);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double totalContribution_3 = svaddv_f64(svptrue_b64(), contribs_3);
			double totalContribution_4 = svaddv_f64(svptrue_b64(), contribs_4);
			double sum_1 = rv[row_1] - totalContribution_1;
			double sum_2 = rv[row_2] - totalContribution_2;
			double sum_3 = rv[row_3] - totalContribution_3;
			double sum_4 = rv[row_4] - totalContribution_4;

			sum_1 += xv[row_1] * currentDiagonal_1;
			sum_2 += xv[row_2] * currentDiagonal_2;
			sum_3 += xv[row_3] * currentDiagonal_3;
			sum_4 += xv[row_4] * currentDiagonal_4;
			xv[row_1] = sum_1 / currentDiagonal_1;
			xv[row_2] = sum_2 / currentDiagonal_2;
			xv[row_3] = sum_3 / currentDiagonal_3;
			xv[row_4] = sum_4 / currentDiagonal_4;
		}
#ifndef HPCG_NO_OPENMP
	}
#endif
	}

//#pragma statement end_scache_isolate_assign
//#pragma statement end_scache_isolate_way	
}
/////////////
inline void SYMGS_VERSION_4(const SparseMatrix& A, double * const& xv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t l = 0; l < A.tdg.size(); l++ ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 6*(tdgLevelSize / 6);

#ifndef HPCG_NO_OPENMP
	#pragma omp parallel
	{
	#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = 0; i < maxLevelSize; i+=6 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i+1];
			local_int_t row_3 = A.tdg[l][i+2];
			local_int_t row_4 = A.tdg[l][i+3];
			local_int_t row_5 = A.tdg[l][i+4];
			local_int_t row_6 = A.tdg[l][i+5];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);

			const double * const currentValues_2 = A.matrixValues[row_2];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			svfloat64_t contribs_2 = svdup_f64(0.0);

			const double * const currentValues_3 = A.matrixValues[row_3];
			const local_int_t * const currentColIndices_3 = A.mtxIndL[row_3];
			const int currentNumberOfNonzeros_3 = A.nonzerosInRow[row_3];
			const double currentDiagonal_3 = matrixDiagonal[row_3][0];
			svfloat64_t contribs_3 = svdup_f64(0.0);

			const double * const currentValues_4 = A.matrixValues[row_4];
			const local_int_t * const currentColIndices_4 = A.mtxIndL[row_4];
			const int currentNumberOfNonzeros_4 = A.nonzerosInRow[row_4];
			const double currentDiagonal_4 = matrixDiagonal[row_4][0];
			svfloat64_t contribs_4 = svdup_f64(0.0);

			const double * const currentValues_5 = A.matrixValues[row_5];
			const local_int_t * const currentColIndices_5 = A.mtxIndL[row_5];
			const int currentNumberOfNonzeros_5 = A.nonzerosInRow[row_5];
			const double currentDiagonal_5 = matrixDiagonal[row_5][0];
			svfloat64_t contribs_5 = svdup_f64(0.0);

			const double * const currentValues_6 = A.matrixValues[row_6];
			const local_int_t * const currentColIndices_6 = A.mtxIndL[row_6];
			const int currentNumberOfNonzeros_6 = A.nonzerosInRow[row_6];
			const double currentDiagonal_6 = matrixDiagonal[row_6][0];
			svfloat64_t contribs_6 = svdup_f64(0.0);

			const int maxNumberOfNonzeros1 = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);
			const int maxNumberOfNonzeros2 = std::max(currentNumberOfNonzeros_3, currentNumberOfNonzeros_4);
			const int maxNumberOfNonzeros3 = std::max(currentNumberOfNonzeros_5, currentNumberOfNonzeros_6);
			const int maxNumberOfNonzeros4 = std::max(maxNumberOfNonzeros1, maxNumberOfNonzeros2);
			const int maxNumberOfNonzeros = std::max(maxNumberOfNonzeros4, maxNumberOfNonzeros3);

			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);		
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);

				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);

				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);

				svbool_t pg_3 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_3);
				svfloat64_t mtxValues_3 = svld1_f64(pg_3, &currentValues_3[j]);
				svuint64_t indices_3 = svld1sw_u64(pg_3, &currentColIndices_3[j]);
				svfloat64_t xvv_3 = svld1_gather_u64index_f64(pg_3, xv, indices_3);

				contribs_3 = svmla_f64_m(pg_3, contribs_3, xvv_3, mtxValues_3);

				svbool_t pg_4 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_4);
				svfloat64_t mtxValues_4 = svld1_f64(pg_4, &currentValues_4[j]);
				svuint64_t indices_4 = svld1sw_u64(pg_4, &currentColIndices_4[j]);
				svfloat64_t xvv_4 = svld1_gather_u64index_f64(pg_4, xv, indices_4);

				contribs_4 = svmla_f64_m(pg_4, contribs_4, xvv_4, mtxValues_4);

				svbool_t pg_5 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_5);
				svfloat64_t mtxValues_5 = svld1_f64(pg_5, &currentValues_5[j]);
				svuint64_t indices_5 = svld1sw_u64(pg_5, &currentColIndices_5[j]);
				svfloat64_t xvv_5 = svld1_gather_u64index_f64(pg_5, xv, indices_5);

				contribs_5 = svmla_f64_m(pg_5, contribs_5, xvv_5, mtxValues_5);

				svbool_t pg_6 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_6);
				svfloat64_t mtxValues_6 = svld1_f64(pg_6, &currentValues_6[j]);
				svuint64_t indices_6 = svld1sw_u64(pg_6, &currentColIndices_6[j]);
				svfloat64_t xvv_6 = svld1_gather_u64index_f64(pg_6, xv, indices_6);

				contribs_6 = svmla_f64_m(pg_6, contribs_6, xvv_6, mtxValues_6);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double sum_1 = rv[row_1] - totalContribution_1;

			sum_1 += xv[row_1] * currentDiagonal_1;
			xv[row_1] = sum_1 / currentDiagonal_1;

			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double sum_2 = rv[row_2] - totalContribution_2;

			sum_2 += xv[row_2] * currentDiagonal_2;
			xv[row_2] = sum_2 / currentDiagonal_2;

			double totalContribution_3 = svaddv_f64(svptrue_b64(), contribs_3);
			double sum_3 = rv[row_3] - totalContribution_3;

			sum_3 += xv[row_3] * currentDiagonal_3;
			xv[row_3] = sum_3 / currentDiagonal_3;

			double totalContribution_4 = svaddv_f64(svptrue_b64(), contribs_4);
			double sum_4 = rv[row_4] - totalContribution_4;

			sum_4 += xv[row_4] * currentDiagonal_4;
			xv[row_4] = sum_4 / currentDiagonal_4;

			double totalContribution_5 = svaddv_f64(svptrue_b64(), contribs_5);
			double sum_5 = rv[row_5] - totalContribution_5;

			sum_5 += xv[row_5] * currentDiagonal_5;
			xv[row_5] = sum_5 / currentDiagonal_5;

			double totalContribution_6 = svaddv_f64(svptrue_b64(), contribs_6);
			double sum_6 = rv[row_6] - totalContribution_6;

			sum_6 += xv[row_6] * currentDiagonal_6;
			xv[row_6] = sum_6 / currentDiagonal_6;
		}

//#pragma omp single
		if (maxLevelSize < tdgLevelSize) {
#ifndef HPCG_NO_OPENMP
#pragma omp for SCHEDULE(runtime)
#endif
		for ( local_int_t i = maxLevelSize; i < tdgLevelSize; i++ ) {

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
		}
#ifndef HPCG_NO_OPENMP
	}
#endif
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 6*(tdgLevelSize / 6);

#ifndef HPCG_NO_OPENMP
#pragma omp parallel 
	{
		//#pragma omp single nowait 
		//{
		#pragma omp for nowait SCHEDULE(runtime)
#endif
		for ( local_int_t i = tdgLevelSize-1; i >= maxLevelSize; i-- ) {

			local_int_t row = A.tdg[l][i];
			const double * const currentValues = A.matrixValues[row];
			const local_int_t * const currentColIndices = A.mtxIndL[row];
			const int currentNumberOfNonzeros = A.nonzerosInRow[row];
			const double currentDiagonal = matrixDiagonal[row][0];
			svfloat64_t contribs = svdup_f64(0.0);

			for ( local_int_t j = 0; j < currentNumberOfNonzeros; j += svcntd()) {
				svbool_t pg = svwhilelt_b64_u64(j, currentNumberOfNonzeros);
				
				svfloat64_t mtxValues = svld1_f64(pg, &currentValues[j]);
				svuint64_t indices = svld1sw_u64(pg, &currentColIndices[j]);
				svfloat64_t xvv = svld1_gather_u64index_f64(pg, xv, indices);

				contribs = svmla_f64_m(pg, contribs, xvv, mtxValues);
			}

			double totalContribution = svaddv_f64(svptrue_b64(), contribs);
			double sum = rv[row] - totalContribution;

			sum += xv[row] * currentDiagonal;
			xv[row] = sum / currentDiagonal;
		}
#ifndef HPCG_NO_OPENMP
		//}
#pragma omp for SCHEDULE(runtime)
#endif
		for ( local_int_t i = maxLevelSize-1; i >= 0; i-= 6 ) {
			local_int_t row_1 = A.tdg[l][i];
			local_int_t row_2 = A.tdg[l][i-1];
			local_int_t row_3 = A.tdg[l][i-2];
			local_int_t row_4 = A.tdg[l][i-3];
			local_int_t row_5 = A.tdg[l][i-4];
			local_int_t row_6 = A.tdg[l][i-5];
			const double * const currentValues_1 = A.matrixValues[row_1];
			const double * const currentValues_2 = A.matrixValues[row_2];
			const double * const currentValues_3 = A.matrixValues[row_3];
			const double * const currentValues_4 = A.matrixValues[row_4];
			const double * const currentValues_5 = A.matrixValues[row_5];
			const double * const currentValues_6 = A.matrixValues[row_6];
			const local_int_t * const currentColIndices_1 = A.mtxIndL[row_1];
			const local_int_t * const currentColIndices_2 = A.mtxIndL[row_2];
			const local_int_t * const currentColIndices_3 = A.mtxIndL[row_3];
			const local_int_t * const currentColIndices_4 = A.mtxIndL[row_4];
			const local_int_t * const currentColIndices_5 = A.mtxIndL[row_5];
			const local_int_t * const currentColIndices_6 = A.mtxIndL[row_6];
			const int currentNumberOfNonzeros_1 = A.nonzerosInRow[row_1];
			const int currentNumberOfNonzeros_2 = A.nonzerosInRow[row_2];
			const int currentNumberOfNonzeros_3 = A.nonzerosInRow[row_3];
			const int currentNumberOfNonzeros_4 = A.nonzerosInRow[row_4];
			const int currentNumberOfNonzeros_5 = A.nonzerosInRow[row_5];
			const int currentNumberOfNonzeros_6 = A.nonzerosInRow[row_6];
			const double currentDiagonal_1 = matrixDiagonal[row_1][0];
			const double currentDiagonal_2 = matrixDiagonal[row_2][0];
			const double currentDiagonal_3 = matrixDiagonal[row_3][0];
			const double currentDiagonal_4 = matrixDiagonal[row_4][0];
			const double currentDiagonal_5 = matrixDiagonal[row_5][0];
			const double currentDiagonal_6 = matrixDiagonal[row_6][0];
			svfloat64_t contribs_1 = svdup_f64(0.0);
			svfloat64_t contribs_2 = svdup_f64(0.0);
			svfloat64_t contribs_3 = svdup_f64(0.0);
			svfloat64_t contribs_4 = svdup_f64(0.0);
			svfloat64_t contribs_5 = svdup_f64(0.0);
			svfloat64_t contribs_6 = svdup_f64(0.0);

			const int maxNumberOfNonzeros1 = std::max(currentNumberOfNonzeros_1, currentNumberOfNonzeros_2);				
			const int maxNumberOfNonzeros2 = std::max(currentNumberOfNonzeros_3, currentNumberOfNonzeros_4);				
			const int maxNumberOfNonzeros3 = std::max(currentNumberOfNonzeros_5, currentNumberOfNonzeros_6);				
			const int maxNumberOfNonzeros4 = std::max(maxNumberOfNonzeros1, maxNumberOfNonzeros2);				
			const int maxNumberOfNonzeros = std::max(maxNumberOfNonzeros3, maxNumberOfNonzeros4);				
							
			for ( local_int_t j = 0; j < maxNumberOfNonzeros; j += svcntd()) {
				svbool_t pg_1 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_1);
				svbool_t pg_2 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_2);
				svbool_t pg_3 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_3);
				svbool_t pg_4 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_4);
				svbool_t pg_5 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_5);
				svbool_t pg_6 = svwhilelt_b64_u64(j, currentNumberOfNonzeros_6);
				
				svfloat64_t mtxValues_1 = svld1_f64(pg_1, &currentValues_1[j]);
				svfloat64_t mtxValues_2 = svld1_f64(pg_2, &currentValues_2[j]);
				svfloat64_t mtxValues_3 = svld1_f64(pg_3, &currentValues_3[j]);
				svfloat64_t mtxValues_4 = svld1_f64(pg_4, &currentValues_4[j]);
				svfloat64_t mtxValues_5 = svld1_f64(pg_5, &currentValues_5[j]);
				svfloat64_t mtxValues_6 = svld1_f64(pg_6, &currentValues_6[j]);
				svuint64_t indices_1 = svld1sw_u64(pg_1, &currentColIndices_1[j]);
				svuint64_t indices_2 = svld1sw_u64(pg_2, &currentColIndices_2[j]);
				svuint64_t indices_3 = svld1sw_u64(pg_3, &currentColIndices_3[j]);
				svuint64_t indices_4 = svld1sw_u64(pg_4, &currentColIndices_4[j]);
				svuint64_t indices_5 = svld1sw_u64(pg_5, &currentColIndices_5[j]);
				svuint64_t indices_6 = svld1sw_u64(pg_6, &currentColIndices_6[j]);
				svfloat64_t xvv_1 = svld1_gather_u64index_f64(pg_1, xv, indices_1);
				svfloat64_t xvv_2 = svld1_gather_u64index_f64(pg_2, xv, indices_2);
				svfloat64_t xvv_3 = svld1_gather_u64index_f64(pg_3, xv, indices_3);
				svfloat64_t xvv_4 = svld1_gather_u64index_f64(pg_4, xv, indices_4);
				svfloat64_t xvv_5 = svld1_gather_u64index_f64(pg_5, xv, indices_5);
				svfloat64_t xvv_6 = svld1_gather_u64index_f64(pg_6, xv, indices_6);

				contribs_1 = svmla_f64_m(pg_1, contribs_1, xvv_1, mtxValues_1);
				contribs_2 = svmla_f64_m(pg_2, contribs_2, xvv_2, mtxValues_2);
				contribs_3 = svmla_f64_m(pg_3, contribs_3, xvv_3, mtxValues_3);
				contribs_4 = svmla_f64_m(pg_4, contribs_4, xvv_4, mtxValues_4);
				contribs_5 = svmla_f64_m(pg_5, contribs_5, xvv_5, mtxValues_5);
				contribs_6 = svmla_f64_m(pg_6, contribs_6, xvv_6, mtxValues_6);
			}

			double totalContribution_1 = svaddv_f64(svptrue_b64(), contribs_1);
			double totalContribution_2 = svaddv_f64(svptrue_b64(), contribs_2);
			double totalContribution_3 = svaddv_f64(svptrue_b64(), contribs_3);
			double totalContribution_4 = svaddv_f64(svptrue_b64(), contribs_4);
			double totalContribution_5 = svaddv_f64(svptrue_b64(), contribs_5);
			double totalContribution_6 = svaddv_f64(svptrue_b64(), contribs_6);
			double sum_1 = rv[row_1] - totalContribution_1;
			double sum_2 = rv[row_2] - totalContribution_2;
			double sum_3 = rv[row_3] - totalContribution_3;
			double sum_4 = rv[row_4] - totalContribution_4;
			double sum_5 = rv[row_5] - totalContribution_5;
			double sum_6 = rv[row_6] - totalContribution_6;

			sum_1 += xv[row_1] * currentDiagonal_1;
			sum_2 += xv[row_2] * currentDiagonal_2;
			sum_3 += xv[row_3] * currentDiagonal_3;
			sum_4 += xv[row_4] * currentDiagonal_4;
			sum_5 += xv[row_5] * currentDiagonal_5;
			sum_6 += xv[row_6] * currentDiagonal_6;
			xv[row_1] = sum_1 / currentDiagonal_1;
			xv[row_2] = sum_2 / currentDiagonal_2;
			xv[row_3] = sum_3 / currentDiagonal_3;
			xv[row_4] = sum_4 / currentDiagonal_4;
			xv[row_5] = sum_5 / currentDiagonal_5;
			xv[row_6] = sum_6 / currentDiagonal_6;
		}
#ifndef HPCG_NO_OPENMP
	}
#endif
	}
}
