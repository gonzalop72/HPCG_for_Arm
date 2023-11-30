inline void SYMGS_SECTION_UNROLL1(const SparseMatrix& A, double * const& xv, const double * const& rv, const local_int_t l, const local_int_t i);
inline void SYMGS_SECTION_UNROLL2(const SparseMatrix& A, double * const& xv, const double * const& rv, const local_int_t l, const local_int_t i);

inline void SYMGS_VERSION_6(const SparseMatrix& A, double * const& xv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;

#ifndef HPCG_NO_OPENMP
	#pragma omp parallel
	{
		local_int_t numThreads = omp_get_num_threads();
		local_int_t threadId = omp_get_thread_num();
#endif

	for ( local_int_t lev = 0; lev < A.tdg.size(); lev++ ) {
		local_int_t tdgLevelSize = A.tdg[lev].size();
		local_int_t maxLevelSize = 6*(tdgLevelSize / 6);

#ifdef MANUAL_TASK_DISTRIBUTION
		//at least 8 tasks per thread 
        const local_int_t blockSize = 6;
        local_int_t maxGroups = tdgLevelSize/blockSize;
		local_int_t groupsPerThread = std::max((maxGroups+numThreads-1)/numThreads, 2);
		local_int_t taskGroup = groupsPerThread*blockSize;
		local_int_t minValue = taskGroup*threadId;
		//local_int_t maxValue = std::min(minValue+taskGroup, maxGroups*blockSize);
		local_int_t maxValue = std::min(minValue+taskGroup, maxLevelSize);

		#pragma fj loop zfill			
		#pragma loop nounroll
		for ( local_int_t i = minValue; i < maxValue; i+=6 )
#else

#ifndef HPCG_NO_OPENMP
		#pragma fj loop zfill			
		#pragma loop nounroll
		#pragma omp for nowait SCH_SYMGS(runtime)
#endif
		for ( local_int_t i = 0; i < maxLevelSize; i+=6 )
#endif
		{
			local_int_t row_1 = A.tdg[lev][i];
			local_int_t row_2 = A.tdg[lev][i+1];
			local_int_t row_3 = A.tdg[lev][i+2];
			local_int_t row_4 = A.tdg[lev][i+3];
			local_int_t row_5 = A.tdg[lev][i+4];
			local_int_t row_6 = A.tdg[lev][i+5];
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

#ifdef dontWork_MANUAL_TASK_DISTRIBUTION
		local_int_t i = maxLevelSize + (numThreads-1) - threadId;
		if (i < tdgLevelSize)
		/*if (maxLevelSize < tdgLevelSize) {
            #pragma omp sections
            {
                #pragma omp section 
                {
					auto dif = tdgLevelSize - maxLevelSize;
					if (dif >= 2)
						SYMGS_SECTION_UNROLL2(A, xv, rv, lev, maxLevelSize);
					else if (dif >= 1)
						SYMGS_SECTION_UNROLL1(A, xv, rv, lev, maxLevelSize);
				}
                #pragma omp section 
                {
					auto dif = tdgLevelSize - maxLevelSize;
					if (dif >= 4)
						SYMGS_SECTION_UNROLL2(A, xv, rv, lev, maxLevelSize+2);
					else if (dif >= 3)
						SYMGS_SECTION_UNROLL1(A, xv, rv, lev, maxLevelSize+2);
				}
                #pragma omp section 
                {
					auto dif = tdgLevelSize - maxLevelSize;
					if (dif >= 5)
						SYMGS_SECTION_UNROLL1(A, xv, rv, lev, maxLevelSize+4);
				}
			}
		}
		else {
			#pragma omp barrier
		}*/
#else
		//if (maxLevelSize < tdgLevelSize) {
		#ifndef HPCG_NO_OPENMP
		#pragma fj loop zfill			
		#pragma loop nounroll
		#pragma omp for
		#endif
		for ( local_int_t i = maxLevelSize; i < tdgLevelSize; i++ ) 
#endif
		{
			local_int_t row = A.tdg[lev][i];
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
		/*}
		else {
			#pragma omp barrier
		}*/
    	//lev += inc;
//#endif
#ifdef dontWork_MANUAL_TASK_DISTRIBUTION
			#pragma omp barrier
#endif
	}

	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
		local_int_t tdgLevelSize = A.tdg[l].size();
		local_int_t maxLevelSize = 6*(tdgLevelSize / 6);

#ifdef MANUAL_TASK_DISTRIBUTION
		//at least 8 tasks per thread 
        const local_int_t blockSize = 6;
        local_int_t maxGroups = tdgLevelSize/blockSize;
		local_int_t groupsPerThread = std::max((maxGroups+numThreads-1)/numThreads, 2);
		local_int_t taskGroup = groupsPerThread*blockSize;
		local_int_t minValue = taskGroup*threadId;
		//local_int_t maxValue = std::min(minValue+taskGroup, maxGroups*blockSize);
		local_int_t maxValue = std::min(minValue+taskGroup, maxLevelSize);
#endif

#ifdef dontWork_MANUAL_TASK_DISTRIBUTION
			local_int_t i = maxLevelSize + (numThreads-1) - threadId;
			if (i < tdgLevelSize)
#else
	#ifndef HPCG_NO_OPENMP
			#pragma fj loop zfill			
			#pragma loop nounroll
			#pragma omp for nowait
	#endif
		for ( local_int_t i = tdgLevelSize-1; i >= maxLevelSize; i-- ) 
#endif
		{
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

#ifdef MANUAL_TASK_DISTRIBUTION
		#pragma fj loop zfill			
		#pragma loop nounroll
		for ( local_int_t i = maxValue-1; i >= minValue; i-=6 )
#else

	#ifndef HPCG_NO_OPENMP
	#pragma fj loop zfill			
	#pragma loop nounroll	
	#pragma omp for SCH_SYMGS(runtime)
	#endif
		for ( local_int_t i = maxLevelSize-1; i >= 0; i-= 6 )
#endif
		{
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
#ifdef MANUAL_TASK_DISTRIBUTION
	#pragma omp barrier
#endif
	}

#ifndef HPCG_NO_OPENMP
	}
#endif
}

inline void SYMGS_SECTION_UNROLL1(const SparseMatrix& A, double * const& xv, const double * const& rv, const local_int_t l, const local_int_t i) {
	
		double **matrixDiagonal = A.matrixDiagonal;

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

inline void SYMGS_SECTION_UNROLL2(const SparseMatrix& A, double * const& xv, const double * const& rv, const local_int_t l, const local_int_t i) {
	
		double **matrixDiagonal = A.matrixDiagonal;

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
