inline void SYMGS_VERSION_5(const SparseMatrix& A, double * const& prxv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;
	double *xv = prxv;

//#pragma statement scache_isolate_way L2=10
//#pragma statement scache_isolate_assign xv
#ifndef HPCG_NO_OPENMP
	#pragma omp parallel
	{
		local_int_t numThreads = omp_get_num_threads();
#endif
        local_int_t lev = 0, inc = 1;
        //local_int_t lev = A.tdg.size()-1, inc=-1;
        for (local_int_t step = 0; step < 2; step++) {

	/*
	 * FORWARD SWEEP
	 */
	for ( local_int_t xx = 0; xx < A.tdg.size(); xx++ ) {
		local_int_t tdgLevelSize = A.tdg[lev].size();
		local_int_t maxLevelSize = 4*(tdgLevelSize / 4);

#ifdef MANUAL_TASK_DISTRIBUTION
		//at least 8 tasks per thread 
        const local_int_t blockSize = 4;
        local_int_t maxGroups = tdgLevelSize/blockSize;
		local_int_t groupsPerThread = std::max((maxGroups+numThreads-1)/numThreads, 2);
		local_int_t taskGroup = groupsPerThread*blockSize;
		local_int_t threadId = omp_get_thread_num();
		local_int_t minValue = taskGroup*threadId;
		//local_int_t maxValue = std::min(minValue+taskGroup, maxGroups*blockSize);
        //maxLevelSize = std::min(taskGroup*numThreads, maxGroups*blockSize);
		local_int_t maxValue = std::min(minValue+taskGroup, maxLevelSize);

		#pragma fj loop zfill			
		#pragma loop nounroll
		for ( local_int_t i = minValue; i < maxValue; i+=4 )
#else
    #ifndef HPCG_NO_OPENMP
        #pragma fj loop zfill			
        #pragma loop nounroll
        #pragma omp for nowait SCH_SYMGS(runtime)
    #endif
		for ( local_int_t i = 0; i < maxLevelSize; i+=4 )
#endif
        {
			local_int_t row_1 = A.tdg[lev][i];
			local_int_t row_2 = A.tdg[lev][i+1];
			local_int_t row_3 = A.tdg[lev][i+2];
			local_int_t row_4 = A.tdg[lev][i+3];
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
            #pragma omp sections nowait
            {
                #pragma fj loop zfill			
                #pragma omp section 
                {
                    local_int_t i = maxLevelSize;
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
                #pragma fj loop zfill			
                #pragma omp section 
                {
                    local_int_t i = maxLevelSize + 1;
                    if (i < tdgLevelSize) {
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
                }
                #pragma fj loop zfill			
                #pragma omp section 
                {
                    local_int_t i = maxLevelSize + 2;
                    if (i < tdgLevelSize) {
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
                }
            }
        #ifndef HPCG_NO_OPENMP
        }
        #pragma omp barrier
        #endif
        lev += inc;
        }
    //break;
    lev = A.tdg.size()-1, inc = -1;
    }
    }
    return;

    local_int_t lev = 0, inc = 1;
    //local_int_t lev = A.tdg.size()-1, inc=-1;
	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t xx = A.tdg.size()-1; xx >= 0; xx-- ) {
        #ifndef HPCG_NO_OPENMP
        #pragma omp parallel for SCH_SYMGS(runtime)
        #endif
		for ( local_int_t i = A.tdg[lev].size()-1; i >= 0; i-- ) {
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
        lev += inc;
    }
}

inline void SYMGS_VERSION_5xx(const SparseMatrix& A, double * const& prxv, const double * const& rv) {
	
	double **matrixDiagonal = A.matrixDiagonal;
	double *xv = prxv;


	/*
	 * BACKWARD SWEEP
	 */
	for ( local_int_t l = A.tdg.size()-1; l >= 0; l-- ) {
        #ifndef HPCG_NO_OPENMP
        #pragma omp parallel for SCH_SYMGS(runtime)
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
    }
}
