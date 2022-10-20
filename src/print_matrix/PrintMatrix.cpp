#pragma once

#include <stdio.h>
#include <stdlib.h>
#include "mmio.c"
#include "SparseMatrix.hpp"
#include "PrintMatrix.hpp"

void PrintRangeMatrx(const SparseMatrix& A, const char* filename, long minRow, long maxRow) {
    MM_typecode matcode;                        

    auto fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error saving matrix file\n");
        return;
    }

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(fp, matcode); 
   
    auto rows = A.localNumberOfRows; //!< total number of matrix rows across all processes
    auto cols = A.localNumberOfColumns;  //!< number of columns local to this process
    auto nz = A.totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes

    mm_write_mtx_crd_size(fp, rows, cols, nz);

    for (local_int_t i=minRow; i<rows && i<maxRow; ++i) {
        for ( local_int_t j = 0; j < A.nonzerosInRow[i]; ++j ) {
            local_int_t curCol = A.mtxIndL[i][j];
            double val = A.matrixValues[i][j];

            fprintf(fp, "%d %d %10.3g\n", i+1, curCol+1, val);
        }
    }

    fclose(fp);
}

void PrintMatrixMarket(const SparseMatrix& A, const char* filename)
{
    MM_typecode matcode;                        

    auto fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error saving matrix file\n");
        return;
    }

    mm_initialize_typecode(&matcode);
    mm_set_matrix(&matcode);
    mm_set_coordinate(&matcode);
    mm_set_real(&matcode);

    mm_write_banner(fp, matcode); 
   
    auto rows = A.localNumberOfRows; //!< total number of matrix rows across all processes
    auto cols = A.localNumberOfColumns;  //!< number of columns local to this process
    auto nz = A.totalNumberOfNonzeros; //!< total number of matrix nonzeros across all processes

    mm_write_mtx_crd_size(fp, rows, cols, nz);

    /* NOTE: matrix market files use 1-based indices, i.e. first element
      of a vector has index 1, not 0.  */
    for (local_int_t i=0; i<rows; ++i) {

        for ( local_int_t j = 0; j < A.nonzerosInRow[i]; ++j ) {
            local_int_t curCol = A.mtxIndL[i][j];
            //double val = A.matrixValues[i][j];
            double val = 100.0;

            fprintf(fp, "%d %d %10.3g\n", i+1, curCol+1, val);
        }
    }

    fclose(fp);
}
