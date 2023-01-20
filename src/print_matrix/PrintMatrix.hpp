#pragma once

void PrintMatrixMarket(const SparseMatrix& A, const char* filename);
void PrintRangeMatrx(const SparseMatrix& A, const char* filename, long minRow, long maxRow);

void PrintTDGGraph(const SparseMatrix& A, const char* filename);
