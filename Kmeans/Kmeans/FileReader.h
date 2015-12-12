#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include <vector>

struct Mat
{
	int n_row, n_col;
	double **val;
};

Matrix GetVector();
Matrix GetVector(int &x,int &y, std::string);
