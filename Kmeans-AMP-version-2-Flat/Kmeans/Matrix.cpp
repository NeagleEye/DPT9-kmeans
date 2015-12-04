#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include <math.h>
#include <iostream>
#include <vector>

Matrix::Matrix(int row, int col, double *val)
{
	n_row_elements = row;
	n_col = col;
	value = val;
	cluster.resize(col);
}

Matrix::~Matrix()
{
	/*if (normalVector != NULL)
		delete[] normalVector;*/
}


void Matrix::ComputeNormalVector()
{
	if (normalVector == NULL)
	{
		normalVector = new double[n_col];
		//memory_used += m_col*sizeof(float);
	}
	for (int i = 0; i < n_col; i++)
	{
		normalVector[i] = 0.0;
		for (int j = 0; j < n_row_elements; j++)
			normalVector[i] += (value[j*n_col + i]) * (value[j*n_col + i]);
	}
}
//add current concept_vector to the original vector
void Matrix::Ith_Add_CV(int i, double *CV, int cluster)
{
	for (int j = 0; j < n_row_elements; j++)
		CV[cluster*n_row_elements+j] += value[j*n_col + i];
}

double Matrix::Euc_Dis(double *x, int i, double norm_x, int cluster)
/* Given squared L2-norms of the vecs and v, norm[i] and norm_v,
compute the Euc-dis between ith vec in the matrix and v,
result is returned.
Used (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
*/
{
	double result = 0.0;
	for (int j = 0; j< n_row_elements; j++)
		result += x[cluster*n_row_elements+j] * value[j*n_col + i];
	result *= -2.0;
	result += normalVector[i] + norm_x;
	return result;
}

void Matrix::Euc_Dis(double *x, double norm_x, double *result, int cluster)
/* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
compute the Euc-dis between each vec in the matrix with x,
results are stored in array 'result'.
Since the matrix is dense, not taking advantage of
(x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
but the abstract class defition needs the parameter of 'norm_x'
*/
{
	for (int i = 0; i < n_col; i++)
		result[cluster*n_col+i] = Euc_Dis(x, i, norm_x, cluster);
}

void Matrix::PassCluster(int clus[])
{
	for (int i = 0; i < GetColumns(); i++)
	{
		setClusterID(i, clus[i]);
	}
}