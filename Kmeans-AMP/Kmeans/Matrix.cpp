#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include <math.h>
#include <iostream>
#include <vector>
#include <amp.h>

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
			normalVector[i] += (value[(i*n_row_elements) + j]) *(value[(i*n_row_elements) + j]); //(value[j][i]) * (value[j][i])
	}
}
//add current concept_vector to the original vector
void Matrix::Ith_Add_CV(int i, double *CV)
{
	for (int j = 0; j < n_row_elements; j++)
		CV[j] += value[(i*n_row_elements) + j];
}

double Matrix::Euc_Dis(double *x, int i, double norm_x)
/* Given squared L2-norms of the vecs and v, norm[i] and norm_v,
compute the Euc-dis between ith vec in the matrix and v,
result is returned.
Used (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
*/
{
	double result = 0.0;
	for (int j = 0; j< n_row_elements; j++)
		result += x[j] * value[(i*n_row_elements) + j];
	result *= -2.0;
	result += normalVector[i] + norm_x;
	return result;
}

void Matrix::Euc_Dis(double *x, double norm_x, double *result)
/* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
compute the Euc-dis between each vec in the matrix with x,
results are stored in array 'result'.
Since the matrix is dense, not taking advantage of
(x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
but the abstract class defition needs the parameter of 'norm_x'
*/
{
	// CPU version
	/*for (int i = 0; i < n_col; i++)
	{
		result[i] = Euc_Dis(x, i, norm_x);
	}*/

	//GPU version

	concurrency::array_view<double, 1> GPU_x(n_col, x);
	concurrency::array_view<double, 1> GPU_result(n_col, result);
	concurrency::array_view<double, 1> GPU_normalVector(n_col, normalVector);
	concurrency::array_view<double, 1> GPU_value(n_row_elements*n_col, value);
	int temp_n_row_elements = n_row_elements;
	concurrency::parallel_for_each(GPU_result.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		double result = 0.0;
		for (int j = 0; j< temp_n_row_elements; j++)
			result += GPU_x[j] * GPU_value[(idx*temp_n_row_elements) + j];
		result *= -2.0;
		result += GPU_normalVector[idx] + norm_x;
		GPU_result[idx] = result;
	});
	GPU_result.synchronize();
	result = GPU_result.data();
}
void Matrix::Euc_Dis(double *x, double norm_x, double *result, int cluster, int n_clusters)
/* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
compute the Euc-dis between each vec in the matrix with x,
results are stored in array 'result'.
Since the matrix is dense, not taking advantage of
(x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
but the abstract class defition needs the parameter of 'norm_x'
*/
{
	// CPU version
	/*for (int i = 0; i < n_col; i++)
	{
		result[(cluster*n_clusters) + i] = Euc_Dis(x, i, norm_x);
	}*/
	//GPU version
	concurrency::array_view<double, 1> GPU_x(n_col, x);
	concurrency::array_view<double, 1> GPU_result(n_col, result);
	concurrency::array_view<double, 1> GPU_normalVector(n_col, normalVector);
	concurrency::array_view<double, 1> GPU_value(n_row_elements*n_col, value);
	int temp_n_row_elements = n_row_elements;
	concurrency::parallel_for_each(GPU_result.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		double result = 0.0;
		for (int j = 0; j< temp_n_row_elements; j++)
			result += GPU_x[j] * GPU_value[(idx*temp_n_row_elements) + j];
		result *= -2.0;
		result += GPU_normalVector[idx] + norm_x;
		GPU_result[(cluster*n_clusters)+ idx] = result;
	});
	GPU_result.synchronize();
	result = GPU_result.data();
}
void Matrix::PassCluster(int clus[])
{
	for (int i = 0; i < GetColumns(); i++)
	{
		setClusterID(i, clus[i]);
	}
}