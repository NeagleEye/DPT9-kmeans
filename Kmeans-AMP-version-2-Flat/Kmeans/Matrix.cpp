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

void Matrix::Euc_Dis(double *x, double norm_x, double *result, int cluster, int n_cluster)
/* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
compute the Euc-dis between each vec in the matrix with x,
results are stored in array 'result'.
Since the matrix is dense, not taking advantage of
(x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
but the abstract class defition needs the parameter of 'norm_x'
*/
{
	//CPU
	/*for (int i = 0; i < n_col; i++)
		result[cluster*n_col+i] = Euc_Dis(x, i, norm_x, cluster);*/
	//GPU
	//GPU version
	concurrency::array_view<double, 1> GPU_x(n_col, x);
	concurrency::array_view<double, 1> GPU_result(n_cluster*n_col, result);
	concurrency::array_view<double, 1> GPU_normalVector(n_col, normalVector);
	concurrency::array_view<double, 1> GPU_value(n_row_elements*n_col, value);
	int temp_n_row_elements = n_row_elements;
	int temp_n_col = n_col;
	concurrency::parallel_for(0, n_col, [=](int i) restrict(cpu, amp) //&GPU_x, &GPU_result, &temp_n_row_elements, &temp_n_col, &cluster, &GPU_normalVector, &norm_x, &GPU_value
	{
		//int i = idx[0];
		double result = 0.0;
		for (int j = 0; j< temp_n_row_elements; j++)
			result += GPU_x[cluster*temp_n_row_elements + j] * GPU_value[j*temp_n_col + i];
		result *= -2.0;
		result += GPU_normalVector[i] + norm_x;
		GPU_result[cluster*temp_n_col + i] = result;
	});
	GPU_normalVector.synchronize();
	result = GPU_result.data();
}


void Matrix::Euc_Dis(double *x, double *normal_ConceptVectors, double *result, int *cluster, int n_cluster)
{
	//for (i = 0; i<col; i++)
	//	sim_Mat[cluster[i] * col + i] = matrix.Euc_Dis(concept_Vectors, i, normal_ConceptVectors[cluster[i]], cluster[i]);

	concurrency::array_view<double, 1> GPU_x(n_col, x);
	concurrency::array_view<double, 1> GPU_result(n_cluster*n_col, result);
	concurrency::array_view<double, 1> GPU_dummy(n_col-1);
	concurrency::array_view<double, 1> GPU_normalVector(n_col, normalVector);
	concurrency::array_view<double, 1> GPU_value(n_row_elements*n_col, value);
	concurrency::array_view<double, 1> GPU_normal_ConceptVectors  (n_cluster, normal_ConceptVectors);
	concurrency::array_view<int, 1> GPU_cluster                   (n_col, cluster);

	int temp_n_row_elements = n_row_elements;
	int temp_n_col = n_col;
	concurrency::parallel_for(0,n_col, 
		[=](int idx) restrict(cpu,amp)//&GPU_x, &GPU_result, &temp_n_row_elements, &temp_n_col, &GPU_normalVector, &GPU_cluster, &GPU_value, &GPU_normal_ConceptVectors
	{
		double result = 0.0;
		for (int j = 0; j < temp_n_row_elements; j++)
			result += GPU_x[GPU_cluster[idx] * temp_n_row_elements + j] * GPU_value[j*temp_n_col + idx];
		result *= -2.0;
		result += GPU_normalVector[idx] + GPU_normal_ConceptVectors[GPU_cluster[idx]];
		GPU_result[GPU_cluster[idx] * temp_n_col + idx] = result;
		//GPU_dummy[idx] = result;
	});

	GPU_dummy.synchronize();
	result = GPU_result.data();
}



/*void Matrix::Euc_Dis_All(double *x, double norm_x, double *result, int n_cluster)
{
	concurrency::array_view<double, 1> GPU_x(n_col, x);
	concurrency::array_view<double, 1> GPU_result(n_col * n_cluster, result);
	concurrency::array_view<double, 1> GPU_normalVector(n_col, normalVector);
	concurrency::array_view<double, 1> GPU_value(n_row_elements*n_col, value);
	int temp_n_row_elements = n_row_elements;
	int temp_n_col = n_col;
	concurrency::parallel_for_each(GPU_result.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		int cluster = idx[0] / temp_n_col;
		int col_inCluster = idx[0] % temp_n_col;

		double result = 0.0;
		for (int j = 0; j< temp_n_row_elements; j++)
			result += GPU_x[cluster*temp_n_row_elements + j] * GPU_value[j*temp_n_col + col_inCluster];
		result *= -2.0;
		result += GPU_normalVector[col_inCluster] + norm_x;
		GPU_result[idx[0]] = result;
	});
	GPU_normalVector.synchronize();
	result = GPU_result.data();

}*/

void Matrix::PassCluster(int clus[])
{
	for (int i = 0; i < GetColumns(); i++)
	{
		setClusterID(i, clus[i]);
	}
}