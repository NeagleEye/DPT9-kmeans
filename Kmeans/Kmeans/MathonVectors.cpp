#include "MathonVectors.h"

void average_vec(double vec[], int n, int num)
{
	int i;
	for (i = 0; i< n; i++)
		vec[i] = vec[i] / num;
}

double norm_2(double vec[], int n)
//compute squared L2 norm of vec
{
	double norm;
	int i;
	norm = 0.0;
	for (i = 0; i < n; i++)
		norm += vec[i] * vec[i];
	return norm;
}