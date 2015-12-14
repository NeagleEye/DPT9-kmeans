#include "MathonVectors.h"

//average vector over all elements in the cluster
void average_vec(double vec[], int n, int num, int cluster)
{
	int i;
	for (i = 0; i< n; i++)
		vec[cluster*n + i] = vec[cluster*n + i] / num;
}

//compute the normal vector over the values in the row elements.
double norm_2(double vec[], int n, int cluster)
//compute squared L2 norm of vec
{
	double norm;
	int i;
	norm = 0.0;
	for (i = 0; i < n; i++)
		norm += vec[cluster*n+i] * vec[cluster*n+i];
	return norm;
}