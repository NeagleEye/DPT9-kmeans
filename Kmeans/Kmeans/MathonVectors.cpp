/*   We have based our code on original code from Yuqiang Guan and 
     the original source code of the program is released under the GNU Public License (GPL)
     from:
     http://www.dataminingresearch.com/index.php/2010/06/gmeans-clustering-software-compatible-with-gcc-4/
     Implementation of Euclidean K-means 
     Copyright (c) 2003, Yuqiang Guan
*/
#include "MathonVectors.h"

//average vector over all elements in the cluster
void average_vec(double vec[], int n, int num)
{
	int i;
	for (i = 0; i< n; i++)
		vec[i] = vec[i] / num;
}

//compute the normal vector over the values in the row elements.
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