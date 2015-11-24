#include "MathonVectors.h"
#include <amp.h>
//average vector over all elements in the cluster
// normal version
//void average_vec(double vec[], int n, int num)
//{
//	int i;
//	for (i = 0; i< n; i++)
//		vec[i] = vec[i] / num;
//}

//GPU version
void average_vec(double vec[], int n, int num) //NIELS: ikke sikker på om man skal unik med (num) 
{
	concurrency::array_view<double, 1> AMP_vec(n, vec);

	concurrency::parallel_for_each(AMP_vec.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		AMP_vec[idx] = AMP_vec[idx] / num;
	});

	vec = AMP_vec.data(); // NIELS: ikke sikker på hvor smart det her er men hurra for pointer...
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