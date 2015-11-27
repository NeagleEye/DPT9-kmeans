#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "Kmeans.h"
#include "PrintMatrix.h"
#include <amp.h>

using namespace std;

void kmeansKode()
{
	int *cluster, n_clusters = 4;
	int x = 0, y = 0;
	Matrix matrix = GetVector(x, y);

	cluster = new int[matrix.GetColumns()];
	//Initialize Euclidean kmeans
	Kmeans k(n_clusters, cluster, matrix.GetColumns(), matrix.GetRows());
	//Calculate normal vectors on every column set.
	matrix.ComputeNormalVector();
	k.Initialize_CV(matrix);
	k.Generel_K_Means(matrix);
	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	for (int i = 0; i < matrix.GetColumns(); i++)
	{
		cout << matrix.GetVal(i, 0) << " " << matrix.GetVal(i, 1) << endl;
	}

	PrintMatrix(matrix, x, y);
	PrintMatrix_With_Cluster(matrix, x, y);
}



void AMP_TestCode1()
{
	const int size = 5;
	int aCPP[] = { 1, 2, 3, 4, 5 };
	int bCPP[] = { 6, 7, 8, 9, 10 };
	int sumCPP[size];

	// Create C++ AMP objects.
	concurrency::array_view<const int, 1> a(size, aCPP);
	concurrency::array_view<const int, 1> b(size, bCPP);
	concurrency::array_view<int, 1> sum(size, sumCPP);
	sum.discard_data();

	concurrency::parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		sum.extent,
		// Define the code to run on each thread on the accelerator.
		[=](concurrency::index<1> idx) restrict(amp)
	{
		sum[idx] = a[idx] + b[idx];
	}
	);

	// Print the results. The expected output is "7, 9, 11, 13, 15".
	for (int i = 0; i < size; i++) {
		std::cout << sum[i] << "\n";
	}
}
void AMP_testCode2()
{
	std::vector<int> data0(1024, 1);
	std::vector<int> data1(1024, 2);
	std::vector<int> data_out(data0.size(), 0);

	concurrency::array_view<int, 1> av0(data0.size(), data0);
	concurrency::array_view<int, 1> av1(data1.size(), data1);
	concurrency::array_view<int, 1> av2(data_out.size(), data_out);

	av2.discard_data();

	concurrency::parallel_for_each(av0.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		av2[idx] = av0[idx] + av1[idx];
	});


}
void AMP_testCode3()
{
	int *test;
	std::vector<int> data0(1024, 1);
	int n = 2;
	concurrency::array_view<int, 1> av0(data0.size(), data0);
	concurrency::parallel_for_each(av0.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		av0[idx] += n;
	});
	test = av0.data();

	for (int i = 0; i < 1024; i++) {
		std::cout << test[i] << "\n";
	}
}
void AMP_testCode4()
{
	int length = 4, depths = 5;
	int **BigOne, *littleOne;
	vector<int>largeOne;

	BigOne = new int*[length];
	for (int i = 0; i < length; i++)
	{
		littleOne = new int[depths];
		for (int x = 0; x < depths; x++)
		{
			littleOne[x] = i + x;
		}
		BigOne[i] = littleOne;
	}

	for (int i = 0; i < length; i++)
	{
		for (int x = 0; x < depths; x++)
		{
			cout << "(" << i << "," << x << ")= " << BigOne[i][x] << endl;
		}
	}

	for (int i = 0; i < length; i++)
	{
		for (int x = 0; x < depths; x++)
		{
			largeOne.push_back(BigOne[i][x]);
		}
	}
}
void exampleCode()
{
	int aMatrix[] = { 1, 4, 2, 5, 3, 6 };
	int bMatrix[] = { 7, 8, 9, 10, 11, 12 };
	int productMatrix[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	concurrency::array_view<int, 2> a(3, 2, aMatrix);
	concurrency::array_view<int, 2> b(2, 3, bMatrix);
	concurrency::array_view<int, 2> product(3, 3, productMatrix);

	concurrency::parallel_for_each(
		product.extent,
		[=](concurrency::index<2> idx) restrict(amp) {
		int row = idx[0];
		int col = idx[1];
		for (int inner = 0; inner < 2; inner++) {
			product[idx] += a(row, inner) * b(inner, col);
		}
	}
	);

	product.synchronize();
}

int main()
{
	//AMP_testCode3();
	kmeansKode();
	return 0;
}