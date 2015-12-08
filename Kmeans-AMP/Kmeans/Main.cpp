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
	int test = matrix.GetColumns();
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
	for (int i = 0; i < 1024; i++) {
		std::cout << av2[i] << "\n";
	}
	int i = 0;
}
void AMP_testCode3()
{
	int *test;
	std::vector<int> data0(100, 1);
	int n = 2;
	concurrency::array_view<int, 1> av0(data0.size(), data0);
	concurrency::parallel_for_each(av0.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		av0[idx] += n;
	});
	test = av0.data();

	int lol = 0;
	for (int i = 0; i < 100; i++) {
		std::cout << test[i] << "\n";
	}
}
void AMP_testCode4()
{
	int size = 10;
	std::vector<int> tester1, tester2;
	for (size_t i = 0; i < size; i++)
	{
		tester1.push_back(i);
		tester2.push_back(-1);
	}
	concurrency::array_view<int, 1> GPU_tester1(size, tester1);
	concurrency::array_view<int, 1> GPU_tester2(size, tester2);


	concurrency::parallel_for_each(GPU_tester2.extent, [=](concurrency::index<1> idx) restrict(amp)
	{
		GPU_tester2[idx] = GPU_tester1[idx];
	});
	GPU_tester2.synchronize();
	int lol = 0;
	for (int i = 0; i < size; i++) {
		std::cout << GPU_tester2[i] << "\n";
	}
	lol = 0;
}
void normal_testCode4()
{
	int size = 10;
	std::vector<int> tester1, tester2;
	for (size_t i = 0; i < size; i++)
	{
		tester1.push_back(i);
		tester2.push_back(-1);
	}
	int counter = 0;
	for each (int var in tester1)
	{
		tester2[counter] = var;
		counter++;
	}
	
	int lol = 0;
}

void AMP_testCode5()
{
	int A[] = {1,2,3,4,5,6};
	int **test;
	concurrency::array_view<int, 2> a(2,3, A);
	concurrency::parallel_for_each(
		// Define the compute domain, which is the set of threads that are created.
		a.extent,
		// Define the code to run on each thread on the accelerator.
		[=](concurrency::index<2> idx) restrict(amp)
	{
		int x = idx[0];
		int y = idx[1];
		a[x][y] = a[x][y] +1;
	}
	);
	test =  a.data();
	for (int row = 0; row < 2; row++) {
		for (int col = 0; col < 3; col++) {
			//std::cout << productMatrix[row*3 + col] << "  ";
			std::cout << a(row, col) << "  ";
		}
		std::cout << "\n";
	}
	int lol = 0;
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
	//AMP_testCode2();
	//AMP_testCode3();
	//normal_testCode4();
	//AMP_testCode4();
	AMP_testCode5();
	//kmeansKode();
	return 0;
}