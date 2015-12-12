#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "Kmeans.h"
#include "PrintMatrix.h"
#include <chrono>
#include <fstream>
#include <string>
using namespace std;

int main(int argc, char** argv)
{
	string s = "";
	int c = argc;
	if (argc != 2 && argc != 1)	{s = "AllRandom.mtx";}
	else{
		for (int i = 0; i < argc; ++i) {
			s = argv[i];
		}
	}
		
	int *cluster, n_clusters = 9;
	int x=0, y=0;
	Matrix matrix = GetVector(x,y,s);

	cluster = new int[matrix.GetColumns()];
	//Initialize Euclidean kmeans
	Kmeans k(n_clusters,cluster, matrix.GetColumns(), matrix.GetRows());

	//Calculate normal vectors on every column set.
	matrix.ComputeNormalVector();
	k.Initialize_CV(matrix);
	auto start = std::chrono::high_resolution_clock::now();
	matrix = k.Generel_K_Means(matrix);
	auto finish = std::chrono::high_resolution_clock::now();
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
	cout << "It took: " << milliseconds.count() << endl;

	std::ofstream myfile;
	myfile.open("KmeansCPP-CPU.txt");
	myfile << milliseconds.count();
	myfile.close();
	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	//PrintMatrix(matrix,x,y);
	//PrintMatrix_With_Cluster(matrix, x, y);
	return 0;
}