#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "Kmeans.h"
#include "PrintMatrix.h"

using namespace std;

int main()
{
	int *cluster, n_clusters = 4;
	int x=0, y=0;
	Matrix matrix = GetVector(x,y);

	cluster = new int[matrix.GetColumns()];
	//Initialize Euclidean kmeans
	Kmeans k(n_clusters,cluster, matrix.GetColumns(), matrix.GetRows());

	//Calculate normal vectors on every column set.
	matrix.ComputeNormalVector();
	k.Initialize_CV(matrix);
	matrix = k.Generel_K_Means(matrix);
	cout << matrix.getClusterID(0) << endl;
	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	PrintMatrix(matrix,x,y);

	return 0;
}