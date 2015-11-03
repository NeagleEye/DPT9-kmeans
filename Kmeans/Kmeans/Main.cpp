#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "Kmeans.h"
//#include "PrintMatrix.h"

using namespace std;

int main()
{
	int *cluster,*e_d_ID = NULL, n_Empty_Docs=0, n_clusters = 4;
	
	Matrix matrix = GetVector();

	e_d_ID = new int[n_Empty_Docs + 1];
	e_d_ID[0] = matrix.GetColumns();
	cluster = new int[matrix.GetColumns()];
	//Initialize Euclidean kmeans
	Kmeans k(n_clusters,cluster, matrix.GetColumns(), matrix.GetRows());
	//Calculate normal vectors on every column set.
	matrix.ComputeNormalVector();
	k.SetEmptyDocs(n_Empty_Docs, e_d_ID);

	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	//DynamicPrintMatrix(matrix,row,col);
	return 0;
}