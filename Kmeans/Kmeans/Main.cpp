#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "PrintMatrix.h"

using namespace std;

int main()
{
	int col=0, row=0;
	vector<Matrix> matrix = GetVector();



	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	GetRowandColumn(row, col, matrix);
	DynamicPrintMatrix(matrix,row,col);
	return 0;
}