#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include <math.h>
#include <iostream>
#include <vector>



void ComputeNormalVector(double normalVector[], std::vector<Matrix> mat, int rows, int numberofColumn)
{
	if ( normalVector == NULL)
	{
		normalVector = new double[numberofColumn];
		//memory_used += m_col*sizeof(float);
	}
	for (int i = 0; i < numberofColumn; i++)
	{
		normalVector[i] = 0.0;
		for (int j = 0; j < rows; j++)
			normalVector[i] += (mat[j].value) * (mat[j].value);
	}
}