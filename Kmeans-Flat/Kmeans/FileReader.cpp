#include <iostream>
#include <vector>
#include <fstream>
#include "FileReader.h"
#include <sstream>
#include <string>

/*
*This should read file that looks like .mtx
*
*/

Matrix GetVector()
{
	int a, b;
	return GetVector(a, b);
}

Matrix GetVector(int &x, int &y)
{
	Mat result;
	std::string line;
	std::ifstream myfile("AllRandom.mtx");
	if (myfile.is_open())
	{
		//the reader takes the first number and uses as rows >> jump over space, uses next number as column >> jumps over space to next number on next line.
		myfile >> result.n_row >> result.n_col;
		//Init the value to have nrow elements [inner element]
		result.val = new double[result.n_row * result.n_col];

		for (int i = 0; i < result.n_col; i++)
		{
			for (int j = 0; j < result.n_row; j++)
			{
				myfile >> result.val[j*result.n_col+i];
				if (j == 0){ if (result.val[j*result.n_col + i] > x)x = result.val[j*result.n_col + i]; }
				else if (j == 1){ if (result.val[j*result.n_col + i] > y)y = result.val[j*result.n_col + i]; }
			}
		}
		myfile.close();
	}

	Matrix matrix(result.n_row, result.n_col, result.val);
	return matrix;
}