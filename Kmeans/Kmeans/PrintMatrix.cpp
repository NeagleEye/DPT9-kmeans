#include <vector>
#include <iostream>
#include <fstream>
#include "printMatrix.h"


const int sizeOfArray = 1000;

void PrintMatrix(std::vector<Matrix> imput)
{
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "P1\n";
	myfile << sizeOfArray << " " << sizeOfArray << "\n";
	int position = 0;
	for (int i = 0; i < sizeOfArray; i++)
	{
		int test = 0;
		for (int j = 0; j < sizeOfArray; j++)
		{
			if (imput[position].n_row == i && imput[position].n_col == j)
			{
				myfile << "1 ";
				position++;
			}
			else
			{
				myfile << "0 ";
			}
		}
		myfile << "\n";
	}
	myfile.close();

}