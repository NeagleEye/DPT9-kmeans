#include <vector>
#include <iostream>
#include <fstream>
#include "printMatrix.h"


const int sizeOfArray = 1000;

/*void PrintMatrix(std::vector<Matrix> input)
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
			if (input[position].n_row == i && input[position].n_col == j)
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

void DynamicPrintMatrix(Matrix input)
{
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "P1\n";
	myfile << input.GetRows() << " " << input.GetColumns() << "\n";
	int position = 0;
	for (int i = 0; i < input.GetRows(); i++)
	{
		int test = 0;
		for (int j = 0; j < input.GetColumns(); j++)
		{
			if (input.[position].n_row == i && input[position].n_col == j)
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

}*/