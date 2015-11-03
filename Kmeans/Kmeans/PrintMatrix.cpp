#include <vector>
#include <iostream>
#include <fstream>
#include "printMatrix.h"

void PrintMatrix(Matrix input)
{
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "P1\n";
	myfile << input.GetColumns() << " " << input.GetColumns() << "\n";
	int position = 0;
	bool noMorePoints = false;
	for (int i = 0; i < input.GetColumns(); i++)
	{
		int test = 0;
		for (int j = 0; j < input.GetColumns(); j++)
		{
			if (!noMorePoints)
			{
				int XPosition = input.GetVal(0, position);
				int YPosition = input.GetVal(1, position);
				if (YPosition == i && XPosition == j)
				{
					myfile << "1 ";
					position++;
				}
				else
				{
					myfile << "0 ";
				}
				if (position >= input.GetColumns())
				{
					noMorePoints = true;
				}
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