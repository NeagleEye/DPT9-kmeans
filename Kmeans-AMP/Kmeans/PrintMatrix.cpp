#include <vector>
#include <iostream>
#include <fstream>
#include "printMatrix.h"

void PrintMatrix(Matrix input, int MAX_X, int MAX_Y)
{
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "P1\n";
	myfile << MAX_X << " " << MAX_Y << "\n";
	int test = input.GetColumns();
	int position = 0;
	bool noMorePoints = false;
	for (int i = 0; i < MAX_Y; i++)
	{
		for (int j = 0; j < MAX_X; j++)
		{
			if (!noMorePoints)
			{
				int XPosition = input.GetVal(0, position);
				int YPosition = input.GetVal(1, position);
				if (XPosition == i && YPosition == j)
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