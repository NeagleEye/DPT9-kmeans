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
	int XPosition, YPosition;
	bool noMorePoints = false;
	for (int i = 0; i <= MAX_X; i++)
	{
		for (int j = 0; j <= MAX_Y; j++)
		{

			if (!noMorePoints)
			{
				XPosition = input.GetVal(0, position);
				YPosition = input.GetVal(1, position);
				if (XPosition == i && YPosition == j)
				{
					myfile << "1 ";
					position++;
					while (XPosition == input.GetVal(0, position) && YPosition == input.GetVal(1, position))
					{
						position++;
					}
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

void PrintMatrix_With_Cluster(Matrix input, int MAX_X, int MAX_Y)
{
	std::ofstream myfile;
	myfile.open("exampleCluster.txt");
	myfile << "P3\n";
	myfile << MAX_X << " " << MAX_Y << "\n";
	int test = input.GetColumns();
	int position = 0;
	bool noMorePoints = false;
	std::vector<std::string> color = { "255 255 255 ", "255   0   0 ", "0 255   0 ", "0   0 255 ", "255 255   0 ", "0   0   0 " };
	for (int i = 0; i <= MAX_X; i++)
	{
		for (int j = 0; j <= MAX_Y; j++)
		{
			if (!noMorePoints)
			{
				int XPosition = input.GetVal(0, position);
				int YPosition = input.GetVal(1, position);
				if (XPosition == i && YPosition == j)
				{
					if (input.getClusterID(position) == 0)
					{
						myfile << "255   0   0 ";
					}
					else if (input.getClusterID(position) == 1)
					{
						myfile << "  0 255   0 ";
					}
					else if (input.getClusterID(position) == 2)
					{
						myfile << "  0   0 255 ";
					}
					else if (input.getClusterID(position) == 3)
					{
						myfile << "255 255   0 ";
					}
					else
					{
						myfile << "  0   0   0 ";
					}
					position++;
					while (XPosition == input.GetVal(0, position) && YPosition == input.GetVal(1, position))
					{
						position++;
					}
				}
				else
				{
					myfile << "255 255 255 ";
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