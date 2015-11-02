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

std::vector<Matrix> GetVector()
{
	std::vector<Matrix> result;
	std::string line;
	std::ifstream myfile("AllRandom.mtx");
	if (myfile.is_open())
	{	
		while (getline(myfile, line))
		{
			Matrix tempResult;
			std::istringstream iss(line);
			iss >> tempResult.n_row >> tempResult.n_col >> tempResult.value;
			result.push_back(tempResult);
		}
		myfile.close();
	}

	return result;
}

void GetRowandColumn(int &row, int &column, std::vector<Matrix> mat)
{
	for (int i = 0; i < mat.size(); i++)
	{
		if (mat[i].n_row > row)
			row = mat[i].n_row;
		if (mat[i].n_col > column)
			column = mat[i].n_col;
	}
}

 