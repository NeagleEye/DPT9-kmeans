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

 