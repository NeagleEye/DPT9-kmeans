#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>

#define ROWS 7

std::vector<std::string> split(std::string str, char delimiter) {
	std::vector<std::string> internal;
	std::stringstream ss(str); // Turn the string into a stream.
	std::string tok;
	
	while (getline(ss, tok, delimiter)) {
		internal.push_back(tok);
	}

	return internal;
}

int  main(int argc, char *argv[])
{
	std::string s, line, fname = "i5-3320m-635m-8gbRam1600mhz-win8-1-AlgAvgTime.csv";//argv[1];
	std::ifstream thisfile(fname);
	std::string* MultiStrings;//This is just the number corresponding to our test this is specific code.
	std::vector<std::string> sep;
	std::vector<std::vector<std::string>> Multisplit;
	char start = 'a';
	start--;
	if (thisfile.is_open())
	{
		
		while (std::getline(thisfile, line))
		{
			sep = split(line, ',');
			Multisplit.push_back(sep);
			s = "";
		}
		
			std::ofstream NewFile;
			NewFile.open(fname + std::to_string(1) + ".dat");
			for (int it = 0; it < Multisplit[1].size(); ++it)
			{
				for (int i = 1; i < Multisplit.size(); ++i)
				{
					if (it == 0 && i == 1)
						NewFile << "x " << (char)(start+i);
					else if(it == 0)
						NewFile << " " << (char)(start + i);
					else
						NewFile << it << " " << Multisplit[i][it];
				}
				NewFile << std::endl;
			}
			NewFile.close();
	}

	std::cout << "DONE";
	return 0;
}