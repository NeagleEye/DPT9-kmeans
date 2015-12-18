#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <vector>
#define NumberofTests 10
#define Programs 2 // 2 to exclude CUDA 3 to include CUDA
#define FPrograms 3 // 3 for all tests
#define Matrix 4
#define TimeOut 900000 // 15 mins
void _tmain(int argc, TCHAR *argv[])
{
	std::vector<std::vector<std::string>> PreCSVAlg;
	std::vector<std::vector<std::string>> PreCSVAlgAverage;
	std::vector<std::vector<std::string>> PreFullCSV;
	std::vector<std::vector<std::string>> PreFullCSVAverage;
	PreCSVAlg.resize(Programs + FPrograms+1);
	PreCSVAlgAverage.resize(Programs + FPrograms+1);
	PreFullCSV.resize(Programs + FPrograms+1);
	PreFullCSVAverage.resize(Programs + FPrograms+1);

	for (int i = 0; i < Programs; i++)
	{
		PreCSVAlg[i+1].resize(Programs + FPrograms + 1);
		PreCSVAlgAverage[i+1].resize(Programs + FPrograms + 1);
		PreFullCSV[i+1].resize(Programs + FPrograms + 1);
		PreFullCSVAverage[i+1].resize(Programs + FPrograms + 1);
	}

	for (int i = 0; i < FPrograms; i++)
	{
		PreCSVAlg[i+Programs+1].resize(Programs + FPrograms + 1);
		PreCSVAlgAverage[i + Programs+1].resize(Programs + FPrograms + 1);
		PreFullCSV[i + Programs+1].resize(Programs + FPrograms + 1);
		PreFullCSVAverage[i + Programs+1].resize(Programs + FPrograms + 1);
	}

	std::string line;
	float time,ftime;
	std::ofstream myfile;
	myfile.open("KmeansAlgTime-test.txt");
	std::ofstream FullTest;
	FullTest.open("KmeansFullTime-test.txt");

	FullTest << "\\begin{table}[]" << std::endl;
	FullTest << "\\centering" << std::endl;
	FullTest << "\\begin{tabular} {|";
	myfile << "\\begin{table}[]" << std::endl;
	myfile << "\\centering" << std::endl;
	myfile << "\\begin{tabular} {|" ;
	for (int i = 0; i < NumberofTests+2; i++)
	{
		myfile << "L{1.1cm}|";
		FullTest << "L{1.1cm}|";
	}
	myfile << "}" << std::endl;
	myfile << "\\hline" << std::endl;
	FullTest << "}" << std::endl;
	FullTest << "\\hline" << std::endl;
	
	PreCSVAlg[0].push_back("Average Time in miliseconds");
	PreFullCSV[0].push_back("Average Time in seconds");
	PreCSVAlgAverage[0].push_back("Average Time in miliseconds");
	PreFullCSVAverage[0].push_back("Average Time in seconds");

	if (argc != 2)
	{
		printf("Usage: %s [cmdline]\n", argv[0]);
		//return;
	}
	std::string name,filename, fFile= "AllRandom.mtx", cmd = "";
	
	int number;
	float average;
	float fullAverage;
	LPCWSTR NameofProgram = L"";
	LPWSTR command = L"", LfFile = L"AllRandom.mtx";

	for (int j = 0; j < Matrix; j++)
	{

		if (j == 0)
		{
			if (remove(fFile.c_str()) != 0)
			{/*Do nothing*/
			}

			command = L"AllRandom-10000.mtx";
			CopyFile(command, LfFile, FALSE);
			number = 10000;
		}
		else if (j == 1)
		{
			if (remove(fFile.c_str()) != 0)
			{/*Do nothing*/
			}
			command = L"AllRandom-99856.mtx";
			CopyFile(command, LfFile, FALSE);
			number = 99856;
		}
		else if (j == 2)
		{
			if (remove(fFile.c_str()) != 0)
			{/*Do nothing*/
			}
			command = L"AllRandom-1000000.mtx";
			CopyFile(command, LfFile, FALSE);
			number = 1000000;
		}
		else if (j == 3)
		{
			if (remove(fFile.c_str()) != 0)
			{/*Do nothing*/
			}
			command = L"AllRandom-9998244.mtx";
			CopyFile(command, LfFile, FALSE);
			number = 9998244;
		}

		for (int k = 0; k < Programs; k++)
		{

			if (k == 0)
			{
				NameofProgram = L"KmeansCPP-CPU.exe";
				name = "C++ Single Thread ";
				filename = "KmeansCPP-CPU.txt";
				PreCSVAlg[k + 1][0] = "C++ Single Thread";
				PreCSVAlgAverage[k + 1][0] = "C++ Single Thread";
				PreFullCSV[k + 1][0] = "C++ Single Thread";
				PreFullCSVAverage[k + 1][0] = "C++ Single Thread";
			}
			else if (k == 1)
			{
				NameofProgram = L"Kmeans-AMP.exe";
				name = "C++ AMP ";
				filename = "Kmeans-AMP.txt";
				PreCSVAlg[k + 1][0] = "C++ AMP";
				PreCSVAlgAverage[k + 1][0] = "C++ AMP";
				PreFullCSV[k + 1][0] = "C++ AMP";
				PreFullCSVAverage[k + 1][0] = "C++ AMP";
			}
			else if (k == 2)
			{
				NameofProgram = L"Kmeans-Cuda.exe";
				name = "C++ CUDA ";
				filename = "Kmeans-Cuda.txt";
				PreCSVAlg[k + 1][0] = "C++ CUDA";
				PreCSVAlgAverage[k + 1][0] = "C++ CUDA";
				PreFullCSV[k + 1][0] = "C++ CUDA";
				PreFullCSVAverage[k + 1][0] = "C++ CUDA";
			}

		

			myfile << name << std::to_string(number) << " & ";
			FullTest << name << std::to_string(number) << " & ";

			PreCSVAlg[k+1].push_back(",");
			PreCSVAlgAverage[k+1].push_back(",");
			PreFullCSV[k+1].push_back(",");
			PreFullCSVAverage[k+1].push_back(",");

			average = 0;
			fullAverage = 0;

			for (int i = 0; i < NumberofTests; i++)
			{
				time = 0;
				ftime = 0;
				STARTUPINFO si;
				PROCESS_INFORMATION pi;

				ZeroMemory(&si, sizeof(si));
				si.cb = sizeof(si);
				ZeroMemory(&pi, sizeof(pi));

				auto start = std::chrono::high_resolution_clock::now();
				// Start the child process. 
				if (!CreateProcess(
					NameofProgram,   // No module name (use command line) //
					command,        // Command line
					NULL,           // Process handle not inheritable
					NULL,           // Thread handle not inheritable
					FALSE,          // Set handle inheritance to FALSE
					0,              // No creation flags
					NULL,           // Use parent's environment block
					NULL,           // Use parent's starting directory 
					&si,            // Pointer to STARTUPINFO structure
					&pi)           // Pointer to PROCESS_INFORMATION structure
					)
				{
					printf("CreateProcess failed (%d).\n", GetLastError());
					//return;
				}
				// Wait until child process exits.
				
				WaitForSingleObject(pi.hProcess, TimeOut);

				// Close process and thread handles. 
				CloseHandle(pi.hProcess);
				CloseHandle(pi.hThread);
				auto finish = std::chrono::high_resolution_clock::now();
				auto seconds = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
				ftime = seconds.count();
				std::ifstream thisfile(filename);
				if (thisfile.is_open())
				{
					thisfile >> time;
				}
				thisfile.close();
				average += time;
				fullAverage += ftime;
				if (remove(filename.c_str()) != 0)
				{
					myfile << "f/TO" << " & ";
					FullTest << "f/TO" << " & ";
					PreCSVAlg[k+1].push_back("f/TO,");
					PreFullCSV[k+1].push_back("f/TO,");
				}
				else
				{
					myfile << time << " ms" << " & ";
					FullTest << ftime << " sec" << " & ";
					if (i == NumberofTests - 1)
					{
						PreCSVAlg[k + 1].push_back(std::to_string((int)time));
						PreFullCSV[k + 1].push_back(std::to_string((int)ftime));
					}
					else
					{
						PreCSVAlg[k + 1].push_back(std::to_string((int)time));
						PreFullCSV[k + 1].push_back(std::to_string((int)ftime));
						PreCSVAlg[k + 1].push_back(",");
						PreFullCSV[k + 1].push_back(",");
					}
				}

			}
			myfile << "Average: " << (average / NumberofTests) << " ms" << " \\\\ \\hline" << std::endl;
			FullTest << "Average: " << (fullAverage / NumberofTests) << " sec" << " \\\\ \\hline" << std::endl;
			PreCSVAlgAverage[k+1].push_back(std::to_string((int)(average / NumberofTests)));
			PreFullCSVAverage[k+1].push_back(std::to_string((int)(fullAverage / NumberofTests)));
		}

		for (int k = 0; k < FPrograms; k++)
		{

			if (k == 0)
			{
				NameofProgram = L"Kmeans_FSCPU.exe";
				name = "F\\# Single Thread ";
				filename = "Kmeans_FSCPU.txt";
				PreCSVAlg[k + Programs + 1][0] = "F# Single Thread";
				PreCSVAlgAverage[k + Programs + 1][0] = "F# Single Thread";
				PreFullCSV[k + Programs + 1][0] = "F# Single Thread";
				PreFullCSVAverage[k + Programs + 1][0] = "F# Single Thread";
			}
			else if (k == 1)
			{
				NameofProgram = L"Kmeans_FMCPU.exe";
				name = "F\\# Multi Thread ";
				filename = "Kmeans_FMCPU.txt";
				PreCSVAlg[k + Programs + 1][0] = "F# Multi Thread";
				PreCSVAlgAverage[k + Programs + 1][0] = "F# Multi Thread";
				PreFullCSV[k + Programs + 1][0] = "F# Multi Thread";
				PreFullCSVAverage[k + Programs + 1][0] = "F# Multi Thread";
			}
			else if (k == 2)
			{
				NameofProgram = L"Kmeans_FSharp_GPU.exe";
				name = "F\\# Brahma (OpenCL) ";
				filename = "Kmeans-FOpenCL.txt";
				PreCSVAlg[k + Programs + 1][0] = "F# Brahma (OpenCL)";
				PreCSVAlgAverage[k + Programs + 1][0] = "F# Brahma (OpenCL)";
				PreFullCSV[k + Programs + 1][0] = "F# Brahma (OpenCL)";
				PreFullCSVAverage[k + Programs + 1][0] = "F# Brahma (OpenCL)";
			}

			myfile << name << std::to_string(number) << " & ";
			FullTest << name << std::to_string(number) << " & ";

			PreCSVAlg[k + Programs+1].push_back(",");
			PreCSVAlgAverage[k + Programs+1].push_back(",");
			PreFullCSV[k + Programs+1].push_back(",");
			PreFullCSVAverage[k + Programs+1].push_back(",");

			average = 0;
			fullAverage = 0;

			for (int i = 0; i < NumberofTests; i++)
			{
				time = 0;
				ftime = 0;
				STARTUPINFO si;
				PROCESS_INFORMATION pi;

				ZeroMemory(&si, sizeof(si));
				si.cb = sizeof(si);
				ZeroMemory(&pi, sizeof(pi));

				auto start = std::chrono::high_resolution_clock::now();
				// Start the child process. 
				if (!CreateProcess(
					NameofProgram,   // No module name (use command line) //
					NULL,        // Command line
					NULL,           // Process handle not inheritable
					NULL,           // Thread handle not inheritable
					FALSE,          // Set handle inheritance to FALSE
					0,              // No creation flags
					NULL,           // Use parent's environment block
					NULL,           // Use parent's starting directory 
					&si,            // Pointer to STARTUPINFO structure
					&pi)           // Pointer to PROCESS_INFORMATION structure
					)
				{
					printf("CreateProcess failed (%d).\n", GetLastError());
					//return;
				}
				// Wait until child process exits.
				WaitForSingleObject(pi.hProcess, TimeOut);

				// Close process and thread handles. 
				CloseHandle(pi.hProcess);
				CloseHandle(pi.hThread);
				auto finish = std::chrono::high_resolution_clock::now();
				auto seconds = std::chrono::duration_cast<std::chrono::seconds>(finish - start);
				ftime = seconds.count();
				std::ifstream thisfile(filename);
				if (thisfile.is_open())
				{
					thisfile >> time;
				}
				thisfile.close();
				average += time;
				fullAverage += ftime;
				if (remove(filename.c_str()) != 0)
				{
					myfile << "f/TO" << " & ";
					FullTest << "f/TO" << " & ";
					PreCSVAlg[k + Programs+1].push_back("f/TO,");
					PreFullCSV[k + Programs+1].push_back("f/TO,");
				}
				else
				{
					myfile << time << " ms" << " & ";
					FullTest << ftime << " sec" << " & ";
					if (i == NumberofTests-1)
					{
						PreCSVAlg[k + Programs + 1].push_back(std::to_string((int)time));
						PreFullCSV[k + Programs + 1].push_back(std::to_string((int)ftime));
					}
					else
					{
						PreCSVAlg[k + Programs + 1].push_back(std::to_string((int)time));
						PreFullCSV[k + Programs + 1].push_back(std::to_string((int)ftime));
						PreCSVAlg[k + Programs + 1].push_back(",");
						PreFullCSV[k + Programs + 1].push_back(",");
					}
					
				}

			}
			myfile << "Average: " << (average / NumberofTests) << " ms" << " \\\\ \\hline" << std::endl;
			FullTest << "Average: " << (fullAverage / NumberofTests) << " sec" << " \\\\ \\hline" << std::endl;
			PreCSVAlgAverage[k + Programs+1].push_back(std::to_string((int)(average / NumberofTests)));
			PreFullCSVAverage[k + Programs+1].push_back(std::to_string((int)(fullAverage / NumberofTests)));
		}	
	}
	 


	myfile << "\\end{tabular}" << std::endl;
	myfile << "\\caption{This table shows " << NumberofTests << " of the different tests.}" << std::endl;
	myfile << "\\label{Tabtests}" << std::endl;
	myfile << "\\end{table}" << std::endl;
	FullTest << "\\end{tabular}" << std::endl;
	FullTest << "\\caption{This table shows " << NumberofTests << " of the different tests.}" << std::endl;
	FullTest << "\\label{Tabtests}" << std::endl;
	FullTest << "\\end{table}" << std::endl;
	myfile.close();
	FullTest.close();


	std::ofstream AlgCSV;
	AlgCSV.open("KmeansAlgTime.csv");
	std::ofstream AlgCSVAvg;
	AlgCSVAvg.open("KmeansAlgAvgTime.csv");
	std::ofstream AlgFullCSV;
	AlgFullCSV.open("KmeansFullTime.csv");
	std::ofstream AlgFullCSVAvg;
	AlgFullCSVAvg.open("KmeansFullAvgTime.csv");

	for (int i = 0; i < PreCSVAlg.size(); i++)
	{
		for (int j = 0; j < PreCSVAlg[i].size(); j++)
			AlgCSV << PreCSVAlg[i][j];
		AlgCSV << std::endl;
	}

	for (int i = 0; i < PreCSVAlgAverage.size(); i++)
	{
		for (int j = 0; j < PreCSVAlgAverage[i].size(); j++)
			AlgCSVAvg << PreCSVAlgAverage[i][j];
		AlgCSVAvg << std::endl;
	}

	for (int i = 0; i < PreFullCSV.size(); i++)
	{
		for (int j = 0; j < PreFullCSV[i].size(); j++)
			AlgFullCSV << PreFullCSV[i][j];
		AlgFullCSV << std::endl;
	}

	for (int i = 0; i < PreFullCSVAverage.size(); i++)
	{
		for (int j = 0; j < PreFullCSVAverage[i].size(); j++)
			AlgFullCSVAvg << PreFullCSVAverage[i][j];
		AlgFullCSVAvg << std::endl;
	}

	AlgCSV.close();

	AlgCSVAvg.close();

	AlgFullCSV.close();

	AlgFullCSVAvg.close();

}