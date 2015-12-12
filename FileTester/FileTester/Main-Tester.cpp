#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#define NumberofTests 10
#define Programs 2 // 3 to include CUDA
#define FPrograms 3
#define Matrix 4
#define TimeOut 900000 // 15 mins
void _tmain(int argc, TCHAR *argv[])
{
	std::string line;
	float time;
	std::ofstream myfile;
	myfile.open("Tester.txt");
	myfile << "\\begin{table}[]" << std::endl;
	myfile << "\\centering" << std::endl;
	myfile << "\\caption{This table shows " << NumberofTests << " of the different tests.}" << std::endl;
	myfile << "\\label{Tabtests}" << std::endl;
	myfile << "\\begin{tabular} {|" ;
	for (int i = 0; i < NumberofTests+1; i++)
	{
		myfile << "l|";
	}
	myfile << "}" << std::endl;
	myfile << "\\hline" << std::endl;
	if (argc != 2)
	{
		printf("Usage: %s [cmdline]\n", argv[0]);
		//return;
	}
	std::string name,filename, fFile= "AllRandom.mtx", cmd = "";
	int number;
	float average;
	LPCWSTR NameofProgram = L"";
	LPWSTR command = L"", LfFile = L"AllRandom.mtx";
	for (int k = 0; k < Programs; k++)
	{

		if (k == 0)
		{
			NameofProgram = L"KmeansCPP-CPU.exe";
			name = "Kmeans C++ single CPU ";
			filename = "KmeansCPP-CPU.txt";
		}
		else if (k == 1)
		{
			NameofProgram = L"Kmeans-AMP.exe";
			name = "Kmeans C++ AMP ";
			filename = "Kmeans-AMP.txt";
		}
		else if (k == 2)
		{
			NameofProgram = L"Kmeans-Cuda.exe";
			name = "Kmeans C++ Cuda ";
			filename = "Kmeans-Cuda.txt";
		}

		for (int j = 0; j < Matrix; j++)
		{

			if (j == 0)
			{
				command = L"AllRandom-10000.mtx";
				number = 10000;
			}
			else if (j == 1)
			{
				command = L"AllRandom-99856.mtx";
				number = 99856;
			}
			else if (j == 2)
			{
				command = L"AllRandom-1000000.mtx";
				number = 1000000;
			}
			else if (j == 3)
			{
				command = L"AllRandom-9998244.mtx";
				number = 9998244;
			}

			myfile << name << std::to_string(number) << " & ";

			average = 0;

			for (int i = 0; i < NumberofTests; i++)
			{
				time = 0;
				STARTUPINFO si;
				PROCESS_INFORMATION pi;

				ZeroMemory(&si, sizeof(si));
				si.cb = sizeof(si);
				ZeroMemory(&pi, sizeof(pi));

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

				std::ifstream thisfile(filename);
				if (thisfile.is_open())
				{
					thisfile >> time;
				}
				thisfile.close();
				average += time;
				if (remove(filename.c_str()) != 0)
				{
					myfile << "f/TO" << " & ";
				}
				else
				{
					myfile << time << " ms" << " & ";
				}

			}
			myfile << "Average: "<< (average/NumberofTests) << " ms" << " \\\\ \\hline" << std::endl;
		}
	}

	for (int k = 0; k < FPrograms; k++)
	{

		if (k == 0)
		{
			NameofProgram = L"Kmeans_FSCPU.exe";
			name = "Kmeans F# single CPU ";
			filename = "Kmeans_FSCPU.txt";
		}
		else if (k == 1)
		{
			NameofProgram = L"Kmeans_FMCPU.exe";
			name = "Kmeans F# multi CPU ";
			filename = "Kmeans_FMCPU.txt";
		}
		else if (k == 2)
		{
			NameofProgram = L"Kmeans_FSharp_GPU.exe";
			name = "Kmeans F# Brahma(Opencl) ";
			filename = "Kmeans-FOpenCL.txt";
		}

		for (int j = 0; j < Matrix; j++)
		{

			if (j == 0)
			{
				if (remove(fFile.c_str()) != 0)
				{/*Do nothing*/ }

				command = L"AllRandom-10000.mtx";
				CopyFile(command, LfFile,FALSE);
				number = 10000;
			}
			else if (j == 1)
			{
				if (remove(fFile.c_str()) != 0)
				{/*Do nothing*/	}
				command = L"AllRandom-99856.mtx";
				CopyFile(command, LfFile, FALSE);
				number = 99856;
			}
			else if (j == 2)
			{
				if (remove(fFile.c_str()) != 0)
				{/*Do nothing*/	}
				command = L"AllRandom-1000000.mtx";
				CopyFile(command, LfFile, FALSE);
				number = 1000000;
			}
			else if (j == 3)
			{
				if (remove(fFile.c_str()) != 0)
				{/*Do nothing*/	}
				command = L"AllRandom-9998244.mtx";
				CopyFile(command, LfFile, FALSE);
				number = 9998244;
			}

			average = 0;
			myfile << name << std::to_string(number) << " & ";
			for (int i = 0; i < NumberofTests; i++)
			{
				time = 0;
				STARTUPINFO si;
				PROCESS_INFORMATION pi;

				ZeroMemory(&si, sizeof(si));
				si.cb = sizeof(si);
				ZeroMemory(&pi, sizeof(pi));

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

				std::ifstream thisfile(filename);
				if (thisfile.is_open())
				{
					thisfile >> time;
				}
				thisfile.close();
				average += time;
				if (remove(filename.c_str()) != 0)
				{
					myfile << "f/TO" << " & ";
				}
				else
				{
					myfile << time << " ms" << " & ";
				}

			}
			myfile << "Average: " << (average / NumberofTests) << " ms" << " \\\\ \\hline" << std::endl;
		}
	}
	 


	myfile << "\end{tabular}" << std::endl;
	myfile << "\end{table}" << std::endl;
	myfile.close();
}