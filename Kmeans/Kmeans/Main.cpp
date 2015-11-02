#include "FileReader.h"
#include <Vector>
#include <Iostream>
#include "PrintMatrix.h"

using namespace std;

int main()
{
	vector<Matrix> matrix = GetVector();
	PrintMatrix(matrix);
	return 0;