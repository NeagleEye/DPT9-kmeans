#include <Vector>
//Should only be 16 bytes per data.
//This will perhaps be changed to a struct
struct Matrix
{
	int n_row, n_col;
	double value, normalVector;
};

void ComputeNormalVector(double normalVector[], std::vector<Matrix> mat, int rows, int numberofColumn);
