/*   We have based our code on original code from Yuqiang Guan and 
     the original source code of the program is released under the GNU Public License (GPL)
     from:
     http://www.dataminingresearch.com/index.php/2010/06/gmeans-clustering-software-compatible-with-gcc-4/
     Implementation of Euclidean K-means 
     Copyright (c) 2003, Yuqiang Guan
*/
#include <Vector>
//Should only be 16 bytes per data.
//This will perhaps be changed to a struct
/*struct Matrix
{
	int n_row, n_col;
	double value, normalVector;
};*/
class Matrix
{
public:
	Matrix(int row, int col, double **val);
	~Matrix();
	void ComputeNormalVector();
	int GetRows(){ return n_row_elements; };
	int GetColumns(){ return n_col; };
	
	/*
	Getval(i,j)
	getval(0,0) = 1
	getval(1,0) = 2
	getval(0,1) = 3
	getval(1,1) = 4

	mtxfiles values (beyond first line)
	1 2
	3 4
	*/
	double GetVal(int i, int j) { return value[i][j]; }
	void Ith_Add_CV(int i, double *CV);

	//Calculate distance
	double Euc_Dis(double *x, int i, double norm_x);
	void Euc_Dis(double *x, double norm_x, double *result);
	double GetNorm(int i) { return normalVector[i]; }
	void PassCluster(int *clus);
	int getClusterID(int position){ return cluster[position]; };
	void setClusterID(int position, int value){ cluster[position] = value; }
private:
	//Row elements = value value value (3d or) value value (2d can be xd)
	//Column how many datasets is in the dataset
	int n_row_elements, n_col;
	//**value = value[row][col] (rowth element, column)
	//normalVector = normalvector over row of elements (vector)
	double **value, *normalVector;
	//This next part is just to split up the cluster visually
	std::vector<int> cluster;
};
