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
	Matrix(int row, int col, double *val);
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
	double GetVal(int i) { return value[i]; }
	void Ith_Add_CV(int i, double *CV, int cluster);

	//Calculate distance
	double Euc_Dis(double *x, int i, double norm_x, int cluster);
	void Euc_Dis(double *x, double norm_x, double *result, int cluster, int n_cluster);
	void Euc_Dis(double *x, double *normal_ConceptVectors, double *result, int *cluster, int n_cluster);
	void Euc_Dis_All(double *x, double norm_x, double *result, int n_cluster);
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
	double *value, *normalVector;
	std::vector<int> cluster;
};