#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include "RandomGenerator.h"

class Kmeans
{
public:
	Kmeans(int n_cluster, int cluster[], int columns, int rows);
	~Kmeans();
	/*void SetEmptyDocs(int n_E_Docs, int *e_D_ID)
	{
		n_Empty_Docs = n_E_Docs;
		empty_Docs_ID = e_D_ID;
	};*/
	void Initialize_CV(Matrix matrix);
	int Assign_Cluster(Matrix matrix, bool simi_est);
	void Update_Centroids(Matrix matrix);
	void Well_Separated_Centroids(Matrix matrix);
	void General_K_Means(Matrix matrix);
	void Compute_Cluster_Size();
	double Coherence(int n_clus);
	double Delta_X(Matrix matrix, int x, int c_ID);
	void Update_Quality_Change_Mat(Matrix matrix, int c_ID);
private:
	RandomGenerator_MT19937 rand_gen;
	double **sim_Mat, *normal_ConceptVectors, **concept_Vectors, **old_ConceptVectors, *cluster_quality, *cv_Norm;
	double *difference, epsilon = 0.001, delta = 0.000001, omega = 0.0, pre_Result, result, initial_obj_fun_val, **quality_change_mat;
	double fv_threshold;
	//Row = words
	//col = docs
	int n_Clusters,col,row,*clusterSize,*cluster,/* n_Empty_Docs, *empty_Docs_ID,*/ EST_START = 5, f_v_times = 0;
	bool stablized;
};

