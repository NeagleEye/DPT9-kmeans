#ifndef Matrix_h
#define Matrix_h
#include "Matrix.h"
#endif
#include "RandomGenerator.h"

class Kmeans
{
public:
	//Initialze the Matrix
	Kmeans(int n_cluster, int cluster[], int columns, int rows);
	//destructur for matrix data
	~Kmeans();
	//Init the concept vectors
	void Initialize_CV(Matrix matrix);
	//assign new cluster based on distances.
	int Assign_Cluster(Matrix matrix);
	int InitAssignCluster(Matrix matrix);
	//Update the centroids of the clusters
	void Update_Centroids(Matrix matrix);
	//seperate the centroids of the concept vectors
	void Well_Separated_Centroids(Matrix matrix);
	//The generel K means, assign -> update until no new assignments
	void Generel_K_Means(Matrix matrix);
	//Compute the size of the cluster (how many point are pointing to a specific cluster)
	void Compute_Cluster_Size();
	//Coherence is based on the total cluster quality
	double Coherence(int n_clus);
	//Calculate the difference from each point to the nearest clusters
	double Delta_X(Matrix matrix, int x, int c_ID);
	//Update to the quality matrix with Delta_X
	void Update_Quality_Change_Mat(Matrix matrix, int c_ID);
	//void SetDebugger(bool debugoption){ debug = debugoption; };
private:
	RandomGenerator_MT19937 rand_gen;
	//sim_Mat normally contains the distance, Concept_vectors contain a point with addition of another concept vector.
	//cluster_Quality determines how likely the different points har close to the correct closter.
	double **sim_Mat, *normal_ConceptVectors, **concept_Vectors, **old_ConceptVectors, *cluster_quality, *cv_Norm;
	//Difference is the difference between two concept vectors, epsilon,delta and omega are user specific parameters adjusting the critic analysis of the program
	double *difference, epsilon = 0.001, delta = 0.000001, omega = 0.0, pre_Result, result, initial_obj_fun_val, **quality_change_mat;
	double fv_threshold;
	//Row = words
	//col = docs
	int /*number of clusters*/n_Clusters,col,row,
		/*indicator of how many elements belong to each cluster*/ *clusterSize, 
		/*pointer to which cluster the column belongs*/*cluster, 
		/*Estimate of how long it will take maximum*/EST_START = 5;
	void GPU_update_cluster_quality();
};

