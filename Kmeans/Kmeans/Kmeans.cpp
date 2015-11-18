#include "Kmeans.h"
#include "MathonVectors.h"
#include <time.h>
#include <iostream>

Kmeans::Kmeans(int n_cluster, int clusterinit[], int columns, int rows)
{
	//Initialize values
	sim_Mat = new double*[n_cluster];
	for (int i = 0; i < n_cluster; i++)
		sim_Mat[i] = new double[columns];
	normal_ConceptVectors = new double[n_cluster];
	n_Clusters = n_cluster;
	cluster = clusterinit;
	col = columns;
	row = rows;
	cluster_quality = new double[n_Clusters];
	cv_Norm = cluster_quality;

	concept_Vectors = new double*[n_Clusters];
	for (int i = 0; i < n_Clusters; i++)
		concept_Vectors[i] = new double[rows];

	old_ConceptVectors = new double*[n_Clusters];
	for (int i = 0; i < n_Clusters; i++)
		old_ConceptVectors[i] = new double[rows];

	difference = new double[n_Clusters];


	clusterSize = new int[n_Clusters];


	/*if (!random_seeding)
		rand_gen.Set((unsigned)seed);
	else*/
	rand_gen.Set((unsigned)time(NULL));
}

//destructor remove all allocated memory for pointers.
Kmeans::~Kmeans()
{
	delete[] normal_ConceptVectors;

	int i;
	for (i = 0; i < n_Clusters; i++)
	{
		delete[] sim_Mat[i];
		delete[] concept_Vectors[i];
		delete[] old_ConceptVectors[i];
	}
	delete[] sim_Mat;
	delete[] concept_Vectors;
	delete[] old_ConceptVectors;

	delete[] cluster_quality;
	delete[] difference;
	delete[] clusterSize;

}

Matrix Kmeans::Generel_K_Means(Matrix matrix)
{
	//Kmeans will only work with its assigned values, whereas gmeans can modify kmeans.
	int n_Iters, i, j;
	bool no_assignment_change;

	n_Iters = 0;
	no_assignment_change = true;

	//DO while of the algorithm to assign to new cluster is closer, update the centroid.
	//Before we get here we will have used Well_Seperated_Centroids which will be the initial partition of data
	do
	{
		pre_Result = result;
		n_Iters++;
		//We estimate its a stable cluster if it can be finished within 5 steps
		if (n_Iters >EST_START)
			stabilized = true;

		//If there is no new assignment we have finished clustering.
		if (Assign_Cluster(matrix, stabilized) == 0){}
		else
		{
			/*There was an assignment thus we have to compute new centroids and update the size of each cluster,
			*as values will be assigned to other clusters*/
			no_assignment_change = false;
			
			Compute_Cluster_Size();

			//If unstable copy concept_vectors onto old_conceptVectors
			if (n_Iters >= EST_START)
				for (i = 0; i < n_Clusters; i++)
					for (j = 0; j < row; j++)
						old_ConceptVectors[i][j] = concept_Vectors[i][j];

			//Update Centroids based on the new clusters
			Update_Centroids(matrix);

			//Average vector for all clusters (concept_vectors now contain the average vector of each cluster)
			for (i = 0; i < n_Clusters; i++)
				average_vec(concept_Vectors[i], row, clusterSize[i]);
			//normal Vector gets the squared average vector
			for (i = 0; i < n_Clusters; i++)
				normal_ConceptVectors[i] = norm_2(concept_Vectors[i], row);
			//if unstable run
			if (n_Iters >= EST_START)
			{
				//Calculate the difference between the previous average vector versus the new vector
				for (i = 0; i < n_Clusters; i++)
				{
					difference[i] = 0.0;
					for (j = 0; j < row; j++)
						difference[i] += (old_ConceptVectors[i][j] - concept_Vectors[i][j]) * (old_ConceptVectors[i][j] - concept_Vectors[i][j]);
				}

				//returning distance between the squared average vector and the average vector to sim_mat
				if (n_Iters > EST_START)
					for (i = 0; i<col; i++)
						sim_Mat[cluster[i]][i] = matrix.Euc_Dis(concept_Vectors[cluster[i]], i, normal_ConceptVectors[cluster[i]]);
				else
					for (i = 0; i < n_Clusters; i++)
						matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);
			}
			else
				for (i = 0; i < n_Clusters; i++)//returning distance between the squared average vector and the average vector to sim_mat
					matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);

			//initialize cluster_quality
			for (i = 0; i<n_Clusters; i++)
				cluster_quality[i] = 0.0;

			//update the cluster quality based on the collected simulation matrix
			for (i = 0; i < col; i++)
			{
				cluster_quality[cluster[i]] += sim_Mat[cluster[i]][i];
			}
			//Coherence is based on the total cluster quality
			result = Coherence(n_Clusters);

			std::cout << "E";
		}//epsilon is a user defined function default set to 0.0001, initial_obj_fun_val is defined by the initial partioning.
	} while ((pre_Result - result) > epsilon*initial_obj_fun_val);
	std::cout << std::endl;

	//if the kmeans has run and it was stabil we retrieve the euclidean distance of concept_vectors and the normal_ConceptVectors
	if (stabilized)
		for (i = 0; i < n_Clusters; i++)
			matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);
	//if it required to assign new changes it will update quility change matrix
	if ((!no_assignment_change) && (f_v_times >0))
		for (i = 0; i<n_Clusters; i++)
			Update_Quality_Change_Mat(matrix, i);

	matrix.PassCluster(cluster);
	return matrix;
}

int Kmeans::Assign_Cluster(Matrix matrix, bool stabilized)
{

	int i, j, multi = 0, changed = 0, temp_Cluster_ID;
	double temp_sim;

	//if stabil (run less than 
	if (stabilized)
	{
		//Init the distance based on old distances
		for (i = 0; i < n_Clusters; i++)
			for (j = 0; j < col; j++)
				if (i != cluster[j])
					sim_Mat[i][j] += difference[i] - 2 * sqrt(difference[i] * sim_Mat[i][j]);

		for (i = 0; i < col; i++)
		{
			temp_sim = sim_Mat[cluster[i]][i];
			temp_Cluster_ID = cluster[i];

			for (j = 0; j < n_Clusters; j++)
				if (j != cluster[i])
				{
					//if current placement is furthere away than new possible placement, assign new cluster if new one is closer
					if (sim_Mat[j][i] < temp_sim)
					{
						multi++;
						//recalculate the new vector based on new formula
						sim_Mat[j][i] = matrix.Euc_Dis(concept_Vectors[j], i, normal_ConceptVectors[j]);
						//if current placement is furthere away than new possible placement, assign new cluster if new one is closer
						if (sim_Mat[j][i] < temp_sim)
						{
							temp_sim = sim_Mat[j][i];
							temp_Cluster_ID = j;
						}
					}
				}
			//Assign new cluster if closer than previous cluster
			if (temp_Cluster_ID != cluster[i])
			{
				cluster[i] = temp_Cluster_ID;
				sim_Mat[cluster[i]][i] = temp_sim;
				changed++;
			}
		}
	}
	//if unstable 
	else
	{
		for (i = 0; i < col; i++)
		{
			temp_sim = sim_Mat[cluster[i]][i];
			temp_Cluster_ID = cluster[i];

			for (j = 0; j < n_Clusters; j++)
				//if point does not belong to cluster do
				if (j != cluster[i])
				{
					multi++;
					//if current placement is furthere away than new possible placement, assign new cluster if new one is closer
					if (sim_Mat[j][i] < temp_sim)
					{
						temp_sim = sim_Mat[j][i];
						temp_Cluster_ID = j;
					}
				}
			//Assign new cluster if closer than previous cluster
			if (temp_Cluster_ID != cluster[i])
			{
				cluster[i] = temp_Cluster_ID;
				sim_Mat[cluster[i]][i] = temp_sim;
				changed++;
			}
		}
	}
	return changed;
}

//Initialize the Concept Vectors
void Kmeans::Initialize_CV(Matrix matrix)
{

	int i, j, k;

	Well_Separated_Centroids(matrix);

	// reset concept vectors

	for (i = 0; i < n_Clusters; i++)
		for (j = 0; j < row; j++)
			concept_Vectors[i][j] = 0.0;
	for (i = 0; i < col; i++)
	{
		//add current concept_vector to the original vector
		if ((cluster[i] >= 0) && (cluster[i] < n_Clusters))
			matrix.Ith_Add_CV(i, concept_Vectors[cluster[i]]);
		else
			cluster[i] = 0;
	}
	//compute how many points belongs to the different cluster
	Compute_Cluster_Size();

	//average vector for each cluster
	for (i = 0; i < n_Clusters; i++)
		average_vec(concept_Vectors[i], row, clusterSize[i]);

	//Compute the normal vector for n_clusters number of vectors = ^2 
	for (i = 0; i < n_Clusters; i++)
		normal_ConceptVectors[i] = norm_2(concept_Vectors[i], row);

	//calculate the distance from the concept vectors to the normal vectors
	for (i = 0; i < n_Clusters; i++)
		matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);

	//Init Cluster quality which is the distance between normal CV and concept_vectors
	for (i = 0; i<n_Clusters; i++)
		cluster_quality[i] = 0.0;
	k = 0;
	for (i = 0; i < col; i++)
	{
		cluster_quality[cluster[i]] += sim_Mat[cluster[i]][i];
	}
	//for (i = 0; i < n_Clusters; i++)
	//diff[i] = 0.0;

	//A random constant figured out based on cluster_quality
	initial_obj_fun_val = result = Coherence(n_Clusters);
	fv_threshold = -1.0*initial_obj_fun_val*delta;

	if (f_v_times >0)
	{
		//init and use quality change matrix (used to change quality Matrix)
		quality_change_mat = new double *[n_Clusters];
		// VT 2009-11-28
		for (int j = 0; j < n_Clusters; j++)
			quality_change_mat[j] = new double[col];

		for (i = 0; i < n_Clusters; i++)
			Update_Quality_Change_Mat(matrix, i);
	}
}

//Centroids should be separated
void Kmeans::Well_Separated_Centroids(Matrix matrix)
{
	int i, j, k, min_ind, *cv = new int[n_Clusters];
	double min, cos_sum;
	bool *mark = new bool[col];

	for (i = 0; i< col; i++)
		mark[i] = false;


	for (i = 0; i < n_Clusters; i++)
		for (j = 0; j < row; j++)
			concept_Vectors[i][j] = 0.0;

	//Random vectors generation
	do{
		cv[0] = rand_gen.GetUniformInt(col);
	} while (mark[cv[0]]);

	//add current concept_vector to the original vector
	matrix.Ith_Add_CV(cv[0], concept_Vectors[0]);
	mark[cv[0]] = true;

	//get normal CV
	normal_ConceptVectors[0] = matrix.GetNorm(cv[0]);
	//Euclidean Distance between the vectors
	matrix.Euc_Dis(concept_Vectors[0], normal_ConceptVectors[0], sim_Mat[0]);

	//Create random concept vectors (roughly in the same area)
	for (i = 1; i<n_Clusters; i++)
	{
		min_ind = 0;
		min = 0.0;
		for (j = 0; j<col; j++)
		{
			if (!mark[j])
			{
				cos_sum = 0.0;
				for (k = 0; k<i; k++)
					cos_sum += sim_Mat[k][j];
				if (cos_sum > min)
				{
					min = cos_sum;
					min_ind = j;
				}
			}
		}
		cv[i] = min_ind;
		matrix.Ith_Add_CV(cv[i], concept_Vectors[i]);

		normal_ConceptVectors[i] = matrix.GetNorm(cv[i]);
		matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);
		mark[cv[i]] = true;
	}

	//assign the points to clusters
	for (i = 0; i<col; i++)
		cluster[i] = 0;
	Assign_Cluster(matrix, false);

	delete[] cv;
	delete[] mark;
}

//Update the centroids
void Kmeans::Update_Centroids(Matrix matrix)
{

	int i, j;
	//reset concept_Vectors
	for (i = 0; i < n_Clusters; i++)
		for (j = 0; j < row; j++)
			concept_Vectors[i][j] = 0.0;
	//Calculate new concept_Vectors
	//get all the vectors specific to the specific cluster
	for (i = 0; i < col; i++)
	{
		matrix.Ith_Add_CV(i, concept_Vectors[cluster[i]]);
	}
}


void Kmeans::Compute_Cluster_Size()
{
	int i, k;

	k = 0;
	//Reset clusterSize
	for (i = 0; i < n_Clusters; i++)
		clusterSize[i] = 0;
	//clustersize is how many values is assigned to that specific cluster
	//cluster is pointer to what cluster that it belongs to (a pointer to cluster
	for (i = 0; i < col; i++)
	{
		clusterSize[cluster[i]]++;
	}
}

//Gives the result of the cluster_quaility added together.
double Kmeans::Coherence(int n_clus)
{
	int i;
	double value = 0.0;

	for (i = 0; i < n_clus; i++)
		value += cluster_quality[i];

	return value + n_clus*omega;
}

//Quality_change based on the different clusters.

double Kmeans::Delta_X(Matrix matrix, int x, int c_ID)
{
	double quality_change = 0.0;

	//This will only happen when we are on the same cluster.
	if (cluster[x] == c_ID)
		return 0;
	//This will happen for all elements that does not belong to the cluster, this will decrease the quality of the cluster whenever it does not have a point.
	if (cluster[x] >= 0)
		quality_change = -1.0* clusterSize[cluster[x]] * sim_Mat[cluster[x]][x] / (clusterSize[cluster[x]] - 1);
	//this will always happen as long as it belongs to a cluster within range.
	if (c_ID >= 0)
		quality_change += clusterSize[c_ID] * sim_Mat[c_ID][x] / (clusterSize[c_ID] + 1);

	return quality_change;
}

// update the quality_change_matrix for a particular cluster
void Kmeans::Update_Quality_Change_Mat(Matrix matrix, int c_ID) 
{
	int k, i;

	k = 0;

	for (i = 0; i < col; i++)
	{
		quality_change_mat[c_ID][i] = Delta_X(matrix, i, c_ID);
	}
}