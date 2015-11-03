#include "Kmeans.h"
#include "MathonVectors.h"
#include <time.h>
#include <iostream>

Kmeans::Kmeans(int n_cluster, int clusterinit[], int columns, int rows)
{
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

void Kmeans::General_K_Means(Matrix matrix)
{

	int n_Iters, i, j, k;
	bool no_assignment_change;

	//if (dumpswitch)
	//	cout << endl << "- Start Euclidean K-Means loop. -" << endl << endl;
	n_Iters = 0;
	no_assignment_change = true;

	do
	{
		pre_Result = result;
		n_Iters++;
		if (n_Iters >EST_START)
			stablized = true;
		//if (dumpswitch && stablized)
		//	cout << "(Similarity estimation used.)" << endl;
		if (Assign_Cluster(matrix, stablized) == 0)
		{
			//if (dumpswitch)
				//cout << "No points are moved in the last step " << endl << "@" << endl << endl;
		}
		else
		{
			no_assignment_change = false;
			Compute_Cluster_Size();
			if (n_Iters >= EST_START)
				for (i = 0; i < n_Clusters; i++)
					for (j = 0; j < row; j++)
						old_ConceptVectors[i][j] = concept_Vectors[i][j];

			Update_Centroids(matrix);

			for (i = 0; i < n_Clusters; i++)
				average_vec(concept_Vectors[i], row, clusterSize[i]);
			for (i = 0; i < n_Clusters; i++)
				normal_ConceptVectors[i] = norm_2(concept_Vectors[i], row);

			if (n_Iters >= EST_START)
			{
				for (i = 0; i < n_Clusters; i++)
				{
					difference[i] = 0.0;
					for (j = 0; j < row; j++)
						difference[i] += (old_ConceptVectors[i][j] - concept_Vectors[i][j]) * (old_ConceptVectors[i][j] - concept_Vectors[i][j]);
					//  diff[i] = diff[i] - 2*sqrt(diff[i]*Sim_Mat[Cluster[i]][i]);
				}
				if (n_Iters > EST_START)
					for (i = 0; i<col; i++)
						sim_Mat[cluster[i]][i] = matrix.Euc_Dis(concept_Vectors[cluster[i]], i, normal_ConceptVectors[cluster[i]]);
				else
					for (i = 0; i < n_Clusters; i++)
						matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);
			}
			else
				for (i = 0; i < n_Clusters; i++)
					matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);

			for (i = 0; i<n_Clusters; i++)
				cluster_quality[i] = 0.0;
			k = 0;
			for (i = 0; i < col; i++)
			{
				while (i<empty_Docs_ID[k])
				{
					cluster_quality[cluster[i]] += sim_Mat[cluster[i]][i];
					i++;
				}
				k++;
			}
			result = Coherence(n_Clusters);
			/*if (dumpswitch)
			{
				find_worst_vectors(false);
				cout << "Obj. func. ( + Omiga * n_Clusters) = " << Result << endl << "@" << endl << endl;
				if (verify)
					cout << "Verify obj. func. : " << verify_obj_func(p_Docs, n_Clusters) << endl;
			}*/
			std::cout << "E";
		}
	} while ((pre_Result - result) > epsilon*initial_obj_fun_val);
	std::cout << std::endl;
	/*if (dumpswitch)
	{
		cout << "Euclidean K-Means loop stoped with " << n_Iters << " iterations." << endl;
		generate_Confusion_Matrix(label, n_Class);
	}*/

	if (stablized)
		for (i = 0; i < n_Clusters; i++)
			matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);

	if ((!no_assignment_change) && (f_v_times >0))
		for (i = 0; i<n_Clusters; i++)
			Update_Quality_Change_Mat(matrix, i);

}

int Kmeans::Assign_Cluster(Matrix matrix, bool simi_est)
{

	int i, j, k, multi = 0, changed = 0, temp_Cluster_ID;
	double temp_sim;

	k = 0;

	if (simi_est)
	{
		for (i = 0; i < n_Clusters; i++)
			for (j = 0; j < col; j++)
				if (i != cluster[j])
					sim_Mat[i][j] += difference[i] - 2 * sqrt(difference[i] * sim_Mat[i][j]);

		for (i = 0; i < col; i++)
		{
			while (i<empty_Docs_ID[k])
			{
				temp_sim = sim_Mat[cluster[i]][i];
				temp_Cluster_ID = cluster[i];

				for (j = 0; j < n_Clusters; j++)
					if (j != cluster[i])
					{
						if (sim_Mat[j][i] < temp_sim)
						{
							multi++;
							sim_Mat[j][i] = matrix.Euc_Dis(concept_Vectors[j], i, normal_ConceptVectors[j]);
							if (sim_Mat[j][i] < temp_sim)
							{
								temp_sim = sim_Mat[j][i];
								temp_Cluster_ID = j;
							}
						}
					}

				if (temp_Cluster_ID != cluster[i])
				{
					cluster[i] = temp_Cluster_ID;
					sim_Mat[cluster[i]][i] = temp_sim;
					changed++;
				}
				i++;
			}
			k++;
		}
	}
	else
	{
		for (i = 0; i < col; i++)
		{
			while (i<empty_Docs_ID[k])
			{
				temp_sim = sim_Mat[cluster[i]][i];
				temp_Cluster_ID = cluster[i];

				for (j = 0; j < n_Clusters; j++)
					if (j != cluster[i])
					{
						multi++;
						if (sim_Mat[j][i] < temp_sim)
						{
							temp_sim = sim_Mat[j][i];
							temp_Cluster_ID = j;
						}
					}
				if (temp_Cluster_ID != cluster[i])
				{
					cluster[i] = temp_Cluster_ID;
					sim_Mat[cluster[i]][i] = temp_sim;
					changed++;
				}
				i++;
			}
			k++;
		}
	}
	/*if (dumpswitch)
	{
		cout << multi << " Euclidean distance computation\n";
		cout << changed << " assignment changes\n";
	}*/
	return changed;
}

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
		if ((cluster[i] >= 0) && (cluster[i] < n_Clusters))
			matrix.Ith_Add_CV(i, concept_Vectors[cluster[i]]);
		else
			cluster[i] = 0;
	}

	Compute_Cluster_Size();

	for (i = 0; i < n_Clusters; i++)
		average_vec(concept_Vectors[i], row, clusterSize[i]);

	for (i = 0; i < n_Clusters; i++)
		normal_ConceptVectors[i] = norm_2(concept_Vectors[i], row);

	for (i = 0; i < n_Clusters; i++)
		matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);

	for (i = 0; i<n_Clusters; i++)
		cluster_quality[i] = 0.0;
	k = 0;
	for (i = 0; i < col; i++)
	{
		while (i<empty_Docs_ID[k])
		{
			cluster_quality[cluster[i]] += sim_Mat[cluster[i]][i];
			i++;
		}
		k++;
	}
	//for (i = 0; i < n_Clusters; i++)
	//diff[i] = 0.0;

	// because we need give the coherence here.

	initial_obj_fun_val = result = Coherence(n_Clusters);
	fv_threshold = -1.0*initial_obj_fun_val*delta;
	/*if (dumpswitch || evaluate)
	{
		outputClusterSize();
		cout << "Initial Obj. func.: " << Result << endl;
		if (n_Class >0)
			cout << "Initial confusion matrix :" << endl;
		generate_Confusion_Matrix(label, n_Class);
	}*/

	/*if (evaluate)
	{
		purity_Entropy_MutInfo();
		F_measure(label, n_Class);
		micro_avg_precision_recall();
		cout << endl;
		cout << "* Evaluation done. *" << endl;
		exit(0);
	}*/

	if (f_v_times >0)
	{
		// VT 2009-11-28
		// quality_change_mat = new (float *)[n_Clusters];
		quality_change_mat = new double *[n_Clusters];
		// VT 2009-11-28
		for (int j = 0; j < n_Clusters; j++)
			quality_change_mat[j] = new double[col];
		//memory_consume += (n_Clusters*n_Docs)*sizeof(float);
		for (i = 0; i < n_Clusters; i++)
			Update_Quality_Change_Mat(matrix, i);
	}
	//memory_consume += matrix->GetMemoryUsed();
}

void Kmeans::Well_Separated_Centroids(Matrix matrix)
// VT 2009-11-28
{
	int i, j, k, min_ind, *cv = new int[n_Clusters];
	double min, cos_sum;
	bool *mark = new bool[col];

	for (i = 0; i< col; i++)
		mark[i] = false;
	for (i = 0; i< n_Empty_Docs; i++)
		mark[empty_Docs_ID[i]] = true;

	for (i = 0; i < n_Clusters; i++)
		for (j = 0; j < row; j++)
			concept_Vectors[i][j] = 0.0;

	/*switch (choice)
	{
	case 0:*/
		do{
			cv[0] = rand_gen.GetUniformInt(col);
		} while (mark[cv[0]]);
		/*if (dumpswitch)
		{
			cout << "Cluster centroids are chosen to be well separated from each other." << endl;
			cout << "Start with a random chosen vector" << endl;
		}
		/*break;
	case 1:
	default:
		float *v, min;
		int min_ID = 0;
		v = new float[n_Words];

		for (i = 0; i < n_Words; i++)
			v[i] = 0.0;
		for (i = 0; i < n_Docs; i++)
			p_Docs->ith_add_CV(i, v);

		float temp, temp_norm;
		k = 0;
		average_vec(v, n_Words, n_Docs);
		temp_norm = norm_2(v, n_Words);
		min = 0.0;
		min_ID = 0;
		for (i = 0; i<n_Docs; i++)
		{
			while (i<empty_Docs_ID[k])
			{
				temp = p_Docs->euc_dis(v, i, temp_norm);
				if (temp > min)
				{
					min = temp;
					min_ID = i;
				}
				i++;
			}
			k++;
		}

		cv[0] = min_ID;
		delete[] v;
		if (dumpswitch)
		{
			cout << "Cluster centroids are chosen to be well separated from each other." << endl;
			cout << "Start with a vector farthest from the centroid of the whole data set" << endl;
		}
		break;
	}*/

	matrix.Ith_Add_CV(cv[0], concept_Vectors[0]);
	mark[cv[0]] = true;

	normal_ConceptVectors[0] = matrix.GetNorm(cv[0]);
	matrix.Euc_Dis(concept_Vectors[0], normal_ConceptVectors[0], sim_Mat[0]);
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

	for (i = 0; i<col; i++)
		cluster[i] = 0;
	Assign_Cluster(matrix, false);

	/*if (dumpswitch)
	{
		cout << "Vectors chosen to be the centroids are : ";
		for (i = 0; i<n_Clusters; i++)
			cout << cv[i] << " ";
		cout << endl;
	}*/
	delete[] cv;
	delete[] mark;
}

void Kmeans::Update_Centroids(Matrix matrix)
{

	int i, j, k;

	for (i = 0; i < n_Clusters; i++)
		for (j = 0; j < row; j++)
			concept_Vectors[i][j] = 0.0;
	k = 0;
	for (i = 0; i < col; i++)
	{
		while (i<empty_Docs_ID[k])
		{
			matrix.Ith_Add_CV(i, concept_Vectors[cluster[i]]);
			i++;
		}
		k++;
	}
}


void Kmeans::Compute_Cluster_Size()
{
	int i, k;

	k = 0;
	for (i = 0; i < n_Clusters; i++)
		clusterSize[i] = 0;
	for (i = 0; i < col; i++)
	{
		while (i<empty_Docs_ID[k])
		{
			clusterSize[cluster[i]]++;
			i++;
		}
		k++;
	}
}

double Kmeans::Coherence(int n_clus)
{
	int i;
	double value = 0.0;

	for (i = 0; i < n_clus; i++)
		value += cluster_quality[i];

	return value + n_clus*omega;
}

double Kmeans::Delta_X(Matrix matrix, int x, int c_ID)
{
	double quality_change = 0.0;

	if (cluster[x] == c_ID)
		return 0;
	if (cluster[x] >= 0)
		quality_change = -1.0* clusterSize[cluster[x]] * sim_Mat[cluster[x]][x] / (clusterSize[cluster[x]] - 1);

	if (c_ID >= 0)
		quality_change += clusterSize[c_ID] * sim_Mat[c_ID][x] / (clusterSize[c_ID] + 1);

	return quality_change;
}

void Kmeans::Update_Quality_Change_Mat(Matrix matrix, int c_ID)
// update the quality_change_matrix for a particular cluster 
{
	int k, i;

	k = 0;

	for (i = 0; i < col; i++)
	{
		while (i<empty_Docs_ID[k])
		{
			quality_change_mat[c_ID][i] = Delta_X(matrix, i, c_ID);
			i++;
		}
		k++;
	}
}