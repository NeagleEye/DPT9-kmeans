
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>
/**
 * Host main routine
 */
#define SIZEOFELEMENTS 20
#define SIZEOFTEST 10
#define NUMTHREADS 1
void AddVector( float *a,float *b, float *c, int rangemin,int rangemax)
{
	for (int i = rangemin; i < rangemax; ++i){
		c[i] = a[i]+b[i];
    	}
}
//Implement other multi thread solution
int
main(void)
{
	clock_t t;


	printf("Starting Global Timer now");
	t = clock();
	bool domultithread = false;//Multithread was naive, a better solution has to be implemented before use

	for(int test = 1; test <= SIZEOFELEMENTS; test++){

		// Print the vector length to be used, and compute its size
		int numElements = test * 1000000;
		for(int test2 = 0; test2 < SIZEOFTEST; test2++){
			size_t size = numElements * sizeof(float);
			printf("[Vector addition of %d elements]\n", numElements);
			float *h_A = new float[size];
			float *h_B = new float[size];
			float *h_C = new float[size];

			// Initialize the vectors
			for (int i = 0; i < numElements; ++i)
			{
				h_A[i] = 1.0;//rand()/(float)RAND_MAX;
				h_B[i] = 1.0;//rand()/(float)RAND_MAX;
			}
			if(domultithread){
				std::vector<std::thread *> threads;
				//std::cout << "minrange = "<<  numElements/NUMTHREADS*0 << " maxrange: " << numElements/NUMTHREADS*(1)<< std::endl;
				for(int i = 1; i < NUMTHREADS;i++){
				  threads.push_back(new std::thread(AddVector,h_A,h_B,h_C,numElements/NUMTHREADS*i,numElements/NUMTHREADS*(i+1)+1));
				}

				AddVector(h_A,h_B,h_C,numElements/NUMTHREADS*0,numElements/NUMTHREADS*(1));
				for(int i = 0; i < NUMTHREADS-1;i++){
				  threads[i]->join();
				  delete threads[i];
				}
			}
			else
    				for (int i = 0; i < numElements; ++i){
    					h_C[i] = h_A[i]+h_B[i];
    				}


    		printf("Test PASSED\n");

    		// Free host memory
    		delete h_A;
    		delete h_B;
    		delete h_C;
    	    }
	}
	t = clock() - t;
	printf ("It took %f seconds.\n",((float)t)/CLOCKS_PER_SEC);

    printf("Done\n");
    return 0;
}

