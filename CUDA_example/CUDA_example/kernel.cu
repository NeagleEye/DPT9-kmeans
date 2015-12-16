
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>

__global__ void CUDAadd(int *a, int *b, int *c, int SizeOfArray)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < SizeOfArray)
	{
		c[i] = a[i] + b[i];
	}
}

int main()
{
	const unsigned int SizeOfArray = 5;
	int a[SizeOfArray];
	int b[SizeOfArray];
	int c[SizeOfArray] = { 0 };

	for (int i = 0; i < SizeOfArray; i++)
	{
		a[i] = i;
		b[i] = i*10;
	}

	int *dev_a;
	int *dev_b;
	int *dev_c;

	//add(a, b, c, SizeOfArray);

	cudaMalloc((void**)&dev_a, SizeOfArray * sizeof(int));
	cudaMalloc((void**)&dev_b, SizeOfArray * sizeof(int));
	cudaMalloc((void**)&dev_c, SizeOfArray * sizeof(int));

	cudaMemcpy(dev_a, a, SizeOfArray * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, SizeOfArray * sizeof(int), cudaMemcpyHostToDevice);

	//This is creating the error:int numberOfBlocks = ceil(col / MaxThreadsPerBlock); // ceil is there just to be save
	//this solves the problem as it dynamic changes size based on the size of the number of points.
	int threadsPerBlock = 256;
	int blocksPerGrid = ((SizeOfArray)+threadsPerBlock - 1) / threadsPerBlock;
	//Block size may not exceed ~ 65000
	for (; blocksPerGrid > 65000;)
	{
		threadsPerBlock *= 2;
		blocksPerGrid = ((SizeOfArray)+threadsPerBlock - 1) / threadsPerBlock;
	}
	//blocksPerGrid, threadsPerBlock
	CUDAadd << < 1, SizeOfArray >> >(dev_a, dev_b, dev_c, SizeOfArray);
	cudaDeviceSynchronize();
	cudaMemcpy(c, dev_c, SizeOfArray * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < SizeOfArray; i++)
	{
		std::cout << "c: " << c[i] << std::endl;;
	}
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.