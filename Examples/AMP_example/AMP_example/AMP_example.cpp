// AMP_example.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <amp.h>
#include <string>
#include <iostream>


int _tmain(int argc, _TCHAR* argv[])
{
	const unsigned int SizeOfArray = 5;
	int a[SizeOfArray];
	int b[SizeOfArray];
	int c[SizeOfArray] = { 0 };
	int *result;

	for (int i = 0; i < SizeOfArray; i++)
	{
		a[i] = i;
		b[i] = i * 10;
	}
	concurrency::array_view<int, 1> dev_a(SizeOfArray, a);
	concurrency::array_view<int, 1> dev_b(SizeOfArray, b);
	concurrency::array_view<int, 1> dev_c(SizeOfArray, c);

	/*concurrency::parallel_for(0, SizeOfArray, [=](int i) restrict(cpu, amp)
	{
		dev_c[i] = dev_a[i] + dev_b[i];
	});*/

	concurrency::parallel_for_each(
		dev_c.extent, [=](concurrency::index<1> idx) restrict(/*cpu, */ amp)
	{
		dev_c[idx] = dev_a[idx] + dev_b[idx];
	}
	);
	dev_c.synchronize();
	result = dev_c.data();
	for (int i = 0; i < SizeOfArray; i++)
	{
		std::cout << "result: " << result[i] << std::endl;;
	}

	return 0;
}

