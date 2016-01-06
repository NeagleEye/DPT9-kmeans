#include <iostream>

int main()
{
	int const sizeofarray = 5;
	int a[sizeofarray], b[sizeofarray], c[sizeofarray];
	for (int i = 0; i < sizeofarray; i++)
	{
		a[i] = i;
		b[i] = i * 10;
	}

	for (int i = 0; i < sizeofarray; i++)
	{
		c[i] = a[i] + b[i];
		std::cout << c[i] << std::endl;
	}
	return 0;
}