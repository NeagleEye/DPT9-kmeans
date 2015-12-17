#include <iostream>

int main()
{
	int a[20],b[20],c[20];
	for (int i = 0; i < 20; i++)
	{
		a[i] = i * 5;
		b[i] = i * 9;
		c[i] = a[i] + b[i];
		std::cout << c[i] << std::endl;
	}
	return 0;
}