#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#pragma region const
int MaxThreadsPerBlock = 1024; // in CUDA 1.3 and lower it is 512

#pragma endregion

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstddef>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>


#pragma region funcktions naming
void average_vec(double vec[], int n, int num);
double norm_2(double vec[], int n);

#pragma endregion

#pragma region GPU Functions
__global__ void GPU_Func_average_vec(double *vec, int n, int num)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		vec[i] = vec[i] / num;
	}
}
void GPU_average_vec(double *vec, int n, int num)
{
	double *dev_vec;
	cudaMalloc((void**)&dev_vec, n * sizeof(double));

	cudaMemcpy(dev_vec, vec, n * sizeof(double), cudaMemcpyHostToDevice);

	int numberOfBlocks = ceil(n / MaxThreadsPerBlock); // ceil is there just to be save

	GPU_Func_average_vec << <numberOfBlocks, MaxThreadsPerBlock >> >(dev_vec, n, num);

	cudaMemcpy(vec, dev_vec, n * sizeof(double), cudaMemcpyHostToDevice);

	cudaFree(dev_vec);
}

__global__ void GPU_Func_ComputeNormalVector(double *normalVector, const int n_row_elements, double **value, int n_col)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_col)
	{
		normalVector[i] = 0.0;
		for (int j = 0; j < n_row_elements; j++)
		{
			normalVector[i] += (value[j][i]) * (value[j][i]);
		}
	}
}

__global__ void GPU_Func_Euc_Dis(double *x, double norm_x, double **value, int n_row_elements, double *normalVector, double *dev_result, int n_col)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n_col)
	{
		double result = 0.0;
		for (int j = 0; j < n_row_elements; j++)
			result += x[j] * value[j][i];
		result *= -2.0;
		result += normalVector[i] + norm_x;
		dev_result[i] = result;
	}
}
#pragma endregion


#pragma region RandomGenerator
// RandomGenerator.h: interface for the random number generator classes.
//		These classes are to generate random numbers needed by I/O and
//		BSG floorplan algorithm implementation including simulated
//		annealing.
// Modified from GNU Scientific Library 0.4.1
// The GNU Scientific Library can be downloaded from:
//		ftp://sourceware.cygnus.com/pub/gsl
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_)
#define AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_

#include <cstddef>

#define DEFAULT_SEED 0
//////////////////////////////////////////////////////////////////////
// base class of random number generators
class RandomGenerator
{
protected:
	const char *name;			// name of the generator
	unsigned long int max;		// maximum value of the random number
	unsigned long int min;		// minimum value of the random number
	size_t size;			// the size of state ?? do not have much use

public:
	RandomGenerator();
	virtual ~RandomGenerator();

	inline double GetUniformPos();			/* return a positive double precision doubleing point
											number uniformly distributed in (0, 1) */
	unsigned long int GetUniformInt(unsigned long int n);
	/* return a random integer from 0 to n-1 inclusive,
	all integers in [0, n-1] are equally likely */

	// Gaussian distribution
	double GetGaussian(const double sigma); /* return a gaussian random number, with mean zero and
											standard deviation sigma */
	double GetGaussianPDF(const double x, const double sigma);
	/* compute the probability density at x for a gaussian
	distribution with standard deviation sigma */
	inline double GetUGaussian() { return GetGaussian(1.0); }
	/* return a gaussian random number with mean zero and
	deviation 1.0 */

	// Get properties of the random number generator
	inline unsigned long int GetMax() { return max; };
	inline unsigned long int GetMin() { return min; };
	inline const char* GetName() { return name; };
	inline size_t GetSize() { return size; };

	// functions that will be overrided
	virtual void Set(unsigned long int seed) = 0;		/* initialize the generator */
	virtual unsigned long int Get() = 0;				/* return a random integer value, all integers
														in [min, max] are equally likely */
	virtual double GetUniform() = 0;					/* return a double precision random doubleing point
														number uniformly distributed in [0, 1) */
};

//////////////////////////////////////////////////////////////////////
// derived classes

// MT19937 generator (simulation quality)
/* "Mersenne Twister" generator by Makoto Matsumoto and Takuji Nishimura

Makoto Matsumoto has a web page with more information about the
generator, http://www.math.keio.ac.jp/~matumoto/emt.html.

The paper below has details of the algorithm.

From: Makoto Matsumoto and Takuji Nishimura, "Mersenne Twister: A
623-dimensionally equidistributerd uniform pseudorandom number
generator". ACM Transactions on Modeling and Computer Simulation,
Vol. 8, No. 1 (Jan. 1998), Pages 3-30

You can obtain the paper directly from Makoto Matsumoto's web page.

The period of this generator is 2^{19937} - 1.
*/

class RandomGenerator_MT19937 : public RandomGenerator
{

#define MT_N 624	/* Period parameters */
#define MT_M 397

protected:
	/* most significant w-r bits */
	const unsigned long UPPER_MASK;

	/* least significant r bits */
	const unsigned long LOWER_MASK;

	int mti;						//state
	unsigned long mt[MT_N];			//state

public:
	RandomGenerator_MT19937();
	virtual ~RandomGenerator_MT19937();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual double GetUniform();
};

// Tausworthe generator (simulation quality)
/* The period of this generator is about 2^88.

From: P. L'Ecuyer, "Maximally Equidistributed Combined Tausworthe
Generators", Mathematics of Computation, 65, 213 (1996), 203--213.

This is available on the net from L'Ecuyer's home page,

http://www.iro.umontreal.ca/~lecuyer/myftp/papers/tausme.ps
ftp://ftp.iro.umontreal.ca/pub/simulation/lecuyer/papers/tausme.ps
*/

class RandomGenerator_Taus : public RandomGenerator
{
protected:
	unsigned long int s1, s2, s3;		// state

public:
	RandomGenerator_Taus();
	virtual ~RandomGenerator_Taus();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual double GetUniform();
};

// TT800 generator
/* This is the TT800 twisted GSFR generator for 32 bit integers. It
has been superceded by MT19937 (mt.c). The period is 2^800.

This implementation is based on tt800.c, July 8th 1996 version by
M. Matsumoto, email: matumoto@math.keio.ac.jp

From: Makoto Matsumoto and Yoshiharu Kurita, "Twisted GFSR
Generators II", ACM Transactions on Modelling and Computer
Simulation, Vol. 4, No. 3, 1994, pages 254-266.
*/

class RandomGenerator_TT800 : public RandomGenerator
{

#define TT_N 25
#define TT_M 7

protected:
	int n;						//state
	unsigned long int x[TT_N];		//state

public:
	RandomGenerator_TT800();
	virtual~RandomGenerator_TT800();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual double GetUniform();
};

// R250 generator
/* This is a shift-register random number generator.

The period of this generator is about 2^250.

The algorithm works for any number of bits. It is implemented here
for 32 bits.

From: S. Kirkpatrick and E. Stoll, "A very fast shift-register
sequence random number generator", Journal of Computational Physics,
40, 517-526 (1981).
*/

class RandomGenerator_R250 : public RandomGenerator
{
protected:
	int i;					//state
	unsigned long x[250];	//state

public:
	RandomGenerator_R250();
	virtual ~RandomGenerator_R250();

	virtual void Set(unsigned long int seed);
	inline virtual unsigned long int Get();
	virtual double GetUniform();
};

#endif // !defined(AFX_RANDOMGENERATOR_H__DE1C1AFF_C9AF_4790_B7F6_F5B9B840789F__INCLUDED_)

#define PI 3.14159265358979323846264338328

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

RandomGenerator::RandomGenerator()
{

}

RandomGenerator::~RandomGenerator()
{

}

// Implementation of base class

double RandomGenerator::GetUniformPos()
{
	double x;
	do
	{
		x = GetUniform();
	} while (x == 0);

	return x;
}

unsigned long int RandomGenerator::GetUniformInt(unsigned long int n)
{
	unsigned long int offset = min;
	unsigned long int range = max - offset;
	unsigned long int scale = range / n;
	unsigned long int k;

	if (n > range)
		n = range;

	do
	{
		k = (Get() - offset) / scale;
	} while (k >= n);

	return k;
}

double RandomGenerator::GetGaussian(const double sigma)
{
#if 1 /* Polar (Box-Mueller) method; See Knuth v2, 3rd ed, p122 */
	double x, y, r2;

	do
	{
		/* choose x,y in uniform square (-1,-1) to (+1,+1) */

		x = -1 + 2 * GetUniform();
		y = -1 + 2 * GetUniform();

		/* see if it is in the unit circle */
		r2 = x * x + y * y;
	} while (r2 > 1.0 || r2 == 0);

	/* Box-Muller transform */
	return sigma * y * sqrt(-2.0 * log(r2) / r2); /* only one random deviate is produced, the
												  other is discarded. Because saving the
												  other in a static variable would screw up
												  re-entrant or theaded code. */
#endif
#if 0 /* Ratio method (Kinderman-Monahan); see Knuth v2, 3rd ed, p130 */
	/* K+M, ACM Trans Math Software 3 (1977) 257-260. */
	double u, v, x, xx;

	do {
		v = GetUniform(r);
		do {
			u = GetUniform(r);
		} while (u == 0);
		/* Const 1.715... = sqrt(8/e) */
		x = 1.71552776992141359295*(v - 0.5) / u;
	} while (x*x > -4.0*log(u));
	return sigma*x;
#endif
}

double RandomGenerator::GetGaussianPDF(const double x, const double sigma)
{
	double u = x / fabs(sigma);
	double p = (1 / (sqrt(2 * PI) * fabs(sigma))) * exp(-u * u / 2);
	return p;
}

//////////////////////////////////////////////////////////////////////
// Implementation of MT_19937 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_MT19937::RandomGenerator_MT19937() : UPPER_MASK(0x80000000UL), LOWER_MASK(0x7fffffffUL)
{
	name = "mt19937";	//??
	max = 0xffffffffUL;			/* RAND_MAX  */
	min = 0;						/* RAND_MIN  */
	size = sizeof(mti)+sizeof(mt);	// ??
}

RandomGenerator_MT19937::~RandomGenerator_MT19937()
{
}

unsigned long RandomGenerator_MT19937::Get()
{
	unsigned long k;

#define MAGIC(y) (((y)&0x1) ? 0x9908b0dfUL : 0)

	if (mti >= MT_N)
	{	/* generate N words at one time */
		int kk;

		for (kk = 0; kk < MT_N - MT_M; kk++)
		{
			unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + MT_M] ^ (y >> 1) ^ MAGIC(y);
		}
		for (; kk < MT_N - 1; kk++)
		{
			unsigned long y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
			mt[kk] = mt[kk + (MT_M - MT_N)] ^ (y >> 1) ^ MAGIC(y);
		}

		{
			unsigned long y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
			mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ MAGIC(y);
		}

		mti = 0;
	}

	/* Tempering */

	k = mt[mti];
	k ^= (k >> 11);
	k ^= (k << 7) & 0x9d2c5680UL;
	k ^= (k << 15) & 0xefc60000UL;
	k ^= (k >> 18);

	mti++;

	return k;
}

double RandomGenerator_MT19937::GetUniform()
{
	return Get() / 4294967296.0;
}

void RandomGenerator_MT19937::Set(unsigned long int seed)
{
	int i;

	if (seed == 0)
		seed = 4357;	/* the default seed is 4357 */

	mt[0] = seed & 0xffffffffUL;

	/* We use the congruence s_{n+1} = (69069*s_n) mod 2^32 to
	initialize the state. This works because ANSI-C unsigned long
	integer arithmetic is automatically modulo 2^32 (or a higher
	power of two), so we can safely ignore overflow. */

#define LCG(n) ((69069 * n) & 0xffffffffUL)

	for (i = 1; i < MT_N; i++)
		mt[i] = LCG(mt[i - 1]);

	mti = i;
}


//////////////////////////////////////////////////////////////////////
// Implementation of Taus generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_Taus::RandomGenerator_Taus()
{
	name = "taus";			/* name */
	max = 0xffffffffUL;		/* RAND_MAX */
	min = 0;			        /* RAND_MIN */
	size = sizeof(unsigned long int) * 3;
}

RandomGenerator_Taus::~RandomGenerator_Taus()
{
}

unsigned long int RandomGenerator_Taus::Get()
{
#define MASK 0xffffffffUL
#define TAUSWORTHE(s,a,b,c,d) (((s &c) <<d) &MASK) ^ ((((s <<a) &MASK)^s) >>b)

	s1 = TAUSWORTHE(s1, 13, 19, 4294967294UL, 12);
	s2 = TAUSWORTHE(s2, 2, 25, 4294967288UL, 4);
	s3 = TAUSWORTHE(s3, 3, 11, 4294967280UL, 17);

	return (s1 ^ s2 ^ s3);
}

double RandomGenerator_Taus::GetUniform()
{
	return Get() / 4294967296.0;
}

void RandomGenerator_Taus::Set(unsigned long seed)
{
	if (seed == 0)
		seed = 1;	/* default seed is 1 */

#define LCG(n) ((69069 * n) & 0xffffffffUL)
	s1 = LCG(seed);
	s2 = LCG(s1);
	s3 = LCG(s2);

	/* "warm it up" */
	Get();
	Get();
	Get();
	Get();
	Get();
	Get();
	return;
}


//////////////////////////////////////////////////////////////////////
// Implementation of TT800 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_TT800::RandomGenerator_TT800()
{
	name = "tt800";			/* name */
	max = 0xffffffffUL;		/* RAND_MAX */
	min = 0;			        /* RAND_MIN */
	size = sizeof(n)+sizeof(x);
}

RandomGenerator_TT800::~RandomGenerator_TT800()
{
}

unsigned long int RandomGenerator_TT800::Get()
{
	/* this is the magic vector, a */

	const unsigned long mag01[2] =
	{ 0x00000000, 0x8ebfd028UL };
	unsigned long int y;

	if (n >= TT_N)
	{
		int i;
		for (i = 0; i < TT_N - TT_M; i++)
		{
			x[i] = x[i + TT_M] ^ (x[i] >> 1) ^ mag01[x[i] % 2];
		}
		for (; i < TT_N; i++)
		{
			x[i] = x[i + (TT_M - TT_N)] ^ (x[i] >> 1) ^ mag01[x[i] % 2];
		};
		n = 0;
	}

	y = x[n];
	y ^= (y << 7) & 0x2b5b2500UL;		/* s and b, magic vectors */
	y ^= (y << 15) & 0xdb8b0000UL;	/* t and c, magic vectors */
	y &= 0xffffffffUL;	/* you may delete this line if word size = 32 */

	/* The following line was added by Makoto Matsumoto in the 1996
	version to improve lower bit's correlation.  Delete this line
	to use the code published in 1994.  */

	y ^= (y >> 16);	/* added to the 1994 version */

	n = n + 1;

	return y;
}

double RandomGenerator_TT800::GetUniform()
{
	return Get() / 4294967296.0;
}

void RandomGenerator_TT800::Set(unsigned long int seed)
{
	const int init_n = 0;
	const unsigned long int init_x[TT_N] =
	{ 0x95f24dabUL, 0x0b685215UL, 0xe76ccae7UL,
	0xaf3ec239UL, 0x715fad23UL, 0x24a590adUL,
	0x69e4b5efUL, 0xbf456141UL, 0x96bc1b7bUL,
	0xa7bdf825UL, 0xc1de75b7UL, 0x8858a9c9UL,
	0x2da87693UL, 0xb657f9ddUL, 0xffdc8a9fUL,
	0x8121da71UL, 0x8b823ecbUL, 0x885d05f5UL,
	0x4e20cd47UL, 0x5a9ad5d9UL, 0x512c0c03UL,
	0xea857ccdUL, 0x4cc1d30fUL, 0x8891a8a1UL,
	0xa6b7aadbUL };

	if (seed == 0)	/* default seed is given explicitly in the original code */
	{
		n = init_n;
		for (int i = 0; i<TT_N; i++) x[i] = init_x[i];
	}
	else
	{
		int i;

		n = 0;

		x[0] = seed & 0xffffffffUL;

		for (i = 1; i < TT_N; i++)
			x[i] = (69069 * x[i - 1]) & 0xffffffffUL;
	}

	return;
}


//////////////////////////////////////////////////////////////////////
// Implementation of R250 generator
//////////////////////////////////////////////////////////////////////

RandomGenerator_R250::RandomGenerator_R250()
{
	name = "r250";			/* name */
	max = 0xffffffffUL;		/* RAND_MAX */
	min = 0;			        /* RAND_MIN */
	size = sizeof(i)+sizeof(x);
}

RandomGenerator_R250::~RandomGenerator_R250()
{
}

unsigned long int RandomGenerator_R250::Get()
{
	unsigned long int k;
	int j;

	if (i >= 147)
	{
		j = i - 147;
	}
	else
	{
		j = i + 103;
	}

	k = x[i] ^ x[j];
	x[i] = k;

	if (i >= 249)
	{
		i = 0;
	}
	else
	{
		i = i + 1;
	}

	return k;
}

double RandomGenerator_R250::GetUniform()
{
	return Get() / 4294967296.0;
}

void RandomGenerator_R250::Set(unsigned long int seed)
{
	int j;

	if (seed == 0)
		seed = 1;	/* default seed is 1 */

	i = 0;

#define LCG(n) ((69069 * n) & 0xffffffffUL)

	for (j = 0; j < 250; j++)	/* Fill the buffer  */
	{
		seed = LCG(seed);
		x[j] = seed;
	}

	{
		/* Masks for turning on the diagonal bit and turning off the
		leftmost bits */

		unsigned long int msb = 0x80000000UL;
		unsigned long int mask = 0xffffffffUL;

		for (j = 0; j < 32; j++)
		{
			int k = 7 * j + 3;	/* Select a word to operate on        */
			x[k] &= mask;	/* Turn off bits left of the diagonal */
			x[k] |= msb;	/* Turn on the diagonal bit           */
			mask >>= 1;
			msb >>= 1;
		}
	}

	return;
}


#pragma endregion

#pragma region Matrix
class Matrix
{
public:
	Matrix(int row, int col, double **val);
	~Matrix();
	void ComputeNormalVector();
	void GPU_ComputeNormalVector();
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
	double GetVal(int i, int j) { return value[i][j]; }
	void Ith_Add_CV(int i, double *CV);

	//Calculate distance
	double Euc_Dis(double *x, int i, double norm_x);
	void Euc_Dis(double *x, double norm_x, double *result);
	void GPU_Euc_Dis(double *x, double norm_x, double *result);
	double GetNorm(int i) { return normalVector[i]; }
private:
	//Row elements = value value value (3d or) value value (2d can be xd)
	//Column how many datasets is in the dataset
	int n_row_elements, n_col;
	//**value = value[row][col] (rowth element, column)
	//normalVector = normalvector over row of elements (vector)
	double **value, *normalVector;
};
Matrix::Matrix(int row, int col, double **val)
{
	n_row_elements = row;
	n_col = col;
	value = val;
}

Matrix::~Matrix()
{
	/*if (normalVector != NULL)
	delete[] normalVector;*/
}

void Matrix::ComputeNormalVector()
{
	//if (normalVector == NULL) // NIELS: i cuda kan det her nummer ikke bruges, det bliver ikke null men 0xccccc...
	//{
		normalVector = new double[n_col];
		//memory_used += m_col*sizeof(float);
	//}
	for (int i = 0; i < n_col; i++)
	{
		normalVector[i] = 0.0;
		for (int j = 0; j < n_row_elements; j++)
			normalVector[i] += (value[j][i]) * (value[j][i]);
	}
}

void Matrix::GPU_ComputeNormalVector()
{
	double *dev_normalVector;
	double **dev_value;

	cudaMemcpy(dev_normalVector, normalVector, n_col * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_value, value, n_row_elements * n_col * sizeof(double), cudaMemcpyHostToDevice);

	int numberOfBlocks = ceil(n_col / MaxThreadsPerBlock); // ceil is there just to be save 

	GPU_Func_ComputeNormalVector <<<numberOfBlocks, MaxThreadsPerBlock >>>(dev_normalVector, n_row_elements, dev_value, n_col);

	cudaMemcpy(normalVector, dev_normalVector, n_col * sizeof(double), cudaMemcpyHostToDevice);

	cudaFree(dev_normalVector);
	cudaFree(dev_value);
}

//add current concept_vector to the original vector
void Matrix::Ith_Add_CV(int i, double *CV) // NIELS: paralleliseret?
{
	for (int j = 0; j < n_row_elements; j++)
		CV[j] += value[j][i];
}

double Matrix::Euc_Dis(double *x, int i, double norm_x) // NIELS: ikke smart at paralleliser
/* Given squared L2-norms of the vecs and v, norm[i] and norm_v,
compute the Euc-dis between ith vec in the matrix and v,
result is returned.
Used (x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
*/
{
	double result = 0.0;
	for (int j = 0; j< n_row_elements; j++)
		result += x[j] * value[j][i];
	result *= -2.0;
	result += normalVector[i] + norm_x;
	return result;
}

void Matrix::Euc_Dis(double *x, double norm_x, double *result) // NIELS: paralleliseret?
/* Given squared L2-norms of the vecs and x, norm[i] and norm_x,
compute the Euc-dis between each vec in the matrix with x,
results are stored in array 'result'.
Since the matrix is dense, not taking advantage of
(x-c)^T (x-c) = x^T x - 2 x^T c + c^T c
but the abstract class defition needs the parameter of 'norm_x'
*/
{
	for (int i = 0; i < n_col; i++)
		result[i] = Euc_Dis(x, i, norm_x);
}

void Matrix::GPU_Euc_Dis(double *x, double norm_x, double *result)
{
	double *dev_X;
	double *dev_Result;
	double **dev_value;

	cudaMemcpy(dev_X, normalVector, n_col * sizeof(double), cudaMemcpyHostToDevice); // NIELS: IKKE HELT SIKKER PÅ STØRRELSERNE HER
	cudaMemcpy(dev_Result, normalVector, n_col * sizeof(double), cudaMemcpyHostToDevice); // NIELS: IKKE HELT SIKKER PÅ STØRRELSERNE HER
	cudaMemcpy(dev_value, value, n_row_elements * n_col * sizeof(double), cudaMemcpyHostToDevice);

	int numberOfBlocks = ceil(n_col / MaxThreadsPerBlock); // ceil is there just to be save

	GPU_Func_Euc_Dis << <numberOfBlocks, MaxThreadsPerBlock >> >(x, norm_x, dev_value, n_row_elements, normalVector, dev_Result, n_col);

	cudaMemcpy(result, dev_Result, n_col * sizeof(double), cudaMemcpyHostToDevice);

	cudaFree(dev_X);
	cudaFree(dev_Result);
	cudaFree(dev_value);
}

#pragma endregion

#pragma region fileReader
struct Mat
{
	int n_row, n_col;
	double **val;
};

Matrix GetVector();
Matrix GetVector(int &x, int &y);

Matrix GetVector()
{
	int a, b;
	return GetVector(a, b);
}

Matrix GetVector(int &x, int &y)
{
	Mat result;
	std::string line;
	std::ifstream myfile("AllRandom.mtx");
	if (myfile.is_open())
	{
		//the reader takes the first number and uses as rows >> jump over space, uses next number as column >> jumps over space to next number on next line.
		myfile >> result.n_row >> result.n_col;
		//Init the value to have nrow elements [inner element]
		result.val = new double*[result.n_row];
		//initialize every column for every row a column data[outerelement]
		for (int i = 0; i < result.n_row; i++)
			result.val[i] = new double[result.n_col];

		for (int i = 0; i < result.n_col; i++)
		{
			for (int j = 0; j < result.n_row; j++)
			{
				myfile >> result.val[j][i];
				if (j == 0){ if (result.val[j][i] > x)x = result.val[j][i]; }
				else if (j == 1){ if (result.val[j][i] > y)y = result.val[j][i]; }
			}
		}
		myfile.close();
	}

	Matrix matrix(result.n_row, result.n_col, result.val);
	return matrix;
}

#pragma endregion

#pragma region Kmeans
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
	int Assign_Cluster(Matrix matrix, bool simi_est);
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
	int /*number of clusters*/n_Clusters, col, row,
		/*indicator of how many elements belong to each cluster*/ *clusterSize,
		/*pointer to which cluster the column belongs*/*cluster,
		/*Estimate of how long it will take maximum*/EST_START = 5, f_v_times = 0;
	bool stabilized /*debug=false*/;
};

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

void Kmeans::Generel_K_Means(Matrix matrix)
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
		if (Assign_Cluster(matrix, stabilized) == 0){ 
			int lala = 0; 
		}
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
				for (i = 0; i<col; i++) // NIELS: paralleliseret?
					sim_Mat[cluster[i]][i] = matrix.Euc_Dis(concept_Vectors[cluster[i]], i, normal_ConceptVectors[cluster[i]]); 
				else
				for (i = 0; i < n_Clusters; i++)
					matrix.Euc_Dis(concept_Vectors[i], normal_ConceptVectors[i], sim_Mat[i]);
			}
			else
			for (i = 0; i < n_Clusters; i++)//returning distance between the squared average vector and the average vector to sim_mat // NIELS: paralleliseret?
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

#pragma endregion

#pragma region PrintMatrix

void PrintMatrix(Matrix input, int MAX_X, int MAX_Y)
{
	std::ofstream myfile;
	myfile.open("example.txt");
	myfile << "P1\n";
	myfile << MAX_X << " " << MAX_Y << "\n";
	int test = input.GetColumns();
	int position = 0;
	bool noMorePoints = false;
	for (int i = 0; i < MAX_Y; i++)
	{
		for (int j = 0; j < MAX_X; j++)
		{
			if (!noMorePoints)
			{
				int XPosition = input.GetVal(0, position);
				int YPosition = input.GetVal(1, position);
				if (XPosition == i && YPosition == j)
				{
					myfile << "1 ";
					position++;
				}
				else
				{
					myfile << "0 ";
				}
				if (position >= input.GetColumns())
				{
					noMorePoints = true;
				}
			}
			else
			{
				myfile << "0 ";
			}
		}
		myfile << "\n";
	}
	myfile.close();

}

#pragma endregion

#pragma region MathonVectors
void average_vec(double vec[], int n, int num)
{
	int i;
	for (i = 0; i< n; i++)
		vec[i] = vec[i] / num;
}

//compute the normal vector over the values in the row elements.
double norm_2(double vec[], int n)
//compute squared L2 norm of vec
{
	double norm;
	int i;
	norm = 0.0;
	for (i = 0; i < n; i++)
		norm += vec[i] * vec[i];
	return norm;
}

#pragma endregion

#pragma region class test
__global__ void addKernel(int c[], const int a[], const int b[], int size)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i] + size;
}

class testClass
{
public:
	 testClass();
	 ~testClass();
	 cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

private:

};

testClass::testClass()
{
}

testClass::~testClass()
{
}

cudaError_t testClass::addWithCuda(int c[], const int a[], const int b[], unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Allocate GPU buffers for three vectors (two input, one output).
	cudaMalloc((void**)&dev_c, size * sizeof(int));
	cudaMalloc((void**)&dev_a, size * sizeof(int));
	cudaMalloc((void**)&dev_b, size * sizeof(int));

	// Copy input vectors from host memory to GPU buffers.
	cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c, dev_a, dev_b, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();

	// Copy output vector from GPU buffer to host memory.
	cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
#pragma endregion

#pragma region main
void testcode()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	testClass test;
	cudaError_t cudaStatus = test.addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}

void kmeanscode()
{
	
	int *cluster, n_clusters = 4;
	int x = 0, y = 0;
	Matrix matrix = GetVector(x, y);

	cluster = new int[matrix.GetColumns()];
	//Initialize Euclidean kmeans
	Kmeans k(n_clusters, cluster, matrix.GetColumns(), matrix.GetRows());
	//Calculate normal vectors on every column set.
	matrix.ComputeNormalVector();
	k.Initialize_CV(matrix);
	k.Generel_K_Means(matrix);
	/*
	*Printing out the matrix only 2d is available and 2d dataset
	*/
	PrintMatrix(matrix, x, y);
	
}

int main()
{
	testcode();
	kmeanscode();
    return 0;
}
#pragma endregion



#pragma region
#pragma endregion