#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#define MIN_ELEMENT 0.000005
#define BLOCK_SIZE 512 // needs to be checked for proper values
#define MAX_ELEMENT 15

__global__ void swap(double* matrix, int rows, int temp,int j)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < rows * 2)
	{
		double temporary = matrix[j*rows * 2 + x];
		matrix[j*rows * 2 + x] = matrix[temp*rows * 2 + x];
		matrix[temp*rows * 2 + x] = temporary;
	}
}

__global__ void gjAlgorithm(double* matrix, int rows, double temp, int i,int j)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < rows * 2)
	{
		matrix[i*rows * 2 + x] -= (matrix[j*rows * 2 + x] / matrix[j*rows * 2 + j])*temp;
	}
}

__global__ void gjAlgorithm2(double* matrix, int rows, double temp, int i, int j )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < rows * 2)
	{
		matrix[i*rows * 2 + x] /= temp;
	}
}

void randomMatrixToFile(int rows)
{
	CFileStream file;
	const int SEED = int(time(NULL));
	srand(SEED);
	double *matrix = new double[rows*rows];

	for (int i = 0; i < rows*rows; ++i)
	{	
		matrix[i] = rand() % MAX_ELEMENT + 1;
	}

	file.write(matrix, rows);
	delete[] matrix;
}

double* readFromFile(int &rows);
void showMatrix(double *matrix, int rows);
void showFullMatrix(double *matrix, int rows);
double* stickMatrix(double* matrix, double* ident_matrix, int rows);

double* gaussJordan(double *matrix, int rows)
{
	int temp;
	double *d_matrix = 0;
	double *ident_matrix = 0;
	int size_bytes = rows * rows * 2 * sizeof(double);		// number of bytes allocated on device mem
	dim3 threadsPerBlock(BLOCK_SIZE);
	dim3 numBlocks((2*rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

	auto err = cudaMalloc(&d_matrix, size_bytes);
	if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

	ident_matrix = new double[rows*rows];
	for (int i = 0; i < rows; i++) 
	{
		for (int j = 0; j < rows; j++)
		{
			if (j == i)
				ident_matrix[i*rows + j] = 1;
			else
				ident_matrix[i*rows + j] = 0;
		}
	}

	double *augmentedmatrix = stickMatrix(matrix, ident_matrix, rows);


//=================================================================================================================================================
//															Obliczanie rownolegle - petla GJ
//=================================================================================================================================================

	for (int j = 0; j<rows; j++) 
	{
		temp = j;

		/* finding maximum jth column element in last (rows-j) rows */
		for (int i = j + 1; i < rows; i++) {
			if (augmentedmatrix[i*rows * 2 + j] > augmentedmatrix[temp*rows * 2 + j])
				temp = i;
		}
		
		if (fabs(augmentedmatrix[temp*rows*2 + j])<MIN_ELEMENT) {
			printf("\n Elements are too small to deal with !!!\n");
			break;
		}

		/* swapping row which has maximum jth column element */
		//KERNEL?
		double temporary;
		if (temp != j)
		{
			err = cudaMemcpy(d_matrix, augmentedmatrix, size_bytes, cudaMemcpyHostToDevice);
			if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

			swap << < numBlocks, threadsPerBlock >> > (d_matrix, rows, temp, j);

			err = cudaMemcpy(augmentedmatrix, d_matrix, size_bytes, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }
			
		}
	
		
		/* performing row operations to form required identity matrix out of the input matrix */
		//KERNEL?
		double r;
		for (int i = 0; i < rows; i++)
		{
			r = augmentedmatrix[i*rows * 2 + j];
			if (i != j)
			{
				err = cudaMemcpy(d_matrix, augmentedmatrix, size_bytes, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }

				gjAlgorithm << < numBlocks, threadsPerBlock >> > (d_matrix, rows, r, i, j);

				err = cudaMemcpy(augmentedmatrix, d_matrix, size_bytes, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) 
					<< " in " << __FILE__ << " at line " << __LINE__ << std::endl; }
			}
				
			else
			{
				err = cudaMemcpy(d_matrix, augmentedmatrix, size_bytes, cudaMemcpyHostToDevice);
				if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }
				
				gjAlgorithm2 << < numBlocks, threadsPerBlock >> > (d_matrix, rows, r, i, j);

				err = cudaMemcpy(augmentedmatrix, d_matrix, size_bytes, cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) { std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl; }
			}
		}
	}

	cudaFree(d_matrix);

	return augmentedmatrix;
}

double* stickMatrix(double* matrix, double* ident_matrix, int rows)
{
	double* aug_matrix = new double[rows*rows * 2];
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < rows; j++)
		{
			aug_matrix[i*rows*2 + j] = matrix[i*rows + j];
		}
		for (int j = rows; j < rows * 2; j++)
		{
			aug_matrix[i*rows*2 + j] = ident_matrix[i*rows + (j-rows)];
		}
	}
			
	return aug_matrix;
}

int main()
{
	double * matrix;
	CMeasure time;
	long long int time_table[3];
	int rows = 500;
	randomMatrixToFile(rows);
	matrix = readFromFile(rows);
	//showMatrix(matrix, rows);
	std::cout << "\n\n";
	showFullMatrix(gaussJordan(matrix, rows), rows);
	std::cout << "\n\n\n";
	return 0;
}



//==========================================================================
//==========================================================================

double* readFromFile(int &rows)
{
	CFileStream file;
	double *matrix = 0;
	std::cout << "Insert file path\n";
	file.openFile();
	matrix = file.readData(matrix, rows);		// !!! size * 2 in allocation
	return matrix;
}

void showMatrix(double *matrix, int rows)
{
	if (rows > 0)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < rows; j++)
			{
				std::cout << matrix[i*rows + j] << " ";
			}
			std::cout << std::endl;
		}
	}
}

void showFullMatrix(double *matrix, int rows)
{
	if (rows > 0)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < 2*rows; j++)
			{
				std::cout << matrix[i*2*rows + j] << " ";
			}
			std::cout << std::endl;
		}
	}
}


//class CMatrix
//{
//	CMatrix(int size) : size(size)
//	{
//		matrix = new double[size];
//	}
//	~CMatrix()
//	{
//		delete[] matrix;
//	}
//
//	CMatrix operator+(const CMatrix &augmented)
//	{
//		CMatrix augmentedMatrix(size * 2);
//		// sssssssss
//	}
//
//	double *matrix;
//
//private:
//
//	int size;
//
//};
