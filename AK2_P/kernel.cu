#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

//_global_ void ... -> device

double* readFromFile(int &rows)
{
	CFileStream file;
	double *matrix = 0;
	std::cout << "Insert file path\n";
	file.openFile();
	matrix = file.readData(matrix, rows);
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
				std::cout << matrix[i*rows+j] << " ";
			}
			std::cout << std::endl;
		}
	}
}
int main()
{
	CMeasure time;
	long long int time_table[3];

	double *matrix = 0, *d_matrix = 0;
	int rows;
	double *ident_matrix;


	matrix = readFromFile(rows);
	
	ident_matrix = new double[rows*rows];

	for(int i = 0 ; i < rows; i++)
		for (int j = 0; j < rows; j++)
		{
			if (j == i)
				ident_matrix[i*rows + j] = 1;
			else
				ident_matrix[i*rows + j] = 0;
		}


	showMatrix(matrix, rows);
	
	int size_bytes = rows * rows * 4;		// number of bytes allocated on device mem
	dim3 threadsPerBlock(size_bytes, size_bytes);
	dim3 numBlocks((rows + size_bytes - 1) / size_bytes, (rows + size_bytes - 1) / size_bytes);

	cudaMalloc(&d_matrix, size_bytes); 
	cudaMemcpy(d_matrix, matrix, size_bytes, cudaMemcpyHostToDevice);

	time.start();
	// TODO:
	time_table[0] = time.elapsed();

	cudaMemcpy(matrix, d_matrix, size_bytes, cudaMemcpyDeviceToHost);

	showMatrix(matrix, rows);				// results

	cudaFree(d_matrix);
	delete[] matrix;
	delete[] ident_matrix;
	system("PAUSE");
	return 0;
}
