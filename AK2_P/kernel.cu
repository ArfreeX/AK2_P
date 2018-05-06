#include "stdafx.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

//_global_ void ... -> device
 


int** readFromFile(int &rows)
{
	CFileStream file;
	int **matrix = 0;
	std::cout << "Insert file path\n";
	file.openFile();
	matrix = file.readData(matrix, rows);
	return matrix;
}

void showMatrix(int **matrix, int rows)
{
	if (rows > 0)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < rows; j++)
				std::cout << matrix[i][j] << " ";
			std::cout << std::endl;
		}
		
	}
}
int main()
{
	CMeasure time;
	long long int time_table[3];

	int **matrix = 0, **d_matrix = 0, rows; // todo: delete []
	
	matrix = readFromFile(rows);
	showMatrix(matrix, rows);
	
	int size_bytes = rows * rows * 4;		// number of bytes allocated on device mem

	cudaMalloc(&d_matrix, size_bytes); 
	cudaMemcpy(d_matrix, matrix, size_bytes, cudaMemcpyHostToDevice);

	time.start();
	// todo
	time_table[0] = time.elapsed();

	cudaMemcpy(matrix, d_matrix, size_bytes, cudaMemcpyDeviceToHost);

	showMatrix(matrix, rows);				// results

	cudaFree(d_matrix);
	for (int i = 0; i < rows; i++)
		delete [] matrix[i];
	delete[] matrix;

	system("PAUSE");
	return 0;
}