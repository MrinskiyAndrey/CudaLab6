
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "wb.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Вычисление C = A * B
__global__ void matrixMultiply(float* A, float* B, float* C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) 
{
    //@@ Вставьте ваш код произведения матриц
        // assert(numAColumns == numBRows);
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < numCRows) && (Col < numCColumns)) {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < numAColumns; ++k) {
            Pvalue += A[Row * numAColumns + k] * B[k * numAColumns + Col];
        }
        C[Row * numCColumns + Col] = Pvalue;
    }
}



int main(int argc, char** argv) {
    wbArg_t args;
    float* hostA; // Матрица A
    float* hostB; // Матрица B
    float* hostC; // Выходная матрица C
    float* deviceA;
    float* deviceB;
    float* deviceC;
    int numARows;    // количество строк матрицы A
    int numAColumns; // количество столбцов матрицы A
    int numBRows;    // количество строк матрицы B
    int numBColumns; // количество столбцов матрицы B
    int numCRows;    // количество строк матрицы  C (установите
    // это значение сами)
    int numCColumns; // количество столбцов матрицы C (установите 
    //это значение сами)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float*)wbImport(wbArg_getInputFile(args, 0), &numARows,
        &numAColumns);
    hostB = (float*)wbImport(wbArg_getInputFile(args, 1), &numBRows,
        &numBColumns);
    //@@ Установите numCRows и numCColumns
    numCRows = 8;
    numCColumns = 8;
    //@@ Выделение памяти под матрицу hostC
    hostC = (float*)malloc(numCColumns * numCRows * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Выделите память GPU
    cudaMalloc(&deviceA, numAColumns * numARows * sizeof(float));
    cudaMalloc(&deviceB, numBColumns * numBRows * sizeof(float));
    cudaMalloc(&deviceC, numCColumns * numCRows * sizeof(float));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Скопируйте память с хоста на GPU 
    cudaMemcpy(deviceA, hostA, numAColumns * numARows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, numBColumns * numBRows * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Инициализируйте размерности блоков и сетки
    dim3 block(32, 32);
    dim3 grid(ceil(static_cast<float>(numCColumns) / block.x), ceil(static_cast<float>(numCRows) / block.y));

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ запустите ядро GPU
    matrixMultiply << <grid, block >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Скопируйте память обратно с GPU на хост
    cudaMemcpy(hostC, deviceC, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Освободите память GPU

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
