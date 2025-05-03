#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define N 1024  // Size of the vectors and matrix

// Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        C[index] = A[index] + B[index];
    }
}

// Kernel for matrix multiplication
__global__ void matrixMul(int *A, int *B, int *C, int width) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < width && col < width) {
        int value = 0;
        for (int i = 0; i < width; i++) {
            value += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = value;
    }
}

int main() {
    // Allocate memory for vectors and matrix
    int *h_A = (int*)malloc(N * sizeof(int));
    int *h_B = (int*)malloc(N * sizeof(int));
    int *h_C = (int*)malloc(N * sizeof(int));
    int *h_MatrixA = (int*)malloc(N * N * sizeof(int));
    int *h_MatrixB = (int*)malloc(N * N * sizeof(int));
    int *h_MatrixC = (int*)malloc(N * N * sizeof(int));

    // Initialize vectors and matrices
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = N - i;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_MatrixA[i * N + j] = i + j;
            h_MatrixB[i * N + j] = (i + j) % 5;
        }
    }

    // Allocate memory on the device for vectors and matrices
    int *d_A, *d_B, *d_C, *d_MatrixA, *d_MatrixB, *d_MatrixC;
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));
    cudaMalloc((void**)&d_MatrixA, N * N * sizeof(int));
    cudaMalloc((void**)&d_MatrixB, N * N * sizeof(int));
    cudaMalloc((void**)&d_MatrixC, N * N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatrixA, h_MatrixA, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatrixB, h_MatrixB, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Set up the execution configuration for vector addition (1D)
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernel for vector addition
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first 10 results of vector addition
    std::cout << "Vector Addition (First 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << "\n";

    // Set up the execution configuration for matrix multiplication (2D)
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel for matrix multiplication
    matrixMul<<<gridDim, blockDim>>>(d_MatrixA, d_MatrixB, d_MatrixC, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_MatrixC, d_MatrixC, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first 10 elements of matrix multiplication result (flattened)
    std::cout << "Matrix Multiplication (First 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_MatrixC[i] << " ";
    }
    std::cout << "\n";

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_MatrixA);
    free(h_MatrixB);
    free(h_MatrixC);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_MatrixA);
    cudaFree(d_MatrixB);
    cudaFree(d_MatrixC);

    return 0;
}
