%%writefile matmul.cu
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

__global__ void matmul(int* A, int* B, int* C, int N) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if (Row < N && Col < N) {
        int Pvalue = 0;
        for (int k = 0; k < N; k++) {
            Pvalue += A[Row * N + k] * B[k * N + Col];
        }
        C[Row * N + Col] = Pvalue;
    }
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }
}

void inputMatrix(int* matrix, int N, const string& name) {
    cout << "Enter values for matrix " << name << " (" << N << "x" << N << "):\n";
    for (int i = 0; i < N * N; i++) {
        cout << "Enter value for " << name << "[" << i / N << "][" << i % N << "]: ";
        cin >> matrix[i];
    }
}

void printMatrix(const int* matrix, int N, const string& name) {
    cout << "\nMatrix " << name << ":\n";
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << matrix[i * N + j] << " ";
        }
        cout << endl;
    }
}

int main() {
    int N = 2;  // Size of the matrix (2x2)
    int size = N * N * sizeof(int);
    
    int *A, *B, *C;   // Host matrices
    int *dev_A, *dev_B, *dev_C;  // Device matrices

    // Allocate memory on host and device
    checkCudaError(cudaMallocHost(&A, size), "cudaMallocHost A");
    checkCudaError(cudaMallocHost(&B, size), "cudaMallocHost B");
    checkCudaError(cudaMallocHost(&C, size), "cudaMallocHost C");
    checkCudaError(cudaMalloc(&dev_A, size), "cudaMalloc A");
    checkCudaError(cudaMalloc(&dev_B, size), "cudaMalloc B");
    checkCudaError(cudaMalloc(&dev_C, size), "cudaMalloc C");

    // Input matrices A and B
    inputMatrix(A, N, "A");
    inputMatrix(B, N, "B");

    // Copy matrices from host to device
    checkCudaError(cudaMemcpy(dev_A, A, size, cudaMemcpyHostToDevice), "Memcpy A");
    checkCudaError(cudaMemcpy(dev_B, B, size, cudaMemcpyHostToDevice), "Memcpy B");

    // Launch kernel
    matmul<<<dim3(N, N), dim3(1, 1)>>>(dev_A, dev_B, dev_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    // Copy result from device to host
    checkCudaError(cudaMemcpy(C, dev_C, size, cudaMemcpyDeviceToHost), "Memcpy C");

    // Print matrices
    printMatrix(A, N, "A");
    printMatrix(B, N, "B");
    printMatrix(C, N, "C");

    // Free memory
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);

    return 0;
}



// !nvcc -arch=sm_75 matmul.cu -o matmul
// !./matmul
