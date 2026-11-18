# PCA-EXP-2-Matrix-Summation-using-2D-Grids-and-2D-Blocks-AY-23-24

<h3>AIM:</h3>
<h3>ENTER YOUR NAME</h3> PRIYANKA P
<h3>ENTER YOUR REGISTER NO</h3> 212224230212
<h3>EX. NO</h3> 2
<h3>DATE</h3> 18/11/26
<h1> <align=center> MATRIX SUMMATION WITH A 2D GRID AND 2D BLOCKS </h3>
i.  Use the file sumMatrixOnGPU-2D-grid-2D-block.cu
ii. Matrix summation with a 2D grid and 2D blocks. Adapt it to integer matrix addition. Find the best execution configuration. </h3>

## AIM:
To perform  matrix summation with a 2D grid and 2D blocks and adapting it to integer matrix addition.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1.	Initialize the data: Generate random data for two input arrays using the initialData function.
2.	Perform the sum on the host: Use the sumMatrixOnHost function to calculate the sum of the two input arrays on the host (CPU) for later verification of the GPU results.
3.	Allocate memory on the device: Allocate memory on the GPU for the two input arrays and the output array using cudaMalloc.
4.	Transfer data from the host to the device: Copy the input arrays from the host to the device using cudaMemcpy.
5.	Set up the execution configuration: Define the size of the grid and blocks. Each block contains multiple threads, and the grid contains multiple blocks. The total number of threads is equal to the size of the grid times the size of the block.
6.	Perform the sum on the device: Launch the sumMatrixOnGPU2D kernel on the GPU. This kernel function calculates the sum of the two input arrays on the device (GPU).
7.	Synchronize the device: Use cudaDeviceSynchronize to ensure that the device has finished all tasks before proceeding.
8.	Transfer data from the device to the host: Copy the output array from the device back to the host using cudaMemcpy.
9.	Check the results: Use the checkResult function to verify that the output array calculated on the GPU matches the output array calculated on the host.
10.	Free the device memory: Deallocate the memory that was previously allocated on the GPU using cudaFree.
11.	Free the host memory: Deallocate the memory that was previously allocated on the host.
12.	Reset the device: Reset the device using cudaDeviceReset to ensure that all resources are cleaned up before the program exits.

## PROGRAM:
```PY

%%writefile matrix.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CHECK(call) \
{ \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s\n", cudaGetErrorString(error)); \
        exit(1); \
    } \
}

double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

void initialData(int *ip, int size)
{
    for (int i = 0; i < size; i++)
        ip[i] = rand() % 100;
}

void sumMatrixOnHost(int *A, int *B, int *C, int nx, int ny)
{
    for (int iy = 0; iy < ny; iy++)
        for (int ix = 0; ix < nx; ix++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
}

__global__ void sumMatrixOnGPU2D(int *A, int *B, int *C,
                                  int NX, int NY)
{
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = iy * NX + ix;

    if (ix < NX && iy < NY)
        C[idx] = A[idx] + B[idx];
}

void checkResult(int *hostRef, int *gpuRef, int N)
{
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] != gpuRef[i])
        {
            printf("Arrays do not match.\n");
            return;
        }
    }
    printf("Arrays match.\n");
}

int main()
{
    int dev = 0;
    cudaDeviceProp deviceProp;

    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    CHECK(cudaSetDevice(dev));

    printf("Using Device: %s\n", deviceProp.name);

    int nx = 1 << 10;
    int ny = 1 << 10;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    printf("Matrix size: %d x %d\n", nx, ny);

    int *h_A, *h_B, *hostRef, *gpuRef;

    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    hostRef = (int *)malloc(nBytes);
    gpuRef = (int *)malloc(nBytes);

    if (!h_A || !h_B || !hostRef || !gpuRef)
    {
        printf("Host memory allocation failed\n");
        return 1;
    }

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    double start = seconds();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    printf("CPU time: %f sec\n", seconds() - start);

    int *d_A, *d_B, *d_C;

    CHECK(cudaMalloc((void **)&d_A, nBytes));
    CHECK(cudaMalloc((void **)&d_B, nBytes));
    CHECK(cudaMalloc((void **)&d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32, 32);
    dim3 grid((nx + 31) / 32, (ny + 31) / 32);

    start = seconds();

    sumMatrixOnGPU2D<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    printf("GPU time: %f sec\n", seconds() - start);

    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    checkResult(hostRef, gpuRef, nxy);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    CHECK(cudaDeviceReset());

    return 0;
}
```

## OUTPUT:
<img width="291" height="56" alt="image" src="https://github.com/user-attachments/assets/85ba66bc-fcf2-4c05-8d8f-e1949fa35830" />
<img width="397" height="137" alt="image" src="https://github.com/user-attachments/assets/6e27ff88-41e2-4905-89f4-9365ad562b02" />


## RESULT:
The host took _________ seconds to complete it’s computation, while the GPU outperforms the host and completes the computation in ________ seconds. Therefore, float variables in the GPU will result in the best possible result. Thus, matrix summation using 2D grids and 2D blocks has been performed successfully.
