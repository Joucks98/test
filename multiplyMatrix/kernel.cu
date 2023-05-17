#include <stdio.h>
#include <math.h>
//#include <time.h>

//CUDA RunTime API
#include <cuda_runtime.h>
#include <omp.h>
#include "kernel.h"

#define TX 8
#define TY 128

#define TX1 1
#define TY1 1024

//打印设备信息
void printDeviceProp(const cudaDeviceProp &prop)
{
    printf("Device Name : %s.\n", prop.name);
    printf("totalGlobalMem : %zd.\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock : %zd.\n", prop.sharedMemPerBlock);
    printf("regsPerBlock : %d.\n", prop.regsPerBlock);
    printf("warpSize : %d.\n", prop.warpSize);
    printf("memPitch : %zd.\n", prop.memPitch);
    printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem : %zd.\n", prop.totalConstMem);
    printf("major.minor : %d.%d.\n", prop.major, prop.minor);
    printf("clockRate : %d.\n", prop.clockRate);
    printf("textureAlignment : %zd.\n", prop.textureAlignment);
    printf("deviceOverlap : %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//CUDA 初始化
bool InitCUDA()
{
    int count;

    //取得支持Cuda的装置的数目
    cudaGetDeviceCount(&count);

    if (count == 0)
    {
        fprintf(stderr, "There is no device.\n");

        return false;
    }
    int i;
    for (i = 0; i < count; ++i)
    {

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        //打印设备信息
        printDeviceProp(prop);

        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess)
        {
            if (prop.major >= 1)
            {
                break;
            }
        }
    }

    if (i == count)
    {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;

}



// __global__ 函数 并行计算矩阵乘法
__global__ void matMultCUDA(const float* a, const float* b, float* c, int m, int n, int f, int rowMajor)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    //计算矩阵乘法
    if (row < m && col < n)
    {
        float sum = 0.0;
        float re = 0.0;
        if (rowMajor == 0) // default
        {
            for (int i = 0; i < f; ++i)
            {
                re -= a[row * f + i] * b[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            c[row * n + col] = sum;
        }
        if (rowMajor == 1)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= a[row * f + i] * b[i * n + col];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            c[row * n + col] = sum;
        }
        if (rowMajor == 2)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= a[row + i * m] * b[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            c[col*m + row] = sum;
        }
        
    }
}

//bool matMultiply(const float* a, const float* b, float* c, int h_a, int w_a, int w_b)
//{
//    float time_elapsed = 0;
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    cudaEventRecord(start, 0);
//
//    const dim3 blockSize(TX, TY);
//    const dim3 gridSize = dim3((w_b + blockSize.x - 1) / blockSize.x, (h_a + blockSize.y - 1) / blockSize.y);
//
//    cudaError_t cudaStatus;
//    /*把数据复制到显卡内存中*/
//    float *cuda_a = 0, *cuda_b = 0, *cuda_c = 0;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    //cudaMalloc 取得一块显卡内存 
//    cudaStatus = cudaMalloc((void**)&cuda_a, sizeof(float)* h_a * w_a);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//    cudaStatus = cudaMalloc((void**)&cuda_b, sizeof(float)* w_a * w_b);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//    cudaStatus = cudaMalloc((void**)&cuda_c, sizeof(float)* h_a * w_b);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//
//    //cudaMemcpy 将产生的矩阵复制到显卡内存中
//    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
//    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
//    cudaStatus = cudaMemcpy(cuda_a, a, sizeof(float)* h_a * w_a, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//    cudaStatus = cudaMemcpy(cuda_b, b, sizeof(float)* w_a * w_b, cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//   
//    // 在CUDA 中执行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(参数...);
//    matMultCUDA<<<gridSize, blockSize >>>(cuda_a, cuda_b, cuda_c, h_a, w_b, w_a);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "matMultCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, cuda_c, h_a * w_b * sizeof(float), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(start);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&time_elapsed, start, stop);    
//    printf("GPU下 matMultiply 的执行时间：%f(ms)\n", time_elapsed);
//
//Error:
//    cudaFree(cuda_a);
//    cudaFree(cuda_b);
//    cudaFree(cuda_c);
//
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//
//    return cudaStatus == cudaSuccess;
//}

// __global__ 函数 并行计算矩阵向量乘法
__global__ void matVecMultCUDA(const float* ma, const float* vb, float* c, int h_a, int w_a)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (row < h_a)
    {
        float sum = 0.0;
        float re = 0.0;
        for (int i = 0; i < w_a; ++i)
        {
            re -= ma[row * w_a + i] * vb[i];
            float ac = sum - re;
            re += (ac - sum);
            sum = ac;
            //sum += ma[row * w_a + i] * vb[i];
        }
        c[row] = sum;
    }
}


__global__ void matAddCUDA(const float *a, const float *b, float *c, int m, int n, bool rowMajor)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < m && col < n)
    {
        int id = rowMajor ? (row * n + col) : (col*m + row);
        c[id] = a[id] + b[id];
    }    
}


__global__ void biasSVDWithCUDA(const float* P, const float* Q, const float* bu, const float* bi, float mu, int m, int n, int f, float* result, int rowMajor)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    //计算矩阵乘法
    if (row < m && col < n)
    {
        float bias = bu[row] + bi[col] + mu;
        float sum = 0.0;
        float re = 0.0;
        if (rowMajor == 0) // default
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row * f + i] * Q[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[row * n + col] = sum + bias;
        }
        if (rowMajor == 1)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row * f + i] * Q[i * n + col];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[row * n + col] = sum + bias;
        }
        if (rowMajor == 2)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row + i * m] * Q[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[col*m + row] = sum + bias;
        }
    }
}

__device__
double sigmoid(double x)
{
    return 1.0 / (1 + exp(-x));
}

__global__ void sigmoidBiasSVDWithCUDA(const float* P, const float* Q, const float* bu, const float* bi, float mu, int m, int n, int f, float* result, int rowMajor)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    //计算矩阵乘法
    if (row < m && col < n)
    {
        float bias = bu[row] + bi[col] + mu;
        float sum = 0.0;
        float re = 0.0;
        if (rowMajor == 0) // default
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row * f + i] * Q[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[row * n + col] = (float)sigmoid(sum + bias);
        }
        if (rowMajor == 1)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row * f + i] * Q[i * n + col];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[row * n + col] = (float)sigmoid(sum + bias);
        }
        if (rowMajor == 2)
        {
            for (int i = 0; i < f; ++i)
            {
                re -= P[row + i * m] * Q[i + col * f];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;
            }
            result[col*m + row] = (float)sigmoid(sum + bias);
        }
    }
}


bool matMult(const float* a, const float* b, float* c, int h_a, int w_a, int w_b, int rowMajor)
{
    dim3 blockSize(TX, TY);
    if (w_b == 1)
    {
        blockSize = dim3(TX1, TY1);
    }
    dim3 gridSize = dim3((w_b + blockSize.x - 1) / blockSize.x, (h_a + blockSize.y - 1) / blockSize.y);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;
    /*把数据复制到显卡内存中*/
    float *cuda_a = 0, *cuda_b = 0, *cuda_c = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //cudaMalloc 取得一块显卡内存 
    cudaStatus = cudaMalloc((void**)&cuda_a, sizeof(float)* h_a * w_a);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc a failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_b, sizeof(float)* w_a * w_b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc b failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_c, sizeof(float)* h_a * w_b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc c failed!");
        goto Error;
    }


    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaStatus = cudaMemcpy(cuda_a, a, sizeof(float)* h_a * w_a, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy a failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_b, b, sizeof(float)* w_a * w_b, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy b failed!");
        goto Error;
    }

    if (w_b == 1)
    {
        matVecMultCUDA << <gridSize, blockSize >> > (cuda_a, cuda_b, cuda_c, h_a, w_a);
    }
    else
    {
        matMultCUDA << <gridSize, blockSize >> > (cuda_a, cuda_b, cuda_c, h_a, w_b, w_a, rowMajor);
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matMultCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, cuda_c, h_a * w_b * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy c failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    printf("GPU matMultiply cost time: %f(ms)\n", time_elapsed);

Error:
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus == cudaSuccess;
}

bool matAdd(const float * a, const float * b, float * c, int m, int n, bool rowMajor)
{
    dim3 blockSize(TX, TY);
    dim3 gridSize = dim3((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;
    /*把数据复制到显卡内存中*/
    float *cuda_a = 0, *cuda_b = 0, *cuda_c = 0;
    size_t byteSize = m * n*sizeof(float);
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //cudaMalloc 取得一块显卡内存 
    cudaStatus = cudaMalloc((void**)&cuda_a, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_b, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_c, byteSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaStatus = cudaMemcpy(cuda_a, a, byteSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_b, b, byteSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    matAddCUDA << <gridSize, blockSize >> > (cuda_a, cuda_b, cuda_c, m, n, rowMajor);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matAddCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, cuda_c, byteSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    printf("GPU matAdd cost time: %f(ms)\n", time_elapsed);

Error:
    cudaFree(cuda_a);
    cudaFree(cuda_b);
    cudaFree(cuda_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus == cudaSuccess;
}

bool biasSVD(const float * P, const float * Q, const float * bu, const float * bi, float mu, int m, int n, int f, float * result, int rowMajor)
{
    dim3 blockSize(TX, TY);
    dim3 gridSize = dim3((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;
    /*把数据复制到显卡内存中*/
    float *cuda_P = 0, *cuda_Q = 0, *cuda_bu = 0, *cuda_bi = 0, *cuda_re = 0;
    
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //cudaMalloc 取得一块显卡内存 
    cudaStatus = cudaMalloc((void**)&cuda_P, m * f * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_Q, f* n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_bu, m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&cuda_bi, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&cuda_re, m * n* sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaStatus = cudaMemcpy(cuda_P, P, m * f * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_Q, Q, f * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(cuda_bu, bu, m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_bi, bi, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    biasSVDWithCUDA << <gridSize, blockSize >> > (cuda_P, cuda_Q, cuda_bu, cuda_bi, mu, m, n, f, cuda_re, rowMajor);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "biasSVDWithCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, cuda_re, m*n*sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    printf("call biasSVD in: %f(ms)\n", time_elapsed);

Error:
    cudaFree(cuda_P);
    cudaFree(cuda_Q);
    cudaFree(cuda_bu);
    cudaFree(cuda_bi);
    cudaFree(cuda_re);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus == cudaSuccess;
}

bool sigmoidBiasSVD(const float * P, const float * Q, const float * bu, const float * bi, float mu, int m, int n, int f, float * result, int rowMajor)
{
    dim3 blockSize(TX, TY);
    dim3 gridSize = dim3((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;
    /*把数据复制到显卡内存中*/
    float *cuda_P = 0, *cuda_Q = 0, *cuda_bu = 0, *cuda_bi = 0, *cuda_re = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    //cudaMalloc 取得一块显卡内存 
    cudaStatus = cudaMalloc((void**)&cuda_P, m * f * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_Q, f* n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&cuda_bu, m * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&cuda_bi, n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&cuda_re, m * n * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaStatus = cudaMemcpy(cuda_P, P, m * f * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_Q, Q, f * n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(cuda_bu, bu, m * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(cuda_bi, bi, n * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    sigmoidBiasSVDWithCUDA << <gridSize, blockSize >> > (cuda_P, cuda_Q, cuda_bu, cuda_bi, mu, m, n, f, cuda_re, rowMajor);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "biasSVDWithCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, cuda_re, m*n * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    printf("call biasSVD in: %f(ms)\n", time_elapsed);

Error:
    cudaFree(cuda_P);
    cudaFree(cuda_Q);
    cudaFree(cuda_bu);
    cudaFree(cuda_bi);
    cudaFree(cuda_re);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus == cudaSuccess;
}

bool matMult(const float * a, const float * cuda_b, float * c, int h_a, int w_a, int w_b)
{
    dim3 blockSize(TX, TY);
    dim3 gridSize = dim3((w_b + blockSize.x - 1) / blockSize.x, (h_a + blockSize.y - 1) / blockSize.y);

    float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    /*把数据复制到显卡内存中*/
    float *cuda_a = 0, *cuda_c = 0;
    //cudaMalloc 取得一块显卡内存 
    cudaStatus = cudaMalloc((void**)&cuda_a, sizeof(float)* h_a * w_a);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc a failed!");
        goto Error;
    }
    
    cudaStatus = cudaMalloc((void**)&cuda_c, sizeof(float)* h_a * w_b);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc c failed!");
        goto Error;
    }


    //cudaMemcpy 将产生的矩阵复制到显卡内存中
    //cudaMemcpyHostToDevice - 从内存复制到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存复制到内存
    cudaStatus = cudaMemcpy(cuda_a, a, sizeof(float)* h_a * w_a, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy a failed!");
        goto Error;
    }
    

    matMultCUDA << <gridSize, blockSize >> > (cuda_a, cuda_b, cuda_c, h_a, w_b, w_a, 0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "matMultCUDA launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, cuda_c, h_a * w_b * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy c failed!");
        goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);
    printf("GPU matMultiply cost time: %f(ms)\n", time_elapsed);

Error:
    cudaFree(cuda_a);
    cudaFree(cuda_c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return cudaStatus == cudaSuccess;
}

