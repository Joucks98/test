#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "kernel.h"

//#include <iostream>
#include <vector>

#include <chrono>   
using namespace std;
using namespace chrono;

#define H 451089
#define W 2676
#define F 100

//#define H 351089
//#define F 2676
//#define W 1

//生成随机矩阵
void matgen(float* a, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            a[i * n + j] = (float)rand() / RAND_MAX + (float)rand() / (RAND_MAX * RAND_MAX);
        }
    }
}


int main()
{
    //CUDA 初始化
    if (!InitCUDA()) return 0;    


    auto quarter_point = [](int l)->vector<int> { return { 0, l / 4, l / 2, l * 3 / 4, l }; };
    int tn = H / 2;
    vector<int> len1 = quarter_point(tn);    
    auto len2 = quarter_point(H - tn);
    bool flag = true;

    cudaError_t cudaStatus;
    float *cuda_Q = 0;


    //定义矩阵
    float *a = 0, *b = 0, *c = 0, *d = 0;

    //分配内存
    a = (float*)malloc(sizeof(float)* H * F);
    if (a == NULL)
    {
        goto Error2;
    }
    b = (float*)malloc(sizeof(float)* F * W);
    if (b == NULL)
    {
        goto Error2;
    }
    c = (float*)malloc(sizeof(float)* H * W);
    if (c == NULL)
    {
        goto Error2;
    }
    d = (float*)malloc(sizeof(float)* H * W);
    if (d == NULL)
    {
        goto Error2;
    }
    //设置随机数种子
    srand(9);
    //随机生成矩阵
    matgen(a, H, F);
    matgen(b, F, W);
    
    /*if (matMult(a, b, c, H, F, W, 0) == false)
    {
        goto Error2;
    }*/


    

    cudaStatus = cudaMalloc((void**)&cuda_Q, F* W * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error2;
    }
    cudaStatus = cudaMemcpy(cuda_Q, b, sizeof(float)* F * W, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy b failed!");
        goto Error2;
    }

    
#pragma omp parallel num_threads(10)
    {
#pragma omp sections
        {
#pragma omp section
            {
                printf("section 1 线程ID: %d\n", omp_get_thread_num());
                matMult(a + len1[0] * F, cuda_Q, c + len1[0] * W, len1[1] - len1[0], F, W);
            }
#pragma omp section
            {
                printf("section 2 线程ID: %d\n", omp_get_thread_num());
                matMult(a + len1[1] * F, cuda_Q, c + len1[1] * W, len1[2] - len1[1], F, W);
            }
#pragma omp section
            {
                printf("section 3 线程ID: %d\n", omp_get_thread_num());
                matMult(a + len1[2] * F, cuda_Q, c + len1[2] * W, len1[3] - len1[2], F, W);
            }
#pragma omp section
            {
                printf("section 4 线程ID: %d\n", omp_get_thread_num());
                matMult(a + len1[3] * F, cuda_Q, c + len1[3] * W, len1[4] - len1[3], F, W);
            }
        }
#pragma omp sections
        {
#pragma omp section
            {
                printf("section 5 线程ID: %d\n", omp_get_thread_num());
                matMult(a + (tn + len2[0]) * F, cuda_Q, c + (tn + len2[0])*W, len2[1] - len2[0], F, W);
            }
#pragma omp section
            {
                printf("section 6 线程ID: %d\n", omp_get_thread_num());
                matMult(a + (tn + len2[1]) * F, cuda_Q, c + (tn + len2[1])*W, len2[2] - len2[1], F, W);
            }
#pragma omp section
            {
                printf("section 7 线程ID: %d\n", omp_get_thread_num());
                matMult(a + (tn + len2[2]) * F, cuda_Q, c + (tn + len2[2])*W, len2[3] - len2[2], F, W);
            }
#pragma omp section
            {
                printf("section 8 线程ID: %d\n", omp_get_thread_num());
                matMult(a + (tn + len2[3]) * F, cuda_Q, c + (tn + len2[3])*W, len2[4] - len2[3], F, W);
            }
        }
    }




    auto start = system_clock::now();
    //CPU矩阵乘法，存入矩阵d
#pragma omp parallel for
    for (int i = 0; i < H; ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < W; ++j)
        {
            float sum = 0;
            float re = 0.0;
#pragma omp parallel for reduction(+:sum)
            for (int k = 0; k < F; ++k)
            {
                /*re -= a[i * F + k] * b[k * W + j];
                float ac = sum - re;
                re += (ac - sum);
                sum = ac;*/
                //sum += a[i * F + k] * b[k * W + j];
                sum += a[i * F + k] * b[k + j*F];
            }
            d[i * W + j] = sum;
        }
    }
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    printf("CPU下 花费的执行时间：%f(s)\n", double(duration.count()) * microseconds::period::num / microseconds::period::den);

    //验证正确性与精确性
    double max_err = 0;
    double average_err = 0;


    for (int i = 0; i < H; ++i)
    {
        for (int j = 0; j < W; ++j)
        {
            if (d[i * W + j] != 0)
            {
                //fabs求浮点数x的绝对值
                double err = fabs((c[i * W + j] - d[i * W + j]) / d[i * W + j]);
                if (max_err < err) max_err = err;
                average_err += err;
            }
        }
    }

    printf("Max error: %g Average error: %g\n", max_err, average_err / (H * W));
    

Error2:
    free(a);
    free(b);
    free(c);
    free(d);

    cudaFree(cuda_Q);

    return 0;

}
