#ifndef KERNEL_H
#define KERNEL_H
extern "C"

// rowMajor 0: default type, result is row ordered.
// rowMajor 1: row type
// rowMajor 2: col type
bool __declspec (dllexport) matMult(const float* a, const float* b, float* c, int h_a, int w_a, int w_b, int rowMajor);
bool __declspec (dllexport) matAdd(const float* a, const float* b, float* c, int m, int n, bool rowMajor = true);

bool __declspec (dllexport) biasSVD(const float* P, const float* Q, const float* bu, const float* bi, float mu, int m, int n, int f, float* result, int rowMajor);
bool __declspec (dllexport) sigmoidBiasSVD(const float* P, const float* Q, const float* bu, const float* bi, float mu, int m, int n, int f, float* result, int rowMajor);


#endif // !KERNEL_H