#ifndef _GEMMUKERNELS_H
#define _GEMMUKERNELS_H

void sgemm128_ukernel(const float* A, const float* B, float* C);
void sgemm32_ukernel (const float* A, const float* B, float* C);
void sgemm4_ukernel (const float* A, const float* B, float* C);
#endif
