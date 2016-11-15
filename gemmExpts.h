/* Header file for GEMM declarations */

#ifndef __gemmEXPTS_H__
#define __gemmEXPTS_H__

void sgemm(obj_t a, obj_t b, obj_t c);

void sgemm32(obj_t a, obj_t b, obj_t c);

void sgemm128(obj_t a, obj_t b, obj_t c);

void sgemm4(obj_t a, obj_t b, obj_t c);

#endif // end of __gemmEXPTS_H__
