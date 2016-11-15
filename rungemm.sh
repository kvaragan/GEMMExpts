
gcc -mavx2 -mfma -mfpmath=sse -march=core-avx2 -O3 -std=c99 -m64 -Wall -lm -I/home/vkprofile/blis/include/blis -I. -l:/home/vkprofile/blis/lib/libblis.a basicGEMM.c test_gemm.c

