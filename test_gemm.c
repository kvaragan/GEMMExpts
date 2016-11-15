/* Kiran Varaganti - GEMM implementation for small matirces */
#include <unistd.h>
#include "blis.h"
#include "gemmExpts.h"

#define  TESTING_

#ifdef TESTING_
FILE *fp = NULL;
#endif

int main(int argc, char*argv[])
{
  obj_t a, b, c, c_save;
  dim_t m, n, k;
  inc_t rs, cs;

  double dtime;
  double dtime_save;
  double gflops;
  num_t dt;
  int n_repeats = 3;
  void (*gemmfptr)(obj_t, obj_t, obj_t);
  char storage = 'c'; // column storage
  
  if (argc < 3)
    {
      printf ("Enter matrix dimension and storage(c or r)\n");
      exit(1);
    }

  bli_init();
  dt = BLIS_FLOAT;
  trans_t  transa = BLIS_NO_TRANSPOSE;
  trans_t  transb = BLIS_NO_TRANSPOSE;

  m = atoi(argv[1]);
  n = m;
  k = m;

  storage = argv[2];
  if (storage == 'c')
    {
      rs = 1;
      cs = m;
    }
  else if (storage == 'r')
    {
      // row-major
      cs = 1;
      rs = n;
    }
  else
    {
      rs = 0;
      cs = 0;
    }
  
  if(m == 128)
    gemmfptr = sgemm128;
  else if (m == 32)
    gemmfptr = sgemm32;
  else if (m == 4)
    gemmfptr = sgemm4;
else
    gemmfptr = sgemm;

  bli_obj_create(dt, m, k, rs, cs, &a);
  bli_obj_create(dt, k, n, rs, cs, &b);
  bli_obj_create(dt, m, n, rs, cs, &c);
  bli_obj_create(dt, m, n, rs, cs, &c_save);

  bli_randm(&a);
  bli_randm(&b);
#ifndef TESTING_
  bli_randm(&c);
  bli_copym(&c, &c_save);
#endif
  dtime_save = 1.0e9;

#ifdef TESTING_
  float* apin = bli_obj_buffer(a);
  float* bpin = bli_obj_buffer(b);
  float* csp    = bli_obj_buffer(c_save);

  for (int i = 0; i < m*k; i++)
    {
      apin[i] = 2.0; //(float)(1 << (i%3));
    }

  for (int i = 0; i < k*n; i++)
    {
      bpin[i] = 2.0; //(float)(1 << (i%5));
    }

  for (int i = 0; i < m*n; i++)
    {
      csp[i] = 0.0;
    }
   bli_copym(&c_save, &c);

  fp = fopen("ref.txt","wt");
  if (fp == NULL) 
    {
      printf("Error opening the file\n");
      exit(1);
    }
  fprintf(fp, "Matrix A\n");
 for (int i = 0; i < m; i++)
   {
    for(int j= 0; j < k; j++)
      fprintf(fp,"%3.3f ", apin[i*rs + j*cs]);
    fprintf(fp,"\n");
   }
    
  fprintf(fp, "\nMatrix B\n");
 for (int i = 0; i < k; i++)
   {
    for(int j= 0; j < n; j++)
      fprintf(fp,"%3.3f ", bpin[i*rs + j*cs]);
    fprintf(fp,"\n");
   }

#endif
  
  for (int r = 0; r < n_repeats; ++r)
    {
      bli_copym(&c_save, &c);
     
      dtime = bli_clock();
      gemmfptr(a, b, c);
      dtime_save = bli_clock_min_diff(dtime_save, dtime);
    }

  gflops = (2.0 * m* n* k)/(dtime_save * 1.0e9);
  printf("[%4lu %4lu %4lu %10.3e %6.3f];\n", (unsigned long)m,
	 (unsigned long)k,
         (unsigned long)n,
         dtime_save,
         gflops);

#ifdef TESTING_
  obj_t cref;
  bli_obj_create(dt, m, n, 0, 0, &cref);
 bli_copym(&c_save, &cref);
  sgemm(a, b, cref);
  float* ap   = bli_obj_buffer(a);
  float* bp   = bli_obj_buffer(b);
  float* cp   = bli_obj_buffer(c);
  float* cr   = bli_obj_buffer(cref);

  float err = 0.0;
  for (int i = 0; i < m*n; i++)
    {
      // err = cp[i] - cr[i];
      //      printf("Error @ [%d] = %6.3f\n", i, err);
    }

 fprintf(fp, "\nMatrix Cref\n"); 
 for (int i = 0; i < m; i++)
   {
    for(int j= 0; j < n; j++)
      fprintf(fp,"%3.3f ", cr[i*rs + j*cs]);
    fprintf(fp,"\n");
   }
 

 fprintf(fp, "\nMatrix C\n"); 
 for (int i = 0; i < m; i++)
   {
    for(int j= 0; j < n; j++)
      fprintf(fp,"%3.3f ", cp[i*rs + j*cs]);
    fprintf(fp,"\n");
   }
 
 fclose(fp);
  bli_obj_free(&cref);
#endif

  bli_obj_free(&a);
  bli_obj_free(&b);
  bli_obj_free(&c);
  bli_obj_free(&c_save);

  bli_finalize();

  return 0;
}//end main
