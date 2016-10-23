/* Kiran Varaganti - GEMM implementation for small matirces */
#include <unistd.h>
#include "blis.h"
#include "gemmExpts.h"

int main(void)
{
  obj_t a, b, c, c_save;
  dim_t m, n, k;

  double dtime;
  double dtime_save;
  double gflops;
  num_t dt;
  int num_repeats = 3;
  

  bli_init();
  dt = BLIS_FLOAT;
  trans_t  transa = BLIS_NO_TRANSPOSE;
  trans_t  transb = BLIS_NO_TRANSPOSE;

  m = 128;
  n = 128;
  k = 128;

  bli_obj_create(dt, m, k, 0, 0, &a);
  bli_obj_create(dt, k, n, 0, 0, &b);
  bli_obj_create(dt, m, n, 0, 0, &c);
  bli_obj_create(dt, m, n, 0, 0, &c_save);

  bli_randm(&a);
  bli_randm(&b);
  bli_randm(&c);
  bli_copym(&c, &c_save);
  dtime_save = 1.0e9;
  
  for (int r = 0; r < n_repeats; ++r)
    {
      bli_copym(&c_save, &c);
     
      dtime = bli_clock();

      sgemm(a, b, c);
      dtime_save = bli_clock_min_diff(dtime_save, dtime);
    }
  gflops = (2.0 * m* n* k)/(dtime_save * 1.0e9);
  printf("[%4lu %4lu %4lu %10.3e %6.3f];\n", (unsigned long)m,
	 (unsigned long)k,
         (unsigned long)n,
         dtime_save,
         gflops);

  bli_obj_free(&a);
  bli_obj_free(&b);
  bli_obj_free(&c);
  bli_obj_free(&c_save);

  bli_finalize();

  return 0;


}
