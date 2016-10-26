/* Kiran Varaganti - GEMM implementation for small matirces */
#include <unistd.h>
#include "blis.h"
#include "gemmExpts.h"
#include "gemmUkernels.h"


void sgemm_Ukernel_ref(
                    float *a, 
                    float *b, 
                    float* c,
                    inc_t m,
                    inc_t n,
                    inc_t k,  
                    inc_t rs_a, inc_t cs_a,
                    inc_t rs_b, inc_t cs_b,
                    inc_t rs_c, 
                    inc_t cs_c
		   );

void sgemm(obj_t a, obj_t b, obj_t c)
{
  dim_t m  = bli_obj_length(c);
  dim_t k  = bli_obj_width_after_trans(a);
  dim_t n  = bli_obj_width(c);

  dim_t lda = bli_obj_col_stride(a);
  dim_t ldb = bli_obj_col_stride(b);
  dim_t ldc = bli_obj_col_stride(c);

  float* ap = bli_obj_buffer(a);
  float* bp = bli_obj_buffer(b);
  float* cp = bli_obj_buffer(c);

  inc_t rsa = a.rs;
  inc_t csa = a.cs;
 
  inc_t rsb = b.rs;
  inc_t csb = b.cs;

  inc_t rsc = c.rs;
  inc_t csc = c.cs;
  
  sgemm_Ukernel_ref ( ap, bp, cp, m, n, k, rsa, csa, rsb, csb, rsc, csc);
}// End of function

void sgemm32(obj_t a, obj_t b, obj_t c)
{
  dim_t m  = bli_obj_length(c);
  dim_t k  = bli_obj_width_after_trans(a);
  dim_t n  = bli_obj_width(c);

  dim_t lda = bli_obj_col_stride(a);
  dim_t ldb = bli_obj_col_stride(b);
  dim_t ldc = bli_obj_col_stride(c);

  float* ap = bli_obj_buffer(a);
  float* bp = bli_obj_buffer(b);
  float* cp = bli_obj_buffer(c);

  inc_t rsa = a.rs;
  inc_t csa = a.cs;
 
  inc_t rsb = b.rs;
  inc_t csb = b.cs;

  inc_t rsc = c.rs;
  inc_t csc = c.cs;
  
  sgemm32_ukernel ( ap, bp, cp);
}// End of function

void sgemm128(obj_t a, obj_t b, obj_t c)
{
  dim_t m  = bli_obj_length(c);
  dim_t k  = bli_obj_width_after_trans(a);
  dim_t n  = bli_obj_width(c);

  dim_t lda = bli_obj_col_stride(a);
  dim_t ldb = bli_obj_col_stride(b);
  dim_t ldc = bli_obj_col_stride(c);

  float* ap = bli_obj_buffer(a);
  float* bp = bli_obj_buffer(b);
  float* cp = bli_obj_buffer(c);

  inc_t rsa = a.rs;
  inc_t csa = a.cs;
 
  inc_t rsb = b.rs;
  inc_t csb = b.cs;

  inc_t rsc = c.rs;
  inc_t csc = c.cs;
  
  sgemm128_ukernel ( ap, bp, cp);
}// End of function





void sgemm_Ukernel_ref (
                    float *a, 
                    float *b, 
                    float* c,
                    inc_t m,
                    inc_t n,
                    inc_t k, 
                    inc_t rs_a, inc_t cs_a,
                    inc_t rs_b, inc_t cs_b,
                    inc_t rs_c, 
                    inc_t cs_c
		   )
{
  for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
	{
	  for(int p = 0; p < k; p++)
	    {
	      c[i*rs_c + j*cs_c] += a[i*rs_a + p* cs_a] * b[p * rs_b + j * cs_b];
	    }
	}
    }
}
