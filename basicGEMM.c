/* Kiran Varaganti - GEMM implementation for small matrices */
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
  dim_t k  = bli_obj_width(a);
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

  printf("(m, n, k) = (%u, %u, %u)\n", m, n, k);
  printf ("a : (rs, cs, lda) = (%u, %u, %u)\n", rsa, csa, lda);
  printf ("b : (rs, cs, lda) = (%u, %u, %u)\n", rsb, csb, ldb);
  printf ("c : (rs, cs, lda) = (%u, %u, %u)\n", rsc, csc, ldc);
  
  sgemm_Ukernel_ref ( ap, bp, cp, m, n, k, rsa, csa, rsb, csb, rsc, csc);
}// End of function

void sgemm32(obj_t a, obj_t b, obj_t c)
{
  float* ap = bli_obj_buffer(a);
  float* bp = bli_obj_buffer(b);
  float* cp = bli_obj_buffer(c);
    
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

void sgemm4(obj_t a, obj_t b, obj_t c)
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
  
  sgemm4_ukernel ( ap, bp, cp);
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

 if((m == 4) && (rs_c == 1) )
    {
      // this is done because in BLIS, for when blis matrix objects are created with 4 x 4 dimesions
      // the column_stride (column major) is getting doubled. Hence to handle this case we are doing this small fix
      for (int i = 0; i < m; i++)
	{
	  for (int j = 0; j < n; j++)
	    {
	      float sum = 0.0;
	      for(int p = 0; p < k; p++)
		{
		  sum += a[i*rs_a + p * k] * b[p*rs_b + j*n];
		}
	      c[i*rs_c + j*n] += sum;
	    }
	}
      return;
    }


  for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
	{
	  float sum = 0.0;
	  for(int p = 0; p < k; p++)
	    {
	      sum += a[i*rs_a + p * cs_a] * b[p*rs_b + j*cs_b];	      
	    }
	  c[i*rs_c + j*cs_c] += sum;
	}
    }
}// end of function
