/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Mark Gates
       
       @precisions normal z -> s d c

       auto-converted from zlaset_band.cu

*/
#include "kernels_header.h"
#include "zlaset_band.h"

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 super-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil((m+k-1)/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread k-1 does the main diagonal, thread k-2 the first super-diagonal, etc.

      block 0           block 1
      0                           => skip above matrix
      1 0                         => skip above matrix
      2 1 0                       => skip above matrix
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                 | 3 2 1 0 ]
    [                 |   3 2 1 ]
                      |     3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=10, n=12, k=4, nb=8. Each column is done in parallel.
*/
__kernel
void zlaset_band_upper(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    __global magmaDoubleComplex *A, unsigned long A_offset, magma_int_t lda)
{
    A += A_offset;

    int k   = get_local_size(0);
    int ibx = get_group_id(0) * NB;
    int ind = ibx + get_local_id(0) - k + 1;
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if ( (int)(get_local_id(0)) == k-1)
        value = diag;

    #pragma unroll
    for (int j=0; j < NB; j++) {
        if (ibx + j < n && ind + j >= 0 && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}

/* ////////////////////////////////////////////////////////////////////////////
 -- GPU kernel for setting the k-1 sub-diagonals to OFFDIAG
    and the main diagonal to DIAG.
    Divides matrix into min( ceil(m/nb), ceil(n/nb) ) block-columns,
    with k threads in each block.
    Each thread iterates across one diagonal.
    Thread 0 does the main diagonal, thread 1 the first sub-diagonal, etc.
    
      block 0           block 1
    [ 0               |         ]
    [ 1 0             |         ]
    [ 2 1 0           |         ]
    [ 3 2 1 0         |         ]
    [   3 2 1 0       |         ]
    [     3 2 1 0     |         ]
    [       3 2 1 0   |         ]
    [         3 2 1 0 |         ]
    [           3 2 1 | 0       ]
    [             3 2 | 1 0     ]
    [               3 | 2 1 0   ]
    [                   3 2 1 0 ]
    [                     3 2 1 ]
                            3 2   => skip below matrix
                              3   => skip below matrix
    
    Thread assignment for m=13, n=12, k=4, nb=8. Each column is done in parallel.
*/

__kernel
void zlaset_band_lower(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    __global magmaDoubleComplex *A, unsigned long A_offset, magma_int_t lda)
{
    A += A_offset;

    //int k   = get_local_size(0);
    int ibx = get_group_id(0) * NB;
    int ind = ibx + get_local_id(0);
    
    A += ind + ibx*lda;
    
    magmaDoubleComplex value = offdiag;
    if (get_local_id(0) == 0)
        value = diag;

    #pragma unroll
    for (int j=0; j < NB; j++) {
        if (ibx + j < n && ind + j < m) {
            A[j*(lda+1)] = value;
        }
    }
}
