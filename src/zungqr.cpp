/*
    -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zungqr(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *a, magma_int_t lda,
    magmaDoubleComplex *tau, magmaDoubleComplex_ptr dT, size_t dT_offset,
    magma_int_t nb,
    magma_queue_t queue,
    magma_int_t *info )
{
/*  -- clMAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZUNGQR generates an M-by-N COMPLEX_16 matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix Q. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    K       (input) INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    A       (input/output) COMPLEX_16 array A, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by ZGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    LDA     (input) INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    TAU     (input) COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF_GPU.

    DT      (input) COMPLEX_16 array on the GPU device.
            DT contains the T matrices used in blocking the elementary
            reflectors H(i), e.g., this can be the 6th argument of
            magma_zgeqrf_gpu.

    NB      (input) INTEGER
            This is the block size used in ZGEQRF_GPU, and correspondingly
            the size of the T matrices, used in the factorization, and
            stored in DT.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument has an illegal value
    =====================================================================    */

    #define  a_ref(i,j)     ( a + (j)*lda  + (i))
    #define da_ref(i,j)     da, (da_offset + (j)*ldda + (i))
    #define t_ref(a_1)      dT, (dT_offset + (a_1)*nb)

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    magma_int_t  i__1, i__2, i__3;
    magma_int_t lwork, ldda;
    magma_int_t i, ib, ki, kk, iinfo;
    magma_int_t lddwork = min(m, n);
    magmaDoubleComplex *work;
    magmaDoubleComplex_ptr da, dwork;
    size_t da_offset, dwork_offset;
    magma_event_t event = NULL;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if ((n < 0) || (n > m)) {
        *info = -2;
    } else if ((k < 0) || (k > n)) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if (n <= 0)
      return *info;

    /* Allocate GPU work space */
    ldda = magma_roundup( m, 32 );
    lddwork = magma_roundup( lddwork, 32 );
    if (MAGMA_SUCCESS != magma_zmalloc( &da, ((n)*ldda + nb*lddwork ) )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    da_offset = 0;
    dwork = da;
    dwork_offset = da_offset + (n)*ldda;

    /* Allocate CPU work space */
    lwork = n * nb;
    magma_zmalloc_cpu( &work, lwork );
    if( work == NULL ) {
        magma_free( da );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    if ( (nb > 1) && (nb < k) )
      {
        /*  Use blocked code after the last block.
            The first kk columns are handled by the block method. */
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);

        /* Set A(1:kk,kk+1:n) to zero. */
        magmablas_zlaset(MagmaFull, kk, n-kk, c_zero, c_zero, da_ref(0,kk), ldda, queue);
      }
    else
      kk = 0;

    /* Use unblocked code for the last or only block. */
    if (kk < n)
      {
        i__1 = m - kk;
        i__2 = n - kk;
        i__3 = k - kk;
        lapackf77_zungqr(&i__1, &i__2, &i__3,
                         a_ref(kk, kk), &lda,
                         &tau[kk], work, &lwork, &iinfo);
        
        magma_zsetmatrix(i__1, i__2, a_ref(kk, kk), lda, da_ref(kk, kk), ldda, queue);
      }

    if (kk > 0)
      {
        /* Use blocked code */
        for (i = ki; i >= 0; i-=nb)
          {
            ib = min(nb, k - i);

            /* Send the current panel to the GPU */
            i__2 = m - i;
            magma_zpanel_to_q( MagmaUpper, ib, a_ref(i,i), lda, work );
            magma_zsetmatrix(i__2, ib, a_ref(i, i), lda, da_ref(i, i), ldda, queue);
                             
            if (i + ib < n)
              {
                /* Apply H to A(i:m,i+ib:n) from the left */
                i__3 = n - i - ib;
                magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  i__2, i__3, ib,
                                  da_ref(i, i   ), ldda, t_ref(i),      nb,
                                  da_ref(i, i+ib), ldda,    dwork, dwork_offset, lddwork, queue);
              }

            /* Apply H to rows i:m of current block on the CPU */
            lapackf77_zungqr(&i__2, &ib, &ib,
                             a_ref(i, i), &lda,
                             &tau[i], work, &lwork, &iinfo);
            magma_zsetmatrix_async( i__2, ib,
                                    a_ref(i,i), lda,
                                    da_ref(i,i), ldda, queue, &event );

            /* Set rows 1:i-1 of current block to zero */
            i__2 = i + ib;
            magmablas_zlaset(MagmaFull, i, i__2 - i, c_zero, c_zero, da_ref(0,i), ldda, queue);
          }
      }
    
    magma_zgetmatrix(m, n, da_ref(0, 0), ldda, a_ref(0, 0), lda, queue);
    
    //cudaStreamDestroy(stream);
    magma_free( da );
    magma_free_cpu(work);

    return *info;
} /* magma_zungqr */

#undef da_ref
#undef a_ref
#undef t_ref
