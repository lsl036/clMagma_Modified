/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

/**
    Purpose
    -------
    ZUNGQR generates an M-by-N COMPLEX_16 matrix Q with orthonormal columns,
    which is defined as the first N columns of a product of K elementary
    reflectors of order M

          Q  =  H(1) H(2) . . . H(k)

    as returned by ZGEQRF.

    This version recomputes the T matrices on the CPU and sends them to the GPU.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix Q. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix Q. M >= N >= 0.

    @param[in]
    k       INTEGER
            The number of elementary reflectors whose product defines the
            matrix Q. N >= K >= 0.

    @param[in,out]
    A       COMPLEX_16 array A, dimension (LDDA,N).
            On entry, the i-th column must contain the vector
            which defines the elementary reflector H(i), for
            i = 1,2,...,k, as returned by ZGEQRF_GPU in the
            first k columns of its array argument A.
            On exit, the M-by-N matrix Q.

    @param[in]
    lda     INTEGER
            The first dimension of the array A. LDA >= max(1,M).

    @param[in]
    tau     COMPLEX_16 array, dimension (K)
            TAU(i) must contain the scalar factor of the elementary
            reflector H(i), as returned by ZGEQRF_GPU.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument has an illegal value

    @ingroup magma_zgeqrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zungqr2(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *tau,
    magma_queue_t queue,
    magma_int_t *info)
{
#define  A(i,j) ( A + (i) + (j)*lda )
#define dA(i,j) dA, ((i) + (j)*ldda)  // pointer and offset
#define dV(i,j) dA, ((i) + (j)*ldda + ldda*n)
#define dW(i,j) dA, ((i) + (j)*ldda + ldda*n + ldda*nb)
#define dT(i,j) dA, ((i) + (j)*ldda + ldda*n + ldda*nb + lddwork*nb)

    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magma_int_t nb = magma_get_zgeqrf_nb(min(m, n));

    magma_int_t  m_kk, n_kk, k_kk, mi;
    magma_int_t lwork, ldda;
    magma_int_t i, ib, ki, kk;  //, iinfo;
    magma_int_t lddwork;
    magmaDoubleComplex_ptr dA;  //, dV, dW, dT;
    magmaDoubleComplex *T;
    magmaDoubleComplex *work;

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

    if (n <= 0) {
        return *info;
    }

    // first kk columns are handled by blocked method.
    // ki is start of 2nd-to-last block
    if ((nb > 1) && (nb < k)) {
        ki = (k - nb - 1) / nb * nb;
        kk = min(k, ki + nb);
    } else {
        ki = 0;
        kk = 0;
    }

    // Allocate GPU work space
    // ldda*n     for matrix dA
    // ldda*nb    for dV
    // lddwork*nb for dW larfb workspace
    ldda    = magma_roundup( m, 32 );
    lddwork = magma_roundup( n, 32 );
    if (MAGMA_SUCCESS != magma_zmalloc( &dA, ldda*n + ldda*nb + lddwork*nb + nb*nb)) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    // Allocate CPU work space
    lwork = (n+m+nb) * nb;
    magma_zmalloc_cpu( &work, lwork );

    T = work;

    if (work == NULL) {
        magma_free( dA );
        magma_free_cpu( work );
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    magmaDoubleComplex *V = work + (n+nb)*nb;

    // Use unblocked code for the last or only block.
    if (kk < n) {
        m_kk = m - kk;
        n_kk = n - kk;
        k_kk = k - kk;
        /*
            lapackf77_zungqr( &m_kk, &n_kk, &k_kk,
                              A(kk, kk), &lda,
                              &tau[kk], work, &lwork, &iinfo );
        */
        lapackf77_zlacpy( MagmaUpperLowerStr, &m_kk, &k_kk, A(kk,kk), &lda, V, &m_kk);
        lapackf77_zlaset( MagmaUpperLowerStr, &m_kk, &n_kk, &c_zero, &c_one, A(kk, kk), &lda );

        lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &k_kk,
                          V, &m_kk, &tau[kk], work, &k_kk);
        lapackf77_zlarfb( MagmaLeftStr, MagmaNoTransStr, MagmaForwardStr, MagmaColumnwiseStr,
                          &m_kk, &n_kk, &k_kk,
                          V, &m_kk, work, &k_kk, A(kk, kk), &lda, work+k_kk*k_kk, &n_kk );
        
        if (kk > 0) {
            magma_zsetmatrix( m_kk, n_kk,
                              A(kk, kk),  lda,
                              dA(kk, kk), ldda, queue );
        
            // Set A(1:kk,kk+1:n) to zero.
            magmablas_zlaset( MagmaFull, kk, n - kk, c_zero, c_zero, dA(0, kk), ldda, queue );
        }
    }

    if (kk > 0) {
        // Use blocked code
        // queue: set Aii (V) --> laset --> laset --> larfb --> [next]
        // CPU has no computation
        for (i = ki; i >= 0; i -= nb) {
            ib = min(nb, k - i);

            // Send current panel to the GPU
            mi = m - i;
            lapackf77_zlaset( "Upper", &ib, &ib, &c_zero, &c_one, A(i, i), &lda );
            magma_zsetmatrix_async( mi, ib,
                                    A(i, i), lda,
                                    dV(0,0), ldda, queue, NULL );
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &mi, &ib,
                              A(i,i), &lda, &tau[i], T, &nb);
            magma_zsetmatrix_async( ib, ib,
                                    T, nb,
                                    dT(0,0), nb, queue, NULL );

            // set panel to identity
            magmablas_zlaset( MagmaFull, i,  ib, c_zero, c_zero, dA(0, i), ldda, queue );
            magmablas_zlaset( MagmaFull, mi, ib, c_zero, c_one,  dA(i, i), ldda, queue );
            
            magma_queue_sync( queue );
            if (i < n) {
                // Apply H to A(i:m,i:n) from the left
                magma_zlarfb_gpu( MagmaLeft, MagmaNoTrans, MagmaForward, MagmaColumnwise,
                                  mi, n-i, ib,
                                  dV(0,0),  ldda, dT(0,0), nb,
                                  dA(i, i), ldda, dW(0,0), lddwork, queue );
            }
        }
    
        // copy result back to CPU
        magma_zgetmatrix( m, n,
                          dA(0, 0), ldda, A(0, 0), lda, queue );
    }

    magma_free( dA );
    magma_free_cpu( work );

    return *info;
} /* magma_zungqr */
