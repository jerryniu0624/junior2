/****************************************************************************
 *
 * cuda-matmul.cu - Dense matrix-matrix multiplication with CUDA
 *
 * Last modified in 2018 by Moreno Marzolla <moreno.marzolla(at)unibo.it>
 *
 * To the extent possible under law, the author(s) have dedicated all 
 * copyright and related and neighboring rights to this software to the 
 * public domain worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication
 * along with this software. If not, see 
 * <http://creativecommons.org/publicdomain/zero/1.0/>. 
 *
 * ---------------------------------------------------------------------------
 *
 * Dense matrix-matrix multiplication kernel with CUDA. Two versions
 * of the kernel are provided: one that does not use shared memory,
 * and one that does. A third version uses shared memory and does not
 * require that the matrix size is multiple of BLKDIM.
 *
 * Compile with:
 * nvcc cuda-matmul.cu -o cuda-matmul -lm
 *
 * Run with:
 * ./cuda-matmul [N]
 *
 ****************************************************************************/

#include <stdio.h>
#include <math.h>       /* for fabsf()  */
#include <cuda_runtime.h>

#define BLKDIM 32



/* Compute r = p * q, for square nxn matrices p, q, r; this version
   does not use shared memory. This kernel does not require that n is
   a multiple of BLKDIM */
__global__ void matmul( const float *p, const float *q, float *r, int n )
{
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    float val = 0.0;
    if ( i < n && j < n ) {
        for (k=0; k<n; k++) {
            val += p[i*n + k] * q[k*n + j];
        }
        r[i*n + j] = val;
    }
}

/* Compute r = p * q, for square n x n matrices p, q, r; this version
   uses shared memory. This kernel requires that n is a multiple of
   BLKDIM */
__global__ void matmulb( const float *p, const float *q, float *r, int n )
{

}

__device__ int cuda_min(int a, int b)
{
    return (a < b ? a : b);
}

/* Same as above, but does not require that n is a multiple of
   BLKDIM. To do so, it fills shared buffers so that values outside
   the matrices are treated as zeros. */
__global__ void matmulb_generic( const float *p, const float *q, float *r, int n )
{

}


/* Initialize square matrix q */
void mat_init( float *q, int n )
{
    int i;
    for (i=0; i<n*n; i++) {
        q[i] = 1.0;
    }
}

int check_result( const float *r, int n ) 
{
    /* Check result */
    int i, j;
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            if (fabsf(r[i*n+j] - n) > 1e-5) {
                printf("Check failed: r[%d][%d] = %f, expected %f\n", i, j, r[i*n+j], (float)n);
                return 0;
            }
        }
    }
    printf("Check OK\n");
    return 1;
}

int main( int argc, char* argv[] ) 
{

	cudaEvent_t start, stop; float elapsedTime;
    float *p, *q, *r;	          /* host copies of p, q, r */ 
    int N = 1024;

    dim3 block(BLKDIM, BLKDIM);
    dim3 grid((N+BLKDIM-1)/BLKDIM, (N+BLKDIM-1)/BLKDIM);
    const size_t size = N*N*sizeof(float);

    /* Allocate space for device copies of p, q, r */
	cudaMallocManaged((void **)&p, size);
	cudaMallocManaged((void **)&q, size);
	cudaMallocManaged((void **)&r, size);

    mat_init(p, N);
    mat_init(q, N);
   



    printf("Matrix-Matrix multiplication (%dx%d)\n", N, N);

    /**
     ** Matrix-matrix multiply WITHOUT shared memory
     **/
    printf("No shared memory:\t");
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start, 0);
	matmul <<<grid, block >>>(p, q, r, N);	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU Elapsed time:%.6f ms.\n", elapsedTime);
    /* Copy result back to host and check correctness */
    check_result(r, N);

    /* zero out r and d_r, to ensure that we don't read old results */
    cudaMemset(r, 0, size);


    /**
     ** Matrix-matrix multiply WITH shared memory
     **/
    printf("Shared memory:\t\t");
	cudaEventCreate(&start);	cudaEventCreate(&stop);	cudaEventRecord(start, 0);
	matmulb_generic <<<grid, block >>>(p, q, r, N);    cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU Elapsed time:%.6f ms.\n", elapsedTime);   
    /* Copy result back to host and check correctness */
    check_result(r, N);

    /* Cleanup */
    cudaFree(p); cudaFree(q); cudaFree(r);
    return 0;
}
