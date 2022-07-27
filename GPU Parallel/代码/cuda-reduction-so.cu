#include <stdio.h>
#include "cuda_runtime.h"
#include<time.h>

#define BLKDIM 1024
#define N_OF_BLOCKS   1024
/* N must be an integer multiple of BLKDIM */
#define N ((N_OF_BLOCKS)*(BLKDIM))

#define NUM_REPS  100

int reduceCPU(int *data, int size)
{
	int sum = data[0];
	for (int i = 1; i < size; i++)
		sum += data[i];
	return sum;
}


__global__ void reduce1(int *g_idata, int *g_odata, int n) {
	__shared__ int sdata[BLKDIM];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
	}
			__syncthreads();
	}
	//write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce2(int *g_idata, int *g_odata, int n) {
	__shared__ int sdata[BLKDIM];

	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// __global__ void reduce0( int *a, int* sums, int n )
// {
//    __shared__ int temp[BLKDIM];
//    temp[threadIdx.x] = a[threadIdx.x + blockIdx.x * blockDim.x];
//    __syncthreads(); 
//    if ( 0 == threadIdx.x) {
//        int i, my_sum = 0;
//        for (i=0; i<blockDim.x; i++)
// 	    {my_sum += temp[i];}
//        sums[blockIdx.x] = my_sum;
//    }
// }

//__global__ void reduce1(int *a, int* sums, int n)
//{
//	__shared__ int temp[BLKDIM];
//	int lindex = threadIdx.x;
//	int bindex = blockIdx.x;
//	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
//	int bsize = blockDim.x / 2;
//
//	temp[lindex] = a[gindex];  __syncthreads();
//
//	/* All threads within the block cooperate to compute the local sum */
//	while (bsize > 0) {
//		if (lindex < bsize) {temp[lindex] += temp[lindex + bsize];}
//		bsize = bsize / 2;	
//		__syncthreads();
//	}
//	if (0 == lindex) {sums[bindex] = temp[0];}
//}

__global__ void reduce3(int *g_idata, int *g_odata, unsigned int n)
{
	__shared__ int sdata[BLKDIM];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce4(int *g_idata, int *g_odata, unsigned int n)
{
	__shared__ int sdata[BLKDIM];

	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = 0;
	if ( i < n ) 
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void reduce6( int* g_idata,  int* g_odata,  int n) {
	__shared__ int sdata[BLKDIM];
	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = 0;
	if (i < n)
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	if (tid < 512) 
		sdata[tid] += sdata[tid + 512]; __syncthreads();
	if(tid < 256)
		sdata[tid] += sdata[tid + 256]; __syncthreads();
	if (tid < 128)
		sdata[tid] += sdata[tid + 128]; __syncthreads();
	if (tid < 64)
		sdata[tid] += sdata[tid + 64];  __syncthreads();

	//	write result for this block to global mem
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32]; __syncthreads();
		sdata[tid] += sdata[tid + 16]; __syncthreads();
		sdata[tid] += sdata[tid + 8]; __syncthreads();
		sdata[tid] += sdata[tid + 4]; __syncthreads();
		sdata[tid] += sdata[tid + 2]; __syncthreads();
		sdata[tid] += sdata[tid + 1]; __syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce5(int *g_idata, int *g_odata, int n)
{

	__shared__ int sdata[BLKDIM];
	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	sdata[tid] = 0;
	if ( i < n ) 
		sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	//	write result for this block to global mem
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32]; __syncthreads();
		sdata[tid] += sdata[tid + 16]; __syncthreads();
		sdata[tid] += sdata[tid + 8]; __syncthreads();
		sdata[tid] += sdata[tid + 4]; __syncthreads();
		sdata[tid] += sdata[tid + 2]; __syncthreads();
		sdata[tid] += sdata[tid + 1]; __syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
	    g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce51(int *g_idata, int *g_odata, int n)
{

	__shared__ int sdata[BLKDIM];
	// load shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x) + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s>32; s >>= 1)
	{
		if (tid < s)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	//	write result for this block to global mem
	if (tid < 32)
	{
		sdata[tid] += sdata[tid + 32]; __syncthreads();
		sdata[tid] += sdata[tid + 16]; __syncthreads();
		sdata[tid] += sdata[tid + 8]; __syncthreads();
		sdata[tid] += sdata[tid + 4]; __syncthreads();
		sdata[tid] += sdata[tid + 2]; __syncthreads();
		sdata[tid] += sdata[tid + 1]; __syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}



int main( void ) 
{

	// step0: CUDA events
	cudaEvent_t start, stop; cudaEventCreate(&start);	cudaEventCreate(&stop); float outerTime;

	// step0: malloc space
	int *a, *b; int i, s=0;
    cudaMallocManaged(&a,N*sizeof(int));    cudaMallocManaged(&b,N_OF_BLOCKS *sizeof(int));
    
	// step1: initialization
	for (i = 0; i < N; i++) { a[i] = 2; }
	
	cudaMemset(&b, 0, N_OF_BLOCKS);

	clock_t start_t = clock();
	for (int i = 0; i < NUM_REPS; i++)
		s = reduceCPU(a,N);
	clock_t finish_t = clock();
	outerTime = (float)(finish_t - start_t) / CLOCKS_PER_SEC * 1000;
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction with CPU:  Elapsed time:%.6f ms.\n", outerTime); }



	// warm up 
	reduce1 <<<N_OF_BLOCKS, BLKDIM >>>(a, b, N);

	// step2: running  sum0
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce1 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N); 
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	for (i = 0; i < N_OF_BLOCKS; i++) { s += b[i]; }
    if ( s != 2*N ) {printf("Check FAILED: Expected %d, got %d\n", 2*N, s);    } 
	else {  printf("Reduction1: GPU Elapsed time:%.6f ms.\n", outerTime); }

	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce2 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N); 
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	for (i = 0; i < N_OF_BLOCKS; i++) { s += b[i]; }
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction2: GPU Elapsed time:%.6f ms.\n", outerTime); }
 

	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce3 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	for (i = 0; i < N_OF_BLOCKS; i++) { s += b[i]; }
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction3: GPU Elapsed time:%.6f ms.\n", outerTime); }


	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce4 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N);
		cudaDeviceSynchronize();		
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	for (i = 0; i < N_OF_BLOCKS; i++) { s += b[i]; }
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction4: GPU Elapsed time:%.6f ms.\n", outerTime); }


	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce5<<<N_OF_BLOCKS, BLKDIM >> > (a, b, N);
		cudaDeviceSynchronize();	
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	
	for (i = 0; i < N_OF_BLOCKS; i++) { 
		s += b[i];
	}
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction5: GPU Elapsed time:%.6f ms.\n", outerTime); }



	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce6 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	
	for (i = 0; i < N_OF_BLOCKS; i++) {
		s += b[i];
	}
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction5-1: GPU Elapsed time:%.6f ms.\n", outerTime); }

	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++) {
		reduce6 << <N_OF_BLOCKS, BLKDIM >> > (a, b, N);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	cudaEventElapsedTime(&outerTime, start, stop);
	s = 0; 	for (i = 0; i < N_OF_BLOCKS; i++) { s += b[i]; }
	if (s != 2 * N) { printf("Check FAILED: Expected %d, got %d\n", 2 * N, s); }
	else { printf("Reduction6: GPU Elapsed time:%.6f ms.\n", outerTime); }

	// step4: free
	cudaFree(a); 	cudaFree(b);


    return 0;
}
