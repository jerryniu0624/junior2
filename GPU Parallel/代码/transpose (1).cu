#include <stdio.h>
#include<time.h>
#include<math.h>
#include <cuda_runtime.h>


#define TILE_DIM    32 // block_size
#define BLOCK_ROWS  32 // thread_size

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;


#define NUM_REPS  10


// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------
__global__ void copy(float *odata, float *idata, int width, int height){
	int idx = TILE_DIM * blockIdx.x + threadIdx.x;
	int idy = TILE_DIM * blockIdx.y + threadIdx.y;
	if(idx < width && idy < height){
		odata[idx * width + idy]  = idata[idx * width + idy];
	}
}

__global__ void copySharedMem(float *odata, float *idata, int width, int height)
{
	int idx = TILE_DIM * blockIdx.x + threadIdx.x;
	int idy = TILE_DIM * blockIdx.y + threadIdx.y;
	__shared__ int temp[TILE_DIM][TILE_DIM];
	if(idx < width && idy < height){
		temp[threadIdx.x][threadIdx.y] = idata[idx * width + idy];
	}
	__syncthreads();
	if(idx < width && idy < height){
		odata[idx * width + idy] = temp[threadIdx.x][threadIdx.y];
	}
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void transposeNaive(float *odata, float *idata, int width, int height){
	int idx = TILE_DIM * blockIdx.x + threadIdx.x;
	int idy = TILE_DIM * blockIdx.y + threadIdx.y;
	if(idx < width && idy < height){
		odata[idx * width + idy]  = idata[idy * width + idx];
	}
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
	int idx = TILE_DIM * blockIdx.x + threadIdx.x;
	int idy = TILE_DIM * blockIdx.y + threadIdx.y;
	int idxs = threadIdx.x;
	int idys = threadIdx.y;
	__shared__ int temp[TILE_DIM][TILE_DIM];
	if(idx < width && idy < height){
		temp[idxs][idys] = idata[idy * width + idx];
	}

	__syncthreads();

	if(idx < width && idy < height){
		odata[idx * width + idy] = temp[idxs][idys];
	}

}

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
	int idx = TILE_DIM * blockIdx.x + threadIdx.x;
	int idy = TILE_DIM * blockIdx.y + threadIdx.y;
	int idxs = threadIdx.x;
	int idys = threadIdx.y;
	__shared__ int temp[TILE_DIM][TILE_DIM];
	if(idx < width && idy < height){
		temp[idxs][idys] = idata[idx * width + idy];
	}

	__syncthreads();

	if(idx < width && idy < height){
		odata[idy * width + idx] = temp[idxs][idys];
	}
}

void computeTransposeGold(float *gold, float *idata,const  int size_x, const  int size_y){
	for (int y = 0; y < size_y; ++y)
		for (int x = 0; x < size_x; ++x)
			gold[(x * size_y) + y] = idata[(y * size_x) + x];
}


bool compareData(float* a, float* b, int n) {
	for (int i = 0; i < n; i++)
		if (abs(a[i] - b[i]) > 0.0001)
			return false;
	return true;
}


int main(){	
	int size_x = MATRIX_SIZE_X;
	int size_y = MATRIX_SIZE_Y;
	// CUDA events
	cudaEvent_t start, stop; cudaEventCreate(&start);	cudaEventCreate(&stop); float outerTime;
	// size of memory required to store the matrix
	const  int mem_size = sizeof(float) * MATRIX_SIZE_X*MATRIX_SIZE_Y;
	float *idata, *odata, *transposeGold, *gold;
	cudaMallocManaged(&idata, mem_size);
	cudaMallocManaged(&odata, mem_size);
	cudaMallocManaged(&transposeGold, mem_size);
	cudaMallocManaged(&gold, mem_size);



	// step1: initalize host data
	for (int i = 0; i < (MATRIX_SIZE_X*MATRIX_SIZE_Y); ++i)
		idata[i] = (float)i;

	// step2: Compute reference transpose solution
	clock_t start_t = clock();
	for (int i = 0; i < NUM_REPS; i++)
		computeTransposeGold(transposeGold, idata, MATRIX_SIZE_X, MATRIX_SIZE_Y);
	clock_t finish_t = clock();
	float total_t = (float)(finish_t - start_t) / CLOCKS_PER_SEC*1000; 
	printf("Transposed: CPU Elapsed time:%.6f ms.\n", total_t);


	// execution configuration parameters
	dim3 grid(size_x / TILE_DIM+1, size_y / TILE_DIM+1), threads(TILE_DIM, BLOCK_ROWS);
	//step3:  warmup to avoid timing startup
	copy<<<grid,threads>>>(odata,idata,size_x,size_y);
	// take measurements for loop over kernel launches
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++)
		copy <<<grid, threads >>>(odata, idata, size_x, size_y);
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&outerTime, start, stop);
	printf("Simple Copy: GPU Elapsed time:%.6f ms.\n", outerTime);

	//Step4: copySharedMem
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++)
		copySharedMem << <grid, threads >> >(odata, idata, size_x, size_y);
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);	
	cudaEventElapsedTime(&outerTime, start, stop);
	printf("SharedMem Copy: GPU Elapsed time:%.6f ms.\n", outerTime);

	//Step5: transposeNaive
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++)
		transposeNaive <<<grid, threads >>>(odata, idata, size_x, size_y);
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTime, start, stop);
	if (compareData(transposeGold, odata, size_x*size_y))
		printf("TransposeNaive (True): GPU Elapsed time:%.6f ms.\n", outerTime);

	//Step6: transposeCoalesced
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++)
		transposeCoalesced <<<grid, threads >> >(odata, idata, size_x, size_y);
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTime, start, stop);
	if (compareData(transposeGold, odata, size_x*size_y))
		printf("TransposeCoalesced(True): GPU Elapsed time:%.6f ms.\n", outerTime);


	//Step7: transposeNoBankConflicts
	cudaEventRecord(start, 0);
	for (int i = 0; i < NUM_REPS; i++)
		transposeNoBankConflicts << <grid, threads >> >(odata, idata, size_x, size_y);
	cudaEventRecord(stop, 0);	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&outerTime, start, stop);
	if (compareData(transposeGold, odata, size_x*size_y))
		printf("TransposeNoBankConflicts(True): GPU Elapsed time:%.6f ms.\n", outerTime);
	cudaFree(idata); cudaFree(odata); cudaFree(transposeGold); cudaFree(gold);
	return 1;
}