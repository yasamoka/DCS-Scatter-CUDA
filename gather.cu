
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

#include <stdio.h>
#include <iostream>
#include <ctime>
#include <random>

#include "GpuTimer.h"

#include "tclap/CmdLine.h"

#define EPS 0.03
#define BLOCK_WIDTH 16
#define NUM_OF_POINTS_DCS2 8

#define LATTICE_DATA_TYPE float

__global__ void DCSKernel(LATTICE_DATA_TYPE *slice, const float *atomXs, const float *atomYs, const float *atomZs, const float *charges, const unsigned short int z, const unsigned int numOfAtoms, const unsigned short int latticeX, const unsigned short int latticeY, const LATTICE_DATA_TYPE latticeGridSpacing)
{
	const unsigned int sliceGridSize = latticeX * latticeY;
	const unsigned short int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned short int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < latticeX && y < latticeY) {
		float atomX, atomY, atomZ, charge;
		LATTICE_DATA_TYPE dx, dy, dz, distance;
		LATTICE_DATA_TYPE potential = 0;
		for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
			atomX = atomXs[atomIdx];
			atomY = atomYs[atomIdx];
			atomZ = atomZs[atomIdx];
			charge = charges[atomIdx];

			dx = atomX - x * latticeGridSpacing;
			dy = atomY - y * latticeGridSpacing;
			dz = atomZ - z * latticeGridSpacing;
			distance = sqrt(dx * dx + dy * dy + dz * dz);
			potential += charge / distance;
		}
		const unsigned int idx = latticeX * y + x;
		slice[idx] = potential;
	}
}

// thread coarsening and memory coalescing
__global__ void DCS2Kernel(LATTICE_DATA_TYPE *slice, const float *atomXs, const float *atomYs, const float *atomZs, const float *charges, const unsigned short int z, const unsigned int numOfAtoms, const unsigned short int latticeX, const unsigned short int latticeY, const LATTICE_DATA_TYPE latticeGridSpacing)
{
	const unsigned int sliceGridSize = latticeX * latticeY;
	const unsigned short int x = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned short int y = blockDim.y * blockIdx.y + threadIdx.y;
	if (x < latticeX && y < latticeY) {
		float atomX, atomY, atomZ, charge;
		LATTICE_DATA_TYPE dx, dy, dz, dy2dz2;
		LATTICE_DATA_TYPE potential[NUM_OF_POINTS_DCS2];
		unsigned short int pointIdx;
		for (pointIdx = 0; pointIdx < NUM_OF_POINTS_DCS2; pointIdx++) {
			potential[pointIdx] = 0;
		}
		unsigned short int x1;
		for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
			atomX = atomXs[atomIdx];
			atomY = atomYs[atomIdx];
			atomZ = atomZs[atomIdx];
			charge = charges[atomIdx];

			dy = atomY - y * latticeGridSpacing;
			dz = atomZ - z * latticeGridSpacing;
			dy2dz2 = dy * dy + dz * dz;

			x1 = x;
			for (pointIdx = 0; pointIdx < NUM_OF_POINTS_DCS2; pointIdx++) {
				dx = atomX - x1 * latticeGridSpacing;
				potential[pointIdx] += charge / sqrt(dx * dx + dy2dz2);
				x1 += blockDim.x;
			}
		}
		unsigned int sliceYOffset = latticeX * y;
		x1 = x;
		for (pointIdx = 0; pointIdx < NUM_OF_POINTS_DCS2; pointIdx++) {
			slice[sliceYOffset + x1] = potential[pointIdx];
			x1 += blockDim.x;
			if (x1 >= latticeX) {
				break;
			}
		}
	}
}

void CPU(LATTICE_DATA_TYPE *lattice, const float *atomXs, const float *atomYs, const float *atomZs, const float *charges, const unsigned short int z, const unsigned int numOfAtoms, const unsigned short int latticeX, const unsigned short int latticeY, const LATTICE_DATA_TYPE latticeGridSpacing)
{
	float atomX, atomY, atomZ, charge;

	const unsigned int latticeSliceGridSize = latticeX * latticeY;
	unsigned long int latticeZOffset;
	unsigned int latticeYOffset;
	unsigned long int latticeOffset;
	unsigned long int latticeIdx;

	LATTICE_DATA_TYPE dx, dy, dz, distance;
	LATTICE_DATA_TYPE potential;

	latticeZOffset = latticeSliceGridSize * z;

	for (unsigned short int y = 0; y < latticeY; y++) {
		latticeYOffset = latticeX * y;
		latticeOffset = latticeZOffset + latticeYOffset;
		for (unsigned short int x = 0; x < latticeX; x++) {
			latticeIdx = latticeOffset + x;
			potential = 0;
			for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
				atomX = atomXs[atomIdx];
				atomY = atomYs[atomIdx];
				atomZ = atomZs[atomIdx];
				charge = charges[atomIdx];

				dx = atomX - x * latticeGridSpacing;
				dy = atomY - y * latticeGridSpacing;
				dz = atomZ - z * latticeGridSpacing;
				distance = sqrt(dx * dx + dy * dy + dz * dz);
				potential += charge / distance;
			}
			lattice[latticeIdx] = potential;
		}
	}
}

// using pre-computed dy2 + dz2
void CPU2(LATTICE_DATA_TYPE *lattice, const float *atomXs, const float *atomYs, const float *atomZs, const float *charges, const unsigned short int z, const unsigned int numOfAtoms, const unsigned short int latticeX, const unsigned short int latticeY, const LATTICE_DATA_TYPE latticeGridSpacing)
{
	float atomX, atomY, atomZ, charge;
	
	const unsigned int latticeSliceGridSize = latticeX * latticeY;
	unsigned long int latticeZOffset;
	unsigned int latticeYOffset;
	unsigned long int latticeOffset;
	unsigned long int latticeIdx;

	LATTICE_DATA_TYPE dx, dy, dz, distance;
	LATTICE_DATA_TYPE *dz2 = (LATTICE_DATA_TYPE*)malloc(numOfAtoms * sizeof(LATTICE_DATA_TYPE));
	LATTICE_DATA_TYPE *dy2dz2 = (LATTICE_DATA_TYPE*)malloc(numOfAtoms * sizeof(LATTICE_DATA_TYPE));
	LATTICE_DATA_TYPE potential;

	latticeZOffset = latticeSliceGridSize * z;
	for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
		dz = atomZs[atomIdx] - z * latticeGridSpacing;
		dz2[atomIdx] = dz * dz;
	}
	for (unsigned short int y = 0; y < latticeY; y++) {
		latticeYOffset = latticeX * y;
		latticeOffset = latticeZOffset + latticeYOffset;
		for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
			dy = atomYs[atomIdx] - y * latticeGridSpacing;
			dy2dz2[atomIdx] = dy * dy + dz2[atomIdx];
		}
		for (unsigned short int x = 0; x < latticeX; x++) {
			latticeIdx = latticeOffset + x;
			potential = 0;
			for (unsigned int atomIdx = 0; atomIdx < numOfAtoms; atomIdx++) {
				dx = atomXs[atomIdx] - x * latticeGridSpacing;
				distance = sqrt(dx * dx + dy2dz2[atomIdx]);
				potential += charges[atomIdx] / distance;
			}
			lattice[latticeIdx] = potential;
		}
	}
}

int main(int argc, char *argv[])
{
	double latticeW;
	double latticeH;
	double latticeD;
	double latticeGridSpacing;
	unsigned int numOfAtoms;
	float maxCharge;
	uint8_t numOfStreams;

	try {
		TCLAP::CmdLine cmd("Runs the Direct Couloumb Summation algorithm on the CPU & GPU (CUDA).", ' ', "1.0");
		TCLAP::ValueArg<double> latticeWArg("x", "width", "Lattice width", true, 1.0f, "double");
		TCLAP::ValueArg<double> latticeHArg("y", "height", "Lattice height", true, 1.0f, "double");
		TCLAP::ValueArg<double> latticeDArg("z", "depth", "Lattice depth", true, 1.0f, "double");
		TCLAP::ValueArg<double> latticeGridSpacingArg("g", "spacing", "Lattice grid spacing", true, 0.1f, "double");
		TCLAP::ValueArg<unsigned int> numOfAtomsArg("a", "atoms", "Number of atoms", true, 1, "int");
		TCLAP::ValueArg<double> maxChargeArg("c", "charge", "Maximum charge", true, 1.0f, "double");
		TCLAP::ValueArg<unsigned int> numOfStreamsArg("n", "streams", "Number of CUDA streams", false, 2, "int");

		cmd.add(numOfStreamsArg);
		cmd.add(maxChargeArg);
		cmd.add(numOfAtomsArg);
		cmd.add(latticeGridSpacingArg);
		cmd.add(latticeDArg);
		cmd.add(latticeHArg);
		cmd.add(latticeWArg);

		cmd.parse(argc, argv);

		latticeW = latticeWArg.getValue();
		latticeH = latticeHArg.getValue();
		latticeD = latticeDArg.getValue();
		latticeGridSpacing = latticeGridSpacingArg.getValue();
		numOfAtoms = numOfAtomsArg.getValue();
		maxCharge = maxChargeArg.getValue();
		numOfStreams = numOfStreamsArg.getValue();
	}
	catch (TCLAP::ArgException &e) {
		fprintf(stderr, "Error in argument(s): %s\n", e.what());
		return 1;
	}

	const unsigned short int latticeX = floor(latticeW / latticeGridSpacing) + 1;
	const unsigned short int latticeY = floor(latticeH / latticeGridSpacing) + 1;
	const unsigned short int latticeZ = floor(latticeD / latticeGridSpacing) + 1;
	const unsigned long int sliceGridSize = latticeX * latticeY;
	const unsigned long int latticeGridSize = sliceGridSize * latticeZ;
	
	float *h_AtomX, *h_AtomY, *h_AtomZ;
	float *h_Charge;
	float *d_AtomX, *d_AtomY, *d_AtomZ;
	float *d_Charge;

	LATTICE_DATA_TYPE *latticeCPU;
	LATTICE_DATA_TYPE *latticeCPU2;
	LATTICE_DATA_TYPE *h_LatticeDCS;
	LATTICE_DATA_TYPE **d_SliceDCS;
	LATTICE_DATA_TYPE *h_LatticeDCS2;
	LATTICE_DATA_TYPE **d_SliceDCS2;

	cudaError_t cudaStatus;

	std::default_random_engine generator;
	std::uniform_real_distribution<float> latticeWDistribution(0, latticeW - 1);
	std::uniform_real_distribution<float> latticeHDistribution(0, latticeH - 1);
	std::uniform_real_distribution<float> latticeDDistribution(0, latticeD - 1);
	std::uniform_real_distribution<float> chargeDistribution(0, maxCharge);

	uint8_t numOfRemainingLaunches;
	uint8_t streamIdx;
	cudaStream_t *stream;
	unsigned long h_LatticeDCSOffset;

	clock_t mallocClock;
	double mallocDuration;
	GpuTimer cudaMallocTimer;
	GpuTimer cudaMemcpyHostDeviceTimer;
	clock_t randomGenerationClock;
	double randomGenerationDuration;
	GpuTimer DCSKernelTimer;
	GpuTimer DCS2KernelTimer;
	clock_t CPUClock;
	double CPUDuration;
	clock_t CPU2Clock;
	double CPU2Duration;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	mallocClock = clock();

	stream = (cudaStream_t*)malloc(numOfStreams * sizeof(cudaStream_t));

	cudaHostAlloc((void**)&h_AtomX, numOfAtoms * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_AtomY, numOfAtoms * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_AtomZ, numOfAtoms * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_Charge, numOfAtoms * sizeof(float), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_LatticeDCS, latticeGridSize * sizeof(LATTICE_DATA_TYPE), cudaHostAllocDefault);
	cudaHostAlloc((void**)&h_LatticeDCS2, latticeGridSize * sizeof(LATTICE_DATA_TYPE), cudaHostAllocDefault);
	cudaHostAlloc((void**)&d_SliceDCS, numOfStreams * sizeof(LATTICE_DATA_TYPE*), cudaHostAllocDefault);

	latticeCPU = (LATTICE_DATA_TYPE*)malloc(latticeGridSize * sizeof(LATTICE_DATA_TYPE));

	latticeCPU2 = (LATTICE_DATA_TYPE*)malloc(latticeGridSize * sizeof(LATTICE_DATA_TYPE));

	mallocDuration = (clock() - mallocClock) / (double)CLOCKS_PER_SEC;
	printf("Memory allocation (host): %f ms\n", mallocDuration * 1000);

	cudaMallocTimer.Start();

	cudaStatus = cudaMalloc((void**)&d_AtomX, numOfAtoms * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (atomX) failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_AtomY, numOfAtoms * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (atomY) failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_AtomZ, numOfAtoms * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (atomZ) failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_Charge, numOfAtoms * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc (charge) failed!");
		goto Error;	
	}

	for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
		cudaStatus = cudaMalloc((void**)&d_SliceDCS[streamIdx], sliceGridSize * sizeof(LATTICE_DATA_TYPE));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc (DCS, slice[%i]) failed!", streamIdx);
			goto Error;
		}
	}

	cudaMallocTimer.Stop();
	printf("Memory allocation (device): %f ms\n", cudaMallocTimer.Elapsed());

	randomGenerationClock = clock();
	for (unsigned int i = 0; i < numOfAtoms; i++) {
		h_AtomX[i] = latticeWDistribution(generator);
		h_AtomY[i] = latticeHDistribution(generator);
		h_AtomZ[i] = latticeDDistribution(generator);
		h_Charge[i] = chargeDistribution(generator);
	}
	randomGenerationDuration = (clock() - randomGenerationClock) / (double)CLOCKS_PER_SEC;
	printf("Random generation (CPU): %f ms\n", randomGenerationDuration * 1000);

	cudaMemcpyHostDeviceTimer.Start();

	cudaStatus = cudaMemcpy(d_AtomX, h_AtomX, numOfAtoms * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (atomX, host -> device) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_AtomY, h_AtomY, numOfAtoms * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (atomY, host -> device) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_AtomZ, h_AtomZ, numOfAtoms * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (atomZ, host -> device) failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_Charge, h_Charge, numOfAtoms * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy (charge, host -> device) failed!");
		goto Error;
	}

	cudaMemcpyHostDeviceTimer.Stop();
	printf("Memory copy (host -> device): %f ms\n", cudaMemcpyHostDeviceTimer.Elapsed());

	for (uint8_t streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
		cudaStreamCreate(&stream[streamIdx]);
	}

	//DCS
	dim3 dimBlockDCS(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	dim3 dimGridDCS((latticeX - 1) / BLOCK_WIDTH + 1, (latticeY - 1) / BLOCK_WIDTH + 1, 1);

	DCSKernelTimer.Start();
	h_LatticeDCSOffset = 0;
	if (latticeZ > 1) {
		for (unsigned short int z = 0; z < latticeZ; z += numOfStreams) {
			for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
				DCSKernel << <dimGridDCS, dimBlockDCS, 0, stream[streamIdx] >> >(d_SliceDCS[streamIdx], d_AtomX, d_AtomY, d_AtomZ, d_Charge, z + streamIdx, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
			}

			for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
				h_LatticeDCSOffset = (z + streamIdx) * sliceGridSize;
				cudaMemcpyAsync(h_LatticeDCS + h_LatticeDCSOffset, d_SliceDCS[streamIdx], sliceGridSize * sizeof(LATTICE_DATA_TYPE), cudaMemcpyDeviceToHost, stream[streamIdx]);
			}
		}
	}
	
	numOfRemainingLaunches = latticeZ % numOfStreams;
	if (numOfRemainingLaunches != 0) {
		unsigned short int z = (latticeZ - numOfStreams);
		for (streamIdx = 0; streamIdx < numOfRemainingLaunches; streamIdx++) {
			DCSKernel << <dimGridDCS, dimBlockDCS, 0, stream[streamIdx] >> >(d_SliceDCS[streamIdx], d_AtomX, d_AtomY, d_AtomZ, d_Charge, z + streamIdx, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
		}

		for (streamIdx = 0; streamIdx < numOfRemainingLaunches; streamIdx++) {
			h_LatticeDCSOffset = (z + streamIdx) * sliceGridSize;
			cudaMemcpyAsync(h_LatticeDCS + h_LatticeDCSOffset, d_SliceDCS[streamIdx], sliceGridSize * sizeof(LATTICE_DATA_TYPE), cudaMemcpyDeviceToHost, stream[streamIdx]);
		}
	}

	for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
		cudaStreamSynchronize(stream[streamIdx]);
	}

	DCSKernelTimer.Stop();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DCSKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	printf("DCSKernel duration: %f ms\n", DCSKernelTimer.Elapsed());

	//DCS2
	dim3 dimGridDCS2((latticeX - 1) / (BLOCK_WIDTH * NUM_OF_POINTS_DCS2) + 1, (latticeY - 1) / BLOCK_WIDTH + 1, 1);

	DCS2KernelTimer.Start();
	h_LatticeDCSOffset = 0;
	if (latticeZ > 1) {
		for (unsigned short int z = 0; z < latticeZ; z += numOfStreams) {
			for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
				DCS2Kernel << <dimGridDCS2, dimBlockDCS, 0, stream[streamIdx] >> >(d_SliceDCS[streamIdx], d_AtomX, d_AtomY, d_AtomZ, d_Charge, z + streamIdx, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
			}

			for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
				h_LatticeDCSOffset = (z + streamIdx) * sliceGridSize;
				cudaMemcpyAsync(h_LatticeDCS2 + h_LatticeDCSOffset, d_SliceDCS[streamIdx], sliceGridSize * sizeof(LATTICE_DATA_TYPE), cudaMemcpyDeviceToHost, stream[streamIdx]);
			}
		}
	}

	numOfRemainingLaunches = latticeZ % numOfStreams;
	if (numOfRemainingLaunches != 0) {
		unsigned short int z = (latticeZ - numOfStreams);
		for (streamIdx = 0; streamIdx < numOfRemainingLaunches; streamIdx++) {
			DCS2Kernel << <dimGridDCS2, dimBlockDCS, 0, stream[streamIdx] >> >(d_SliceDCS[streamIdx], d_AtomX, d_AtomY, d_AtomZ, d_Charge, z + streamIdx, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
		}

		for (streamIdx = 0; streamIdx < numOfRemainingLaunches; streamIdx++) {
			h_LatticeDCSOffset = (z + streamIdx) * sliceGridSize;
			cudaMemcpyAsync(h_LatticeDCS2 + h_LatticeDCSOffset, d_SliceDCS[streamIdx], sliceGridSize * sizeof(LATTICE_DATA_TYPE), cudaMemcpyDeviceToHost, stream[streamIdx]);
		}
	}

	for (streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
		cudaStreamSynchronize(stream[streamIdx]);
	}

	DCS2KernelTimer.Stop();

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "DCS2Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	printf("DCS2Kernel duration: %f ms\n", DCS2KernelTimer.Elapsed());

	//CPU
	CPUClock = clock();
	for (unsigned short int z = 0; z < latticeZ; z++) {
		CPU(latticeCPU, h_AtomX, h_AtomY, h_AtomZ, h_Charge, z, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
	}
	CPUDuration = (clock() - CPUClock) / (double)CLOCKS_PER_SEC;
	printf("CPU duration: %f ms\n", CPUDuration * 1000);

	//CPU2
	CPU2Clock = clock();
	for (unsigned short int z = 0; z < latticeZ; z++) {
		CPU2(latticeCPU2, h_AtomX, h_AtomY, h_AtomZ, h_Charge, z, numOfAtoms, latticeX, latticeY, latticeGridSpacing);
	}
	CPU2Duration = (clock() - CPU2Clock) / (double)CLOCKS_PER_SEC;
	printf("CPU2 duration: %f ms\n", CPU2Duration * 1000);

	printf("CPU2 verification started.\n");
	for (unsigned int i = 0; i < latticeGridSize; i++) {
		if (abs(latticeCPU[i] - latticeCPU2[i]) > EPS) {
			fprintf(stderr, "DCS Verification failed at element %i! latticeCPU[%i] = %f, latticeCPU2[%i] = %f\n", i, i, latticeCPU[i], i, latticeCPU2[i]);
			return 1;
		}
	}
	printf("CPU2 verification PASSED.\n");
	
	printf("DCS verification started.\n");
	for (unsigned int i = 0; i < latticeGridSize; i++) {
		if (abs(latticeCPU[i] - h_LatticeDCS[i]) > EPS) {
			fprintf(stderr, "DCS Verification failed at element %i! latticeCPU[%i] = %f, latticeDCS[%i] = %f\n", i, i, latticeCPU[i], i, h_LatticeDCS[i]);
			return 1;
		}
	}
	printf("DCS verification PASSED.\n");

	printf("DCS2 verification started.\n");
	for (unsigned int i = 0; i < latticeGridSize; i++) {
		if (abs(latticeCPU[i] - h_LatticeDCS2[i]) > EPS) {
			fprintf(stderr, "DCS2 Verification failed at element %i! latticeCPU[%i] = %f, latticeDCS2[%i] = %f\n", i, i, latticeCPU[i], i, h_LatticeDCS2[i]);
			return 1;
		}
	}
	printf("DCS2 verification PASSED.\n");

Error:
	free(h_AtomX);
	free(h_AtomY);
	free(h_AtomZ);
	free(latticeCPU);
	free(h_LatticeDCS);
	
	cudaFree(d_AtomX);
	cudaFree(d_AtomY);
	cudaFree(d_AtomZ);
	for (uint8_t streamIdx = 0; streamIdx < numOfStreams; streamIdx++) {
		cudaStreamDestroy(stream[streamIdx]);
		cudaFree(d_SliceDCS[streamIdx]);
	}

    cudaError_t cudaStatusReset = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	return cudaStatus;
}