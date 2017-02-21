/*! 
  @file rx_cu_funcs.cuh
	
  @brief CUDA関数の宣言
 
*/
// FILE --rx_cu_funcs.cuh--

#ifndef _RX_CU_FUNCS_CUH_
#define _RX_CU_FUNCS_CUH_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_cu_common.cuh"


//-----------------------------------------------------------------------------
// CUDA関数
//-----------------------------------------------------------------------------
extern "C"
{
void CuInit(int argc, char **argv);
void CuSetDevice(int id);

void CuDeviceProp(void);

void CuAllocateArray(void **devPtr, int size);
void CuSetArrayValue(void *devPtr, int val, size_t size);
void CuFreeArray(void *devPtr);

void CuCopyArrayD2D(void *dDst, void *dSrc, int size);
void CuThreadSync(void);

void CuCopyArrayFromDevice(void* host, const void* device, cudaGraphicsResource **resource, int size);
void CuCopyArrayToDevice(void* device, const void* host, int offset, int size);
void CuRegisterGLBufferObject(unsigned int vbo, cudaGraphicsResource **resource);
void CuUnregisterGLBufferObject(cudaGraphicsResource *resource);
void *CuMapGLBufferObject(cudaGraphicsResource **resource);
void CuUnmapGLBufferObject(cudaGraphicsResource *resource);

void CuScan(unsigned int* output, unsigned int* input, unsigned int numElements);
void CuSort(unsigned int *dHash, uint *dIndex, uint numParticles);

//-----------------------------------------------------------------------------
// 3D SPH
//-----------------------------------------------------------------------------
void CuSPHInit(int max_particles);
void CuSPHClean(void);

void CuSetParameters(rxSimParams *hostParams);
void CuClearData(void);

// 近傍パーティクル検索用グリッド
void CuCalcHash(uint*  gridParticleHash, uint*  gridParticleIndex, float* pos, int numParticles);
void CuReorderDataAndFindCellStart(rxParticleCell cell, float* oldPos, float* oldVel);

// SPH計算
void CuSphDensity(float* dDens, float* dPres, rxParticleCell cell);
void CuSphForces(float* dDens, float* dPres, float* dFrc, rxParticleCell cell, float dt);
void CuSphNormal(float* dNrms, float* dDens, rxParticleCell cell);
void CuSphIntegrate(float* pos, float* vel, float* frc, float* dens, float dt, uint numParticles);
void CuSphIntegrateWithPolygon(float* pos, float* vel, float* frc, float* dens, 
							   float* vrts, int* tris, int tri_num, float dt, rxParticleCell cell);

// グリッド
void CuSphGridDensity(float *dGridD, rxParticleCell cell, 
					  int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz);

// Anisotropic Kernel
void CuSphCalUpdatedPosition(float* dUpPos, float* dPosW, float lambda, float h, rxParticleCell cell);
void CuSphCalCovarianceMatrix(float* dPosW, float* dCMat, float h, rxParticleCell cell);
void CuSphSVDecomposition(float* dC, float* dPosW, float* dEigen, float* dR, uint numParticles);
void CuSphCalTransformMatrix(float* dEigen, float* dR, float *dG, uint numParticles);

void CuSphGridDensityAniso(float *dGridD, float *dG, float Emax, rxParticleCell cell, 
						   int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz);



//-----------------------------------------------------------------------------
// MC法によるメッシュ化
//-----------------------------------------------------------------------------
#ifdef RX_CUMC_USE_GEOMETRY
void CuMCCreateMesh(float threshold, unsigned int &nvrts, unsigned int &ntris);
#else
void CuMCCreateMesh(GLuint pvbo, GLuint nvbo, float threshold, unsigned int &nvrts, unsigned int &ntris);
#endif

void CuInitMCTable(void);
void CuCleanMCTable(void);

void CuMCCalTriNum(float *dVolume, uint *dVoxBit, uint *dVoxVNum, uint *dVoxVNumScan, 
				   uint *dVoxTNum, uint *dVoxTNumScan, uint *dVoxOcc, uint *dVoxOccScan, uint *dCompactedVox, 
				   uint3 grid_size, uint num_voxels, float3 grid_width, float threshold, 
				   uint &num_active_voxels, uint &nvrts, uint &ntris);
void CuMCCalEdgeVrts(float *dVolume, float *dEdgeVrts, float *dCompactedEdgeVrts, 
					 uint *dEdgeOcc, uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4], 
					 uint3 grid_size, uint num_voxels, float3 grid_width, float3 grid_min, float threshold, 
					 uint &nvrts);
void CuMCCalTri(uint *dTris, uint *dVoxBit, uint *dVoxTNumScan, uint *dCompactedVox, 
				uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4], 
				uint3 grid_size, uint num_voxels, float3 grid_width, float threshold, 
				uint num_active_voxels, uint nvrts, uint ntris);
void CuMCCalNrm(float *dNrms, uint *dTris, float *dCompactedEdgeVrts, uint nvrts, uint ntris);


} // extern "C"


#endif // #ifdef _RX_CU_FUNCS_CUH_