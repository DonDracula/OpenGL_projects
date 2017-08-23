/*! 
  @file rx_cu_funcs.cuh
	
  @brief CUDA関数の宣言
 
  @author Makoto Fujisawa
  @date 2009-08, 2011-06
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

void CuCopyArrayFromDevice(void* host, const void* device, cudaGraphicsResource **resource, int offset, int size);
void CuCopyArrayToDevice(void* device, const void* host, int offset, int size);
void CuRegisterGLBufferObject(unsigned int vbo, cudaGraphicsResource **resource);
void CuUnregisterGLBufferObject(cudaGraphicsResource *resource);
void *CuMapGLBufferObject(cudaGraphicsResource **resource);
void CuUnmapGLBufferObject(cudaGraphicsResource *resource);

void CuScan(unsigned int* output, unsigned int* input, unsigned int numElements);
void CuSort(unsigned int *dHash, uint *dIndex, uint numParticles);

void CuScanf(float* output, float* input, unsigned int numElements);


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

void CuCalcHashB(uint* dGridParticleHash, uint* dSortedIndex, float* dPos, 
				 float3 world_origin, float3 cell_width, uint3 grid_size, int nprts);
void CuReorderDataAndFindCellStartB(rxParticleCell cell, float* oldPos);

// 境界パーティクル
void CuSphBoundaryVolume(float* dVolB, float mass, rxParticleCell cell);
void CuSphBoundaryDensity(float* dDens, float* dPres, float* dPos, float* dVolB, rxParticleCell bcell, uint pnum);
void CuSphBoundaryForces(float* dDens, float* dPres, float* dPos, float* dVolB, float* dFrc, rxParticleCell bcell, uint pnum);


//-----------------------------------------------------------------------------
// PBDSPH
//-----------------------------------------------------------------------------
void CuPbfDensity(float* dDens, rxParticleCell cell);
void CuPbfExternalForces(float* dDens, float* dFrc, rxParticleCell cell, float dt);

void CuPbfScalingFactor(float* dPos, float* dDens, float* dScl, float eps, rxParticleCell cell);
void CuPbfPositionCorrection(float* dPos, float* dScl, float* dDp, rxParticleCell cell);

void CuPbfCorrectPosition(float* dPos, float* dDp, uint nprts);
void CuPbfUpdatePosition(float* pos, float* new_pos, float* new_vel, float dt, uint nprts);
void CuPbfUpdateVelocity(float* pos, float* new_pos, float* new_vel, float dt, uint nprts);

void CuXSphViscosity(float* dPos, float* dVel, float* dNewVel, float* dDens, float c, rxParticleCell cell);

float CuPbfCalDensityFluctuation(float* dErrScan, float* dErr, float* dDens, float rest_dens, uint nprts);

void CuPbfIntegrate(float* dPos, float* dVel, float* dAcc, 
					float* dNewPos, float* dNewVel, float dt, uint nprts);
void CuPbfIntegrateWithPolygon(float* dPos, float* dVel, float* dAcc, 
							   float* dNewPos, float* dNewVel, 
							   float* dVrts, int* dTris, int tri_num, float dt, rxParticleCell cell);
void CuPbfIntegrate2(float* dPos, float* dVel, float* dAcc, 
					 float* dNewPos, float* dNewVel, float dt, uint nprts);
void CuPbfIntegrateWithPolygon2(float* dPos, float* dVel, float* dAcc, 
								float* dNewPos, float* dNewVel, 
								float* dVrts, int* dTris, int tri_num, float dt, rxParticleCell cell);

void CuPbfGridDensity(float *dGridD, rxParticleCell cell, 
						 int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz);
void CuPbfNormal(float* dNrms, float* dDens, rxParticleCell cell);

// 境界パーティクル
void CuPbfBoundaryDensity(float* dDens, float* dPos, float* dVolB, rxParticleCell bcell, uint pnum);
void CuPbfScalingFactorWithBoundary(float* dPos, float* dDens, float* dScl, float eps, rxParticleCell cell, 
									   float* dVolB, float* dSclB, rxParticleCell bcell);
void CuPbfPositionCorrectionWithBoundary(float* dPos, float* dScl, float* dDp, rxParticleCell cell, 
											float* dVolB, float* dSclB, rxParticleCell bcell);


//-----------------------------------------------------------------------------
// MC法によるメッシュ化
//-----------------------------------------------------------------------------
#ifdef RX_CUMC_USE_GEOMETRY
void CuMCCreateMesh(float threshold, unsigned int &nvrts, unsigned int &ntris);
#else
void CuMCCreateMesh(GLuint pvbo, GLuint nvbo, float threshold, unsigned int &nvrts, unsigned int &ntris);
#endif


bool CuMCInit(void);

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