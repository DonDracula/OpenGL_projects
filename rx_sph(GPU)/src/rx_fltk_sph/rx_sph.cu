/*! 
  @file rx_sph.cu
	
  @brief CUDAによるSPH

*/
// FILE --rx_sph.cu--


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <GL/glew.h>

#include <GL/freeglut.h>

#include "rx_sph_kernel.cu"

//#include "rx_cu_funcs.cuh"
#include <thrust/device_vector.h>
#include <thrust/scan.h>



//-----------------------------------------------------------------------------
// MARK:グローバル変数
//-----------------------------------------------------------------------------
cudaArray *g_caNoiseTile = 0;
float *g_dNoiseTile[3] = {0, 0, 0};
uint g_udNoiseTileSize = 0;
uint g_uNoiseTileNum[3*3] = {0, 0, 0,  0, 0, 0,  0, 0, 0};


//-----------------------------------------------------------------------------
// CUDA関数
//-----------------------------------------------------------------------------
extern "C"
{
void CuSetParameters(rxSimParams *hostParams)
{
	// copy parameters to constant memory
	RX_CUCHECK( cudaMemcpyToSymbol(params, hostParams, sizeof(rxSimParams)) );
}

void CuClearData(void)
{
}


// グリッド内ブロック数，ブロック内スレッド数の計算
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = DivCeil(n, numThreads);
}


//-----------------------------------------------------------------------------
// MARK:3D SPH
//-----------------------------------------------------------------------------
/*!
 * 分割セルのハッシュを計算
 * @param[in] 
 * @return 
 */
void CuCalcHash(uint* dGridParticleHash, uint* dSortedIndex, float* dPos, int nprts)
{
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	calcHashD<<< numBlocks, numThreads >>>(dGridParticleHash,
										   dSortedIndex,
										   (float4*)dPos,
										   nprts);
	
	RX_CUERROR("Kernel execution failed");	// カーネルエラーチェック
}

/*!
 * パーティクル配列をソートされた順番に並び替え，
 * 各セルの始まりと終わりのインデックスを検索
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] oldPos パーティクル位置
 * @param[in] oldVel パーティクル速度
 */
void CuReorderDataAndFindCellStart(rxParticleCell cell, float* oldPos, float* oldVel)
{
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	RX_CUCHECK(cudaMemset(cell.dCellStart, 0xffffffff, cell.uNumCells*sizeof(uint)));

#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, oldPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dSortedVelTex, oldVel, cell.uNumParticles*sizeof(float4)));
#endif

	uint smemSize = sizeof(uint)*(numThreads+1);

	// カーネル実行
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(cell, (float4*)oldPos, (float4*)oldVel);

	RX_CUERROR("Kernel execution failed: CuReorderDataAndFindCellStartD");
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dSortedVelTex));
#endif
}


/*!
 * パーティクル密度の計算(カーネル呼び出し)
 * @param[out] dDens パーティクル密度
 * @param[out] dPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphDensity(float* dDens, float* dPres, rxParticleCell cell)
{
	// MRK:CuSphDensity2D
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif
	//RX_CUCHECK(cudaMemset((void*)dNewDens, 0, sizeof(float2)*cell.uNumParticles));

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalDensity<<< numBlocks, numThreads >>>(dDens, dPres, cell);

	RX_CUERROR("sphCalDensity kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}

/*!
 * パーティクル法線の計算
 * @param[out] dNewDens パーティクル密度
 * @param[out] dNewPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphNormal(float* dNrms, float* dDens, rxParticleCell cell)
{
	// MRK:CuSphNormal

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalNormal<<< numBlocks, numThreads >>>((float4*)dNrms, dDens, cell);

	RX_CUERROR("sphCalNormal kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

}

/*!
 * パーティクルにかかる力の計算(カーネル呼び出し)
 * @param[in] dDens パーティクル密度
 * @param[in] dPres パーティクル圧力
 * @param[out] dFrc パーティクルにかかる力
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] dt 時間ステップ幅
 */
void CuSphForces(float* dDens, float* dPres, float* dFrc, rxParticleCell cell, float dt)
{
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dSortedVelTex, cell.dSortedVel, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalForces<<< numBlocks, numThreads >>>(dDens, dPres, (float4*)dFrc, cell);

	RX_CUERROR("calForcesSPH kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dSortedVelTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}

/*!
 * パーティクル位置，速度の更新
 * @param[inout] pos パーティクル位置
 * @param[inout] vel パーティクル速度
 * @param[inout] velOld 前ステップのパーティクル速度
 * @param[in] frc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuSphIntegrate(float* pos, float* vel, float* frc, float* dens, 
					float dt, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphIntegrate<<< numBlocks, numThreads >>>((float4*)pos, (float4*)vel, (float4*)frc, dens, 
											  dt, nprts);
	
	RX_CUERROR("sphIntegrate kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * パーティクル位置，速度の更新
 * @param[inout] pos パーティクル位置
 * @param[inout] vel パーティクル速度
 * @param[inout] velOld 前ステップのパーティクル速度
 * @param[in] frc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuSphIntegrateWithPolygon(float* pos, float* vel, float* frc, float* dens, 
							   float* vrts, int* tris, int tri_num, float dt, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphIntegrateWithPolygon<<< numBlocks, numThreads >>>((float4*)pos, (float4*)vel, (float4*)frc, dens, 
											   (float3*)vrts, (int3*)tris, tri_num, dt, cell);
	
	RX_CUERROR("sphIntegrate kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * グリッド上の密度を算出
 * @param[out] dGridD グリッド上の密度値
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] nx,ny グリッド数
 * @param[in] x0,y0 グリッド最小座標
 * @param[in] dx,dy グリッド幅
 */
void CuSphGridDensity(float *dGridD, rxParticleCell cell, 
					  int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz)
{
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif

	uint3  gnum = make_uint3(nx, ny, nz);
	float3 gmin = make_float3(x0, y0, z0);
	float3 glen = make_float3(dx, dy, dz);

	int numcell = gnum.x*gnum.y*gnum.z;

	int threads = 128;
	dim3 grid((numcell+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// カーネル実行
	sphCalDensityInGrid<<<grid, threads>>>(dGridD, cell, gnum, gmin, glen);

	RX_CUERROR("Kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}


//-----------------------------------------------------------------------------
// MARK:Anisotropic Kernel
//-----------------------------------------------------------------------------
/*!
 * カーネル中心位置の更新と重み付き平均の計算(カーネル関数)
 * @param[out] dUpPos 更新カーネル中心
 * @param[out] dPosW 重み付き平均パーティクル座標 
 * @param[in]  lambda 平滑化のための定数
 * @param[in]  h 探索半径
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphCalUpdatedPosition(float* dUpPos, float* dPosW, float lambda, float h, rxParticleCell cell)
{
	// MRK:CuSphCalUpdatedPosition
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalUpdatedPosition<<< numBlocks, numThreads >>>((float4*)dUpPos, (float4*)dPosW, lambda, h, cell);

	RX_CUERROR("sphCalUpdatedPosition kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}

/*!
 * 平滑化位置での重み付き平均位置の再計算とcovariance matrixの計算
 * @param[out] dPosW 重み付き平均パーティクル座標 
 * @param[out] dCMat Covariance Matrix
 * @param[in]  h 探索半径
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphCalCovarianceMatrix(float* dPosW, float* dCMat, float h, rxParticleCell cell)
{
	// MRK:CuSphCalCovarianceMatrix
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalCovarianceMatrix<<< numBlocks, numThreads >>>((float4*)dPosW, (matrix3x3*)dCMat, h, cell);

	RX_CUERROR("sphCalCovarianceMatrix kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}

/*!
 * 特異値分解により固有値を計算
 * @param[in]  dC Covariance Matrix
 * @param[in]  dPosW 重み付き平均位置
 * @param[out] dEigen 固有値
 * @param[out] dR 固有ベクトル(回転行列)
 * @param[in]  numParticles パーティクル数
 */
void CuSphSVDecomposition(float* dC, float* dPosW, float* dEigen, float* dR, uint numParticles)
{
	// MRK:CuSphCalTransformMatrix

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(numParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphSVDecomposition<<< numBlocks, numThreads >>>((matrix3x3*)dC, (float4*)dPosW, (float3*)dEigen, (matrix3x3*)dR, numParticles);

	RX_CUERROR("sphCalTransformMatrix kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * 固有値，固有ベクトル(回転行列)から変形行列を計算
 * @param[in]  dEigen 固有値
 * @param[in]  dR 固有ベクトル(回転行列)
 * @param[out] dG 変形行列
 * @param[in]  numParticles パーティクル数
 */
void CuSphCalTransformMatrix(float* dEigen, float* dR, float *dG, uint numParticles)
{
	// MRK:CuSphCalTransformMatrix

	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(numParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalTransformMatrix<<< numBlocks, numThreads >>>((float3*)dEigen, (matrix3x3*)dR, (matrix3x3*)dG, numParticles);

	RX_CUERROR("sphCalTransformMatrix kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * グリッド上の密度を算出
 * @param[out] dGridD グリッド上の密度値
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] nx,ny グリッド数
 * @param[in] x0,y0 グリッド最小座標
 * @param[in] dx,dy グリッド幅
 */
void CuSphGridDensityAniso(float *dGridD, float *dG, float Emax, rxParticleCell cell, 
						   int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz)
{
#if USE_TEX
	RX_CUCHECK(cudaBindTexture(0, dSortedPosTex, cell.dSortedPos, cell.uNumParticles*sizeof(float4)));
	RX_CUCHECK(cudaBindTexture(0, dCellStartTex, cell.dCellStart, cell.uNumCells*sizeof(uint)));
	RX_CUCHECK(cudaBindTexture(0, dCellEndTex, cell.dCellEnd, cell.uNumCells*sizeof(uint)));	
#endif

	uint3  gnum = make_uint3(nx, ny, nz);
	float3 gmin = make_float3(x0, y0, z0);
	float3 glen = make_float3(dx, dy, dz);

	int numcell = gnum.x*gnum.y*gnum.z;

	int threads = THREAD_NUM;
	dim3 grid((numcell+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// カーネル実行
	sphCalDensityAnisoInGrid<<<grid, threads>>>(dGridD, (matrix3x3*)dG, Emax, cell, gnum, gmin, glen);

	RX_CUERROR("Kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

#if USE_TEX
	RX_CUCHECK(cudaUnbindTexture(dSortedPosTex));
	RX_CUCHECK(cudaUnbindTexture(dCellStartTex));
	RX_CUCHECK(cudaUnbindTexture(dCellEndTex));
#endif
}


}   // extern "C"
