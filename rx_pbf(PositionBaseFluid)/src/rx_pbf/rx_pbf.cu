/*! 
  @file rx_sph.cu
	
  @brief CUDAによるSPH

  @author Makoto Fujisawa
  @date 2009-08, 2011-06
*/
// FILE --rx_sph.cu--


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_pbf_kernel.cu"

#include <thrust/device_vector.h>
#include <thrust/scan.h>



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

/*!
 * thrust::exclusive_scanの呼び出し
 * @param[out] dScanData scan後のデータ
 * @param[in] dData 元データ
 * @param[in] num データ数
 */
void CuScanf(float* dScanData, float* dData, unsigned int num)
{
	thrust::exclusive_scan(thrust::device_ptr<float>(dData), 
						   thrust::device_ptr<float>(dData+num),
						   thrust::device_ptr<float>(dScanData));
}


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
	
	RX_CUERROR("calcHashD kernel execution failed");	// カーネルエラーチェック
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

	uint smemSize = sizeof(uint)*(numThreads+1);

	// カーネル実行
	reorderDataAndFindCellStartD<<< numBlocks, numThreads, smemSize>>>(cell, (float4*)oldPos, (float4*)oldVel);

	RX_CUERROR("reorderDataAndFindCellStartD kernel execution failed");
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * 分割セルのハッシュを計算
 * @param[in] 
 * @return 
 */
void CuCalcHashB(uint* dGridParticleHash, uint* dSortedIndex, float* dPos, 
				 float3 world_origin, float3 cell_width, uint3 grid_size, int nprts)
{
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	calcHashB<<< numBlocks, numThreads >>>(dGridParticleHash,
										   dSortedIndex,
										   (float4*)dPos,
										   world_origin, 
										   cell_width, 
										   grid_size, 
										   nprts);
	
	RX_CUERROR("Kernel execution failed : calcHashB");	// カーネルエラーチェック
}

/*!
 * パーティクル配列をソートされた順番に並び替え，
 * 各セルの始まりと終わりのインデックスを検索
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] oldPos パーティクル位置
 */
void CuReorderDataAndFindCellStartB(rxParticleCell cell, float* oldPos)
{
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	RX_CUCHECK(cudaMemset(cell.dCellStart, 0xffffffff, cell.uNumCells*sizeof(uint)));

	uint smemSize = sizeof(uint)*(numThreads+1);

	// カーネル実行
	reorderDataAndFindCellStartB<<< numBlocks, numThreads, smemSize>>>(cell, (float4*)oldPos);

	RX_CUERROR("Kernel execution failed: CuReorderDataAndFindCellStartB");
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}










//-----------------------------------------------------------------------------
// 境界パーティクル処理
//-----------------------------------------------------------------------------
/*!
 * 境界パーティクルの体積を計算
 *  - "Versatile Rigid-Fluid Coupling for Incompressible SPH", 2.2 式(3)の上
 * @param[out] dVolB 境界パーティクルの体積
 * @param[in]  mass パーティクル質量
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphBoundaryVolume(float* dVolB, float mass, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalBoundaryVolume<<< numBlocks, numThreads >>>(dVolB, cell);

	RX_CUERROR("kernel execution failed : sphCalBoundaryVolume");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクル密度の計算(カーネル呼び出し)
 * @param[out] dDens パーティクル密度
 * @param[out] dPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuSphBoundaryDensity(float* dDens, float* dPres, float* dPos, float* dVolB, rxParticleCell bcell, uint pnum)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(pnum, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalBoundaryDensity<<< numBlocks, numThreads >>>(dDens, dPres, (float4*)dPos, dVolB, bcell, pnum);

	RX_CUERROR("kernel execution failed : sphCalBoundaryDensity");	// カーネル実行エラーチェック
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
void CuSphBoundaryForces(float* dDens, float* dPres, float* dPos, float* dVolB, float* dFrc, rxParticleCell bcell, uint pnum)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(pnum, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	sphCalBoundaryForce<<< numBlocks, numThreads >>>(dDens, dPres, (float4*)dPos, dVolB, (float4*)dFrc, bcell, pnum);

	RX_CUERROR("kernel execution failed : sphCalBoundaryForce");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}





//-----------------------------------------------------------------------------
// PBF
//-----------------------------------------------------------------------------

/*!
 * パーティクル密度の計算(カーネル呼び出し)
 * @param[out] dDens パーティクル密度
 * @param[out] dPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuPbfDensity(float* dDens, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalDensity<<< numBlocks, numThreads >>>(dDens, cell);

	RX_CUERROR("pbfCalDensity kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクルにかかる力の計算(カーネル呼び出し)
 * @param[in] dDens パーティクル密度
 * @param[out] dFrc パーティクルにかかる力
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] dt 時間ステップ幅
 */
void CuPbfExternalForces(float* dDens, float* dFrc, rxParticleCell cell, float dt)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalExternalForces<<< numBlocks, numThreads >>>(dDens, (float4*)dFrc, cell);

	RX_CUERROR("pbfCalExternalForces kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * スケーリングファクタの計算
 * @param[in] dPos パーティクル中心座標
 * @param[out] dDens パーティクル密度
 * @param[out] dScl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in] cell パーティクルグリッドデータ
 */
void CuPbfScalingFactor(float* dPos, float* dDens, float* dScl, float eps, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalScalingFactor<<< numBlocks, numThreads >>>((float4*)dPos, dDens, dScl, eps, cell);

	RX_CUERROR("pbfCalScalingFactor kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * 平均密度変動の計算
 *  - すべてのパーティクル密度の初期密度との差をカーネルで計算し，Prefix Sum (Scan)でその合計を求める
 * @param[out] dErrScan 変動値のScan結果を格納する配列
 * @param[out] dErr パーティクル密度変動値
 * @param[in] dDens パーティクル密度
 * @param[in] rest_dens 初期密度
 * @param[in] nprts パーティクル数
 * @return 平均密度変動
 */
float CuPbfCalDensityFluctuation(float* dErrScan, float* dErr, float* dDens, float rest_dens, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfDensityFluctuation<<< numBlocks, numThreads >>>(dErr, dDens, rest_dens, nprts);

	RX_CUERROR("pbfDensityFluctuation kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

	// 各パーティクルの密度変動をScan
	CuScanf(dErrScan, dErr, nprts);

	// Exclusive scan (最後の要素が0番目からn-2番目までの合計になっている)なので，
	// Scan前配列の最後(n-1番目)の要素と合計することで密度変動の合計を計算
	float lval, lsval;
	RX_CUCHECK(cudaMemcpy((void*)&lval, (void*)(dErr+nprts-1), sizeof(float), cudaMemcpyDeviceToHost));
	RX_CUCHECK(cudaMemcpy((void*)&lsval, (void*)(dErrScan+nprts-1), sizeof(float), cudaMemcpyDeviceToHost));
	float dens_var = lval+lsval;

	return dens_var/(float)nprts;
}

/*!
 * 位置修正量の計算
 * @param[in] dPos パーティクル中心座標
 * @param[in] dScl スケーリングファクタ
 * @param[out] dDp 位置修正量
 * @param[in] cell パーティクルグリッドデータ
 */
void CuPbfPositionCorrection(float* dPos, float* dScl, float* dDp, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfPositionCorrection<<< numBlocks, numThreads >>>((float4*)dPos, dScl, (float4*)dDp, cell);

	RX_CUERROR("pbfPositionCorrection kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクル位置を更新
 * @param[inout] dPos パーティクル位置
 * @param[in] dDp 位置修正量
 * @param[in] nprts パーティクル数
 */
void CuPbfCorrectPosition(float* dPos, float* dDp, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCorrectPosition<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dDp, nprts);
	
	RX_CUERROR("pbfCorrectPosition kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}



/*!
 * 境界パーティクル密度を従来のパーティクル密度に加える
 * @param[inout] dDens 流体パーティクル密度
 * @param[in] dPos  流体パーティクル圧力
 * @param[in] dVolB 境界パーティクル体積
 * @param[in] bcell 境界パーティクルグリッドデータ
 */
void CuPbfBoundaryDensity(float* dDens, float* dPos, float* dVolB, rxParticleCell bcell, uint pnum)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(pnum, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalBoundaryDensity<<< numBlocks, numThreads >>>(dDens, (float4*)dPos, dVolB, bcell, pnum);

	RX_CUERROR("pbfCalBoundaryDensity kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * スケーリングファクタの計算(境界パーティクル含む)
 * @param[in] dPos 流体パーティクル中心座標
 * @param[out] dDens 流体パーティクル密度
 * @param[out] dScl 流体パーティクルのスケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in] cell 流体パーティクルグリッドデータ
 * @param[in] dVolB 境界パーティクル体積
 * @param[out] dSclB 境界パーティクルのスケーリングファクタ
 * @param[in] bcell 境界パーティクルグリッドデータ
 */
void CuPbfScalingFactorWithBoundary(float* dPos, float* dDens, float* dScl, float eps, rxParticleCell cell, 
									   float* dVolB, float* dSclB, rxParticleCell bcell)
{
	// 流体パーティクルの数だけスレッドを立てる
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalScalingFactorWithBoundary<<< numBlocks, numThreads >>>((float4*)dPos, dDens, dScl, eps, cell, dVolB, bcell);

	RX_CUERROR("pbfCalScalingFactorWithBoundary kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

	// 境界パーティクルのスケーリングファクタの計算
	// 境界パーティクルの数だけスレッドを立てる
	computeGridSize(bcell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalBoundaryScalingFactor<<< numBlocks, numThreads >>>((float4*)dPos, dDens, eps, cell, dVolB, dSclB, bcell);

	RX_CUERROR("pbfCalScalingFactorWithBoundary kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

}

/*!
 * 位置修正量の計算(境界パーティクル含む)
 * @param[in] dPos 流体パーティクル中心座標
 * @param[in] dScl 流体パーティクルのスケーリングファクタ
 * @param[out] dDens 流体パーティクル位置修正量
 * @param[in] cell 流体パーティクルグリッドデータ
 * @param[in] dVolB 境界パーティクル体積
 * @param[in] dSclB 境界パーティクルのスケーリングファクタ
 * @param[in] bcell 境界パーティクルグリッドデータ
 */
void CuPbfPositionCorrectionWithBoundary(float* dPos, float* dScl, float* dDp, rxParticleCell cell, 
											float* dVolB, float* dSclB, rxParticleCell bcell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfPositionCorrectionWithBoundary<<< numBlocks, numThreads >>>((float4*)dPos, dScl, (float4*)dDp, cell, 
																	  dVolB, dSclB, bcell);

	RX_CUERROR("pbfPositionCorrectionWithBoundary kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}



/*!
 * パーティクル位置，速度の更新
 *  - 三角形ポリゴン境界版
 *  - 現在の位置のみ使って衝突判定
 * @param[in] dPos パーティクル位置
 * @param[in] dVel パーティクル速度
 * @param[in] dAcc パーティクル加速度
 * @param[out] dNewPos 更新されたパーティクル位置
 * @param[out] dNewVel 更新されたパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuPbfIntegrate(float* dPos, float* dVel, float* dAcc, 
					float* dNewPos, float* dNewVel, float dt, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfIntegrate<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dVel, (float4*)dAcc, 
												 (float4*)dNewPos, (float4*)dNewVel, dt, nprts);
	
	RX_CUERROR("pbfIntegrate kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクル位置，速度の更新
 *  - 三角形ポリゴン境界版
 *  - 現在の位置のみ使って衝突判定
 * @param[in] dPos パーティクル位置
 * @param[in] dVel パーティクル速度
 * @param[in] dAcc パーティクル加速度
 * @param[out] dNewPos 更新されたパーティクル位置
 * @param[out] dNewVel 更新されたパーティクル速度
 * @param[in] dVrts 三角形ポリゴン頂点
 * @param[in] dTris 三角形ポリゴンインデックス
 * @param[in] tri_num 三角形ポリゴン数
 * @param[in] dt 時間ステップ幅
 * @param[in] cell パーティクルグリッドデータ
 */
void CuPbfIntegrateWithPolygon(float* dPos, float* dVel, float* dAcc, 
							   float* dNewPos, float* dNewVel, 
							   float* dVrts, int* dTris, int tri_num, float dt, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfIntegrateWithPolygon<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dVel, (float4*)dAcc, 
															(float4*)dNewPos, (float4*)dNewVel, (float3*)dVrts, (int3*)dTris, tri_num, dt, cell);
	
	RX_CUERROR("pbfIntegrateWithPolygon kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * パーティクル位置，速度の更新
 *  - 現在の位置と修正後の位置の2つを使って衝突判定(PBF反復時に使用)
 * @param[in] dPos パーティクル位置
 * @param[in] dVel パーティクル速度
 * @param[in] dAcc パーティクル加速度
 * @param[out] dNewPos 更新されたパーティクル位置
 * @param[out] dNewVel 更新されたパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuPbfIntegrate2(float* dPos, float* dVel, float* dAcc, 
					 float* dNewPos, float* dNewVel, float dt, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfIntegrate2<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dVel, (float4*)dAcc, 
												  (float4*)dNewPos, (float4*)dNewVel, dt, nprts);
	
	RX_CUERROR("pbfIntegrate2 kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}
/*!
 * パーティクル位置，速度の更新
 *  - 三角形ポリゴン境界版
 *  - 現在の位置と修正後の位置の2つを使って衝突判定(PBF反復時に使用)
 * @param[in] dPos パーティクル位置
 * @param[in] dVel パーティクル速度
 * @param[in] dAcc パーティクル加速度
 * @param[out] dNewPos 更新されたパーティクル位置
 * @param[out] dNewVel 更新されたパーティクル速度
 * @param[in] dVrts 三角形ポリゴン頂点
 * @param[in] dTris 三角形ポリゴンインデックス
 * @param[in] tri_num 三角形ポリゴン数
 * @param[in] dt 時間ステップ幅
 * @param[in] cell パーティクルグリッドデータ
 */
void CuPbfIntegrateWithPolygon2(float* dPos, float* dVel, float* dAcc, 
								float* dNewPos, float* dNewVel, 
								float* dVrts, int* dTris, int tri_num, float dt, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfIntegrateWithPolygon2<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dVel, (float4*)dAcc, 
															 (float4*)dNewPos, (float4*)dNewVel, (float3*)dVrts, (int3*)dTris, tri_num, dt, cell);
	
	RX_CUERROR("pbfIntegrateWithPolygon2 kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}


/*!
 * パーティクル位置，速度の更新
 * @param[in] pos 更新されたパーティクル位置
 * @param[inout] new_pos ステップ最初のパーティクル位置/新しいパーティクル速度
 * @param[out] new_vel 新しいパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuPbfUpdatePosition(float* dPos, float* dNewPos, float* dNewVel, float dt, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfUpdatePosition<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dNewPos, (float4*)dNewVel, dt, nprts);
	
	RX_CUERROR("CupbfUpdatePosition kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクル位置，速度の更新
 * @param[in] pos 更新されたパーティクル位置
 * @param[inout] new_pos ステップ最初のパーティクル位置/新しいパーティクル速度
 * @param[out] new_vel 新しいパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
void CuPbfUpdateVelocity(float* dPos, float* dNewPos, float* dNewVel, float dt, uint nprts)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(nprts, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfUpdateVelocity<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dNewPos, (float4*)dNewVel, dt, nprts);
	
	RX_CUERROR("pbfUpdateVelocity kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * XSPHによる粘性計算
 * @param[in] dPos パーティクル中心座標
 * @param[in] dVel パーティクル速度
 * @param[out] dNewVel 更新されたパーティクル速度
 * @param[in] dDens パーティクル密度
 * @param[in] c 粘性計算用パラメータ
 * @param[in] cell パーティクルグリッドデータ
 */
void CuXSphViscosity(float* dPos, float* dVel, float* dNewVel, float* dDens, float c, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	xsphVisocosity<<< numBlocks, numThreads >>>((float4*)dPos, (float4*)dVel, (float4*)dNewVel, dDens, c, cell);

	RX_CUERROR("xsphVisocosity kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * グリッド上の密度を算出
 * @param[out] dGridD グリッド上の密度値
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] nx,ny,nz グリッド数
 * @param[in] x0,y0,z0 グリッド最小座標
 * @param[in] dx,dy,dz グリッド幅
 */
void CuPbfGridDensity(float *dGridD, rxParticleCell cell, 
					  int nx, int ny, int nz, float x0, float y0, float z0, float dx, float dy, float dz)
{
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
	pbfCalDensityInGrid<<<grid, threads>>>(dGridD, cell, gnum, gmin, glen);

	RX_CUERROR("pbfCalDensityInGrid Kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ
}

/*!
 * パーティクル法線の計算
 * @param[out] dNrms パーティクル法線
 * @param[int] dDens パーティクル密度
 * @param[in]  cell パーティクルグリッドデータ
 */
void CuPbfNormal(float* dNrms, float* dDens, rxParticleCell cell)
{
	// 1スレッド/パーティクル
	uint numThreads, numBlocks;
	computeGridSize(cell.uNumParticles, THREAD_NUM, numBlocks, numThreads);

	// カーネル実行
	pbfCalNormal<<< numBlocks, numThreads >>>((float4*)dNrms, dDens, cell);

	RX_CUERROR("pbfCalNormal kernel execution failed");	// カーネル実行エラーチェック
	RX_CUCHECK(cudaThreadSynchronize());		// 全てのスレッドが終わるのを待つ

}










}   // extern "C"
