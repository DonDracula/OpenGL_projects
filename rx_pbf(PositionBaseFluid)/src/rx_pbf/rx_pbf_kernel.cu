/*! 
  @file rx_pvsph_kernel.cu
	
  @brief CUDAによるPVSPH
 
  @author Makoto Fujisawa
  @date 2014-12
*/

#ifndef _RX_PVSPH_KERNEL_CU_
#define _RX_PVSPH_KERNEL_CU_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_cu_common.cu"

//-----------------------------------------------------------------------------
// ハッシュ
//-----------------------------------------------------------------------------
/*!
 * 各パーティクルのグリッドハッシュ値の計算
 * @param[out] gridParticleHash ハッシュ値
 * @param[out] dSortedIndex パーティクルインデックス値(ソート前の初期値が入れられる)
 * @param[in] pos パーティクル位置を格納した配列
 * @param[in] nprts パーティクル数
 */
__global__
void calcHashD(uint*   dGridParticleHash, 
			   uint*   dSortedIndex, 
			   float4* dPos, 
			   uint	   nprts)
{
	uint index = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	if(index >= nprts) return;
	
	volatile float4 p = dPos[index];
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
	uint hash = calcGridHash(gridPos);

	dGridParticleHash[index] = hash;
	dSortedIndex[index] = index;
}

/*!
 * 各パーティクルのグリッドハッシュ値
 *  - paramsを使わないでグリッド情報を引数で与える
 * @param[out] gridParticleHash パーティクルのグリッドハッシュ値
 * @param[out] dSortedIndex パーティクルインデックス値(ソート前の初期値が入れられる)
 * @param[in] dPos パーティクル位置を格納した配列
 * @param[in] world_origin グリッド最小座標値
 * @param[in] cell_width グリッドのセル幅
 * @param[in] grid_size グリッド数
 * @param[in] nprts パーティクル数
 */
__global__
void calcHashB(uint*   dGridParticleHash, 
			   uint*   dSortedIndex, 
			   float4*  dPos, 
			   float3  world_origin, 
			   float3  cell_width, 
			   uint3   grid_size, 
			   uint	   nprts)
{
	uint index = __umul24(blockIdx.x, blockDim.x)+threadIdx.x;
	if(index >= nprts) return;
	
	float3 p = make_float3(dPos[index]);
	int3 gridPos = calcGridPosB(make_float3(p.x, p.y, p.z), world_origin, cell_width, grid_size);
	uint hash = calcGridHashB(gridPos, grid_size);

	dGridParticleHash[index] = hash;
	dSortedIndex[index] = index;
}


/*!
 * パーティクルデータをソートして，ハッシュ内の各セルの最初のアドレスを検索
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] dSortedPos パーティクル位置
 * @param[in] dSortedVel パーティクル速度
 */
__global__
void reorderDataAndFindCellStartD(rxParticleCell cell, float4* dSortedPos, float4* dSortedVel)
{
	extern __shared__ uint sharedHash[];	// サイズ : blockSize+1
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	
	uint hash;
	if(index < cell.uNumParticles){
		hash = cell.dGridParticleHash[index];	// ハッシュ値

		sharedHash[threadIdx.x+1] = hash;	// ハッシュ値をシェアードメモリに格納

		if(index > 0 && threadIdx.x == 0){
			// 各シェアードメモリの最初は隣のグリッドのパーティクルのハッシュ値を格納
			sharedHash[0] = cell.dGridParticleHash[index-1];
		}
	}

	__syncthreads();
	
	if(index < cell.uNumParticles){
		// インデックス0である，もしくは，一つ前のパーティクルのグリッドハッシュ値が異なる場合，
		// パーティクルは分割領域の最初
		if(index == 0 || hash != sharedHash[threadIdx.x]){
			cell.dCellStart[hash] = index;
			if(index > 0){
				// 一つ前のパーティクルは，一つ前の分割領域の最後
				cell.dCellEnd[sharedHash[threadIdx.x]] = index;
			}
		}

		// インデックスが最後ならば，分割領域の最後
		if(index == cell.uNumParticles-1){
			cell.dCellEnd[hash] = index+1;
		}

		// 位置と速度のデータを並び替え
		// ソートしたインデックスで参照も可能だが探索時のグローバルメモリアクセスを極力抑えるためにデータそのものを並び替える
		uint sortedIndex = cell.dSortedIndex[index];
		float4 pos = FETCH(dSortedPos, sortedIndex);
		float4 vel = FETCH(dSortedVel, sortedIndex);

		cell.dSortedPos[index] = pos;
		cell.dSortedVel[index] = vel;
	}
}

/*!
 * パーティクルデータをソートして，ハッシュ内の各セルの最初のアドレスを検索
 *  - 位置のみ
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] dPos パーティクル位置
 */
__global__
void reorderDataAndFindCellStartB(rxParticleCell cell, float4* dPos)
{
	extern __shared__ uint sharedHash[];	// サイズ : blockSize+1
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	
	uint hash;
	if(index < cell.uNumParticles){
		hash = cell.dGridParticleHash[index];	// ハッシュ値

		sharedHash[threadIdx.x+1] = hash;	// ハッシュ値をシェアードメモリに格納

		if(index > 0 && threadIdx.x == 0){
			// 各シェアードメモリの最初は隣のグリッドのパーティクルのハッシュ値を格納
			sharedHash[0] = cell.dGridParticleHash[index-1];
		}
	}

	__syncthreads();
	
	if(index < cell.uNumParticles){
		// インデックス0である，もしくは，一つ前のパーティクルのグリッドハッシュ値が異なる場合，
		// パーティクルは分割領域の最初
		if(index == 0 || hash != sharedHash[threadIdx.x]){
			cell.dCellStart[hash] = index;
			if(index > 0){
				// 一つ前のパーティクルは，一つ前の分割領域の最後
				cell.dCellEnd[sharedHash[threadIdx.x]] = index;
			}
		}

		// インデックスが最後ならば，分割領域の最後
		if(index == cell.uNumParticles-1){
			cell.dCellEnd[hash] = index+1;
		}

		// 位置と速度のデータを並び替え
		// ソートしたインデックスで参照も可能だが探索時のグローバルメモリアクセスを極力抑えるためにデータそのものを並び替える
		uint sortedIndex = cell.dSortedIndex[index];
		float4 pos = dPos[sortedIndex];
		cell.dSortedPos[index] = pos;
	}
}



//-----------------------------------------------------------------------------
// 境界パーティクル処理カーネル
//-----------------------------------------------------------------------------
/*!
 * 与えられたセル内のパーティクルとの距離から境界パーティクルの体積を計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calBoundaryVolumeCell(int3 gridPos, uint i, float3 pos0, rxParticleCell cell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = cell.dCellStart[gridHash];

	float h = params.EffectiveRadius;
	float mw = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = cell.dCellEnd[gridHash];
		for(uint j = startIndex; j < endIndex; ++j){
			float3 pos1 = make_float3(cell.dSortedPos[j]);

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h){
				float q = h*h-r*r;
				mw += params.Mass*params.Wpoly6*q*q*q;
			}
		}
	}

	return mw;
}

/*!
 * 境界パーティクルの体積計算(カーネル関数)
 * @param[out] newVolB パーティクル体積
 * @param[in]  cell 境界パーティクルグリッドデータ
 */
__global__
void sphCalBoundaryVolume(float* newVolB, rxParticleCell cell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(cell.dSortedPos[index]);	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos1 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);


	// 周囲のグリッドも含めて近傍探索
	float mw = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				mw += calBoundaryVolumeCell(n_grid_pos, index, pos, cell);
			}
		}
	}

	// 体積を結果に書き込み
	uint oIdx = cell.dSortedIndex[index];
	newVolB[oIdx] = params.Mass/mw;
}

/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos0 計算座標
 * @param[in] dVolB 境界パーティクル仮想体積
 * @param[in] bcell 境界パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calBoundaryDensityCell(int3 gridPos, uint i, float3 pos0, float* dVolB, rxParticleCell bcell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = bcell.dCellStart[gridHash];

	float h = params.EffectiveRadius;
	float dens = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = bcell.dCellEnd[gridHash];
		for(uint j = startIndex; j < endIndex; ++j){
			//if(j == i) continue;

			float3 pos1 = make_float3(bcell.dSortedPos[j]);
			uint jdx = bcell.dSortedIndex[j];

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h){
				float q = h*h-r*r;
				dens += params.Density*dVolB[jdx]*params.Wpoly6*q*q*q;
			}
		}
	}

	return dens;
}

/*!
 * 境界パーティクル密度計算(カーネル関数)
 * @param[out] newDens 境界パーティクル密度
 * @param[out] newPres 境界パーティクル圧力 - PBFでは使わない
 * @param[in] dPos  境界パーティクル位置
 * @param[in] dVolB 境界パーティクル仮想体積
 * @param[in] bcell 境界パーティクルグリッドデータ
 * @param[in] pnum  境界パーティクル数
 */
__global__
void sphCalBoundaryDensity(float* newDens, float* newPres, float4* dPos, float* dVolB, rxParticleCell bcell, uint pnum)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= pnum) return;	
	
	float3 pos = make_float3(dPos[index]);	// パーティクル位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos1 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 周囲のグリッドも含めて近傍探索，密度計算
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calBoundaryDensityCell(n_grid_pos, index, pos, dVolB, bcell);
			}
		}
	}

	dens += newDens[index];

	// ガス定数を使った圧力算出
	float pres;
	pres = params.GasStiffness*(dens-params.Density);

	// 密度と圧力値を結果に書き込み
	newDens[index] = dens;
	newPres[index] = pres;
}


/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos0 計算座標
 * @param[in] dVolB 境界パーティクル仮想体積
 * @param[in] dens0 パーティクルiの密度
 * @param[in] pres0 パーティクルiの圧力
 * @param[in] bcell 境界パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した外力
 */
__device__
float3 calBoundaryForceCell(int3 gridPos, uint i, float3 pos0, float* dVolB, float dens0, float pres0, rxParticleCell bcell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = bcell.dCellStart[gridHash];

	float h = params.EffectiveRadius;
	float3 bp = make_float3(0.0f);
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = bcell.dCellEnd[gridHash];
		float prsi = pres0/(dens0*dens0);
		for(uint j = startIndex; j < endIndex; ++j){
			float3 pos1 = make_float3(bcell.dSortedPos[j]);
			uint jdx = bcell.dSortedIndex[j];

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h && r > 0.0001){
				float q = h-r;
				bp += -params.Density*dVolB[jdx]*prsi*params.GWspiky*q*q*rij/r;
			}
		}
	}

	return bp;
}

/*!
 * 境界パーティクルによる力の計算(カーネル関数)
 * @param[in] dDens 境界パーティクル密度
 * @param[in] dPres 境界パーティクル圧力
 * @param[in] dPos  境界パーティクル位置
 * @param[in] dVolB 境界パーティクル仮想体積
 * @param[out] outFrc 外力
 * @param[in] bcell 境界パーティクルグリッドデータ
 * @param[in] pnum  境界パーティクル数
 */
__global__
void sphCalBoundaryForce(float* dDens, float* dPres, float4* dPos, float* dVolB, float4* outFrc, rxParticleCell bcell, uint pnum)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= pnum) return;	
	
	float3 pos = make_float3(dPos[index]);	// パーティクル位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos1 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 密度と圧力
	float dens0 = dDens[index];
	float pres0 = dPres[index];

	// 周囲のグリッドも含めて近傍探索，密度計算
	float3 frc = make_float3(0.0f);
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				frc += calBoundaryForceCell(n_grid_pos, index, pos, dVolB, dens0, pres0, bcell);
			}
		}
	}

	// 密度と圧力値を結果に書き込み
	outFrc[index] += make_float4(frc, 0.0f);
}


//-----------------------------------------------------------------------------
// pbf
//-----------------------------------------------------------------------------
/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos0 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calDensityCellPB(int3 gridPos, uint i, float3 pos0, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;
	float dens = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);
		for(uint j = startIndex; j < endIndex; ++j){
			//if(j == i) continue;

			float3 pos1 = make_float3(FETCHC(dSortedPos, j));

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h){
				float q = h*h-r*r;
				dens += params.Mass*params.Wpoly6*q*q*q;
			}
		}
	}

	return dens;
}



/*!
 * パーティクル密度計算(カーネル関数)
 * @param[out] newDens パーティクル密度
 * @param[in]  cell パーティクルグリッドデータ
 */
__global__
void pbfCalDensity(float* newDens, rxParticleCell cell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，密度計算
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calDensityCellPB(n_grid_pos, index, pos, cell);
			}
		}
	}

	// 密度と圧力値を結果に書き込み
	uint oIdx = cell.dSortedIndex[index];
	newDens[oIdx] = dens;
}

/*!
 * 与えられたセル内のパーティクルとの距離から力場を計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos0 計算座標
 * @param[in] vel0 計算座標の速度
 * @param[in] dens0 計算座標の密度
 * @param[in] dens パーティクル密度
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した力場
 */
__device__
float3 calExtForceCell(int3 gridPos, uint i, float3 pos0, float3 vel0, float dens0, float* dens, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;

	float3 frc = make_float3(0.0f);
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);
		for(uint j = startIndex; j < endIndex; ++j){
			if(j != i){
				// 近傍パーティクルのパラメータ
				float3 pos1 = make_float3(FETCHC(dSortedPos, j));
				float3 vel1 = make_float3(FETCHC(dSortedVel, j));

				float3 rij = pos0-pos1;
				float r = length(rij);

				if(r <= h && r > 0.0001){
					float dens1 = dens[cell.dSortedIndex[j]];

					float3 vij = vel1-vel0;

					float q = h-r;

					// 粘性項
					frc += params.Viscosity*params.Mass*(vij/dens1)*params.LWvisc*q;
				}
			}
		}
	}

	return frc;
}

/*!
 * パーティクルにかかる外力の計算(カーネル関数)
 * @param[in] dens パーティクル密度
 * @param[out] outFrc パーティクルにかかる力
 * @param[in] cell パーティクルグリッドデータ
 */
__global__
void pbfCalExternalForces(float* dens, float4* outFrc, rxParticleCell cell)
{
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	// ソート済み配列からパーティクルデータを取得
	float3 pos0 = make_float3(FETCHC(dSortedPos, index));
	float3 vel0 = make_float3(FETCHC(dSortedVel, index));
	float h = params.EffectiveRadius;

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = cell.dSortedIndex[index];

	float3 frc = make_float3(0.0f);
	float dens0 = dens[oIdx];

	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos0-make_float3(h));
	grid_pos1 = calcGridPos(pos0+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，圧力項，粘性項を計算
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);

				frc += calExtForceCell(n_grid_pos, index, pos0, vel0, dens0, dens, cell);
			}
		}
	}

	// 外力(重力)
	frc += params.Gravity;

	outFrc[oIdx] = make_float4(frc, 0.0f);
}


/*!
 * 与えられたセル内のパーティクルとの距離からスケーリングファクタの分母項計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos0 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calScalingFactorCell(int3 gridPos, uint i, float3 pos0, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;
	float r0 = params.Density;
	float sd = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);
		for(uint j = startIndex; j < endIndex; ++j){
			if(j == i) continue;

			float3 pos1 = make_float3(FETCHC(dSortedPos, j));

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h && r > 0.0){
				float q = h-r;

				// Spikyカーネルで位置変動を計算
				float3 dp = (params.GWspiky*q*q*rij/r)/r0;

				sd += dot(dp, dp);
			}

		}
	}

	return sd;
}

/*!
 * スケーリングファクタの計算
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in] cell パーティクルグリッドデータ
 */
__global__
void pbfCalScalingFactor(float4* ppos, float* pdens, float* pscl, float eps, rxParticleCell cell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置

	float h = params.EffectiveRadius;
	float r0 = params.Density;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，密度計算
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calDensityCellPB(n_grid_pos, index, pos, cell);
			}
		}
	}

	// 密度拘束条件(式(1))
	float C = dens/r0-1.0;

	// 周囲のグリッドも含めて近傍探索，スケーリングファクタの分母項計算
	float sd = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				sd += calScalingFactorCell(n_grid_pos, index, pos, cell);
			}
		}
	}

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = cell.dSortedIndex[index];

	// スケーリングファクタの計算(式(11))
	pscl[oIdx] = -C/(sd+eps);

	// 更新された密度
	pdens[oIdx] = dens;
}


/*!
 * 与えられたセル内のパーティクルとの距離からスケーリングファクタの分母項計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float3 calPositionCorrectionCell(int3 gridPos, uint i, float3 pos0, float* pscl, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float k = params.AP_K;
	float n = params.AP_N;
	float wq = params.AP_WQ;

	float h = params.EffectiveRadius;
	float r0 = params.Density;
	float3 dp = make_float3(0.0);

	float dt = params.Dt;

	float si = pscl[cell.dSortedIndex[i]];

	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);
		for(uint j = startIndex; j < endIndex; ++j){
			if(j == i) continue;

			float3 pos1 = make_float3(FETCHC(dSortedPos, j));

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h && r > 0.0){
				float scorr = 0.0f;

				if(params.AP){
					float q1 = h*h-r*r;
					float ww = params.Wpoly6*q1*q1*q1/wq;
					scorr = -k*pow(ww, n)*dt*dt;
				}
				float q = h-r;
				float sj = pscl[cell.dSortedIndex[j]];

				// Spikyカーネルで位置修正量を計算
				dp += (si+sj+scorr)*(params.GWspiky*q*q*rij/r)/r0;
			}

		}
	}

	return dp;
}

/*!
 * スケーリングファクタの計算
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in]  cell パーティクルグリッドデータ
 */
__global__
void pbfPositionCorrection(float4* ppos, float* pscl, float4* pdp, rxParticleCell cell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置

	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，位置修正量を計算
	float3 dpij = make_float3(0.0f);
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dpij += calPositionCorrectionCell(n_grid_pos, index, pos, pscl, cell);
			}
		}
	}

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = cell.dSortedIndex[index];

	// 位置修正量
	pdp[oIdx] = make_float4(dpij, 0.0);
}

/*!
 * パーティクル位置修正
 * @param[inout] pos パーティクル位置
 * @param[in] pdp 位置修正量
 * @param[in] nprts パーティクル数
 */
__global__
void pbfCorrectPosition(float4* ppos, float4* pdp, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	// 位置修正
	ppos[index] += pdp[index];
}

/*!
 * 密度変動の計算
 * @param[inout] pos パーティクル位置
 * @param[in] pdp 位置修正量
 * @param[in] nprts パーティクル数
 */
__global__
void pbfDensityFluctuation(float* perr, float* pdens, float rest_dens, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	// 密度変動
	//perr[index] = fabs(pdens[index]-rest_dens)/rest_dens;
	float err = pdens[index]-rest_dens;
	perr[index] = (err >= 0.0f ? err : 0.0f)/rest_dens;
}




/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calBoundaryDensityCellPB(int3 gridPos, uint i, float3 pos0, float* dVolB, rxParticleCell bcell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = bcell.dCellStart[gridHash];

	float h = params.EffectiveRadius;
	float dens = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = bcell.dCellEnd[gridHash];
		for(uint j = startIndex; j < endIndex; ++j){
			//if(j == i) continue;

			float3 pos1 = make_float3(bcell.dSortedPos[j]);
			uint jdx = bcell.dSortedIndex[j];

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h){
				float q = h*h-r*r;
				dens += params.Density*dVolB[jdx]*params.Wpoly6*q*q*q;
			}
		}
	}

	return dens;
}

/*!
 * パーティクル密度計算(カーネル関数)
 * @param[out] newDens パーティクル密度
 * @param[out] newPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
__global__
void pbfCalBoundaryDensity(float* newDens, float4* dPos, float* dVolB, rxParticleCell bcell, uint pnum)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= pnum) return;	
	
	float3 pos = make_float3(dPos[index]);	// パーティクル位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos1 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 周囲のグリッドも含めて近傍探索，密度計算
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calBoundaryDensityCellPB(n_grid_pos, index, pos, dVolB, bcell);
			}
		}
	}

	// 密度を結果に書き込み
	newDens[index] += dens;
}




/*!
 * 与えられたセル内のパーティクルとの距離からスケーリングファクタの分母項計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calBoundaryScalingFactorCell(int3 gridPos, uint i, float3 pos0, float* dVolB, rxParticleCell bcell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = bcell.dCellStart[gridHash];

	float h = params.EffectiveRadius;
	float r0 = params.Density;
	float sd = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = bcell.dCellEnd[gridHash];
		for(uint j = startIndex; j < endIndex; ++j){
			float3 pos1 = make_float3(bcell.dSortedPos[j]);
			uint jdx = bcell.dSortedIndex[j];

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h && r > 0.0){
				float q = h-r;

				// Spikyカーネルで位置変動を計算
				float3 dp = (params.Density*dVolB[jdx]/params.Mass)*(params.GWspiky*q*q*rij/r)/r0;

				sd += dot(dp, dp);
			}

		}
	}

	return sd;
}

/*!
 * スケーリングファクタの計算(境界パーティクル含む)
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in] cell パーティクルグリッドデータ
 */
__global__
void pbfCalScalingFactorWithBoundary(float4* ppos, float* pdens, float* pscl, float eps, rxParticleCell cell, 
										float* bvol, rxParticleCell bcell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置

	float h = params.EffectiveRadius;
	float r0 = params.Density;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 流体パーティクルによる密度
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calDensityCellPB(n_grid_pos, index, pos, cell);
			}
		}
	}

	// パーティクル周囲のグリッド
	int3 grid_pos2, grid_pos3;
	grid_pos2 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos3 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 境界パーティクルによる密度
	for(int z = grid_pos2.z; z <= grid_pos3.z; ++z){
		for(int y = grid_pos2.y; y <= grid_pos3.y; ++y){
			for(int x = grid_pos2.x; x <= grid_pos3.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calBoundaryDensityCellPB(n_grid_pos, index, pos, bvol, bcell);
			}
		}
	}

	// 密度拘束条件(式(1))
	float C = dens/r0-1.0;

	// 流体パーティクルによるスケーリングファクタの分母項計算
	float sd = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				sd += calScalingFactorCell(n_grid_pos, index, pos, cell);
			}
		}
	}

	// 境界パーティクルによるスケーリングファクタの分母項計算
	for(int z = grid_pos2.z; z <= grid_pos3.z; ++z){
		for(int y = grid_pos2.y; y <= grid_pos3.y; ++y){
			for(int x = grid_pos2.x; x <= grid_pos3.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				sd += calBoundaryScalingFactorCell(n_grid_pos, index, pos, bvol, bcell);
			}
		}
	}

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = cell.dSortedIndex[index];

	// スケーリングファクタの計算(式(11))
	pscl[oIdx] = -C/(sd+eps);

	// 更新された密度
	pdens[oIdx] = dens;
}



/*!
 * スケーリングファクタの計算(境界パーティクル含む)
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in] cell パーティクルグリッドデータ
 */
__global__
void pbfCalBoundaryScalingFactor(float4* ppos, float* pdens, float eps, rxParticleCell cell, 
									float* bvol, float* bscl, rxParticleCell bcell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= bcell.uNumParticles) return;	
	
	float3 pos = make_float3(bcell.dSortedPos[index]);	// パーティクル位置

	float h = params.EffectiveRadius;
	float r0 = params.Density;

	// パーティクル周囲のグリッド(流体パーティクル用)
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 流体パーティクルによる密度
	float dens = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calDensityCellPB(n_grid_pos, index, pos, cell);
			}
		}
	}

	// パーティクル周囲のグリッド(境界パーティクル用)
	int3 grid_pos2, grid_pos3;
	grid_pos2 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos3 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 境界パーティクルによる密度
	for(int z = grid_pos2.z; z <= grid_pos3.z; ++z){
		for(int y = grid_pos2.y; y <= grid_pos3.y; ++y){
			for(int x = grid_pos2.x; x <= grid_pos3.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dens += calBoundaryDensityCellPB(n_grid_pos, index, pos, bvol, bcell);
			}
		}
	}

	// 密度拘束条件(式(1))
	float C = dens/r0-1.0;

	// 流体パーティクルによるスケーリングファクタの分母項計算
	float sd = 0.0f;
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				sd += calScalingFactorCell(n_grid_pos, index, pos, cell);
			}
		}
	}

	// 境界パーティクルによるスケーリングファクタの分母項計算
	for(int z = grid_pos2.z; z <= grid_pos3.z; ++z){
		for(int y = grid_pos2.y; y <= grid_pos3.y; ++y){
			for(int x = grid_pos2.x; x <= grid_pos3.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				sd += calBoundaryScalingFactorCell(n_grid_pos, index, pos, bvol, bcell);
			}
		}
	}

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = bcell.dSortedIndex[index];

	// スケーリングファクタの計算(式(11))
	bscl[oIdx] = -C/(sd+eps);
}



/*!
 * 与えられたセル内のパーティクルとの距離からスケーリングファクタの分母項計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float3 calBoundaryPositionCorrectionCell(int3 gridPos, uint i, float3 pos0, float si, float* bscl, float* bvol, rxParticleCell bcell)
{
	uint gridHash = calcGridHashB(gridPos, params.GridSizeB);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = bcell.dCellStart[gridHash];

	float k = params.AP_K;
	float n = params.AP_N;
	float wq = params.AP_WQ;

	float h = params.EffectiveRadius;
	float r0 = params.Density;
	float3 dp = make_float3(0.0);

	float dt = params.Dt;

	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = bcell.dCellEnd[gridHash];
		for(uint j = startIndex; j < endIndex; ++j){
			float3 pos1 = make_float3(bcell.dSortedPos[j]);
			uint jdx = bcell.dSortedIndex[j];

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h && r > 0.0){
				float scorr = 0.0f;

				if(params.AP){
					float q1 = h*h-r*r;
					float ww = (params.Density*bvol[jdx]/params.Mass)*params.Wpoly6*q1*q1*q1/wq;
					scorr = -k*pow(ww, n)*dt*dt;
				}
				float q = h-r;
				float sj = bscl[jdx];

				// Spikyカーネルで位置修正量を計算
				dp += (si+sj+scorr)*(params.GWspiky*q*q*rij/r)/r0;
			}

		}
	}

	return dp;
}

/*!
 * スケーリングファクタの計算
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] eps 緩和係数
 * @param[in]  cell パーティクルグリッドデータ
 */
__global__
void pbfPositionCorrectionWithBoundary(float4* ppos, float* pscl, float4* pdp, rxParticleCell cell, 
										  float* bvol, float* bscl, rxParticleCell bcell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置

	float h = params.EffectiveRadius;

	float si = pscl[cell.dSortedIndex[index]];


	// パーティクル周囲のグリッド(流体パーティクル用)
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 流体パーティクルによる位置修正量を計算
	float3 dpij = make_float3(0.0f);
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dpij += calPositionCorrectionCell(n_grid_pos, index, pos, pscl, cell);
			}
		}
	}

	// パーティクル周囲のグリッド(境界パーティクル用)
	int3 grid_pos2, grid_pos3;
	grid_pos2 = calcGridPosB(pos-make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);
	grid_pos3 = calcGridPosB(pos+make_float3(h), params.WorldOriginB, params.CellWidthB, params.GridSizeB);

	// 境界パーティクルによる位置修正量を計算
	for(int z = grid_pos2.z; z <= grid_pos3.z; ++z){
		for(int y = grid_pos2.y; y <= grid_pos3.y; ++y){
			for(int x = grid_pos2.x; x <= grid_pos3.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				dpij += calBoundaryPositionCorrectionCell(n_grid_pos, index, pos, si, bscl, bvol, bcell);
			}
		}
	}

	// パーティクルのソートなし配列上でのインデックス
	uint oIdx = cell.dSortedIndex[index];

	// 位置修正量
	pdp[oIdx] = make_float4(dpij, 0.0);
}


__device__
void calCollisionSolidPB(float3 &pos, float3 &vel, float dt)
{
	float d;
	float3 n;
	float3 cp;

	// ボックス形状のオブジェクトとの衝突
#if MAX_BOX_NUM
	for(int i = 0; i < params.BoxNum; ++i){
		if(params.BoxFlg[i] == 0) continue;
		
		collisionPointBox(pos, params.BoxCen[i], params.BoxExt[i], params.BoxRot[i], params.BoxInvRot[i], cp, d, n);

		if(d < 0.0){
			float res = params.Restitution;
			res = (res > 0) ? (res*fabs(d)/(dt*length(vel))) : 0.0f;
			vel -= (1+res)*n*dot(n, vel);
			pos = cp;
		}
	}
#endif

	// 球形状のオブジェクトとの衝突
#if MAX_SPHERE_NUM
	for(int i = 0; i < params.SphereNum; ++i){
		if(params.SphereFlg[i] == 0) continue;

		collisionPointSphere(pos, params.SphereCen[i], params.SphereRad[i], cp, d, n);

		if(d < 0.0){
			float res = params.Restitution;
			res = (res > 0) ? (res*fabs(d)/(dt*length(vel))) : 0.0f;
			vel -= (1+res)*n*dot(n, vel);
			pos = cp;
		}
	}
#endif

	// 周囲の境界との衝突判定
	float3 l0 = params.Boundary[0];
	float3 l1 = params.Boundary[1];
	collisionPointAABB(pos, 0.5*(l1+l0), 0.5*(l1-l0), cp, d, n);

	if(d < 0.0){
		float res = params.Restitution;
		res = (res > 0) ? (res*fabs(d)/(dt*length(vel))) : 0.0f;
		vel -= (1+res)*n*dot(n, vel);
		pos = cp;
	}
}

__device__
inline bool calCollisionPolygonPB(float3 &pos0, float3 &pos1, float3 &vel, float3 v0, float3 v1, float3 v2, float dt)
{
	float3 cp, n;
	if(intersectSegmentTriangle(pos0, pos1, v0, v1, v2, cp, n, params.ParticleRadius) == 1){
		float d = length(pos1-cp);
		n = normalize(n);

		//float res = params.Restitution;
		//res = (res > 0) ? (res*fabs(d)/(dt*length(vel))) : 0.0f;
		//float3 vr = -(1+res)*n*dot(n, vel);

		float3 v = pos1-pos0;
		float l = length(v);
		v /= l;
		float3 vd = v*(l-d);
		float3 vr = vd-2*dot(n, vd)*n;

		pos1 = cp+vr*0.7;

		//vel += vr;//+params.PolyVel[0];
		//vel.x = 1.0;

		return true;
	}
	return false;
}




/*!
 * パーティクル位置，速度の更新
 * @param[inout] ppos パーティクル位置
 * @param[inout] pvel パーティクル速度
 * @param[in] pfrc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfIntegrate(float4* ppos, float4* pvel, float4* pacc, 
					 float4* new_ppos, float4* new_pvel, float dt, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	float3 x = make_float3(ppos[index]);
	float3 v = make_float3(pvel[index]);
	float3 a = make_float3(pacc[index]);
	//float3 v_old = v;

	// 更新位置，速度の更新
	v += dt*a;
	x += dt*v;

	// 固体・境界との衝突
	calCollisionSolidPB(x, v, dt);

	// 位置と速度の更新
	new_ppos[index] = make_float4(x);
	new_pvel[index] = make_float4(v);
}



/*!
 * パーティクル位置，速度の更新(Leap-Frog)
 * @param[inout] ppos パーティクル位置
 * @param[inout] pvel パーティクル速度
 * @param[in] pfrc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] vrts
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfIntegrateWithPolygon(float4* ppos, float4* pvel, float4* pacc, 
								float4* new_ppos, float4* new_pvel, 
								float3* vrts, int3* tris, int tri_num, float dt, rxParticleCell cell)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;

	float3 x = make_float3(ppos[index]);
	float3 v = make_float3(pvel[index]);
	float3 a = make_float3(pacc[index]);
	//float3 v_old = v;
	float3 x_old = x;

	// 更新位置，速度の更新
	v += dt*a;
	x += dt*v;

	// ポリゴンオブジェクトとの衝突
	int3 gridPos[2];
	gridPos[0] = calcGridPos(x_old);	// 位置更新前のパーティクルが属するグリッド
	gridPos[1] = calcGridPos(x);		// 位置更新後のパーティクルが属するグリッド
	for(int i = 0; i < 2; ++i){
		uint grid_hash = calcGridHash(gridPos[i]);
		uint start_index = cell.dPolyCellStart[grid_hash];
		if(start_index != 0xffffffff){	// セルが空でないかのチェック

			uint end_index = cell.dPolyCellEnd[grid_hash];
			for(uint j = start_index; j < end_index; ++j){
				uint pidx = cell.dSortedPolyIdx[j];

				int3 idx = tris[pidx];
				if(calCollisionPolygonPB(x_old, x, v, vrts[idx.x], vrts[idx.y], vrts[idx.z], dt)){
				}
			}
		}
	}

	// 固体・境界との衝突
	calCollisionSolidPB(x, v, dt);

	// 位置と速度の更新
	new_ppos[index] = make_float4(x);
	new_pvel[index] = make_float4(v);
}



/*!
 * パーティクル位置，速度の更新
 * @param[inout] ppos パーティクル位置
 * @param[inout] pvel パーティクル速度
 * @param[in] pfrc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfIntegrate2(float4* ppos, float4* pvel, float4* pacc, 
					  float4* new_ppos, float4* new_pvel, float dt, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	float3 x = make_float3(new_ppos[index]);
	float3 v = make_float3(new_pvel[index]);

	// 固体・境界との衝突
	calCollisionSolidPB(x, v, dt);

	// 位置と速度の更新
	new_ppos[index] = make_float4(x);
}



/*!
 * パーティクル位置，速度の更新(Leap-Frog)
 * @param[inout] ppos パーティクル位置
 * @param[inout] pvel パーティクル速度
 * @param[in] pfrc パーティクルにかかる力
 * @param[in] dens パーティクル密度
 * @param[in] vrts
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfIntegrateWithPolygon2(float4* ppos, float4* pvel, float4* pacc, 
							  float4* new_ppos, float4* new_pvel, 
							  float3* vrts, int3* tris, int tri_num, float dt, rxParticleCell cell)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;

	float3 x = make_float3(new_ppos[index]);
	float3 x_old = make_float3(ppos[index]);
	float3 v = make_float3(new_pvel[index]);

	// ポリゴンオブジェクトとの衝突
	int3 gridPos[2];
	gridPos[0] = calcGridPos(x_old);	// 位置更新前のパーティクルが属するグリッド
	gridPos[1] = calcGridPos(x);		// 位置更新後のパーティクルが属するグリッド
	for(int i = 0; i < 2; ++i){
		uint grid_hash = calcGridHash(gridPos[i]);
		uint start_index = cell.dPolyCellStart[grid_hash];
		if(start_index != 0xffffffff){	// セルが空でないかのチェック

			uint end_index = cell.dPolyCellEnd[grid_hash];
			for(uint j = start_index; j < end_index; ++j){
				uint pidx = cell.dSortedPolyIdx[j];

				int3 idx = tris[pidx];
				if(calCollisionPolygonPB(x_old, x, v, vrts[idx.x], vrts[idx.y], vrts[idx.z], dt)){
				}
			}
		}
	}

	// 固体・境界との衝突
	calCollisionSolidPB(x, v, dt);

	// 位置と速度の更新
	new_ppos[index] = make_float4(x);
	new_pvel[index] = make_float4(v);
}


/*!
 * パーティクル位置，速度の更新
 * @param[in] ppos 更新されたパーティクル位置
 * @param[inout] new_ppos ステップ最初のパーティクル位置/新しいパーティクル速度
 * @param[out] new_pvel 新しいパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfUpdatePosition(float4* ppos, float4* new_ppos, float4* new_pvel, float dt, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	float3 x0 = make_float3(new_ppos[index]);
	float3 x1 = make_float3(ppos[index]);
	float3 v = (x1-x0)/dt;

	// 位置と速度の更新
	new_pvel[index] = make_float4(v);
	new_ppos[index] = make_float4(x1);
}

/*!
 * パーティクル速度の更新
 * @param[in] ppos 更新されたパーティクル位置
 * @param[in] new_ppos ステップ最初のパーティクル位置/新しいパーティクル速度
 * @param[out] new_pvel 新しいパーティクル速度
 * @param[in] dt 時間ステップ幅
 * @param[in] nprts パーティクル数
 */
__global__
void pbfUpdateVelocity(float4* ppos, float4* new_ppos, float4* new_pvel, float dt, uint nprts)
{
	uint index = __umul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= nprts) return;

	float3 x0 = make_float3(new_ppos[index]);
	float3 x1 = make_float3(ppos[index]);
	float3 v = (x1-x0)/dt;

	// 位置と速度の更新
	new_pvel[index] = make_float4(v);
}



/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] index パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float3 calXsphViscosityCell(int3 gridPos, uint i, float3 pos0, float3 vel0, float4* pvel, float* dens, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;
	float3 v = make_float3(0.0);
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);
		for(uint j = startIndex; j < endIndex; ++j){
			//if(j == i) continue;

			float3 pos1 = make_float3(FETCHC(dSortedPos, j));

			float3 rij = pos0-pos1;
			float r = length(rij);

			if(r <= h){
				float3 vel1 = make_float3(pvel[cell.dSortedIndex[j]]);
				float3 rho1 = make_float3(dens[cell.dSortedIndex[j]]);

				float q = h*h-r*r;
				v += (params.Mass/rho1)*(vel1-vel0)*params.Wpoly6*q*q*q;
			}
		}
	}

	return v;
}

/*!
 * パーティクル密度計算(カーネル関数)
 * @param[out] newDens パーティクル密度
 * @param[out] newPres パーティクル圧力
 * @param[in]  cell パーティクルグリッドデータ
 */
__global__
void xsphVisocosity(float4* ppos, float4* pvel, float4* new_pvel, float* dens, float c, rxParticleCell cell)
{
	// パーティクルインデックス
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos0 = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	float3 vel0 = make_float3(pvel[cell.dSortedIndex[index]]);	// パーティクル速度
	//int3 grid_pos = calcGridPos(pos0);	// パーティクルが属するグリッド位置
	float h = params.EffectiveRadius;

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos0-make_float3(h));
	grid_pos1 = calcGridPos(pos0+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，密度計算
	float3 v = make_float3(0.0);
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				v += calXsphViscosityCell(n_grid_pos, index, pos0, vel0, pvel, dens, cell);
			}
		}
	}

	// 密度と圧力値を結果に書き込み
	uint oIdx = cell.dSortedIndex[index];
	new_pvel[oIdx] = make_float4(vel0+c*v);
	//new_pvel[oIdx] = make_float4(vel0);
}




/*!
 * 与えられたセル内のパーティクルとの距離から密度を計算
 * @param[in] gridPos グリッド位置
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float calDensityCellGPB(int3 gridPos, float3 pos0, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;
	float d = 0.0f;
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);

		for(uint j = startIndex; j < endIndex; ++j){
			//if(j != index){
				float3 pos1 = make_float3(FETCHC(dSortedPos, j));

				float3 rij = pos0-pos1;
				float r = length(rij);

				if(r <= h){
					float q = h*h-r*r;

					d += params.Mass*params.Wpoly6*q*q*q;
				}

			//}
		}
	}

	return d;
}

/*!
 * グリッド上での密度を計算
 * @param[out] GridD グリッド密度
 * @param[in] cell パーティクルグリッドデータ
 * @param[in] gnum グリッド数
 * @param[in] gmin グリッド最小座標
 * @param[in] glen グリッド幅
 */
__global__
void pbfCalDensityInGrid(float* GridD, rxParticleCell cell, 
					uint3 gnum, float3 gmin, float3 glen)
{
	uint blockId = __mul24(blockIdx.y, gridDim.x)+blockIdx.x;
	uint i = __mul24(blockId, blockDim.x)+threadIdx.x;

	uint3 gridPos = calcGridPosU(i, gnum);

	if(gridPos.x < gnum.x && gridPos.y < gnum.y && gridPos.z < gnum.z){
		float3 gpos;
		gpos.x = gmin.x+(gridPos.x)*glen.x;
		gpos.y = gmin.y+(gridPos.y)*glen.y;
		gpos.z = gmin.z+(gridPos.z)*glen.z;

		float d = 0.0f;

		int3 pgpos = calcGridPos(gpos);

		float h = params.EffectiveRadius;
		int3 grid_pos0, grid_pos1;
		grid_pos0 = calcGridPos(gpos-make_float3(h));
		grid_pos1 = calcGridPos(gpos+make_float3(h));

		for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
			for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
				for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
					int3 neighbourPos = make_int3(x, y, z);

					d += calDensityCellGPB(neighbourPos, gpos, cell);
				}
			}
		}

		GridD[gridPos.x+gridPos.y*gnum.x+gridPos.z*gnum.x*gnum.y] = d;
	}

}

/*!
 * 与えられたセル内のパーティクルとの距離から法線を計算
 * @param[in] gridPos グリッド位置
 * @param[in] i パーティクルインデックス
 * @param[in] pos 計算座標
 * @param[in] cell パーティクルグリッドデータ
 * @return セル内のパーティクルから計算した密度値
 */
__device__
float3 calNormalCellPB(int3 gridPos, uint i, float3 pos0, float* dens, rxParticleCell cell)
{
	uint gridHash = calcGridHash(gridPos);

	// セル内のパーティクルのスタートインデックス
	uint startIndex = FETCHC(dCellStart, gridHash);

	float h = params.EffectiveRadius;
	float3 nrm = make_float3(0.0f);
	if(startIndex != 0xffffffff){	// セルが空でないかのチェック
		// セル内のパーティクルで反復
		uint endIndex = FETCHC(dCellEnd, gridHash);

		for(uint j = startIndex; j < endIndex; ++j){
			if(j != i){
				float3 pos1 = make_float3(FETCHC(dSortedPos, j));

				float3 rij = pos0-pos1;
				float r = length(rij);

				if(r <= h && r > 0.0001){
					float d1 = dens[cell.dSortedIndex[j]];
					float q = h*h-r*r;

					nrm += (params.Mass/d1)*params.GWpoly6*q*q*rij;
				}

			}
		}
	}

	return nrm;
}


/*!
 * パーティクル法線計算(カーネル関数)
 * @param[out] newNrms パーティクル法線
 * @param[in] dens パーティクル密度
 * @param[in] cell パーティクルグリッドデータ
 */
__global__
void pbfCalNormal(float4* newNrms, float* dens, rxParticleCell cell)
{
	uint index = __mul24(blockIdx.x,blockDim.x)+threadIdx.x;
	if(index >= cell.uNumParticles) return;	
	
	float3 pos = make_float3(FETCHC(dSortedPos, index));	// パーティクル位置
	float h = params.EffectiveRadius;
	//int3 grid_pos = calcGridPos(pos);	// パーティクルが属するグリッド位置

	// パーティクル周囲のグリッド
	int3 grid_pos0, grid_pos1;
	grid_pos0 = calcGridPos(pos-make_float3(h));
	grid_pos1 = calcGridPos(pos+make_float3(h));

	// 周囲のグリッドも含めて近傍探索，密度計算
	float3 nrm = make_float3(0.0f);
	for(int z = grid_pos0.z; z <= grid_pos1.z; ++z){
		for(int y = grid_pos0.y; y <= grid_pos1.y; ++y){
			for(int x = grid_pos0.x; x <= grid_pos1.x; ++x){
				int3 n_grid_pos = make_int3(x, y, z);
				nrm += calNormalCellPB(n_grid_pos, index, pos, dens, cell);
			}
		}
	}

	float l = length(nrm);
	if(l > 0){
		nrm /= l;
	}

	uint oIdx = cell.dSortedIndex[index];
	newNrms[oIdx] = make_float4(nrm, 0.0f);
}





#endif // #ifndef _RX_PVSPH_KERNEL_CU_



