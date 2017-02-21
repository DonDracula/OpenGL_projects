/*! 
  @file rx_mc.cu
	
  @brief CUDAによるメッシュ生成(MC法)

  ボリュームデータからMC(Marching Cube)法を用いて閾値に基づく表面を抽出する．
  並列に処理するためにScan(Prefix Sum)を利用し，
  Scanの高速処理にはCUDPPを用いる．
  
  MC法に関する情報
  http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
  http://en.wikipedia.org/wiki/Marching_cubes
  
  MC法のテーブル
  http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
  
  CUDPP(CUDA Data Parallel Primitives Library)に関する情報
  http://www.gpgpu.org/developer/cudpp

*/
// FILE --rx_mc.cu--


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <GL/glew.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <GL/freeglut.h>
#include "rx_mc_kernel.cu"
#include "rx_mc_tables.h"

#include "rx_cu_funcs.cuh"



//-----------------------------------------------------------------------------
// MARK:グローバル変数

// MC法のテーブル
uint* g_puNumVrtsTable = 0;		//!< ボクセル内の頂点数テーブル
uint* g_puEdgeTable = 0;		//!< ボクセル内のエッジテーブル
uint* g_puTriTable = 0;			//!< ボクセル内のメッシュ構成テーブル


//-----------------------------------------------------------------------------
// CUDA関数
//-----------------------------------------------------------------------------
extern "C"
{

/*!
 * デバイスメモリの中身を画面出力
 * @param[in] d_buffer デバイスメモリポインタ
 * @param[in] nelements データ数
 */
void DumpBuffer(uint *d_buffer, int nelements)
{
	uint bytes = nelements*sizeof(uint);
	uint *h_buffer = (uint *) malloc(bytes);
	RX_CUCHECK(cudaMemcpy(h_buffer, d_buffer, bytes, cudaMemcpyDeviceToHost) );
	for(int i=0; i<nelements; i++) {
		printf("%d: %u\n", i, h_buffer[i]);
	}
	printf("\n");
	free(h_buffer);
}

/*!
 * MC法のテーブル
 */
void CuInitMCTable(void)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);

	RX_CUCHECK(cudaMalloc((void**) &g_puEdgeTable, 256*sizeof(uint)));
	RX_CUCHECK(cudaMemcpy(g_puEdgeTable, edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
	RX_CUCHECK(cudaBindTexture(0, edgeTex, g_puEdgeTable, channelDesc) );

	RX_CUCHECK(cudaMalloc((void**) &g_puTriTable, 256*16*sizeof(uint)));
	RX_CUCHECK(cudaMemcpy(g_puTriTable, triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice) );
	RX_CUCHECK(cudaBindTexture(0, triTex, g_puTriTable, channelDesc) );

	RX_CUCHECK(cudaMalloc((void**) &g_puNumVrtsTable, 256*sizeof(uint)));
	RX_CUCHECK(cudaMemcpy(g_puNumVrtsTable, numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
	RX_CUCHECK(cudaBindTexture(0, numVertsTex, g_puNumVrtsTable, channelDesc) );
}

/*!
 * MC法のテーブルの破棄
 */
void CuCleanMCTable(void)
{
	RX_CUCHECK(cudaFree(g_puEdgeTable));
	RX_CUCHECK(cudaFree(g_puTriTable));
	RX_CUCHECK(cudaFree(g_puNumVrtsTable));
}



#define DEBUG_BUFFERS 0


#ifdef RX_CUMC_USE_GEOMETRY

/*!
 * 各ボクセルの8頂点の内外を判定し，テーブルから頂点数，ポリゴン数を求める
 * @param[in] dVolume 陰関数データを格納するグリッド
 * @param[out] dVoxBit グリッド8頂点の陰関数値が閾値以上かどうかを各ビットに格納した変数
 * @param[out] dVoxVNum グリッドに含まれるメッシュ頂点数
 * @param[out] dVoxVNumScan グリッドに含まれるメッシュ頂点数のScan
 * @param[out] dVoxTNum グリッドごとの三角形メッシュ数
 * @param[out] dVoxTNumScan グリッドごとの三角形メッシュ数のScan
 * @param[out] dVoxOcc グリッドごとのメッシュ存在情報(メッシュがあれば1,そうでなければ0)
 * @param[out] dVoxOccScan グリッドごとのメッシュ存在情報のScan
 * @param[out] dCompactedVox メッシュを含むグリッド情報
 * @param[in] grid_size  グリッド数(nx,ny,nz)
 * @param[in] num_voxels 総グリッド数
 * @param[in] grid_width グリッド幅
 * @param[in] threshold  メッシュ化閾値
 * @param[out] num_active_voxels メッシュを含むボクセルの数
 * @param[out] nvrts 頂点数
 * @param[out] ntris メッシュ数
*/
void CuMCCalTriNum(float *dVolume, uint *dVoxBit, uint *dVoxVNum, uint *dVoxVNumScan, 
				   uint *dVoxTNum, uint *dVoxTNumScan, uint *dVoxOcc, uint *dVoxOccScan, uint *dCompactedVox, 
				   uint3 grid_size, uint num_voxels, float3 grid_width, float threshold, 
				   uint &num_active_voxels, uint &nvrts, uint &ntris)
{
	uint lval, lsval;

	// 1スレッド/ボクセル
	int threads = THREAD_NUM;
	dim3 grid((num_voxels+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// 各ボクセルの8頂点の内外を判定し，テーブルから頂点数，ポリゴン数を求める
	ClassifyVoxel2<<<grid, threads>>>(dVoxBit, dVoxVNum, dVoxTNum, dVoxOcc, 
									  dVolume, grid_size, num_voxels, grid_width, threshold);
	RX_CUERROR("ClassifyVoxel2 failed");

	num_active_voxels = num_voxels;

#if SKIP_EMPTY_VOXELS
	// 空のグリッドをスキップする場合

	// グリッド内のメッシュ有無配列をScan
	CuScan(dVoxOccScan, dVoxOcc, num_voxels);

	// Exclusive scan (最後の要素が0番目からn-2番目までの合計になっている)なので，
	// Scan前配列の最後(n-1番目)の要素と合計することでグリッド数を計算
	RX_CUCHECK(cudaMemcpy((void*)&lval, (void*)(dVoxOcc+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	RX_CUCHECK(cudaMemcpy((void*)&lsval, (void*)(dVoxOccScan+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	num_active_voxels = lval+lsval;

	if(!num_active_voxels){
		nvrts = 0; ntris = 0;
		return;
	}

	CompactVoxels<<<grid, threads>>>(dCompactedVox, dVoxOcc, dVoxOccScan, num_voxels);
	RX_CUERROR("CompactVoxels failed");

#endif // #if SKIP_EMPTY_VOXELS

	// バーテックスカウント用Scan(prefix sum)作成
	CuScan(dVoxVNumScan, dVoxVNum, num_voxels);

	// 三角形メッシュ数情報をScan(prefix sum)
	CuScan(dVoxTNumScan, dVoxTNum, num_voxels);

	// Exclusive scan (最後の要素が0番目からn-2番目までの合計になっている)なので，
	// Scan前配列の最後(n-1番目)の要素と合計することでポリゴン数を計算
	RX_CUCHECK(cudaMemcpy((void*)&lval, (void*)(dVoxVNum+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	RX_CUCHECK(cudaMemcpy((void*)&lsval, (void*)(dVoxVNumScan+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	nvrts = lval+lsval;

	// Exclusive scan (最後の要素が0番目からn-2番目までの合計になっている)なので，
	// Scan前配列の最後(n-1番目)の要素と合計することでポリゴン数を計算
	RX_CUCHECK(cudaMemcpy((void*)&lval, (void*)(dVoxTNum+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	RX_CUCHECK(cudaMemcpy((void*)&lsval, (void*)(dVoxTNumScan+num_voxels-1), sizeof(uint), cudaMemcpyDeviceToHost));
	ntris = lval+lsval;
}
	

/*!
 * エッジごとに頂点座標を計算
 * @param[in] dVolume 陰関数データを格納するグリッド
 * @param[in] dNoise ノイズデータを格納するグリッド
 * @param[out] dEdgeVrts エッジごとに算出した頂点情報
 * @param[out] dCompactedEdgeVrts 隙間をつめた頂点情報
 * @param[out] dEdgeOcc エッジにメッシュ頂点を含むかどうか(x方向，y方向, z方向の順)
 * @param[out] dEdgeOccScan エッジにメッシュ頂点を含むかどうかのScan
 * @param[in] edge_size[3] エッジ数(nx,ny,nz)
 * @param[in] num_edge[4] 総エッジ数
 * @param[in] grid_size  グリッド数(nx,ny,nz)
 * @param[in] num_voxels 総グリッド数
 * @param[in] grid_width グリッド幅
 * @param[in] threshold  メッシュ化閾値
 * @param[inout] nvrts 頂点数
 */
void CuMCCalEdgeVrts(float *dVolume, float *dEdgeVrts, float *dCompactedEdgeVrts, 
					 uint *dEdgeOcc, uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4], 
					 uint3 grid_size, uint num_voxels, float3 grid_width, float3 grid_min, float threshold, 
					 uint &nvrts)
{
	uint lval, lsval;
	//
	// エッジごとに頂点座標を計算
	//
	uint3 dir[3];
	dir[0] = make_uint3(1, 0, 0);
	dir[1] = make_uint3(0, 1, 0);
	dir[2] = make_uint3(0, 0, 1);

	uint cpos = 0;
	int threads = THREAD_NUM;
	dim3 grid;
	for(int i = 0; i < 3; ++i){
		// 1スレッド/エッジ
		grid = dim3((num_edge[i]+threads-1)/threads, 1, 1);
		if(grid.x > 65535){
			grid.y = (grid.x+32768-1)/32768;
			grid.x = 32768;
		}
		CalVertexEdge<<<grid, threads>>>(((float4*)dEdgeVrts)+cpos, dEdgeOcc+cpos, 
										  dVolume, dir[i], edge_size[i], grid_size, 
										  num_voxels, num_edge[i], grid_width, grid_min, threshold);

		cpos += num_edge[i];
	}
	RX_CUERROR("CalVertexEdge failed");
	RX_CUCHECK(cudaThreadSynchronize());


	// 頂点情報を詰める
	CuScan(dEdgeOccScan, dEdgeOcc, num_edge[3]);

	RX_CUCHECK(cudaMemcpy((void*)&lval, (void*)(dEdgeOcc+num_edge[3]-1), sizeof(uint), cudaMemcpyDeviceToHost));
	RX_CUCHECK(cudaMemcpy((void*)&lsval, (void*)(dEdgeOccScan+num_edge[3]-1), sizeof(uint), cudaMemcpyDeviceToHost));
	nvrts = lval + lsval;

	if(nvrts == 0){
		return;
	}

	grid = dim3((num_edge[3]+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// compact edge vertex array
	CompactEdges<<<grid, threads>>>((float4*)dCompactedEdgeVrts, dEdgeOcc, dEdgeOccScan, 
									 (float4*)dEdgeVrts, num_edge[3]);
	RX_CUERROR("CompactEdges failed");
	RX_CUCHECK(cudaThreadSynchronize());

	//cudaMemcpy(m_f4VertPos, m_dfCompactedEdgeVrts, sizeof(float4)*m_uNumTotalVerts, cudaMemcpyDeviceToHost);
}


/*!
 * 位相情報作成
 * @param[out] dTris ポリゴンの幾何情報
 * @param[in] dVoxBit グリッド8頂点の陰関数値が閾値以上かどうかを各ビットに格納した変数
 * @param[in] dVoxTNumScan グリッドごとの三角形メッシュ数のScan
 * @param[in] dCompactedVox メッシュを含むグリッド情報
 * @param[in] dEdgeOccScan エッジにメッシュ頂点を含むかどうかのScan
 * @param[in] edge_size[3] エッジ数(nx,ny,nz)
 * @param[in] num_edge[4] 総エッジ数
 * @param[in] grid_size  グリッド数(nx,ny,nz)
 * @param[in] num_voxels 総グリッド数
 * @param[in] grid_width グリッド幅
 * @param[in] threshold  メッシュ化閾値
 * @param[in] num_active_voxels メッシュを含むボクセルの数
 * @param[in] nvrts 頂点数
 * @param[in] ntris メッシュ数
*/
void CuMCCalTri(uint *dTris, uint *dVoxBit, uint *dVoxTNumScan, uint *dCompactedVox, 
				uint *dEdgeOccScan, uint3 edge_size[3], uint num_edge[4], 
				uint3 grid_size, uint num_voxels, float3 grid_width, float threshold, 
				uint num_active_voxels, uint nvrts, uint ntris)
{
	// 1スレッド/アクティブボクセル
	int threads = NTHREADS;
	dim3 grid((num_active_voxels+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// 位相情報作成
	uint3 numEdge = make_uint3(num_edge[0], num_edge[1], num_edge[2]);
	GenerateTriangles3<<<grid, threads>>>((uint3*)dTris, dVoxTNumScan, dEdgeOccScan, dVoxBit, 
										  edge_size[0], edge_size[1], edge_size[2], numEdge, dCompactedVox, grid_size, 
										  num_active_voxels, grid_width, threshold, nvrts, ntris);
	RX_CUERROR("GenerateTriangles3 failed");
	RX_CUCHECK(cudaThreadSynchronize());

	//cudaMemcpy(m_u3TriIdx, dTris, sizeof(uint3)*m_uMaxVerts*3, cudaMemcpyDeviceToHost);
}

/*!
 * 法線情報作成
 * @param[out] dNrms 頂点法線
 * @param[in] dTris ポリゴンの幾何情報
 * @param[in] dCompactedEdgeVrts 隙間をつめた頂点情報
 * @param[in] nvrts 頂点数
 * @param[in] ntris メッシュ数
*/
void CuMCCalNrm(float *dNrms, uint *dTris, float *dCompactedEdgeVrts, uint nvrts, uint ntris)
{
	RX_CUCHECK(cudaMemset((void*)dNrms, 0, sizeof(float3)*nvrts));

	// 1スレッド/メッシュ
	int threads = NTHREADS;
	dim3 grid((ntris+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// メッシュ法線の計算と頂点への蓄積
	CalVertexNormalKernel<<<grid, threads>>>((float4*)dCompactedEdgeVrts, (uint3*)dTris, (float3*)dNrms, nvrts, ntris);
	RX_CUCHECK(cudaThreadSynchronize());


	// 1スレッド/頂点
	grid = dim3((nvrts+threads-1)/threads, 1, 1);
	if(grid.x > 65535){
		grid.y = (grid.x+32768-1)/32768;
		grid.x = 32768;
	}

	// 頂点法線の正規化
	NormalizeKernel<<<grid, threads>>>((float3*)dNrms, nvrts);
	RX_CUCHECK(cudaThreadSynchronize());
}


#else

void CuMCCreateMesh(GLuint pvbo, GLuint nvbo, float threshold, unsigned int &nvrts, unsigned int &ntris)
{
	// MARK:CuMCCreateMesh
	int threads = 128;
	dim3 grid(m_uNumVoxels/threads, 1, 1);
	// get around maximum grid size of 65535 in each dimension
	if(grid.x > 65535){
		grid.y = grid.x/32768;
		grid.x = 32768;
	}

	//
	// 各ボクセルの8頂点の内外を判定
	//
	classifyVoxel<<<grid, threads>>>(m_duVoxelVerts, m_duVoxelOccupied, g_dfVolume, 
									 m_u3GridSize, m_uNumVoxels, m_f3VoxelH, threshold);
	cutilCheckMsg("classifyVoxel failed");


#if SKIP_EMPTY_VOXELS
	// ボクセル占有配列をscan
	ThrustScanWrapper(m_duVoxelOccupiedScan, m_duVoxelOccupied, m_uNumVoxels);

	// read back values to calculate total number of non-empty voxels
	// since we are using an exclusive scan, the total is the last value of
	// the scan result plus the last value in the input array
	{
		uint lval, lsval;
		RX_CUCHECK(cudaMemcpy((void *) &lval, 
					   (void *) (m_duVoxelOccupied + m_uNumVoxels-1), 
					   sizeof(uint), cudaMemcpyDeviceToHost));
		RX_CUCHECK(cudaMemcpy((void *) &lsval, 
					   (void *) (m_duVoxelOccupiedScan + m_uNumVoxels-1), 
					   sizeof(uint), cudaMemcpyDeviceToHost));
		m_uNumActiveVoxels = lval + lsval;
	}

	if (m_uNumActiveVoxels==0) {
		// return if there are no full voxels
		m_uNumTotalVerts = 0;
		return;
	}

	// compact voxel index array
	compactVoxels<<<grid, threads>>>(m_duCompactedVoxelArray, m_duVoxelOccupied, m_duVoxelOccupiedScan, m_uNumVoxels);
	cutilCheckMsg("compactVoxels failed");

#endif // SKIP_EMPTY_VOXELS

	// scan voxel vertex count array
	ThrustScanWrapper(m_duVoxelVertsScan, m_duVoxelVerts, m_uNumVoxels);

	// readback total number of vertices
	{
		uint lval, lsval;
		RX_CUCHECK(cudaMemcpy((void *) &lval, 
					   (void *) (m_duVoxelVerts + m_uNumVoxels-1), 
					   sizeof(uint), cudaMemcpyDeviceToHost));
		RX_CUCHECK(cudaMemcpy((void *) &lsval, 
					   (void *) (m_duVoxelVertsScan + m_uNumVoxels-1), 
					   sizeof(uint), cudaMemcpyDeviceToHost));
		m_uNumTotalVerts = lval+lsval;

		nvrts = m_uNumTotalVerts;
		ntris = nvrts/3;
	}


	//
	// 三角形メッシュ作成
	//

	// 頂点と法線バッファ
	float4 *d_pos = 0, *d_nrm = 0;
	if(pvbo){
		RX_CUCHECK(cudaGLMapBufferObject((void**)&d_pos, pvbo));
		RX_CUCHECK(cudaGLMapBufferObject((void**)&d_nrm, nvbo));
	}
	else{
		d_pos = m_df4Vrts;
		d_nrm = m_df4Nrms;
	}

#if SKIP_EMPTY_VOXELS
	dim3 grid2((int) ceil(m_uNumActiveVoxels / (float) NTHREADS), 1, 1);
#else
	dim3 grid2((int) ceil(m_uNumVoxels / (float) NTHREADS), 1, 1);
#endif

	while(grid2.x > 65535) {
		grid2.x/=2;
		grid2.y*=2;
	}

	generateTriangles2<<<grid2, NTHREADS>>>(d_pos, d_nrm, 
						   m_duCompactedVoxelArray, m_duVoxelVertsScan, g_dfVolume, 
						   m_u3GridSize, m_f3VoxelH, m_f3VoxelMin, threshold, m_uNumActiveVoxels, m_uMaxVerts);
	cutilCheckMsg("generateTriangles2 failed");

	if(pvbo){
		RX_CUCHECK(cudaGLUnmapBufferObject(nvbo));
		RX_CUCHECK(cudaGLUnmapBufferObject(pvbo));
	}


}

#endif



}   // extern "C"
