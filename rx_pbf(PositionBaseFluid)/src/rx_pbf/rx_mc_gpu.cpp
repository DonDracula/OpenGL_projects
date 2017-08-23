/*!
  @file rx_mc_cpu.cpp
	
  @brief 陰関数表面からのポリゴン生成(MC法)
	
	http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
 
  @author Raghavendra Chandrashekara (basesd on source code
			provided by Paul Bourke and Cory Gene Bloyd)
  @date   2010-03
*/


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cmath>
#include "helper_math.h"

#include "rx_mc.h"

#include "rx_cu_funcs.cuh"
#include <cuda_runtime.h>


//-----------------------------------------------------------------------------
// rxMCMeshGPUの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 */
rxMCMeshGPU::rxMCMeshGPU()
{
	// MARK:コンストラクタ
	m_uNumVoxels = 0;				//!< 総グリッド数
	m_uMaxVerts = 0;				//!< 最大頂点数
	m_uNumActiveVoxels = 0;			//!< メッシュが存在するボクセル数
	m_uNumVrts = 0;			//!< 総頂点数

	// デバイスメモリ
	g_dfVolume = 0;					//!< 陰関数データを格納するグリッド
	g_dfNoise = 0;					//!< ノイズ値を格納するグリッド(描画時の色を決定するのに使用)
	m_duVoxelVerts = 0;				//!< グリッドに含まれるメッシュ頂点数
	m_duVoxelVertsScan = 0;			//!< グリッドに含まれるメッシュ頂点数(Scan)
	m_duCompactedVoxelArray = 0;	//!< メッシュを含むグリッド情報

	m_duVoxelOccupied = 0;			//!< ポリゴンが内部に存在するボクセルのリスト
	m_duVoxelOccupiedScan = 0;		//!< ポリゴンが内部に存在するボクセルのリスト(prefix scan)

#ifdef RX_CUMC_USE_GEOMETRY
	// 幾何情報を生成するときに必要な変数
	m_duVoxelCubeIdx = 0;			//!< グリッド8頂点の陰関数値が閾値以上かどうかを各ビットに格納した変数

	m_duEdgeOccupied = 0;			//!< エッジにメッシュ頂点を含むかどうか(x方向，y方向, z方向の順)
	m_duEdgeOccupiedScan = 0;		//!< エッジにメッシュ頂点を含むかどうか(Scan)
	m_dfEdgeVrts = 0;				//!< エッジごとに算出した頂点情報
	m_dfCompactedEdgeVrts = 0;		//!< 隙間をつめた頂点情報
	m_duIndexArray = 0;			//!< ポリゴンの幾何情報
	m_dfVertexNormals = 0;			//!< 頂点法線
	m_duVoxelTriNum = 0;			//!< グリッドごとの三角形メッシュ数
	m_duVoxelTriNumScan = 0;		//!< グリッドごとの三角形メッシュ数(Scan)
#else
	// 幾何情報を必要としないときのみ用いる
	m_df4Vrts = 0;					//!< ポリゴン頂点座標
	m_df4Nrms = 0;					//!< ポリゴン頂点法線
#endif

	// ホストメモリ
	m_f4VertPos = 0;				//!< 頂点座標
#ifdef RX_CUMC_USE_GEOMETRY
	m_u3TriIdx = 0;					//!< メッシュインデックス
	m_uScan = 0;					//!< デバッグ用
#else
	m_f4VertNrm = 0;				//!< 頂点法線
#endif

	m_uMaxVerts = 0;
	m_iVertexStore = 4;
	m_bSet = false;

	m_ptScalarField = NULL;
	m_fpScalarFunc = 0;
	m_tIsoLevel = 0;
	m_bValidSurface = false;

	CuMCInit();
}

/*!
 * デストラクタ
 */
rxMCMeshGPU::~rxMCMeshGPU()
{
	Clean();
}



/*!
 * メッシュ生成ボクセル分割数の設定とデバイスメモリ，ホストメモリの確保
 * @param[out] max_verts 最大頂点数
 * @param[in]  n ボクセル分割数の乗数(2^n)
 */
bool rxMCMeshGPU::Set(Vec3 vMin, Vec3 vH, int n[3], unsigned int vm)
{
	float minp[3], maxp[3];
	for(int i = 0; i < 3; ++i){
		minp[i] = (float)(vMin[i]);
		maxp[i] = (float)(vMin[i]+vH[i]*n[i]);
	}

	// グリッド数
	m_u3GridSize = make_uint3(n[0], n[1], n[2]);

	// 総グリッド数
	m_uNumVoxels = m_u3GridSize.x*m_u3GridSize.y*m_u3GridSize.z;

	// グリッド全体の大きさ
	m_f3VoxelMin = make_float3(minp[0], minp[1], minp[2]);
	m_f3VoxelMax = make_float3(maxp[0], maxp[1], maxp[2]);
	float3 rng = m_f3VoxelMax-m_f3VoxelMin;
	m_f3VoxelH = make_float3(rng.x/m_u3GridSize.x, rng.y/m_u3GridSize.y, rng.z/m_u3GridSize.z);


#ifdef RX_CUMC_USE_GEOMETRY
	// エッジ数
	m_u3EdgeSize[0] = make_uint3(m_u3GridSize.x, m_u3GridSize.y+1, m_u3GridSize.z+1);
	m_u3EdgeSize[1] = make_uint3(m_u3GridSize.x+1, m_u3GridSize.y, m_u3GridSize.z+1);
	m_u3EdgeSize[2] = make_uint3(m_u3GridSize.x+1, m_u3GridSize.y+1, m_u3GridSize.z);
	m_uNumEdges[0] = m_u3GridSize.x*(m_u3GridSize.y+1)*(m_u3GridSize.z+1);
	m_uNumEdges[1] = (m_u3GridSize.x+1)*m_u3GridSize.y*(m_u3GridSize.z+1);
	m_uNumEdges[2] = (m_u3GridSize.x+1)*(m_u3GridSize.y+1)*m_u3GridSize.z;
	m_uNumEdges[3] = m_uNumEdges[0]+m_uNumEdges[1]+m_uNumEdges[2];
	//printf("mc edge num : %d, %d, %d - %d\n", m_uNumEdges[0], m_uNumEdges[1], m_uNumEdges[2], m_uNumEdges[3]);
#endif

	// 最大頂点数
	m_uMaxVerts = m_u3GridSize.x*m_u3GridSize.y*vm;
	m_iVertexStore = vm;
	cout << "maximum vertex num : " << m_uMaxVerts << ", vertex store : " << m_iVertexStore << endl;

	m_uNumTris = 0;
	m_uNumVrts = 0;

	// 陰関数値を格納するボリューム
	int size = m_u3GridSize.x*m_u3GridSize.y*m_u3GridSize.z*sizeof(float);
	CuAllocateArray((void**)&g_dfVolume, size);
	CuSetArrayValue((void*)g_dfVolume, 0, size);
	CuAllocateArray((void**)&g_dfNoise, size);
	CuSetArrayValue((void*)g_dfNoise, 0, size);

	// テーブル
	CuInitMCTable();

	// デバイスメモリ
	unsigned int memSize = sizeof(uint)*m_uNumVoxels;
	CuAllocateArray((void**)&m_duVoxelVerts,      memSize);
	CuAllocateArray((void**)&m_duVoxelVertsScan,  memSize);

#if SKIP_EMPTY_VOXELS
	CuAllocateArray((void**)&m_duVoxelOccupied, memSize);
	CuAllocateArray((void**)&m_duVoxelOccupiedScan, memSize);
	CuAllocateArray((void**)&m_duCompactedVoxelArray, memSize);
#endif

#ifdef RX_CUMC_USE_GEOMETRY

	CuAllocateArray((void**)&m_duVoxelCubeIdx,    memSize);

	CuAllocateArray((void**)&m_duVoxelTriNum,     memSize);
	CuAllocateArray((void**)&m_duVoxelTriNumScan, memSize);

	CuSetArrayValue((void*)m_duVoxelCubeIdx,    0, memSize);
	CuSetArrayValue((void*)m_duVoxelTriNum,     0, memSize);
	CuSetArrayValue((void*)m_duVoxelTriNumScan, 0, memSize);

	memSize = sizeof(uint)*m_uNumEdges[3];
	CuAllocateArray((void**)&m_duEdgeOccupied,	    memSize);
	CuAllocateArray((void**)&m_duEdgeOccupiedScan, memSize);

	CuSetArrayValue((void*)m_duEdgeOccupied,     0, memSize);
	CuSetArrayValue((void*)m_duEdgeOccupiedScan, 0, memSize);

	memSize = sizeof(float)*4*m_uNumEdges[3];
	CuAllocateArray((void**)&m_dfEdgeVrts,          memSize);
	CuAllocateArray((void**)&m_dfCompactedEdgeVrts, memSize);

	CuSetArrayValue((void*)m_dfEdgeVrts,          0, memSize);
	CuSetArrayValue((void*)m_dfCompactedEdgeVrts, 0, memSize);

	memSize = sizeof(float)*3*m_uNumEdges[3];
	CuAllocateArray((void**)&m_dfVertexNormals,     memSize);
	CuSetArrayValue((void*)m_dfVertexNormals,     0, memSize);
	
	memSize = sizeof(uint)*3*m_uMaxVerts*3;
	CuAllocateArray((void**)&m_duIndexArray, memSize);
	CuSetArrayValue((void*)m_duIndexArray, 0, memSize);

#else

	memSize = sizeof(float4)*m_uMaxVerts;
	CuAllocateArray((void**)&m_df4Vrts, memSize);
	CuAllocateArray((void**)&m_df4Nrms, memSize);

#endif

	// ホストメモリ
	m_f4VertPos = (float4*)malloc(sizeof(float4)*m_uMaxVerts);
	memset(m_f4VertPos, 0, sizeof(float4)*m_uMaxVerts);

#ifdef RX_CUMC_USE_GEOMETRY
	m_u3TriIdx = (uint3*)malloc(sizeof(uint3)*m_uMaxVerts*3);
	memset(m_u3TriIdx, 0, sizeof(uint3)*m_uMaxVerts*3);

	m_uScan = (uint*)malloc(sizeof(uint)*m_uNumEdges[3]);
	memset(m_uScan, 0, sizeof(uint)*m_uNumEdges[3]);
#else
	m_f4VertNrm = (float4*)malloc(sizeof(float4)*m_uMaxVerts);
	memset(m_f4VertNrm, 0, sizeof(float4)*m_uMaxVerts);
#endif

	m_bSet = true;
	return true;
}

/*!
 * 確保した配列の削除
 */
void rxMCMeshGPU::Clean(void)
{
	if(m_bSet){
		CuFreeArray(g_dfVolume);
		CuFreeArray(g_dfNoise);

		CuFreeArray(m_duVoxelVerts);
		CuFreeArray(m_duVoxelVertsScan);

#if SKIP_EMPTY_VOXELS
		CuFreeArray(m_duVoxelOccupied);
		CuFreeArray(m_duVoxelOccupiedScan);
		CuFreeArray(m_duCompactedVoxelArray);
#endif

		CuCleanMCTable();

#ifdef RX_CUMC_USE_GEOMETRY
		CuFreeArray(m_duVoxelCubeIdx);
		CuFreeArray(m_duVoxelTriNum);
		CuFreeArray(m_duVoxelTriNumScan);
		CuFreeArray(m_duIndexArray);

		CuFreeArray(m_duEdgeOccupied);
		CuFreeArray(m_duEdgeOccupiedScan);
		CuFreeArray(m_dfEdgeVrts);
		CuFreeArray(m_dfCompactedEdgeVrts);

		CuFreeArray(m_dfVertexNormals);

		if(m_u3TriIdx != 0) free(m_u3TriIdx);
		if(m_uScan != 0) free(m_uScan);
#else
		CuFreeArray(m_df4Vrts);
		CuFreeArray(m_df4Nrms);

		if(m_f4VertNrm != 0) free(m_f4VertNrm);
#endif

		if(m_f4VertPos != 0) free(m_f4VertPos);

		m_bSet = false;
	}
}

/*!
 * 陰関数から三角形メッシュを生成
 * @param[in] func 陰関数値取得用関数ポインタ
 * @param[in] min_p グリッドの最小座標
 * @param[in] h グリッドの幅
 * @param[in] n[3] グリッド数(x,y,z)
 * @param[in] threshold しきい値(陰関数値がこの値のところをメッシュ化)
 * @param[in] method メッシュ生成方法("mc", "rmt", "bloomenthal")
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 * @retval true  メッシュ生成成功
 * @retval false メッシュ生成失敗
 */
bool rxMCMeshGPU::CreateMesh(float (*func)(double, double, double), Vec3 minp, double h, int n[3], float threshold, string method, 
									vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face)
{
	if(func == NULL) return false;

	// ボリュームデータ作成
	float *field = new float[n[0]*n[1]*n[2]];
	for(int k = 0; k < n[2]; ++k){
		for(int j = 0; j < n[1]; ++j){
			for(int i = 0; i < n[0]; ++i){
				int idx = k*n[0]*n[1]+j*n[0]+i;
				Vec3 pos = minp+Vec3(i, j, k)*h;

				float val = func(pos[0], pos[1], pos[2]);
				field[idx] = val;
			}
		}
	}

	// ボリュームデータをデバイスメモリにコピー
	int size = m_u3GridSize.x*m_u3GridSize.y*m_u3GridSize.z*sizeof(float);
	CuCopyArrayToDevice((void*)g_dfVolume, (void*)field, 0, size);

	// メッシュ生成
	uint nvrts, ntris;
	CreateMeshV(minp, h, n, threshold, nvrts, ntris);

	// データを配列に格納
	SetDataToArray(vrts, nrms, face);

	return true;
}


/*!
 * デバイスボリュームデータから三角形メッシュを生成
 * @param[in] field サンプルボリューム
 * @param[in] min_p グリッドの最小座標
 * @param[in] h グリッドの幅
 * @param[in] n[3] グリッド数(x,y,z)
 * @param[in] threshold しきい値(陰関数値がこの値のところをメッシュ化)
 * @param[in] method メッシュ生成方法("mc", "rmt", "bloomenthal")
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 * @retval true  メッシュ生成成功
 * @retval false メッシュ生成失敗
 */
bool rxMCMeshGPU::CreateMeshV(Vec3 minp, double h, int n[3], float threshold, uint &nvrts, uint &ntris)
{
	m_uNumVrts = 0;

	// ボクセル中の頂点数，メッシュ数の算出
	CuMCCalTriNum(g_dfVolume, m_duVoxelCubeIdx, m_duVoxelVerts, m_duVoxelVertsScan, 
				  m_duVoxelTriNum, m_duVoxelTriNumScan, m_duVoxelOccupied, m_duVoxelOccupiedScan, m_duCompactedVoxelArray, 
				  m_u3GridSize, m_uNumVoxels, m_f3VoxelH, threshold, 
				  m_uNumActiveVoxels, m_uNumVrts, m_uNumTris);

	// エッジ頂点の算出
	CuMCCalEdgeVrts(g_dfVolume, m_dfEdgeVrts, m_dfCompactedEdgeVrts, 
					m_duEdgeOccupied, m_duEdgeOccupiedScan, m_u3EdgeSize, m_uNumEdges, 
					m_u3GridSize, m_uNumVoxels, m_f3VoxelH, m_f3VoxelMin, threshold, 
					m_uNumVrts);

	if(m_uNumVrts){
		// メッシュ生成
		CuMCCalTri(m_duIndexArray, m_duVoxelCubeIdx, m_duVoxelTriNumScan, m_duCompactedVoxelArray, 
				   m_duEdgeOccupiedScan, m_u3EdgeSize, m_uNumEdges, 
				   m_u3GridSize, m_uNumVoxels, m_f3VoxelH, threshold, 
				   m_uNumActiveVoxels, m_uNumVrts, m_uNumTris);

		// 頂点法線計算
		CuMCCalNrm(m_dfVertexNormals, m_duIndexArray, m_dfCompactedEdgeVrts, m_uNumVrts, m_uNumTris);
	}
	else{
		m_uNumTris = 0;
	}


	nvrts = m_uNumVrts;
	ntris = m_uNumTris;

	return true;
}

/*!
 * FBOにデータを設定
 * @param[in] uVrtVBO 頂点FBO
 * @param[in] uNrmVBO 法線FBO
 * @param[in] uTriVBO メッシュFBO
 */
bool rxMCMeshGPU::SetDataToFBO(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO)
{
	int memsize = 0;

	// 頂点情報の取得
	memsize = sizeof(float)*m_uNumVrts*4;

	glBindBuffer(GL_ARRAY_BUFFER, uVrtVBO);
	float *vrt_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	CuCopyArrayFromDevice(vrt_ptr, GetVrtDev(), 0, 0, memsize);

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 法線情報の取得
	memsize = sizeof(float)*m_uNumVrts*4;

	glBindBuffer(GL_ARRAY_BUFFER, uNrmVBO);
	float *nrm_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	CuCopyArrayFromDevice(nrm_ptr, GetNrmDev(), 0, 0, memsize);

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 接続情報の取得
	memsize = sizeof(uint)*m_uNumTris*3;

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uTriVBO);
	uint *tri_ptr = (uint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

	CuCopyArrayFromDevice(tri_ptr, GetIdxDev(), 0, 0, memsize);

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}

/*!
 * ホスト側配列にデータを設定
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 */
bool rxMCMeshGPU::SetDataToArray(vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face)
{
	// 頂点情報の取得
	float *vrt_ptr = new float[m_uNumVrts*4];
	CuCopyArrayFromDevice(vrt_ptr, GetVrtDev(), 0, 0, m_uNumVrts*4*sizeof(float));

	vrts.resize(m_uNumVrts);
	for(uint i = 0; i < m_uNumVrts; ++i){
		vrts[i][0] = vrt_ptr[4*i];
		vrts[i][1] = vrt_ptr[4*i+1];
		vrts[i][2] = vrt_ptr[4*i+2];
	}
	
	// 法線情報の取得
	CuCopyArrayFromDevice(vrt_ptr, GetNrmDev(), 0, 0, m_uNumVrts*4*sizeof(float));

	nrms.resize(m_uNumVrts);
	for(uint i = 0; i < m_uNumVrts; ++i){
		nrms[i][0] = vrt_ptr[4*i];
		nrms[i][1] = vrt_ptr[4*i+1];
		nrms[i][2] = vrt_ptr[4*i+2];
	}

	delete [] vrt_ptr;

	// 接続情報の取得
	uint *tri_ptr = new uint[m_uNumTris*3];
	CuCopyArrayFromDevice(tri_ptr, GetIdxDev(), 0, 0, m_uNumTris*3*sizeof(uint));

	face.resize(m_uNumTris);
	for(uint i = 0; i < m_uNumTris; ++i){
		face[i].vert_idx.resize(3);
		face[i][0] = tri_ptr[3*i];
		face[i][1] = tri_ptr[3*i+1];
		face[i][2] = tri_ptr[3*i+2];
	}

	delete [] tri_ptr;

	return true;
}

/*!
 * サンプルボリュームを設定
 * @param[in] hVolume サンプルボリューム
 */
void rxMCMeshGPU::SetSampleVolumeFromHost(float *hVolume)
{
	int size = m_u3GridSize.x*m_u3GridSize.y*m_u3GridSize.z*sizeof(float);
	CuCopyArrayToDevice((void*)g_dfVolume, (void*)hVolume, 0, size);
}
float* rxMCMeshGPU::GetSampleVolumeDevice(void)
{
	return g_dfVolume;
}

/*!
 * ノイズ付きサンプルボリュームを設定
 * @param[in] 
 * @return 
 */
void rxMCMeshGPU::SetSampleNoiseFromHost(float *hVolume)
{
	int size = m_u3GridSize.x*m_u3GridSize.y*m_u3GridSize.z*sizeof(float);
	CuCopyArrayToDevice((void*)g_dfNoise, (void*)hVolume, 0, size);
}
float* rxMCMeshGPU::GetSampleNoiseDevice(void)
{
	return g_dfNoise;
}


