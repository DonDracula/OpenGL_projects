/*!
  @file rx_nnsearch.h
	
  @brief 矩形グリッド分割による近傍探索
 
  @author Makoto Fujisawa
  @date 2012-08
*/
// FILE --rx_nnsearch.h--

#ifndef _RX_NNSEARCH_H_
#define _RX_NNSEARCH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_utility.h"

#include <algorithm>

#include <GL/glut.h>
#include <vector>
#include <set>

#include "rx_pcube.h"


using namespace std;

typedef unsigned int uint;

#ifndef RXREAL
	#define RXREAL float
#endif


//-----------------------------------------------------------------------------
//! ハッシュ値によるソート用の構造体
//-----------------------------------------------------------------------------
struct rxHashSort
{
	uint hash;
	uint value;
};

/*!
 * ハッシュ値の比較関数
 * @param[in] left,right 比較する値
 * @return left < right
 */
inline bool LessHash(const rxHashSort &left, const rxHashSort &right)
{
	return left.hash < right.hash;
}


//-----------------------------------------------------------------------------
//! 近傍探索結果構造体
//-----------------------------------------------------------------------------
struct rxNeigh
{
	int Idx;		//!< 近傍パーティクルインデックス
	RXREAL Dist;	//!< 近傍パーティクルまでの距離
	RXREAL Dist2;	//!< 近傍パーティクルまでの2乗距離
};


//-----------------------------------------------------------------------------
//! rxNNGridクラス - グリッド分割法による近傍探索(3D)
//-----------------------------------------------------------------------------
class rxNNGrid
{
public:
	//! 探索用空間分割グリッドの各セル
	struct rxCell
	{
		//uint* hSortedIndex;			//!< ハッシュ値でソートしたパーティクルインデックス
		//uint* hGridParticleHash;	//!< 各パーティクルのグリッドハッシュ値
		rxHashSort* hSortedIndex;
		uint* hCellStart;			//!< ソートリスト内の各セルのスタートインデックス
		uint* hCellEnd;				//!< ソートリスト内の各セルのエンドインデックス
		uint  uNumCells;			//!< 総セル数

		uint* hSortedPolyIdx;		//!< ハッシュ値でソートしたポリゴンインデックス(重複有り)
		uint* hGridPolyHash;		//!< 各ポリゴンのグリッドハッシュ値
		uint* hPolyCellStart;		//!< ソートリスト内の各セルのスタートインデックス
		uint* hPolyCellEnd;			//!< ソートリスト内の各セルのエンドインデックス	

		uint uNumPolyHash;			//!< ポリゴンを含むセルの数
	};


protected:
	// 空間分割格子
	rxCell m_hCellData;				//!< 探索用空間分割グリッド
	int m_iGridSize[3];				//!< 格子の数
	double m_fCellWidth[3];			//!< 格子一片の長さ

	int m_iDim;						//!< 次元数(座標配列のステップ幅)
	Vec3 m_v3EnvMin;				//!< 環境最小座標

	int m_iSorted;					//!< 一度でもソートされたかどうかのフラグ

public:
	//! デフォルトコンストラクタ
	rxNNGrid(int dim) : m_iDim(dim)
	{
		m_hCellData.uNumCells = 0;
		m_hCellData.hSortedIndex = 0;
		//m_hCellData.hGridParticleHash = 0;
		m_hCellData.hCellStart = 0;
		m_hCellData.hCellEnd = 0;

		m_hCellData.hSortedPolyIdx = 0;
		m_hCellData.hGridPolyHash = 0;
		m_hCellData.hPolyCellStart = 0;
		m_hCellData.hPolyCellEnd = 0;
		m_hCellData.uNumPolyHash = 0;

		m_iSorted = 0;
	}

	//! デストラクタ
	~rxNNGrid()
	{
		if(m_hCellData.hSortedIndex) delete [] m_hCellData.hSortedIndex;
		//if(m_hCellData.hGridParticleHash) delete [] m_hCellData.hGridParticleHash;
		if(m_hCellData.hCellStart) delete [] m_hCellData.hCellStart;
		if(m_hCellData.hCellEnd) delete [] m_hCellData.hCellEnd;

		if(m_hCellData.hSortedPolyIdx) delete [] m_hCellData.hSortedPolyIdx;
		if(m_hCellData.hGridPolyHash) delete [] m_hCellData.hGridPolyHash;
		if(m_hCellData.hPolyCellStart) delete [] m_hCellData.hPolyCellStart;
		if(m_hCellData.hPolyCellEnd) delete [] m_hCellData.hPolyCellEnd;
	}

public:
	// 分割セルの初期設定
	void Setup(Vec3 vMin, Vec3 vMax, double h, int n);

	// 分割セルへパーティクルを格納
	void SetObjectToCell(RXREAL *p, uint n);
	void SetObjectToCellV(Vec3 *p, uint n);

	// 分割セルへポリゴンを格納
	void SetPolygonsToCell(RXREAL *vrts, int nv, int* tris, int nt);

	// 近傍取得
	void GetNN_Direct(Vec3 pos, RXREAL *p, uint n, vector<rxNeigh> &neighs, RXREAL h = -1.0);
	void GetNN(Vec3 pos, RXREAL *p, uint n, vector<rxNeigh> &neighs, RXREAL h = -1.0);
	void GetNNV(Vec3 pos, Vec3 *p, uint n, vector<rxNeigh> &neighs, RXREAL h = -1.0);

	// セル内のポリゴン取得
	int  GetNNPolygons(Vec3 pos, set<int> &polys, RXREAL h);
	int  GetPolygonsInCell(uint grid_hash, set<int> &polys);
	int  GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys);
	bool IsPolygonsInCell(int gi, int gj, int gk);

	// OpenGL描画
	void DrawCell(int i, int j, int k);
	void DrawCells(Vec3 col, Vec3 col2, int sel = 0, RXREAL *p = 0);
	void DrawCellsV(Vec3 col, Vec3 col2, int sel = 0, Vec3 *p = 0);

	// グリッドハッシュの計算
	uint CalGridHash(int x, int y, int z);
	uint CalGridHash(Vec3 pos);

public:
	rxCell& GetCellData(void){ return m_hCellData; }

protected:
	// 分割セルから近傍パーティクルを取得
	void getNeighborsInCell(Vec3 pos, RXREAL *p, int gi, int gj, int gk, vector<rxNeigh> &neighs, RXREAL h);
	void getNeighborsInCellV(Vec3 pos, Vec3 *p, int gi, int gj, int gk, vector<rxNeigh> &neighs, RXREAL h);
};


	
/*!
 * 空間分割法の準備
 * @param[in] vMin 環境の最小座標
 * @param[in] vMax 環境の最大座標
 * @param[in] h 影響半径
 */
inline void rxNNGrid::Setup(Vec3 vMin, Vec3 vMax, double h, int n)
{
	if(h < RX_EPS) return;

	Vec3 world_size = vMax-vMin;
	Vec3 world_origin = vMin;

	double max_axis = RXFunc::Max3(world_size);

	int d = (int)(log(max_axis/h)/log(2.0)+0.5);
	int m = (int)(pow(2.0, (double)d)+0.5);
	double cell_width = max_axis/m;

	d = (int)(log(world_size[0]/cell_width)/log(2.0)+0.5);
	m_iGridSize[0] = (int)(pow(2.0, (double)d)+0.5);
	d = (int)(log(world_size[1]/cell_width)/log(2.0)+0.5);
	m_iGridSize[1] = (int)(pow(2.0, (double)d)+0.5);;
	d = (int)(log(world_size[2]/cell_width)/log(2.0)+0.5);
	m_iGridSize[2] = (int)(pow(2.0, (double)d)+0.5);;

	m_fCellWidth[0] = cell_width;
	m_fCellWidth[1] = cell_width;
	m_fCellWidth[2] = cell_width;

	m_v3EnvMin = world_origin;

	m_hCellData.uNumCells = m_iGridSize[0]*m_iGridSize[1]*m_iGridSize[2];

	cout << "grid for nn search : " << endl;
	cout << "  size   : " << m_iGridSize[0] << "x" << m_iGridSize[1] << endl;
	cout << "  num    : " << m_hCellData.uNumCells << endl;
	cout << "  origin : " << m_v3EnvMin << endl;
	cout << "  width  : " << m_fCellWidth[0] << "x" << m_fCellWidth[1] << endl;

	// 分割グリッド構造体の配列確保
	m_hCellData.hSortedIndex = new rxHashSort[n];
	//m_hCellData.hSortedIndex = new uint[n];
	//m_hCellData.hGridParticleHash = new uint[n];
	m_hCellData.hCellStart = new uint[m_hCellData.uNumCells];
	m_hCellData.hCellEnd = new uint[m_hCellData.uNumCells];

	int mem_size1 = n*sizeof(uint);
	int mem_size2 = m_hCellData.uNumCells*sizeof(uint);
	memset(m_hCellData.hSortedIndex, 0, n*sizeof(rxHashSort));
	//memset(m_hCellData.hGridParticleHash, 0, mem_size1);
	memset(m_hCellData.hCellStart, 0xffffffff, mem_size2);
	memset(m_hCellData.hCellEnd, 0xffffffff, mem_size2);

	m_hCellData.hPolyCellStart = new uint[m_hCellData.uNumCells];
	m_hCellData.hPolyCellEnd = new uint[m_hCellData.uNumCells];

	memset(m_hCellData.hPolyCellStart, 0xffffffff, mem_size2);
	memset(m_hCellData.hPolyCellEnd, 0, mem_size2);

	m_iSorted = 0;
}

/*!
 * パーティクルを分割セルに格納
 *  - パーティクルの属するグリッドハッシュを計算して格納する
 * @param[in] p 格納したい全パーティクルの座標を記述した配列
 * @param[in] n パーティクル数
 */
inline void rxNNGrid::SetObjectToCell(RXREAL *p, uint n)
{
	int mem_size1 = n*sizeof(uint);
	int mem_size2 = m_hCellData.uNumCells*sizeof(uint);
	//if(!m_iSorted) memset(m_hCellData.hSortedIndex, 0, mem_size1);
	if(!m_iSorted) memset(m_hCellData.hSortedIndex, 0, n*sizeof(rxHashSort));
	//memset(m_hCellData.hGridParticleHash, 0, mem_size1);
	memset(m_hCellData.hCellStart, 0xffffffff, mem_size2);
	memset(m_hCellData.hCellEnd, 0xffffffff, mem_size2);

	if(n == 0) return;

	if(m_iSorted){
		// 各パーティクルのグリッドハッシュの計算
		for(uint ip = 0; ip < n; ++ip){
			int i = m_hCellData.hSortedIndex[ip].value;
			Vec3 pos;
			pos[0] = p[m_iDim*i+0];
			pos[1] = p[m_iDim*i+1];
			pos[2] = p[m_iDim*i+2];

			// ハッシュ値計算
			uint hash = CalGridHash(pos);
			m_hCellData.hSortedIndex[ip].hash = hash;
			//m_hCellData.hGridParticleHash[ip] = hash;
		}
	}
	else{
		// 各パーティクルのグリッドハッシュの計算
		for(uint i = 0; i < n; ++i){
			Vec3 pos;
			pos[0] = p[m_iDim*i+0];
			pos[1] = p[m_iDim*i+1];
			pos[2] = p[m_iDim*i+2];

			// ハッシュ値計算
			uint hash = CalGridHash(pos);

			m_hCellData.hSortedIndex[i].value = i;
			m_hCellData.hSortedIndex[i].hash = hash;
			//m_hCellData.hSortedIndex[i] = i;
			//m_hCellData.hGridParticleHash[i] = hash;
		}
	}

	// グリッドハッシュでソート
	//vector<rxHashSort> hash_and_value;
	//hash_and_value.resize(n);
	//for(uint i = 0; i < n; ++i){
	//	//hash_and_value[i].hash = m_hCellData.hGridParticleHash[i];
	//	//hash_and_value[i].value  = m_hCellData.hSortedIndex[i];
	//	hash_and_value[i]  = m_hCellData.hSortedIndex[i];
	//}
	//std::sort(hash_and_value.begin(), hash_and_value.end(), LessHash);
	//for(uint i = 0; i < n; ++i){
	//	//m_hCellData.hSortedIndex[i] = hash_and_value[i].value;
	//	//m_hCellData.hGridParticleHash[i] = hash_and_value[i].hash;
	//	m_hCellData.hSortedIndex[i] = hash_and_value[i];
	//}
	std::sort(m_hCellData.hSortedIndex, m_hCellData.hSortedIndex+n, LessHash);

	m_iSorted = 1;

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	for(uint i = 0; i < n; i++){
		//int hash = m_hCellData.hGridParticleHash[i];
		int hash = m_hCellData.hSortedIndex[i].hash;

		if(i == 0){
			m_hCellData.hCellStart[hash] = i;
			m_hCellData.hCellEnd[hash] = i;
		}
		else{
			//int prev_hash = m_hCellData.hGridParticleHash[i-1];
			int prev_hash = m_hCellData.hSortedIndex[i-1].hash;

			if(i == 0 || hash != prev_hash){
				m_hCellData.hCellStart[hash] = i;
				if(i > 0){
					m_hCellData.hCellEnd[prev_hash] = i;
				}
			}

			if(i == n-1){
				m_hCellData.hCellEnd[hash] = i+1;
			}
		}
	}
}

/*!
 * パーティクルを分割セルに格納
 *  - パーティクルの属するグリッドハッシュを計算して格納する
 * @param[in] p 格納したい全パーティクルの座標を記述した配列
 * @param[in] n パーティクル数
 */
inline void rxNNGrid::SetObjectToCellV(Vec3 *p, uint n)
{
	int mem_size1 = n*sizeof(uint);
	int mem_size2 = m_hCellData.uNumCells*sizeof(uint);
	//if(!m_iSorted) memset(m_hCellData.hSortedIndex, 0, mem_size1);
	if(!m_iSorted) memset(m_hCellData.hSortedIndex, 0, n*sizeof(rxHashSort));
	//memset(m_hCellData.hGridParticleHash, 0, mem_size1);
	memset(m_hCellData.hCellStart, 0xffffffff, mem_size2);
	memset(m_hCellData.hCellEnd, 0xffffffff, mem_size2);

	if(n == 0) return;
	
	if(m_iSorted){
		// 各パーティクルのグリッドハッシュの計算
		for(uint ip = 0; ip < n; ++ip){
			int i = m_hCellData.hSortedIndex[ip].value;

			// ハッシュ値計算
			uint hash = CalGridHash(p[i]);

			m_hCellData.hSortedIndex[ip].hash = hash;
			//m_hCellData.hGridParticleHash[ip] = hash;
		}
	}
	else{
		// 各パーティクルのグリッドハッシュの計算
		for(uint i = 0; i < n; ++i){
			// ハッシュ値計算
			uint hash = CalGridHash(p[i]);

			m_hCellData.hSortedIndex[i].value = i;
			m_hCellData.hSortedIndex[i].hash = hash;
			//m_hCellData.hSortedIndex[i] = i;
			//m_hCellData.hGridParticleHash[i] = hash;
		}
	}

	// グリッドハッシュでソート
	//vector<rxHashSort> hash_and_value;
	//hash_and_value.resize(n);
	//for(uint i = 0; i < n; ++i){
	//	//hash_and_value[i].hash = m_hCellData.hGridParticleHash[i];
	//	//hash_and_value[i].value  = m_hCellData.hSortedIndex[i];
	//	hash_and_value[i]  = m_hCellData.hSortedIndex[i];
	//}
	//std::sort(hash_and_value.begin(), hash_and_value.end(), LessHash);
	//for(uint i = 0; i < n; ++i){
	//	//m_hCellData.hSortedIndex[i] = hash_and_value[i].value;
	//	//m_hCellData.hGridParticleHash[i] = hash_and_value[i].hash;
	//	m_hCellData.hSortedIndex[i] = hash_and_value[i];
	//}
	std::sort(m_hCellData.hSortedIndex, m_hCellData.hSortedIndex+n, LessHash);

	m_iSorted = 1;

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	for(uint i = 0; i < n; i++){
		//int hash = m_hCellData.hGridParticleHash[i];
		int hash = m_hCellData.hSortedIndex[i].hash;

		if(i == 0){
			m_hCellData.hCellStart[hash] = i;
			m_hCellData.hCellEnd[hash] = i;
		}
		else{
			//int prev_hash = m_hCellData.hGridParticleHash[i-1];
			int prev_hash = m_hCellData.hSortedIndex[i-1].hash;

			if(i == 0 || hash != prev_hash){
				m_hCellData.hCellStart[hash] = i;
				if(i > 0){
					m_hCellData.hCellEnd[prev_hash] = i;
				}
			}

			if(i == n-1){
				m_hCellData.hCellEnd[hash] = i+1;
			}
		}
	}
}

/*!
 * ポリゴンを分割セルに格納
 * @param[in] vrts ポリゴン頂点
 * @param[in] nv 頂点数
 * @param[in] tris メッシュ
 * @param[in] nt メッシュ数
 */
inline void rxNNGrid::SetPolygonsToCell(RXREAL *vrts, int nv, int* tris, int nt)
{
	int mem_size2 = m_hCellData.uNumCells*sizeof(uint);
	memset(m_hCellData.hPolyCellStart, 0xffffffff, mem_size2);
	memset(m_hCellData.hPolyCellEnd, 0, mem_size2);

	int num_hash = 0;

	// 各ポリゴンのグリッドハッシュの計算
	vector<uint> tri_hash, tri_idx;
	vector<Vec3> tri_vrts, tri_vrts_c;
	tri_vrts.resize(3);
	tri_vrts_c.resize(3);
	for(int i = 0; i < nt; i++){
		for(int j = 0; j < 3; ++j){
			Vec3 pos;
			pos[0] = vrts[3*tris[3*i+j]+0];
			pos[1] = vrts[3*tris[3*i+j]+1];
			pos[2] = vrts[3*tris[3*i+j]+2];
			tri_vrts[j] = pos;
		}

		Vec3 nrm = Unit(cross(tri_vrts[1]-tri_vrts[0], tri_vrts[2]-tri_vrts[0]));

		// ポリゴンのBBox
		Vec3 bmin, bmax;
		bmin = tri_vrts[0];
		bmax = tri_vrts[0];
		for(int j = 1; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				if(tri_vrts[j][k] < bmin[k]) bmin[k] = tri_vrts[j][k];
				if(tri_vrts[j][k] > bmax[k]) bmax[k] = tri_vrts[j][k];
			}
		}

		// BBoxと重なるセル
		bmin -= m_v3EnvMin;
		bmax -= m_v3EnvMin;

		// 分割セルインデックスの算出
		int bmin_gidx[3], bmax_gidx[3];
		for(int k = 0; k < 3; ++k){
			bmin_gidx[k] = bmin[k]/m_fCellWidth[k];
			bmax_gidx[k] = bmax[k]/m_fCellWidth[k];

			bmin_gidx[k] = RX_CLAMP(bmin_gidx[k], 0, m_iGridSize[k]-1);
			bmax_gidx[k] = RX_CLAMP(bmax_gidx[k], 0, m_iGridSize[k]-1);
		}

		// 各セルにポリゴンが含まれるかをチェック
		Vec3 len = Vec3(m_fCellWidth[0], m_fCellWidth[1], m_fCellWidth[2]);
		Vec3 cen(0.0);
		for(int x = bmin_gidx[0]; x <= bmax_gidx[0]; ++x){
			for(int y = bmin_gidx[1]; y <= bmax_gidx[1]; ++y){
				for(int z = bmin_gidx[2]; z <= bmax_gidx[2]; ++z){
					cen = m_v3EnvMin+Vec3(x+0.5, y+0.5, z+0.5)*len;

					for(int j = 0; j < 3; ++j){
						tri_vrts_c[j] = (tri_vrts[j]-cen)/len;
					}

					if(RXFunc::polygon_intersects_cube(tri_vrts_c, nrm)){
						// ハッシュ値計算
						uint hash = CalGridHash(x, y, z);

						tri_idx.push_back((uint)i);
						tri_hash.push_back(hash);

						//m_hCellData.hPolyCellStart[hash] = 0;

						num_hash++;
					}
				}
			}
		}
	}

	m_hCellData.uNumPolyHash = (uint)num_hash;

	int mem_size1 = m_hCellData.uNumPolyHash*sizeof(uint);
	m_hCellData.hSortedPolyIdx = new uint[m_hCellData.uNumPolyHash];
	m_hCellData.hGridPolyHash = new uint[m_hCellData.uNumPolyHash];
	memcpy(m_hCellData.hSortedPolyIdx, &tri_idx[0], mem_size1);
	memcpy(m_hCellData.hGridPolyHash, &tri_hash[0], mem_size1);

	// グリッドハッシュでソート
	vector<rxHashSort> hash_and_value;
	hash_and_value.resize(m_hCellData.uNumPolyHash);
	for(int i = 0; i < (int)m_hCellData.uNumPolyHash; ++i){
		hash_and_value[i].hash  = m_hCellData.hGridPolyHash[i];
		hash_and_value[i].value = m_hCellData.hSortedPolyIdx[i];
	}
	std::sort(hash_and_value.begin(), hash_and_value.end(), LessHash);
	for(int i = 0; i < (int)m_hCellData.uNumPolyHash; ++i){
		m_hCellData.hSortedPolyIdx[i] = hash_and_value[i].value;
		m_hCellData.hGridPolyHash[i]  = hash_and_value[i].hash;
	}

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	for(uint i = 0; i < m_hCellData.uNumPolyHash; ++i){
		uint hash = m_hCellData.hGridPolyHash[i];

		if(i == 0){
			m_hCellData.hPolyCellStart[hash] = i;
		}
		else{
			uint prev_hash = m_hCellData.hGridPolyHash[i-1];

			if(i == 0 || hash != prev_hash){
				m_hCellData.hPolyCellStart[hash] = i;
				if(i > 0){
					m_hCellData.hPolyCellEnd[prev_hash] = i;
				}
			}

			if(i == nt-1){
				m_hCellData.hPolyCellEnd[hash] = i+1;
			}
		}
	}
}

/*!
 * 近傍粒子探索(総当たり)
 * @param[in] pos 探索中心
 * @param[in] p パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
inline void rxNNGrid::GetNN_Direct(Vec3 pos0, RXREAL *p, uint n, vector<rxNeigh> &neighs, RXREAL h)
{
	RXREAL h2 = h*h;

	for(uint i = 0; i < n; i++){
		Vec3 pos1;
		pos1[0] = p[m_iDim*i+0];
		pos1[1] = p[m_iDim*i+1];
		pos1[2] = p[m_iDim*i+2];

		rxNeigh neigh;
		neigh.Dist2 = (RXREAL)norm2(pos0-pos1);

		if(neigh.Dist2 <= h2 && neigh.Dist2 > RX_FEQ_EPS){
			neigh.Idx = i;
			neighs.push_back(neigh);
		}
	}
}

/*!
 * 近傍粒子探索
 * @param[in] pos 探索中心
 * @param[in] p パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
inline void rxNNGrid::GetNN(Vec3 pos, RXREAL *p, uint n, vector<rxNeigh> &neighs, RXREAL h)
{
	// 分割セルインデックスの算出
	int x = (pos[0]-m_v3EnvMin[0])/m_fCellWidth[0];
	int y = (pos[1]-m_v3EnvMin[1])/m_fCellWidth[1];
	int z = (pos[2]-m_v3EnvMin[2])/m_fCellWidth[2];

	int numArdGrid = (int)(h/m_fCellWidth[0])+1;
	for(int k = -numArdGrid; k <= numArdGrid; ++k){
		for(int j = -numArdGrid; j <= numArdGrid; ++j){
			for(int i = -numArdGrid; i <= numArdGrid; ++i){
				int i1 = x+i;
				int j1 = y+j;
				int k1 = z+k;
				if(i1 < 0 || i1 >= m_iGridSize[0] || j1 < 0 || j1 >= m_iGridSize[1] || k1 < 0 || k1 >= m_iGridSize[2]){
					continue;
				}

				getNeighborsInCell(pos, p, i1, j1, k1, neighs, h);
			}
		}
	}
}

/*!
 * 分割セル内の粒子から近傍を検出
 * @param[in] pos 探索中心
 * @param[in] p パーティクル位置
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
inline void rxNNGrid::getNeighborsInCell(Vec3 pos, RXREAL *p, int gi, int gj, int gk, vector<rxNeigh> &neighs, RXREAL h)
{
	RXREAL h2 = h*h;

	uint grid_hash = CalGridHash(gi, gj, gk);

	uint start_index = m_hCellData.hCellStart[grid_hash];
	if(start_index != 0xffffffff){	// セルが空でないかのチェック
		uint end_index = m_hCellData.hCellEnd[grid_hash];
		for(uint j = start_index; j < end_index; ++j){
			//uint idx = m_hCellData.hSortedIndex[j];
			uint idx = m_hCellData.hSortedIndex[j].value;

			Vec3 xij;
			xij[0] = pos[0]-p[m_iDim*idx+0];
			xij[1] = pos[1]-p[m_iDim*idx+1];
			xij[2] = pos[2]-p[m_iDim*idx+2];

			rxNeigh neigh;
			neigh.Dist2 = norm2(xij);

			if(neigh.Dist2 <= h2){
				neigh.Idx = idx;
				//neigh.Dist = sqrt(neigh.Dist2);

				neighs.push_back(neigh);
			}
		}
	}
}

/*!
 * 近傍粒子探索
 * @param[in] pos 探索中心
 * @param[in] p パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
inline void rxNNGrid::GetNNV(Vec3 pos, Vec3 *p, uint n, vector<rxNeigh> &neighs, RXREAL h)
{
	// 分割セルインデックスの算出
	int x = (pos[0]-m_v3EnvMin[0])/m_fCellWidth[0];
	int y = (pos[1]-m_v3EnvMin[1])/m_fCellWidth[1];
	int z = (pos[2]-m_v3EnvMin[2])/m_fCellWidth[2];

	int numArdGrid = (int)(h/m_fCellWidth[0])+1;
	for(int k = -numArdGrid; k <= numArdGrid; ++k){
		for(int j = -numArdGrid; j <= numArdGrid; ++j){
			for(int i = -numArdGrid; i <= numArdGrid; ++i){
				int i1 = x+i;
				int j1 = y+j;
				int k1 = z+k;
				if(i1 < 0 || i1 >= m_iGridSize[0] || j1 < 0 || j1 >= m_iGridSize[1] || k1 < 0 || k1 >= m_iGridSize[2]){
					continue;
				}

				getNeighborsInCellV(pos, p, i1, j1, k1, neighs, h);
			}
		}
	}
}


/*!
 * 分割セル内の粒子から近傍を検出
 * @param[in] pos 探索中心
 * @param[in] p パーティクル位置
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
inline void rxNNGrid::getNeighborsInCellV(Vec3 pos, Vec3 *p, int gi, int gj, int gk, vector<rxNeigh> &neighs, RXREAL h)
{
	RXREAL h2 = h*h;

	uint grid_hash = CalGridHash(gi, gj, gk);

	uint start_index = m_hCellData.hCellStart[grid_hash];
	if(start_index != 0xffffffff){	// セルが空でないかのチェック
		uint end_index = m_hCellData.hCellEnd[grid_hash];
		for(uint j = start_index; j < end_index; ++j){
			//uint idx = m_hCellData.hSortedIndex[j];
			uint idx = m_hCellData.hSortedIndex[j].value;

			Vec3 xij = pos-p[idx];

			rxNeigh neigh;
			neigh.Dist2 = norm2(xij);

			if(neigh.Dist2 <= h2){
				neigh.Idx = idx;
				//neigh.Dist = sqrt(neigh.Dist2);

				neighs.push_back(neigh);
			}
		}
	}
}


/*!
 * 近傍ポリゴン情報を取得
 * @param[in] pos 探索中心
 * @param[out] polys ポリゴンリスト(重複なしにするためにset<>を用いている)
 * @param[in] h 有効半径
 */
inline int rxNNGrid::GetNNPolygons(Vec3 pos, std::set<int> &polys, RXREAL h)
{
	// 分割セルインデックスの算出
	int x = (pos[0]-m_v3EnvMin[0])/m_fCellWidth[0];
	int y = (pos[1]-m_v3EnvMin[1])/m_fCellWidth[1];
	int z = (pos[2]-m_v3EnvMin[2])/m_fCellWidth[2];

	int cnt = 0;
	int numArdGrid = (int)(h/m_fCellWidth[0])+1;
	for(int k = -numArdGrid; k <= numArdGrid; ++k){
		for(int j = -numArdGrid; j <= numArdGrid; ++j){
			for(int i = -numArdGrid; i <= numArdGrid; ++i){
				int i1 = x+i;
				int j1 = y+j;
				int k1 = z+k;
				if(i1 < 0 || i1 >= m_iGridSize[0] || j1 < 0 || j1 >= m_iGridSize[1] || k1 < 0 || k1 >= m_iGridSize[2]){
					continue;
				}

				cnt += GetPolygonsInCell(i1, j1, k1, polys);
			}
		}
	}

	return cnt;
}

/*!
 * 分割セルに格納されたポリゴン情報を取得
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] polys ポリゴンリスト(重複なしにするためにset<>を用いている)
 * @return 格納ポリゴン数
 */
inline int rxNNGrid::GetPolygonsInCell(uint grid_hash, set<int> &polys)
{
	uint start_index = m_hCellData.hPolyCellStart[grid_hash];
	if(start_index != 0xffffffff){	// セルが空でないかのチェック

		int cnt = 0;
		uint end_index = m_hCellData.hPolyCellEnd[grid_hash];
		for(uint j = start_index; j < end_index; ++j){
			uint idx = m_hCellData.hSortedPolyIdx[j];

			polys.insert(idx);

			cnt++;
		}

		return cnt;
	}

	return 0;
}
inline int rxNNGrid::GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys)
{
	return GetPolygonsInCell(CalGridHash(gi, gj, gk), polys);
}
/*!
 * 分割セル内のポリゴンの有無を調べる
 * @param[in] gi,gj,gk 対象分割セル
 * @return ポリゴンが格納されていればtrue
 */
inline bool rxNNGrid::IsPolygonsInCell(int gi, int gj, int gk)
{
	uint grid_hash = CalGridHash(gi, gj, gk);

	uint start_index = m_hCellData.hPolyCellStart[grid_hash];
	if(start_index != 0xffffffff){	// セルが空でないかのチェック

		int cnt = 0;
		uint end_index = m_hCellData.hPolyCellEnd[grid_hash];
		for(uint j = start_index; j < end_index; ++j){
			uint idx = m_hCellData.hSortedPolyIdx[j];
			cnt++;
			break;
		}

		return (cnt > 0);
	}

	return false;
}

/*!
 * グリッドハッシュ値の計算
 * @param[in] x,y,z グリッド位置
 * @return グリッドハッシュ値
 */
inline uint rxNNGrid::CalGridHash(int x, int y, int z)
{
	x = (x < 0 ? 0 : (x >= m_iGridSize[0] ? m_iGridSize[0]-1 : x));
	y = (y < 0 ? 0 : (y >= m_iGridSize[1] ? m_iGridSize[1]-1 : y));
	z = (z < 0 ? 0 : (z >= m_iGridSize[2] ? m_iGridSize[2]-1 : z));
	return z*m_iGridSize[1]*m_iGridSize[0]+y*m_iGridSize[0]+x;
}
/*!
 * グリッドハッシュ値の計算
 * @param[in] pos パーティクル座標
 * @return グリッドハッシュ値
 */
inline uint rxNNGrid::CalGridHash(Vec3 pos)
{
	pos -= m_v3EnvMin;

	// 分割セルインデックスの算出
	int x = pos[0]/m_fCellWidth[0];
	int y = pos[1]/m_fCellWidth[1];
	int z = pos[2]/m_fCellWidth[2];
	return CalGridHash(x, y, z);
}


//-----------------------------------------------------------------------------
// OpenGL描画
//-----------------------------------------------------------------------------
/*!
 * 探索用セルの描画
 * @param[in] i,j,k グリッド上のインデックス
 */
inline void rxNNGrid::DrawCell(int i, int j, int k)
{
	glPushMatrix();
	glTranslated(m_v3EnvMin[0], m_v3EnvMin[1], m_v3EnvMin[2]);
	glTranslatef((i+0.5)*m_fCellWidth[0], (j+0.5)*m_fCellWidth[1], (k+0.5)*m_fCellWidth[2]);
	glutWireCube(m_fCellWidth[0]);
	glPopMatrix();
}

/*!
 * 探索用グリッドの描画
 * @param[in] col パーティクルが含まれるセルの色
 * @param[in] col2 ポリゴンが含まれるセルの色
 * @param[in] sel ランダムに選択されたセルのみ描画(1で新しいセルを選択，2ですでに選択されているセルを描画，0ですべてのセルを描画)
 * @param[in] p パーティクル位置
 */
inline void rxNNGrid::DrawCells(Vec3 col, Vec3 col2, int sel, RXREAL *p)
{
	glPushMatrix();

	if(sel){
		// ランダムに選んだセルとその中のパーティクルのみ描画
		static int grid_hash = 0;
		static uint start_index = 0xffffffff;
		if(sel == 1){
			do{
				grid_hash = RXFunc::Nrand(m_hCellData.uNumCells-1);
				start_index = m_hCellData.hCellStart[grid_hash];
			}while(start_index == 0xffffffff);
		}

		uint w = grid_hash%(m_iGridSize[0]*m_iGridSize[1]);
		DrawCell(w%m_iGridSize[0], w/m_iGridSize[0], grid_hash/(m_iGridSize[0]*m_iGridSize[1]));

		if(p){
			glColor3d(1.0, 0.0, 0.0);
			glPointSize(10.0);
			glBegin(GL_POINTS);

			int c = 0;
			uint end_index = m_hCellData.hCellEnd[grid_hash];
			for(uint j = start_index; j < end_index; ++j){
				//uint idx = m_hCellData.hSortedIndex[j];
				uint idx = m_hCellData.hSortedIndex[j].value;
				Vec3 pos;
				pos[0] = p[m_iDim*idx+0];
				pos[1] = p[m_iDim*idx+1];
				pos[2] = p[m_iDim*idx+2];
			
				glVertex3dv(pos);

				c++;
			}
			glEnd();

			cout << "cell(" << grid_hash << ") : " << c << endl;
		}

	}
	else{
		// パーティクル or ポリゴンを含む全セルの描画
		RXFOR3(0, m_iGridSize[0], 0, m_iGridSize[1], 0, m_iGridSize[2]){
			bool disp = false;
			uint grid_hash = CalGridHash(i, j, k);
			uint start_index = m_hCellData.hCellStart[grid_hash];
			uint start_index_poly = 0xffffffff;

			if(m_hCellData.uNumPolyHash) start_index_poly = m_hCellData.hPolyCellStart[grid_hash];
		
			if(start_index != 0xffffffff){
				glColor3dv(col2.data);
				disp = true;
			}
			if(start_index_poly != 0xffffffff){
				glColor3dv(col.data);
				disp = true;
			}

			if(disp){
				DrawCell(i, j, k);
			}
		}
	}

	glPopMatrix();
}

/*!
 * 探索用グリッドの描画
 * @param[in] col パーティクルが含まれるセルの色
 * @param[in] col2 ポリゴンが含まれるセルの色
 * @param[in] sel ランダムに選択されたセルのみ描画(1で新しいセルを選択，2ですでに選択されているセルを描画，0ですべてのセルを描画)
 * @param[in] p パーティクル位置
 */
inline void rxNNGrid::DrawCellsV(Vec3 col, Vec3 col2, int sel, Vec3 *p)
{
	glPushMatrix();

	if(sel){
		// ランダムに選んだセルとその中のパーティクルのみ描画
		static int grid_hash = 0;
		static uint start_index = 0xffffffff;
		if(sel == 1){
			do{
				grid_hash = RXFunc::Nrand(m_hCellData.uNumCells-1);
				start_index = m_hCellData.hCellStart[grid_hash];
			}while(start_index == 0xffffffff);
		}

		uint w = grid_hash%(m_iGridSize[0]*m_iGridSize[1]);
		DrawCell(w%m_iGridSize[0], w/m_iGridSize[0], grid_hash/(m_iGridSize[0]*m_iGridSize[1]));

		if(p){
			glColor3d(1.0, 0.0, 0.0);
			glPointSize(10.0);
			glBegin(GL_POINTS);

			int c = 0;
			uint end_index = m_hCellData.hCellEnd[grid_hash];
			for(uint j = start_index; j < end_index; ++j){
				//uint idx = m_hCellData.hSortedIndex[j];
				uint idx = m_hCellData.hSortedIndex[j].value;
				Vec3 pos = p[idx];
			
				glVertex3dv(pos);

				c++;
			}
			glEnd();

			cout << "cell(" << grid_hash << ") : " << c << endl;
		}

	}
	else{
		// パーティクル or ポリゴンを含む全セルの描画
		RXFOR3(0, m_iGridSize[0], 0, m_iGridSize[1], 0, m_iGridSize[2]){
			bool disp = false;
			uint grid_hash = CalGridHash(i, j, k);
			uint start_index = m_hCellData.hCellStart[grid_hash];
			uint start_index_poly = 0xffffffff;

			if(m_hCellData.uNumPolyHash) start_index_poly = m_hCellData.hPolyCellStart[grid_hash];
		
			if(start_index != 0xffffffff){
				glColor3dv(col2.data);
				disp = true;
			}
			if(start_index_poly != 0xffffffff){
				glColor3dv(col.data);
				disp = true;
			}

			if(disp){
				DrawCell(i, j, k);
			}
		}
	}

	glPopMatrix();
}

#endif // #ifndef _RX_NNSEARCH_H_