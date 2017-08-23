/*!
  @file rx_mc.h
	
  @brief 陰関数表面からのポリゴン生成(MC法)
	
	http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
 
  @author Raghavendra Chandrashekara (basesd on source code
			provided by Paul Bourke and Cory Gene Bloyd)
  @date   2010-03
*/


#ifndef _RX_MC_MESH_H_
#define _RX_MC_MESH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <cstdlib>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// STL
#include <map>
#include <vector>
#include <string>

#include <iostream>

#include "rx_utility.h"
#include "rx_mesh.h"


//-----------------------------------------------------------------------------
// 名前空間
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
typedef unsigned int uint;

#ifndef RXREAL
	#define RXREAL float
#endif


struct RxVertexID
{
	uint newID;
	double x, y, z;
};

typedef std::map<uint, RxVertexID> ID2VertexID;

struct RxTriangle
{
	uint pointID[3];
};

typedef std::vector<RxTriangle> RxTriangleVector;

struct RxScalarField
{
	uint iNum[3];
	Vec3 fWidth;
	Vec3 fMin;
};



//-----------------------------------------------------------------------------
// rxMCMeshCPUクラス
//-----------------------------------------------------------------------------
class rxMCMeshCPU
{
public:
	// コンストラクタ
	rxMCMeshCPU();

	// デストラクタ
	~rxMCMeshCPU();
	
	//! 陰関数から三角形メッシュを生成
	bool CreateMeshF(RXREAL (*func)(double, double, double), Vec3 min_p, double h, int n[3], RXREAL threshold, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから三角形メッシュを生成
	bool CreateMeshV(RXREAL *field, Vec3 min_p, double h, int n[3], RXREAL threshold, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから等値面メッシュ生成
	void GenerateSurface(const RxScalarField sf, RXREAL *field, RXREAL threshold, 
						 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris);

	//! 関数からサンプルボリュームを作成して等値面メッシュ生成
	void GenerateSurfaceV(const RxScalarField sf, RXREAL (*func)(double, double, double), RXREAL threshold, 
						  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris);

	//! 関数から等値面メッシュ生成
	void GenerateSurfaceF(const RxScalarField sf, RXREAL (*func)(double, double, double), RXREAL threshold, 
						  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris);

	//! 等値面作成が成功したらtrueを返す
	bool IsSurfaceValid() const { return m_bValidSurface; }

	//! 作成下等値面メッシュの破棄
	void DeleteSurface();

	//! メッシュ化に用いたグリッドの大きさ(メッシュ作成していない場合は返値が-1)
	int GetVolumeLengths(double& fVolLengthX, double& fVolLengthY, double& fVolLengthZ);

	// 作成したメッシュの情報
	uint GetNumVertices(void) const { return m_nVertices; }
	uint GetNumTriangles(void) const { return m_nTriangles; }
	uint GetNumNormals(void) const { return m_nNormals; }

protected:
	// MARK:メンバ変数
	uint m_nVertices;	//!< 等値面メッシュの頂点数
	uint m_nNormals;	//!< 等値面メッシュの頂点法線数(作成されていれば 法線数=頂点数)
	uint m_nTriangles;	//!< 等値面メッシュの三角形ポリゴン数

	ID2VertexID m_i2pt3idVertices;			//!< 等値面を形成する頂点のリスト
	RxTriangleVector m_trivecTriangles;		//!< 三角形ポリゴンを形成する頂点のリスト

	RxScalarField m_Grid;					//!< 分割グリッド情報

	// 陰関数値(スカラー値)取得用変数(どちらかのみ用いる)
	const RXREAL* m_ptScalarField;				//!< スカラー値を保持するサンプルボリューム
	RXREAL (*m_fpScalarFunc)(double, double, double);	//!< スカラー値を返す関数ポインタ

	RXREAL m_tIsoLevel;							//!< 閾値

	bool m_bValidSurface;					//!< メッシュ生成成功の可否


	// メッシュ構築用のテーブル
	static const uint m_edgeTable[256];
	static const int m_triTable[256][16];



	// MARK:protectedメンバ関数

	//! エッジID
	uint GetEdgeID(uint nX, uint nY, uint nZ, uint nEdgeNo);

	//! 頂点ID
	uint GetVertexID(uint nX, uint nY, uint nZ);

	// エッジ上の等値点を計算
	RxVertexID CalculateIntersection(uint nX, uint nY, uint nZ, uint nEdgeNo);
	RxVertexID CalculateIntersectionF(uint nX, uint nY, uint nZ, uint nEdgeNo);

	//! グリッドエッジ両端の陰関数値から線型補間で等値点を計算
	RxVertexID Interpolate(double fX1, double fY1, double fZ1, double fX2, double fY2, double fZ2, RXREAL tVal1, RXREAL tVal2);

	//! 頂点，メッシュ幾何情報を出力形式で格納
	void RenameVerticesAndTriangles(vector<Vec3> &vrts, uint &nvrts, vector<int> &tris, uint &ntris);

	//! 頂点法線計算
	void CalculateNormals(const vector<Vec3> &vrts, uint nvrts, const vector<int> &tris, uint ntris, 
						  vector<Vec3> &nrms, uint &nnrms);

};


//-----------------------------------------------------------------------------
// rxMCMeshGPUクラス
//-----------------------------------------------------------------------------
class rxMCMeshGPU
{
protected:
	// MC法用
	uint m_u3GridSize[3];			//!< グリッド数(nx,ny,nz)
	uint m_u3GridSizeMask[3];		//!< グリッド/インデックス変換時のマスク
	uint m_u3GridSizeShift[3];		//!< グリッド/インデックス変換時のシフト量

	float m_f3VoxelMin[3];			//!< グリッド最小位置
	float m_f3VoxelMax[3];			//!< グリッド最大位置
	float m_f3VoxelH[3];			//!< グリッド幅
	uint m_uNumVoxels;				//!< 総グリッド数
	uint m_uMaxVerts;				//!< 最大頂点数
	uint m_uNumActiveVoxels;		//!< メッシュが存在するボクセル数
	uint m_uNumVrts;				//!< 総頂点数
	uint m_uNumTris;				//!< 総メッシュ数

	// デバイスメモリ
	float *g_dfVolume;				//!< 陰関数データを格納するグリッド
	float *g_dfNoise;				//!< ノイズ値を格納するグリッド(描画時の色を決定するのに使用)
	uint *m_duVoxelVerts;			//!< グリッドに含まれるメッシュ頂点数
	uint *m_duVoxelVertsScan;		//!< グリッドに含まれるメッシュ頂点数(Scan)
	uint *m_duCompactedVoxelArray;	//!< メッシュを含むグリッド情報

	uint *m_duVoxelOccupied;		//!< ポリゴンが内部に存在するボクセルのリスト
	uint *m_duVoxelOccupiedScan;	//!< ポリゴンが内部に存在するボクセルのリスト(prefix scan)

	// 幾何情報を生成するときに必要な変数
	uint m_u3EdgeSize[3][3];		//!< エッジ数(nx,ny,nz)
	uint m_uNumEdges[4];			//!< 総エッジ数

	// 幾何情報を生成するときに必要な変数
	uint *m_duVoxelCubeIdx;			//!< グリッド8頂点の陰関数値が閾値以上かどうかを各ビットに格納した変数
	uint *m_duEdgeOccupied;			//!< エッジにメッシュ頂点を含むかどうか(x方向，y方向, z方向の順)
	uint *m_duEdgeOccupiedScan;		//!< エッジにメッシュ頂点を含むかどうか(Scan)
	float *m_dfEdgeVrts;			//!< エッジごとに算出した頂点情報
	float *m_dfCompactedEdgeVrts;	//!< 隙間をつめた頂点情報
	uint *m_duIndexArray;			//!< ポリゴンの幾何情報
	float *m_dfVertexNormals;		//!< 頂点法線
	uint *m_duVoxelTriNum;			//!< グリッドごとの三角形メッシュ数
	uint *m_duVoxelTriNumScan;		//!< グリッドごとの三角形メッシュ数(Scan)

	// 幾何情報を必要としないときのみ用いる
	float *m_df4Vrts;				//!< ポリゴン頂点座標
	float *m_df4Nrms;				//!< ポリゴン頂点法線


	// ホストメモリ
	float *m_f4VertPos;			//!< 頂点座標
	uint *m_u3TriIdx;				//!< メッシュインデックス
	uint *m_uScan;					//!< デバッグ用

	float *m_f4VertNrm;			//!< 頂点法線


	int m_iVertexStore;
	bool m_bSet;


	// 陰関数値(スカラー値)取得用変数(どちらかのみ用いる)
	const float* m_ptScalarField;				//!< スカラー値を保持するサンプルボリューム
	float (*m_fpScalarFunc)(double, double, double);	//!< スカラー値を返す関数ポインタ

	float m_tIsoLevel;							//!< 閾値

	bool m_bValidSurface;						//!< メッシュ生成成功の可否

public:
	// コンストラクタ
	rxMCMeshGPU();

	// デストラクタ
	~rxMCMeshGPU();
	
	//! 陰関数から三角形メッシュを生成
	bool CreateMeshF(RXREAL (*func)(double, double, double), Vec3 min_p, double h, int n[3], float threshold, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから三角形メッシュを生成
	bool CreateMeshV(RXREAL *volume, Vec3 minp, double h, int n[3], float threshold, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームから三角形メッシュを生成
	bool CreateMeshV(Vec3 minp, double h, int n[3], float threshold, uint &nvrts, uint &ntris);
	
	//! 配列の確保
	bool Set(Vec3 minp, Vec3 h, int n[3], unsigned int vm = 5);

	//! 配列の削除
	void Clean(void);

	//! FBOにデータを設定
	bool SetDataToFBO(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO);

	//! ホスト側配列にデータを設定
	bool SetDataToArray(vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);

	//! サンプルボリュームセット
	void   SetSampleVolumeFromHost(float *hVolume);
	float* GetSampleVolumeDevice(void);

	void   SetSampleNoiseFromHost(float *hVolume);
	float* GetSampleNoiseDevice(void);

	// 最大頂点数
	uint GetMaxVrts(void){ return m_uMaxVerts; }

	// 最大頂点数計算用係数
	void SetVertexStore(int vs){ m_iVertexStore = vs; }

	//! 頂点データ(デバイスメモリ)
	float* GetVrtDev(void);

	//! メッシュデータ(デバイスメモリ)
	uint* GetIdxDev(void);

	//! 法線データ(デバイスメモリ)
	float* GetNrmDev(void);

	//! 頂点データ(ホストメモリ)
	float GetVertex(int idx, int dim)
	{
		if(dim == 0){
			return m_f4VertPos[4*idx+0];
		}
		else if(dim == 1){
			return m_f4VertPos[4*idx+1];
		}
		else{
			return m_f4VertPos[4*idx+2];
		}
	}
	void GetVertex2(int idx, float *x, float *y, float *z)
	{
		*x = m_f4VertPos[4*idx+0];
		*y = m_f4VertPos[4*idx+1];
		*z = m_f4VertPos[4*idx+2];
	}

	//! メッシュデータ(ホストメモリ)
	void GetTriIdx(int idx, unsigned int *vidx0, unsigned int *vidx1, unsigned int *vidx2)
	{
		*vidx0 = m_u3TriIdx[4*idx+0];
		*vidx1 = m_u3TriIdx[4*idx+1];
		*vidx2 = m_u3TriIdx[4*idx+2];
	}
};




#endif // _RX_MC_MESH_H_

