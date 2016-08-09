/*!
  @file rx_ssm.h
	
  @brief Screen Space Mesh作成
 		- M. Muller et al., Screen space meshes, SCA2007, 2007. 

  @author Makoto Fujisawa
  @date 2011-05
*/
// FILE --rx_ssm.h--

#ifndef _RX_SSM_H_
#define _RX_SSM_H_

#pragma warning (disable: 4244)


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <cmath>
#include <cstdlib>

#include <iostream>
#include <fstream>

// STL
#include <vector>
#include <string>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// データ構造
#include "rx_utility.h"
#include "rx_mesh.h"
#include "rx_matrix.h"


using namespace std;



//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
#ifndef RXREAL
	#define RXREAL float
#endif


//! SSMパラメータ
struct rxSSMParameter
{
	double PrtRad;		//!< パーティクルの半径
	double Spacing;		//!< デプスマップのサンプリング間隔
	double Zmax;		//!< 輪郭となるデプス差の閾値
	int Nfilter;		//!< デプス値平滑化のフィルタサイズ
	int Niters;			//!< 輪郭平滑化の反復回数

	rxSSMParameter()
	{
		PrtRad = 0.02;
		Spacing = 2;
		Zmax = 1.8*PrtRad;
		Nfilter = 1;
		Niters = 3;
	}
};

//! Screen Spaceでのパーティクル
struct rxSSParticle
{
	Vec3 xp;	//!< スクリーンスペースでの中心座標
	Vec3 rp;	//!< スクリーンスペースでの半径
};

//! Screen Space での頂点
struct rxSSVertex
{
	Vec3 pos;		//!< Screen Space座標値
	RXREAL depth;	//!< デプス値
	int edge;		//!< エッジ頂点であるかどうか

	Vec3 avg_pos;	//!< 隣接頂点平均座標値(輪郭平滑化用)
	RXREAL avg_num;	//!< 隣接頂点数(輪郭平滑化用)

	rxSSVertex(Vec3 p, int e = 0) : pos(Vec3(p[0], p[1], p[2])), depth(p[2]), edge(e) {}
};

//! 輪郭エッジ
struct rxSSEdge
{
	Vec3 x0, x1;	//!< 端点座標
	RXREAL depth;	//!< エッジデプス値
	RXREAL d0, d1;	//!< 端点デプス値
	int xy;			//!< エッジ方向(x:0, y:1)
	int silhouette;	
	int front_vertex;	//!< エッジ頂点のインデックス
	RXREAL dx;			//!< デプス値が小さい端点からエッジ頂点までの距離
};

//! メッシュ生成用グリッド
struct rxSSGrid
{
	int i, j;
	int node_vrts[4];	//!< ノード頂点インデックス
	int num_nv;			//!< ノード頂点数
	int edge_vrts[4];	//!< エッジ頂点(front vertex)
	int num_ev;			//!< エッジ頂点数(front vertex)
	int back_vrts[6];	//!< エッジ頂点(back vertex, back-2 vertex)
	int num_bv;			//!< エッジ頂点数(back vertex)

	RXREAL node_depth[4];	//!< ノードのデプス値

	int table_index0;	//!< デバッグ用:メッシュ化のためのインデックス値
	int table_index1;	//!< デバッグ用:メッシュ化のためのインデックス値
	int mesh_num;		//!< デバッグ用:メッシュ数
	int mesh[6];		//!< デバッグ用:メッシュインデックス
	int back2;			//!< デバッグ用
	int v[14];			//!< デバッグ用
};


// ノード頂点
// 3 - 2
// |   |
// 0 - 1
	 
// エッジ頂点
// - 2 -
// 3   1
// - 0 -

//-----------------------------------------------------------------------------
// 関数プロトタイプ宣言
//-----------------------------------------------------------------------------
inline RXREAL GetDepthNearestT(double x, double y, int nx, int ny, double dx, double dy, const vector<RXREAL> &dmap);
inline RXREAL GetDepthInterpT(double x, double y, int nx, int ny, double dx, double dy, const vector<RXREAL> &dmap);


//-----------------------------------------------------------------------------
// Screen Space Mesh - 基底クラス
// MARK:rxSSMesh
//-----------------------------------------------------------------------------
class rxSSMesh
{
protected:
	// メンバ変数
	double m_fDmx, m_fDmy;	//!< デプスマップの各グリッド幅
	int m_iNumNodeVrts;		//!< ノード頂点数
	int m_iNumEdgeVrts;		//!< エッジ頂点数
	int m_iNumMesh;			//!< メッシュ数

	double m_fSpacing;		//!< デプスマップのサンプリング間隔
	double m_fPrtRad;		//!< パーティクルの半径
	double m_fSSZmax;		//!< 輪郭となるデプス差の閾値
	int m_iNfilter;			//!< デプス値平滑化のフィルタサイズ
	int m_iNiters;			//!< 輪郭平滑化の反復回数
	int m_iNgx, m_iNgy;		//!< メッシュ生成用グリッドの解像度

protected:
	//! デフォルトコンストラクタ
	rxSSMesh(){}	// 引数指定なしでオブジェクトが作成されるのを防ぐ

public:
	//! コンストラクタ
	rxSSMesh(double zmax, double h, double r, int n_filter, int n_iter)
	{
		m_fSSZmax = zmax;
		m_fSpacing = h;
		m_fPrtRad = r;
		m_iNfilter = n_filter;
		m_iNiters = n_iter;
		m_iNgx = m_iNgy = 24;

		m_iNumNodeVrts = 0;
		m_iNumEdgeVrts = 0;
		m_iNumMesh = 0;
	}

	rxSSMesh(rxSSMParameter params)
	{
		m_fSSZmax  = params.Zmax;
		m_fSpacing = params.Spacing;
		m_fPrtRad  = params.PrtRad;
		m_iNfilter = params.Nfilter;
		m_iNiters  = params.Niters;
		m_iNgx = m_iNgy = 24;

		m_iNumNodeVrts = 0;
		m_iNumEdgeVrts = 0;
		m_iNumMesh = 0;
	}

	//! デストラクタ
	virtual ~rxSSMesh()
	{
	}

public:
	//
	// アクセスメソッド
	//
	//! グリッド数
	int GetNx(void) const { return m_iNgx; }
	int GetNy(void) const { return m_iNgy; }
	void GetDepthMapSize(int &nx, int &ny){ nx = m_iNgx+1; ny = m_iNgy+1; }

	//! デプスマップのサンプリング間隔
	void   SetSpacing(double h){ m_fSpacing = h; }
	double GetSpacing(void) const { return m_fSpacing; }

	//! パーティクル半径
	void   SetRadius(double r){ m_fPrtRad = r; }
	double GetRadius(void) const { return m_fPrtRad; }

	//! 輪郭となるデプス差の閾値
	void   SetZMax(double r){ m_fSSZmax = r; }
	double GetZMax(void) const { return m_fSSZmax; }

	//! デプス値平滑化のフィルタサイズ
	void SetFilterRadius(int r){ m_iNfilter = r; }
	int  GetFilterRadius(void) const { return m_iNfilter; }

	//! 輪郭平滑化の反復回数
	void SetSmoothIter(int r){ m_iNiters = r; }
	int  GetSmoothIter(void) const { return m_iNiters; }

	//! 頂点数
	int GetVertexNum(void) const { return m_iNumNodeVrts+m_iNumEdgeVrts; }

	//! メッシュ数
	int GetMeshNum(void) const { return m_iNumMesh; }

	//! デプスマップ
	virtual RXREAL* GetDepthMap(void) = 0;

	//! 頂点情報
	virtual rxSSVertex* GetSSVertex(void) = 0;

	//! メッシュ生成用グリッド情報
	virtual rxSSGrid GetMeshGrid(int idx) = 0;
	virtual rxSSGrid GetMeshGrid(int i, int j) = 0;

public:
	//
	// 処理関数
	//

	/*!
	 * スクリーンスペースメッシュ生成
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
							vector<Vec3> &vrts, vector<rxFace> &mesh, int filtering = 0, int debug_output = 0) = 0;

	/*!
	 * スクリーンスペースメッシュ生成(法線計算含む)
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] nrms 頂点法線列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
							vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &mesh, 
							int filtering = 0, int debug_output = 0) = 0;

	/*!
	 * マップやグリッド配列のサイズを変更
	 * @param[in] W,H 画面解像度
	 * @param[in] spacing デプスマップのサンプリング間隔
	 */
	virtual void Resize(int W, int H, int spacing = -1) = 0;

	/*!
	 * 頂点法線計算
	 * @param[in] vrts 頂点座標
	 * @param[in] nvrts 頂点数
	 * @param[in] tris 三角形ポリゴン幾何情報
	 * @param[in] ntris 三角形ポリゴン数
	 * @param[out] nrms 法線
	 */
	virtual void CalVertexNormals(const vector<Vec3> &vrts, unsigned int nvrts, const vector< vector<int> > &tris, unsigned int ntris, 
								  vector<Vec3> &nrms){}

public:
	// OpenGL描画
	virtual void DrawSilhouetteEdge(void){}
	virtual void DrawSSVertex(Vec3 node_color, Vec3 edge_color){}
	virtual void DrawMeshGrid(int grid, const Vec3 colors[]){}
	virtual void DrawField(double minpos[2], double maxpos[2]){}

	virtual void DrawSSEdge(void){}

public:
	// デバッグ用
	virtual void OutputGridInfo(int grid){}
	virtual void OutputGridVertexInfo(int grid){}
};


//-----------------------------------------------------------------------------
// Screen Space Mesh - CPUでの実装
// MARK:rxSSMeshCPU
//-----------------------------------------------------------------------------
class rxSSMeshCPU : public rxSSMesh
{
protected:
	// テーブルインデックス更新関数ポインタ
	typedef void (rxSSMeshCPU::*FuncTableIndex)(int&, int&, int[], rxSSGrid*);

protected:
	// メンバ変数
	vector<RXREAL> m_vSSDMap;		//!< デプスマップ

	vector<rxSSParticle> m_vSSPrts;		//!< 正規化座標系でのパーティクル

	vector<rxSSEdge> m_vSSEdge;			//!< 輪郭エッジ
	vector<rxSSVertex> m_vSSEdgeVertex;	//!< エッジ頂点
	vector<rxSSVertex> m_vSSVertex;		//!< スクリーンスペースメッシュ頂点

	vector<rxSSGrid> m_vSSMGrid;		//!< メッシュ生成用グリッド
	vector<int> m_vMeshGrid;			//!< デバッグ用:メッシュが属するグリッド

	vector< vector<double> > m_vFilter;	//!< 平滑化用binomial係数

	FuncTableIndex m_FuncTableIndex[25];//!< テーブルインデックス更新関数ポインタ

protected:
	//! デフォルトコンストラクタ
	rxSSMeshCPU(){}	// 引数指定なしでオブジェクトが作成されるのを防ぐ

public:
	//! コンストラクタ
	rxSSMeshCPU(double zmax, double h, double r, int n_filter, int n_iter);
	rxSSMeshCPU(rxSSMParameter params);

	//! デストラクタ
	virtual ~rxSSMeshCPU();

public:
	//
	// アクセスメソッド
	//
	
	//! デプスマップ
	virtual RXREAL* GetDepthMap(void);

	//! 頂点情報
	virtual rxSSVertex* GetSSVertex(void);

	//! メッシュ生成用グリッド情報
	virtual rxSSGrid GetMeshGrid(int idx);
	virtual rxSSGrid GetMeshGrid(int i, int j){ return GetMeshGrid(i+j*m_iNgx); }

public:
	//
	// 処理関数
	//

	/*!
	 * スクリーンスペースメッシュ生成
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
					        vector<Vec3> &vrts, vector<rxFace> &mesh, int filtering = 0, int debug_output = 0);

	/*!
	 * スクリーンスペースメッシュ生成(法線計算含む)
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] nrms 頂点法線列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
							vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &mesh, 
							int filtering = 0, int debug_output = 0);

	/*!
	 * マップやグリッド配列のサイズを変更
	 * @param[in] W,H 画面解像度
	 * @param[in] spacing デプスマップのサンプリング間隔
	 */
	virtual void Resize(int W, int H, int spacing = -1);

	/*!
	 * 頂点法線計算
	 * @param[in] vrts 頂点座標
	 * @param[in] nvrts 頂点数
	 * @param[in] tris 三角形ポリゴン幾何情報
	 * @param[in] ntris 三角形ポリゴン数
	 * @param[out] nrms 法線
	 */
	virtual void CalVertexNormals(const vector<Vec3> &vrts, unsigned int nvrts, const vector<rxFace> &tris, unsigned int ntris, vector<Vec3> &nrms);

	/*!
	 * パーティクル座標と半径を透視投影変換してデプスマップを生成
	 * @param[in] P 透視投影行列
	 * @param[in] MV モデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 */
	void CalDepthMap(const rxMatrix4 &P, const rxMatrix4 &MV, int W, int H, vector<Vec3> &prts, int pnum);

	/*!
	 * デプスマップに平滑化を施す
	 * @param[inout] dmap デプスマップ
	 * @param[in] nx,ny マップ解像度
	 * @param[in] n_filter フィルタリング幅
	 */
	void ApplyDepthFilter(vector<RXREAL> &dmap, int nx, int ny, int n_filter);

	/*!
	 * 輪郭に平滑化を施す
	 * @param[inout] ssvrts スクリーン座標系での頂点列
	 * @param[in] polys メッシュ(構成する頂点列)
	 * @param[in] n_iter フィルタリング反復数
	 */
	void ApplySilhoutteSmoothing(vector<rxSSVertex> &ssvrts, const vector<rxFace> &polys, int n_iter);

	/*!
	 * 輪郭エッジの検出とfront edge vertexの計算
	 * @param[in] nx,ny メッシュ生成用グリッド解像度
	 * @param[in] dw,dh メッシュ生成用グリッド幅
	 * @param[in] dgrid メッシュ生成用デプスマップ
	 * @param[in] ssprts Screen Spaceでのパーティクル
	 * @param[in] W,H スクリーン解像度
	 * @return 検出された輪郭エッジ数
	 */
	int DetectSilhouetteEdgeVertex(int nx, int ny, double dw, double dh, const vector<RXREAL> &dgrid, 
								   const vector<rxSSParticle> &ssprts, int W, int H);

	/*!
	 * 輪郭エッジの検出
	 * @param[in] nx,ny メッシュ生成用グリッド解像度
	 * @param[in] dw,dh メッシュ生成用グリッド幅
	 * @param[in] dgrid メッシュ生成用デプスマップ
	 * @return 検出された輪郭エッジ数
	 */
	int DetectSilhouetteEdge(int nx, int ny, double dw, double dh, const vector<RXREAL> &dgrid);


	/*!
	 * ノード頂点生成
	 * @param[in] nx,ny メッシュ生成用グリッド解像度
	 * @param[in] dw,dh メッシュ生成用グリッド幅
	 * @param[in] dgrid メッシュ生成用デプスマップ
	 * @return 生成されたノード頂点数
	 */
	int CalNodeVertex(int nx, int ny, double dw, double dh, const vector<RXREAL> &dgrid);

	/*!
	 * エッジ頂点生成
	 * @param[in] nx,ny メッシュ生成用グリッド解像度
	 * @param[in] dw,dh メッシュ生成用グリッド幅
	 * @param[in] dgrid メッシュ生成用デプスマップ
	 * @param[in] edges 輪郭エッジリスト
	 * @return 生成されたエッジ頂点数
	 */
	int CalEdgeVertex(int nx, int ny, double dw, double dh, const vector<RXREAL> &dgrid, const vector<rxSSEdge> &edges);

	/*!
	 * 三角形メッシュ生成
	 * @param[in] nx,ny メッシュ生成用グリッド解像度
	 * @param[in] dgrid メッシュグリッド
	 * @param[out] polys 三角形メッシュ
	 * @param[in] vstart 頂点インデックスの始点
	 * @return 生成された三角形メッシュ数
	 */
	int CalMesh(int nx, int ny, vector<rxSSGrid> &grid, vector<rxFace> &polys, int vstart = 0);

protected:
	//! エッジテーブル,フィルター用Binomialsの初期化
	void initTable(void);

	// 二分探索による頂点位置計算
	double binarySearchDepth(Vec3 v1, Vec3 v2, Vec3 &vr, double zmax);

	// デプス値の参照
	RXREAL getDepthValue(Vec3 x)
	{
		return GetDepthInterpT(x[0], x[1], m_iNgx+1, m_iNgy+1, m_fDmx, m_fDmy, m_vSSDMap);
		//return GetDepthNearestT(x[0], x[1], m_iNgx+1, m_iNgy+1, m_fDmx, m_fDmy, m_vSSDMap);
	}

	// メッシュテーブルインデックスの更新
	void updateTableIndexAny(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE0N4(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE1(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE2N4(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE3N23(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE3N4(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE4N2(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE4N3(int &table_index, int &vrot, int v[], rxSSGrid *g);
	void updateTableIndexE4N4(int &table_index, int &vrot, int v[], rxSSGrid *g);


public:
	// OpenGL描画
	virtual void DrawSilhouetteEdge(void);
	virtual void DrawSSVertex(Vec3 node_color, Vec3 edge_color);
	virtual void DrawMeshGrid(int grid, const Vec3 colors[]);
	virtual void DrawField(double minpos[2], double maxpos[2]);

	virtual void DrawSSEdge(void);

public:
	// デバッグ用
	virtual void OutputGridInfo(int grid);
	virtual void OutputGridVertexInfo(int grid);

	int Mesh2Grid(int mesh_idx){ return m_vMeshGrid[mesh_idx]; }

};

#if 1
struct rxSSGridG;
struct rxSSEdgeG;

//-----------------------------------------------------------------------------
// Screen Space Mesh - GPUでの実装
// MARK:rxSSMeshGPU
//-----------------------------------------------------------------------------
class rxSSMeshGPU : public rxSSMesh
{
protected:
	// メンバ変数
	vector<float> m_vBinomials;			//!< 平滑化用binomial係数(1D)

	vector<RXREAL> m_vSSDMap;			//!< デプスマップ
	vector<rxSSVertex> m_vSSVertex;		//!< スクリーンスペースメッシュ頂点

	// CUDA用変数
	RXREAL *m_dSSPrtsCen;				//!< スクリーン座標系でのパーティクル座標
	RXREAL *m_dSSPrtsRad;				//!< スクリーン座標系でのパーティクル半径

	RXREAL *m_dSSDMap;					//!< デプスマップ
	RXREAL *m_dSSVertex;				//!< 全頂点情報(ノード頂点，前面エッジ頂点，背面エッジ頂点，最背面エッジ頂点)
	rxSSGridG *m_dSSMGrid;				//!< メッシュ生成用グリッド
	unsigned int *m_dSSTriNum;					//!< グリッド内のメッシュ数
	unsigned int *m_dSSTriNumScan;				//!< グリッド内のメッシュ数のScan
	//unsigned int *m_dSSTriArray;				//!< 三角形メッシュ

	RXREAL *m_hSSPrtsCen;				//!< スクリーン座標系でのパーティクル座標(デバイスメモリとの交換用)
	RXREAL *m_hSSPrtsRad;				//!< スクリーン座標系でのパーティクル半径(デバイスメモリとの交換用)

	rxSSEdgeG *m_hSSEdge;				//!< スクリーン座標系でのパーティクル座標(デバイスメモリとの交換用)
	RXREAL *m_hSSVertex;				//!< スクリーン座標系でのパーティクル座標(デバイスメモリとの交換用)



protected:
	//! デフォルトコンストラクタ
	rxSSMeshGPU(){}	// 引数指定なしでオブジェクトが作成されるのを防ぐ

public:
	//! コンストラクタ
	rxSSMeshGPU(double zmax, double h, double r, int n_filter, int n_iter);
	rxSSMeshGPU(rxSSMParameter params);

	//! デストラクタ
	virtual ~rxSSMeshGPU();

public:
	//
	// アクセスメソッド
	//
	
	//! デプスマップ
	virtual RXREAL* GetDepthMap(void);

	//! 頂点情報
	virtual rxSSVertex* GetSSVertex(void);

	//! メッシュ生成用グリッド情報
	virtual rxSSGrid GetMeshGrid(int idx);
	virtual rxSSGrid GetMeshGrid(int i, int j){ return GetMeshGrid(i+j*m_iNgx); }

public:
	//
	// 処理関数
	//

	/*!
	 * スクリーンスペースメッシュ生成
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
							vector<Vec3> &vrts, vector<rxFace> &mesh, 
							int filtering = 0, int debug_output = 0);

	/*!
	 * スクリーンスペースメッシュ生成(法線計算含む)
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] vrts メッシュ頂点列
	 * @param[out] nrms 頂点法線列
	 * @param[out] polys メッシュ列
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	virtual void CreateMesh(double *proj, double *modelview, int W, int H, vector<Vec3> &prts, int pnum, 
							vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &mesh, 
							int filtering = 0, int debug_output = 0);

	/*!
	 * スクリーンスペースメッシュ生成
	 *  - パーティクルがデバイスメモリに格納されている場合
	 *  - 法線も計算
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] dvrt メッシュ頂点列(デバイスメモリ)
	 * @param[out] dtri メッシュ列(デバイスメモリ)
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	void CreateMeshD(double *proj, double *modelview, int W, int H, RXREAL *dppos, RXREAL *dprad, int pnum, int pdim, 
					 RXREAL* &dvrt, unsigned int* &dtri, int filtering = 0, int debug_output = 0);
	/*!
	 * スクリーンスペースメッシュ生成
	 *  - パーティクルがデバイスメモリに格納されている場合
	 *  - 法線も計算
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] dvrt 頂点列(デバイスメモリ)
	 * @param[out] dnrm 頂点法線列(デバイスメモリ)
	 * @param[out] dtri メッシュ列(デバイスメモリ)
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	void CreateMeshD(double *proj, double *modelview, int W, int H, RXREAL *dppos, RXREAL *dprad, int pnum, int pdim, 
					 RXREAL* &dvrt, RXREAL* &dnrm, unsigned int* &dtri, int filtering = 0, int debug_output = 0);

	/*!
	 * スクリーンスペースメッシュ生成
	 *  - パーティクルがデバイスメモリに格納されている場合
	 *  - VBOに出力
	 * @param[in] proj OpenGL透視投影行列
	 * @param[in] modelview OpenGLモデルビュー行列
	 * @param[in] W,H 画面解像度
	 * @param[in] prts パーティクル座標
	 * @param[in] pnum パーティクル数
	 * @param[out] uvrt_vbo メッシュ頂点列VBO
	 * @param[out] unrm_vbo メッシュ頂点列VBO
	 * @param[out] utri_vbo メッシュ頂点列VBO
	 * @param[in] filtering フィルタリングフラグ(デプス値0x01, 輪郭0x02)
	 * @param[in] debug_output 結果の画面出力の有無
	 */
	void CreateMeshVBO(double *proj, double *modelview, int W, int H, RXREAL *dppos, RXREAL *dprad, int pnum, int pdim, 
					   GLuint &uVrtVBO, GLuint &uNrmVBO, GLuint &uTriVBO, int filtering = 0, int debug_output = 0);

	/*!
	 * マップやグリッド配列のサイズを変更
	 * @param[in] W,H 画面解像度
	 * @param[in] spacing デプスマップのサンプリング間隔
	 */
	virtual void Resize(int W, int H, int spacing = -1);

	/*!
	 * 頂点法線計算
	 * @param[in] vrts 頂点座標
	 * @param[in] nvrts 頂点数
	 * @param[in] tris 三角形ポリゴン幾何情報
	 * @param[in] ntris 三角形ポリゴン数
	 * @param[out] nrms 法線
	 */
	virtual void CalVertexNormals(const vector<Vec3> &vrts, unsigned int nvrts, const vector< vector<int> > &tris, 
								  unsigned int ntris, vector<Vec3> &nrms);

protected:
	//! 初期化
	void init(void);

	//! GPU側のパラメータファイルを更新
	void updateParams(int W, int H, const rxMatrix4 &P, const rxMatrix4 &PMV);

	void calVertexNormalD(RXREAL* dvrt, unsigned int nvrts, unsigned int* dtri, unsigned int ntris, RXREAL* &nrms);

public:
	// OpenGL描画
	virtual void DrawSilhouetteEdge(void);
	virtual void DrawSSVertex(Vec3 node_color, Vec3 edge_color);
	virtual void DrawMeshGrid(int grid, const Vec3 colors[]);
	virtual void DrawField(double minpos[2], double maxpos[2]);

	virtual void DrawSSEdge(void);

public:
	// デバッグ用
	virtual void OutputGridInfo(int grid);
	virtual void OutputGridVertexInfo(int grid);

};
#endif


//-----------------------------------------------------------------------------
// 関数
//-----------------------------------------------------------------------------
/*!
 * クランプ
 * @param[inout] x 値
 * @param[in] a,b [a,b]でクランプ
 */
template<class T> 
inline void RX_CLAMP2(T &x, const T &a, const T &b){ x = (x < a ? a : (x > b ? b : x)); }

inline void Swap(Vec4 &a, Vec4 &b)
{
	Vec4 tmp = a;
	a = b;
	b = tmp;
}


inline RXREAL GetDepthInterpT(double x, double y, int nx, int ny, double dx, double dy, const vector<RXREAL> &dmap)
{
	int ix0 = (int)(x/dx);
	int ix1 = ix0+1;
	double ddx = x/dx-ix0;

	RX_CLAMP2(ix0, 0, nx-1);
	RX_CLAMP2(ix1, 0, nx-1);

	int iy0 = (int)(y/dx);
	int iy1 = iy0+1;
	double ddy = y/dy-iy0;

	RX_CLAMP2(iy0, 0, ny-1);
	RX_CLAMP2(iy1, 0, ny-1);

	RXREAL d00 = dmap[iy0*nx+ix0];
	RXREAL d01 = dmap[iy0*nx+ix1];
	RXREAL d10 = dmap[iy1*nx+ix0];
	RXREAL d11 = dmap[iy1*nx+ix1];

	return (d00*(1.0-ddx)+d01*ddx)*(1.0-ddy)+(d10*(1.0-ddx)+d11*ddx)*ddy;
}


inline RXREAL GetDepthNearestT(double x, double y, int nx, int ny, double dx, double dy, const vector<RXREAL> &dmap)
{
	int ix = (int)(x/dx);
	RX_CLAMP2(ix, 0, nx-1);

	int iy = (int)(y/dx);
	RX_CLAMP2(iy, 0, ny-1);

	return dmap[iy*nx+ix];
}

// 二分探索(1D)
inline double rtbis(double func(const double), const double x1, const double x2, const double xacc)
{
	const int JMAX = 40;
	double dx, f, fmid, xmid, rtb;

	f = func(x1);
	fmid = func(x2);
	if(f*fmid >= 0.0) return 0.0;

	rtb = f < 0.0 ? (dx = x2-x1, x1) : (dx = x1-x2, x2);
	for(int j = 0; j < JMAX; ++j){
		dx *= 0.5;
		xmid = rtb+dx;

		fmid = func(xmid);

		if(fmid <= 0.0){
			rtb = xmid;
		}

		if(fabs(dx) < xacc || fmid == 0.0){
			return rtb;
		}
	}
	
	return 0.0;
}


/*!
 * OpenGL変換行列をrxMatrix4に変換
 *  column major を row major に変換してrxMatrix4に格納
 * @param[in] m OpenGL変換行列(	glGetDoublev(GL_PROJECTION_MATRIX, m); などで取得)
 * @return 変換後の行列
 */
inline rxMatrix4 GetMatrixGL(double *m)
{
	return rxMatrix4(m[0], m[4], m[8],  m[12], 
					 m[1], m[5], m[9],  m[13], 
					 m[2], m[6], m[10], m[14], 
					 m[3], m[7], m[11], m[15]);
}


/*!
 * 整数を2進数文字列に変換
 * @param[in] x 元の整数
 * @param[in] bit 2進数桁数
 * @return 2進数文字列
 */
inline string GetBitArray(int x, int bit)
{
    string s;
    s.resize(bit, '0');
    for(int i = 0; i < bit; ++i){
        s[bit-i-1] = ((x >> i) & 0x01) ? '1' : '0';
    }
    return s;
}

/*!
 * デプス値を[0,1]に変換
 * @param[in] depth デプス値
 * @param[in] w division値
 * @return [0,1]の値
 */
inline double RX_DEPTH2COLORf(RXREAL depth)
{
	if(depth == RX_FEQ_INF){
		depth = 1.0;
	}
	else{
		depth *= 0.5;
	}
	return 1.0-RX_CLAMP(depth, (RXREAL)0.0, (RXREAL)1.0);
}
inline unsigned char RX_DEPTH2COLOR(RXREAL d){ return (unsigned char)(RX_DEPTH2COLORf(d)*255); }


/*!
 * 平滑化用のbinominal係数を計算
 * @param[in] b 次数
 * @return binominal係数列
 */
static vector< vector<double> > CalBinomials(int b)
{
	vector< vector<double> > bs;
	vector<double> f, tmp;
	f.resize(b+1);
	tmp.resize(b+1);

	bs.resize(b+1);

	double a = 1.0;

	for(int i = 0; i < b+1; ++i){
		f[i]   = (i == 0 ? 1 : 0);
		tmp[i] = (i == 0 ? 1 : 0);	
	}

	for(int k = 0; k < b+1; ++k){
		for(int i = 1; i < k+1; ++i){
			tmp[i] = f[i-1]+f[i];
		}

		for(int i = 1; i < k+1; ++i){
			f[i] = tmp[i];
		}

		bs[k].resize(k+1);
		for(int i = 0; i < k+1; ++i){
			bs[k][i] = f[i]*a;
		}

		a *= 0.5;
	}

	return bs;
}

/*!
 * 平滑化用のbinominal係数を計算(1次元配列版, フィルタに用いる物のみ)
 * @param[in] r フィルタ幅(次数b = 2*r+1)
 * @return binominal係数列
 */
static vector<float> CalBinomialsForFilter(int r)
{
	int b = 2*r+1;
	vector<float> bs;
	vector<float> f, tmp;
	f.resize(b+1);
	tmp.resize(b+1);

	bs.resize((r+1)*(r+1));

	float a = 1.0;

	for(int i = 0; i < b+1; ++i){
		f[i]   = (i == 0 ? 1.0f : 0.0f);
		tmp[i] = (i == 0 ? 1.0f : 0.0f);	
	}

	int c = 0;
	for(int k = 0; k < b+1; ++k){
		for(int i = 1; i < k+1; ++i){
			tmp[i] = f[i-1]+f[i];
		}

		for(int i = 1; i < k+1; ++i){
			f[i] = tmp[i];
		}

		if(!(k%2)){
			for(int i = 0; i < k+1; ++i){
				bs[c++] = f[i]*a;
			}
		}

		a *= 0.5;
	}

	return bs;
}

/*!
 * 線分と円の交差判定(2D)
 * @param[in] A,B 線分の両端点座標
 * @param[in] C 円の中心
 * @param[in] r 円の半径
 * @param[out] P 交点座標
 * @return 交点数
 */
static int LineCircleIntersection(const Vec2 &A, const Vec2 &B, const Vec2 &C, const double &r, Vec2 P[2], double t[2])
{
	double rr = r*r;
	Vec2 AC = C-A;
	Vec2 BC = C-B;

	Vec2 v = B-A;
	double l = norm(v);
	v /= l;

	double td = dot(v, AC);
	Vec2 D = A+td*v;
	double dd = norm2(D-C);

	if(dd < rr){
		double dt = sqrt(rr-dd);

		double da = rr-norm2(AC);
		double db = rr-norm2(BC);

		int inter = 0;
		double t1 = td-dt;
		double t2 = td+dt;
		if(t1 >= 0 && t1 <= l){
			P[inter] = A+t1*v;
			t[inter] = t1;
			inter++;
		}
		if(t2 >= 0 && t2 <= l){
			P[inter] = A+t2*v;
			t[inter] = t2;
			inter++;
		}

		return inter;
	}
	else{
		return 0;
	}
}




#endif // #ifndef _RX_SSM_H_