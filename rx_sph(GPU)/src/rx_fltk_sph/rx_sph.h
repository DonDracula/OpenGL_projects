/*!
  @file rx_sph.h
	
  @brief SPH法
 
*/
// FILE --rx_sph.h--

#ifndef _RX_SPH_H_
#define _RX_SPH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph_commons.h"

#include "rx_ps.h"			// パーティクルシステム基底クラス
#include "rx_nnsearch.h"	// グリッド分割による近傍探索

#include "rx_sph_solid.h"

#include "rx_cu_common.cuh"




//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
//#define GL_REAL GL_DOUBLE
#define GL_REAL GL_FLOAT

// 時間計測
class rxTimerAvg;
extern rxTimerAvg g_Time;

#define RX_USE_TIMER

#ifdef RX_USE_TIMER
#define RXTIMER_CLEAR g_Time.ClearTime()
#define RXTIMER_RESET g_Time.ResetTime()
#define RXTIMER(x) g_Time.Split(x)
#define RXTIMER_PRINT g_Time.Print()
#define RXTIMER_STRING(x) g_Time.PrintToString(x)
#else
#define RXTIMER_CLEAR
#define RXTIMER_RESET
#define RXTIMER(x) 
#define RXTIMER_PRINT
#define RXTIMER_STRING(x)
#endif


//! SPHシーンのパラメータ
struct rxSPHEnviroment
{
	int max_particles;			//!< 最大パーティクル数
	Vec3 boundary_cen;			//!< 境界の中心
	Vec3 boundary_ext;			//!< 境界の大きさ(各辺の長さの1/2)
	RXREAL dens;				//!< 初期密度
	RXREAL mass;				//!< パーティクルの質量
	RXREAL kernel_particles;	//!< 有効半径h以内のパーティクル数
	RXREAL dt;					//!< 時間ステップ幅

	int use_inlet;				//!< 流入境界条件の有無

	// 表面メッシュ
	Vec3 mesh_boundary_cen;		//!< メッシュ生成境界の中心
	Vec3 mesh_boundary_ext;		//!< メッシュ生成境界の大きさ(各辺の長さの1/2)
	int mesh_vertex_store;		//!< 頂点数からポリゴン数を予測するときの係数
	int mesh_max_n;				//!< MC法用グリッドの最大分割数

	rxSPHEnviroment()
	{
		max_particles = 50000;
		boundary_cen = Vec3(0.0);
		boundary_ext = Vec3(2.0, 0.8, 0.8);
		dens = (RXREAL)998.29;
		mass = (RXREAL)0.04;
		kernel_particles = (RXREAL)20.0;
		dt = 0.01;

		mesh_vertex_store = 10;
		use_inlet = 0;
		mesh_max_n = 128;
	}
};


//! 表面パーティクル
struct rxSurfaceParticle
{
	Vec3 pos;					//!< 中心座標
	Vec3 nrm;					//!< 法線
	Vec3 vel;					//!< 速度
	RXREAL d;					//!< 探索中心からの距離
	int idx;					//!< パーティクルインデックス
};

extern double g_fSurfThr[2];


//-----------------------------------------------------------------------------
// MARK:rxSPHクラスの宣言
//-----------------------------------------------------------------------------
class rxSPH : public rxParticleSystemBase
{
private:
	// パーティクル
	RXREAL *m_hNrm;					//!< パーティクル法線
	RXREAL *m_hFrc;					//!< パーティクルにかかる力
	RXREAL *m_hDens;				//!< パーティクル密度
	RXREAL *m_hPres;				//!< パーティクル圧力

	// 表面生成用(Anisotropic kernel)
	RXREAL *m_hUpPos;				//!< 平滑化パーティクル位置
	RXREAL *m_hPosW;				//!< 重み付き平均座標
	RXREAL *m_hCMatrix;				//!< 共分散行列
	RXREAL *m_hEigen;				//!< 共分散行列の特異値
	RXREAL *m_hRMatrix;				//!< 回転行列(共分散行列の特異ベクトル)
	RXREAL *m_hG;					//!< 変形行列

	uint *m_hSurf;					//!< 表面パーティクル

	// 境界・固体
	rxSolid *m_pBoundary;			//!< シミュレーション空間の境界
	vector<rxSolid*> m_vSolids;		//!< 固体物体
	RXREAL *m_hVrts;				//!< 固体ポリゴンの頂点
	int m_iNumVrts;					//!< 固体ポリゴンの頂点数
	int *m_hTris;					//!< 固体ポリゴン
	int m_iNumTris;					//!< 固体ポリゴンの数

	// 空間分割格子関連
	rxNNGrid *m_pNNGrid;			//!< 分割グリッドによる近傍探索
	vector< vector<rxNeigh> > m_vNeighs;	//!< 近傍パーティクル

	// 粒子パラメータ
	uint m_iKernelParticles;		//!< カーネル内のパーティクル数
	RXREAL m_fInitDens, m_fMass;	//!< 密度，質量
	RXREAL m_fEffectiveRadius;		//!< 有効半径

	// シミュレーションパラメータ
	RXREAL m_fGasStiffness;			//!< ガス定数
	RXREAL m_fViscosity;			//!< 粘性係数
	RXREAL m_fBuoyancy;				//!< 浮力

	// カーネル関数の計算の際に用いられる定数係数
	double m_fWpoly6;				//!< Pory6カーネルの定数係数
	double m_fGWpoly6;				//!< Pory6カーネルの勾配の定数係数
	double m_fLWpoly6;				//!< Pory6カーネルのラプラシアンの定数係数
	double m_fWspiky;				//!< Spikyカーネルの定数係数
	double m_fGWspiky;				//!< Spikyカーネルの勾配の定数係数
	double m_fLWspiky;				//!< Spikyカーネルのラプラシアンの定数係数
	double m_fWvisc;				//!< Viscosityカーネルの定数係数
	double m_fGWvisc;				//!< Viscosityカーネルの勾配の定数係数
	double m_fLWvisc;				//!< Viscosityカーネルのラプラシアンの定数係数

protected:
	rxSPH(){}

public:
	//! コンストラクタ
	rxSPH(bool use_opengl);

	//! デストラクタ
	virtual ~rxSPH();

	// パーティクル半径
	float GetEffectiveRadius(){ return m_fEffectiveRadius; }

	// 近傍パーティクル
	uint* GetNeighborList(const int &i, int &n);

public:
	//
	// 仮想関数
	//
	// パーティクルデータ
	virtual RXREAL* GetParticle(void);
	virtual RXREAL* GetParticleDevice(void);
	virtual void UnmapParticle(void){}

	// シミュレーションステップ
	virtual bool Update(RXREAL dt, int step = 0);

	// シーンの設定
	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris);
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, int flg);
	virtual void SetSphereObstacle(Vec3 cen, double rad, int flg);
	virtual void MoveSphereObstacle(int b, Vec3 disp);
	virtual Vec3 GetSphereObstaclePos(int b = -1);

	// ホスト<->VBO間転送
	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true);
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count);
	virtual void SetColorVBO(int type);

	// 陰関数値計算
	virtual double GetImplicit(double x, double y, double z);
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF);
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF);

	// SPH情報出力
	virtual void OutputSetting(string fn);

	// 描画関数
	virtual void DrawCell(int i, int j, int k);
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0);
	virtual void DrawObstacles(void);


public:
	// SPH初期化
	void Initialize(const rxSPHEnviroment &env);
	void Allocate(int max_particles);
	void Finalize(void);

	// 近傍取得
	void GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h = -1.0);
	void GetNearestNeighbors(int idx, RXREAL *p, vector<rxNeigh> &neighs, RXREAL h = -1.0);

	// 分割セルにパーティクルを格納
	virtual void SetParticlesToCell(void);
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h);

	// メタボールによる陰関数値
	double CalColorField(double x, double y, double z);

	// 分割セルにポリゴンを格納
	virtual void SetPolygonsToCell(void);

	int  GetPolygonsInCell(int gi, int gj, int gk, vector<int> &polys);
	bool IsPolygonsInCell(int gi, int gj, int gk);

	
	// 表面パーティクル検出
	void DetectSurfaceParticles(void);					// 表面パーティクルの検出
	double CalDistToNormalizedMassCenter(const int i);	// 近傍パーティクルの重心までの距離計算
	uint* GetArraySurf(void);							// 表面パーティクル情報の取得

	// 表面パーティクル情報の取得
	int GetSurfaceParticles(const Vec3 pos, RXREAL h, vector<rxSurfaceParticle> &sp);

	// 法線計算
	void CalNormalFromDensity(void);
	void CalNormal(void);


	//
	// Anisotropic kernel
	//
	virtual void CalAnisotropicKernel(void);

protected:
	// CPUによるSPH計算
	void calDensity(void);
	void calNormal(void);
	void calForce(void);

	// 位置と速度の更新(Leap-Frog)
	void integrate(const RXREAL *pos, const RXREAL *vel, const RXREAL *frc, 
				   RXREAL *pos_new, RXREAL *vel_new, RXREAL dt);

	// 衝突判定
	int calCollisionPolygon(uint grid_hash, Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt);
	int calCollisionSolid(Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt);
};


//-----------------------------------------------------------------------------
// MARK:rxSPH_GPUクラスの宣言
//-----------------------------------------------------------------------------
class rxSPH_GPU : public rxParticleSystemBase
{
private:
	//
	// メンバ変数(GPU変数)
	//
	RXREAL *m_dPos;			//!< パーティクル位置
	RXREAL *m_dVel;			//!< パーティクル速度
	RXREAL *m_dNrm;			//!< パーティクル法線
	RXREAL *m_dFrc;			//!< パーティクルにかかる力
	RXREAL *m_dDens;		//!< パーティクル密度
	RXREAL *m_dPres;		//!< パーティクル圧力

	uint *m_dAttr;			//!< パーティクル属性

	cudaGraphicsResource *m_pPosResource;	//!< OpenGL(のVBO)-CUDA間のデータ転送を扱うためのハンドル


	// 表面生成用(Anisotropic kernel)
	RXREAL *m_dUpPos;		//!< 平滑化パーティクル位置
	RXREAL *m_dPosW;		//!< 重み付き平均座標
	RXREAL *m_dCMatrix;		//!< 共分散行列
	RXREAL *m_dEigen;		//!< 共分散行列の特異値
	RXREAL *m_dRMatrix;		//!< 回転行列(共分散行列の特異ベクトル)
	RXREAL *m_dG;			//!< 変形行列

	// 表面メッシュ
	RXREAL *m_dVrts;		//!< 固体メッシュ頂点
	int    *m_dTris;		//!< 固体メッシュ

	// シミュレーションパラメータ
	rxSimParams m_params;	//!< シミュレーションパラメータ(GPUへのデータ渡し用)
	uint3 m_gridSize;		//!< 近傍探索グリッドの各軸の分割数
	uint m_numGridCells;	//!< 近傍探索グリッド分割総数
	
	// 空間分割(GPU)
	rxParticleCell m_dCellData;	//!< 近傍探索グリッド
	uint m_gridSortBits;		//!< ハッシュ値による基数ソート時の基数桁数

	//
	// メンバ変数(CPU変数)
	//
	RXREAL *m_hNrm;			//!< パーティクル法線
	RXREAL *m_hFrc;			//!< パーティクルにかかる力
	RXREAL *m_hDens;		//!< パーティクル密度
	RXREAL *m_hPres;		//!< パーティクル圧力

	uint *m_hSurf;			//!< 表面パーティクル

	// 表面生成用(Anisotropic kernel)
	RXREAL *m_hUpPos;		//!< 平滑化パーティクル位置
	RXREAL *m_hPosW;		//!< 重み付き平均座標
	RXREAL *m_hCMatrix;		//!< 共分散行列
	RXREAL *m_hEigen;		//!< 共分散行列の特異値
	RXREAL *m_hRMatrix;		//!< 回転行列(共分散行列の特異ベクトル)
	RXREAL *m_hG;			//!< 変形行列
	RXREAL  m_fEigenMax;	//!< 特異値の最大値(探索半径拡張に用いる)

	// 表面メッシュ
	vector<RXREAL> m_vVrts;	//!< 固体メッシュ頂点
	int m_iNumVrts;			//!< 固体メッシュ頂点数
	vector<int> m_vTris;	//!< 固体メッシュ
	int m_iNumTris;			//!< 固体メッシュ数

	bool m_bCalNormal;		//!< 法線計算フラグ

protected:
	rxSPH_GPU(){}

public:
	//! コンストラクタ
	rxSPH_GPU(bool use_opengl);

	//! デストラクタ
	~rxSPH_GPU();

	// パーティクル半径
	float GetEffectiveRadius(){ return m_params.EffectiveRadius; }

	// 近傍パーティクル
	uint* GetNeighborList(const int &i, int &n);

	// シミュレーションパラメータ
	rxSimParams GetParams(void){ return m_params; }
	void UpdateParams(void);

	// フラグ切替
	void ToggleNormalCalc(int t = -1){ RX_TOGGLE(m_bCalNormal, t); }			//!< パーティクル法線の計算
	bool IsNormalCalc(void) const { return m_bCalNormal; }
#if MAX_BOX_NUM
	void ToggleSolidFlg(int t = -1);
#endif

public:
	//
	// 仮想関数
	//
	// パーティクルデータ
	virtual RXREAL* GetParticle(void);
	virtual RXREAL* GetParticleDevice(void);
	virtual void UnmapParticle(void);

	// シミュレーションステップ
	virtual bool Update(RXREAL dt, int step = 0);

	// シーンの設定
	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris);
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, int flg);
	virtual void SetSphereObstacle(Vec3 cen, double rad, int flg);
	virtual void MoveSphereObstacle(int b, Vec3 disp);
	virtual Vec3 GetSphereObstaclePos(int b = -1);

	// ホスト<->VBO間転送
	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true);
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count);
	virtual void SetColorVBO(int type);


	// 陰関数値計算
	virtual double GetImplicit(double x, double y, double z);
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF);
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF);

	// SPH情報出力
	virtual void OutputSetting(string fn);

	// 描画関数
	virtual void DrawCell(int i, int j, int k);
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0);
	virtual void DrawObstacles(void);

protected:
	void setObjectToCell(RXREAL *p);


public:
	// SPH初期化
	void Initialize(const rxSPHEnviroment &env);
	void Allocate(int max_particles);
	void Finalize(void);

	// 分割セルにパーティクルを格納
	virtual void SetParticlesToCell(void);
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h);
	virtual void SetPolygonsToCell(void);

	int  GetPolygonsInCell(int gi, int gj, int gk, vector<int> &polys){ return 0; }
	bool IsPolygonsInCell(int gi, int gj, int gk){ return false; }

	void CalMaxDensity(int k);

	// 表面パーティクル検出
	void DetectSurfaceParticles(void);					// 表面パーティクルの検出
	double CalDistToNormalizedMassCenter(const int i);	// 近傍パーティクルの重心までの距離計算
	uint* GetArraySurf(void);							// 表面パーティクル情報の取得

	// 表面パーティクル情報の取得
	int GetSurfaceParticles(const Vec3 pos, RXREAL h, vector<rxSurfaceParticle> &sp);

	// 法線計算
	void CalNormalFromDensity(void);
	void CalNormal(void);
	
	//
	// Anisotropic kernel
	//
	virtual void CalAnisotropicKernel(void);

protected:
	// 分割セルの初期設定
	void setupCells(const Vec3 &vMin, const Vec3 &vMax, const double &h);

	// グリッドハッシュの計算
	uint calGridHash(uint x, uint y, uint z);
	uint calGridHash(Vec3 pos);

	// ポリゴンを分割セルに格納
	void setPolysToCell(RXREAL *vrts, int nv, int* tris, int nt);
};





#endif	// _SPH_H_

