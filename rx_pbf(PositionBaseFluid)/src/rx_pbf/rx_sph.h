/*!
  @file rx_sph.h
	
  @brief SPH法
 
  @author Makoto Fujisawa
  @date 2008-10,2011-06
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

#include "rx_kernel.h"

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

const RXREAL KOL2 = (RXREAL)0.561231024;	// 2^(-5/6)

// グローバル変数の宣言


inline RXREAL3 MAKE_RXREAL3(RXREAL x, RXREAL y, RXREAL z)
{
	return make_float3(x, y, z);
}
inline RXREAL2 MAKE_RXREAL2(RXREAL x, RXREAL y)
{
	return make_float2(x, y);
}
inline RXREAL3 MAKE_FLOAT3V(Vec3 x)
{
	return make_float3((FLOAT)x[0], (FLOAT)x[1], (FLOAT)x[2]);
}



//! SPHシーンのパラメータ
struct rxEnviroment
{
	#define MAX_DELETE_REGIONS 64

	int max_particles;			//!< 最大パーティクル数
	Vec3 boundary_cen;			//!< 境界の中心
	Vec3 boundary_ext;			//!< 境界の大きさ(各辺の長さの1/2)
	RXREAL dens;				//!< 初期密度
	RXREAL mass;				//!< パーティクルの質量
	RXREAL kernel_particles;	//!< 有効半径h以内のパーティクル数
	RXREAL dt;					//!< 時間ステップ幅
	RXREAL viscosity;			//!< 動粘性係数
	RXREAL gas_k;				//!< ガス定数

	int use_inlet;				//!< 流入境界条件の有無

	RXREAL epsilon;				//!< CFMの緩和係数
	RXREAL eta;					//!< 密度変動許容量
	int min_iter;				//!< ヤコビ反復最小数
	int max_iter;				//!< ヤコビ反復最大数

	int use_ap;					//!< 人工圧力ON/OFF (0 or 1)
	RXREAL ap_k;				//!< 人工圧力のための係数k (倍数)
	RXREAL ap_n;				//!< 人工圧力のための係数n (n乗)
	RXREAL ap_q;				//!< 人工圧力計算時の基準カーネル値計算用係数(有効半径hに対する係数, [0,1])

	// 表面メッシュ
	Vec3 mesh_boundary_cen;		//!< メッシュ生成境界の中心
	Vec3 mesh_boundary_ext;		//!< メッシュ生成境界の大きさ(各辺の長さの1/2)
	int mesh_vertex_store;		//!< 頂点数からポリゴン数を予測するときの係数
	int mesh_max_n;				//!< MC法用グリッドの最大分割数

	rxEnviroment()
	{
		max_particles = 50000;
		boundary_cen = Vec3(0.0);
		boundary_ext = Vec3(2.0, 0.8, 0.8);
		dens = (RXREAL)998.29;
		mass = (RXREAL)0.04;
		kernel_particles = (RXREAL)20.0;
		dt = 0.01;
		viscosity = 1.0e-3;
		gas_k = 3.0;

		mesh_vertex_store = 10;
		use_inlet = 0;
		mesh_max_n = 128;

		epsilon = 0.001;
		eta = 0.05;
		min_iter = 2;
		max_iter = 10;

		use_ap = true;
		ap_k = 0.1;
		ap_n = 4.0;
		ap_q = 0.2;
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
// MARK:rxPBDSPHクラスの宣言
//  - Miles Macklin and Matthias Muller, "Position Based Fluids", Proc. SIGGRAPH 2013, 2013. 
//  - http://blog.mmacklin.com/publications/
//-----------------------------------------------------------------------------
class rxPBDSPH : public rxParticleSystemBase
{
private:
	// パーティクル
	RXREAL *m_hFrc;					//!< パーティクルにかかる力
	RXREAL *m_hDens;				//!< パーティクル密度

	RXREAL *m_hS;					//!< Scaling factor for CFM
	RXREAL *m_hDp;					//!< 位置修正量

	RXREAL *m_hPredictPos;			//!< 予測位置
	RXREAL *m_hPredictVel;			//!< 予測速度

	RXREAL *m_hSb;					//!< 境界パーティクルのScaling factor

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

	rxNNGrid *m_pNNGridB;			//!< 境界パーティクル用分割グリッド
	vector< vector<rxNeigh> > m_vNeighsB;	//!< 境界近傍パーティクル


	// 粒子パラメータ
	uint m_iKernelParticles;		//!< カーネル内のパーティクル数
	RXREAL m_fRestDens, m_fMass;	//!< 密度，質量
	RXREAL m_fEffectiveRadius;		//!< 有効半径
	RXREAL m_fKernelRadius;			//!< カーネルの影響範囲

	RXREAL m_fRestConstraint;		//!< スケーリングファクタの分母項

	// シミュレーションパラメータ
	RXREAL m_fViscosity;			//!< 粘性係数

	RXREAL m_fEpsilon;				//!< CFMの緩和係数
	RXREAL m_fEta;					//!< 密度変動率
	int m_iMinIterations;			//!< ヤコビ反復最小反復回数
	int m_iMaxIterations;			//!< ヤコビ反復最大反復回数

	bool m_bArtificialPressure;		//!< クラスタリングを防ぐためのArtificial Pressure項を追加するフラグ
	RXREAL m_fApK;					//!< 人工圧力のための係数k
	RXREAL m_fApN;					//!< 人工圧力のための係数n
	RXREAL m_fApQ;					//!< 人工圧力計算時の基準カーネル値計算用係数(有効半径hに対する係数, [0,1])


	// カーネル関数の計算の際に用いられる定数係数
	double m_fAw;
	double m_fAg;
	double m_fAl;

	// カーネル関数
	double (*m_fpW)(double, double, double);
	Vec3 (*m_fpGW)(double, double, double, Vec3);
	double (*m_fpLW)(double, double, double, double);

protected:
	rxPBDSPH(){}

public:
	//! コンストラクタ
	rxPBDSPH(bool use_opengl);

	//! デストラクタ
	virtual ~rxPBDSPH();

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
	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris, Vec3 vel);
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel, int flg);
	virtual void SetSphereObstacle(Vec3 cen, double rad, Vec3 vel, int flg);

	// ホスト<->VBO間転送
	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true, int num = -1);
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count);
	virtual void SetColorVBO(int type, int picked);

	// 陰関数値計算
	virtual double GetImplicit(double x, double y, double z);
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF);
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF);

	// SPH情報出力
	virtual void OutputSetting(string fn);

	// 描画関数
	virtual void DrawCell(int i, int j, int k);
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0);
	virtual void DrawObstacles(int drw);


public:
	// SPH初期化
	void Initialize(const rxEnviroment &env);
	void Allocate(int max_particles);
	void Finalize(void);

	// 近傍取得
	void GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h = -1.0);
	void GetNearestNeighbors(int idx, RXREAL *p, vector<rxNeigh> &neighs, RXREAL h = -1.0);

	void GetNearestNeighborsB(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h = -1.0);

	// 分割セルにパーティクルを格納
	virtual void SetParticlesToCell(void);
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h);

	// メタボールによる陰関数値
	double CalColorField(double x, double y, double z);

	// 分割セルにポリゴンを格納
	virtual void SetPolygonsToCell(void);

	int  GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys);
	bool IsPolygonsInCell(int gi, int gj, int gk);

	// 人口圧力項
	bool& GetArtificialPressure(void){ return m_bArtificialPressure; }


protected:
	// CPUによるSPH計算
	void calDensity(const RXREAL *pos, RXREAL *dens, RXREAL h);
	void calForceExtAndVisc(const RXREAL *pos, const RXREAL *vel, const RXREAL *dens, RXREAL *frc, RXREAL h);

	// Scaling factorの計算
	void calScalingFactor(const RXREAL *ppos, RXREAL *pdens, RXREAL *pscl, RXREAL h, RXREAL dt);

	// 一週声量の計算
	void calPositionCorrection(const RXREAL *ppos, const RXREAL *pscl, RXREAL *pdp, RXREAL h, RXREAL dt);

	// rest densityの計算
	RXREAL calRestDensity(RXREAL h);

	// 境界パーティクルの体積を計算
	void calBoundaryVolumes(const RXREAL *bpos, RXREAL *bvol, RXREAL mass, uint n, RXREAL h);

	// 時間ステップ幅の修正
	RXREAL calTimeStep(RXREAL &dt, RXREAL eta_avg, const RXREAL *pfrc, const RXREAL *pvel, const RXREAL *pdens);

	// 位置と速度の更新
	void integrate(const RXREAL *pos, const RXREAL *vel, const RXREAL *dens, const RXREAL *acc, 
				   RXREAL *pos_new, RXREAL *vel_new, RXREAL dt);
	void integrate2(const RXREAL *pos, const RXREAL *vel, const RXREAL *dens, const RXREAL *acc, 
				    RXREAL *pos_new, RXREAL *vel_new, RXREAL dt);

	// 衝突判定
	int calCollisionPolygon(uint grid_hash, Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt);
	int calCollisionSolid(Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt);
};




//-----------------------------------------------------------------------------
// MARK:rxPBDSPH_GPUクラスの宣言
//  - Miles Macklin and Matthias Muller, "Position Based Fluids", Proc. SIGGRAPH 2013, 2013. 
//  - http://blog.mmacklin.com/publications/
//-----------------------------------------------------------------------------
class rxPBDSPH_GPU : public rxParticleSystemBase
{
private:
	//
	// メンバ変数(GPU変数)
	//
	RXREAL *m_dPos;			//!< パーティクル位置
	RXREAL *m_dVel;			//!< パーティクル速度

	RXREAL *m_dFrc;			//!< パーティクルにかかる力
	RXREAL *m_dDens;		//!< パーティクル密度

	RXREAL *m_dPosB;		//!< 境界パーティクル
	RXREAL *m_dVolB;		//!< 境界パーティクルの体積

	RXREAL *m_dS;			//!< Scaling factor for CFM
	RXREAL *m_dDp;			//!< 位置修正量
	RXREAL *m_dPredictPos;	//!< 予測位置
	RXREAL *m_dPredictVel;	//!< 予測速度

	RXREAL *m_dSb;			//!< 境界パーティクルのScaling factor

	RXREAL *m_dErr;			//!< 密度変動値	
	RXREAL *m_dErrScan;		//!< 密度変動値のScan結果

	cudaGraphicsResource *m_pPosResource;	//!< OpenGL(のVBO)-CUDA間のデータ転送を扱うためのハンドル

	// 表面メッシュ
	RXREAL *m_dVrts;		//!< 固体メッシュ頂点
	int    *m_dTris;		//!< 固体メッシュ


	// シミュレーションパラメータ
	rxSimParams m_params;	//!< シミュレーションパラメータ(GPUへのデータ渡し用)
	uint3 m_gridSize;		//!< 近傍探索グリッドの各軸の分割数
	uint m_numGridCells;	//!< 近傍探索グリッド分割総数
	
	// 空間分割(GPU)
	rxParticleCell m_dCellData;			//!< 近傍探索グリッド
	uint m_gridSortBits;				//!< ハッシュ値による基数ソート時の基数桁数

	rxParticleCell m_dCellDataB;		//!< 境界パーティクル用近傍探索グリッド
	uint3 m_gridSizeB;					//!< 境界パーティクル用近傍探索グリッドの各軸の分割数

	//
	// メンバ変数(CPU変数)
	//
	RXREAL *m_hFrc;			//!< パーティクルにかかる力
	RXREAL *m_hDens;		//!< パーティクル密度

	RXREAL *m_hS;			//!< Scaling factor for CFM
	RXREAL *m_hDp;			//!< 位置修正量
	RXREAL *m_hPredictPos;	//!< 予測位置
	RXREAL *m_hPredictVel;	//!< 予測速度

	RXREAL *m_hSb;			//!< 境界パーティクルのScaling factor

	// 表面メッシュ
	rxPolygons m_Poly;
	vector<RXREAL> m_vVrts;	//!< 固体メッシュ頂点
	int m_iNumVrts;			//!< 固体メッシュ頂点数
	vector<int> m_vTris;	//!< 固体メッシュ
	int m_iNumTris;			//!< 固体メッシュ数

	RXREAL m_fEpsilon;				//!< CFMの緩和係数
	RXREAL m_fEta;					//!< 密度変動率
	int m_iMinIterations;			//!< ヤコビ反復最小反復回数
	int m_iMaxIterations;			//!< ヤコビ反復最大反復回数

	bool m_bArtificialPressure;		//!< クラスタリングを防ぐためのArtificial Pressure項を追加するフラグ

	vector<rxSolid*> m_vSolids;		//!< 固体物体(固体パーティクル生成用)

protected:
	rxPBDSPH_GPU(){}

public:
	//! コンストラクタ
	rxPBDSPH_GPU(bool use_opengl);

	//! デストラクタ
	~rxPBDSPH_GPU();

	// パーティクル半径
	float GetEffectiveRadius(){ return m_params.EffectiveRadius; }

	// 近傍パーティクル
	uint* GetNeighborList(const int &i, int &n);

	// シミュレーションパラメータ
	rxSimParams GetParams(void){ return m_params; }
	void UpdateParams(void);

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
	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris, Vec3 vel);
	virtual void SetPolygonObstacle(const string &filename, Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel);
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel, int flg);
	virtual void SetSphereObstacle(Vec3 cen, double rad, Vec3 vel, int flg);

	// ホスト<->VBO間転送
	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true, int num = -1);
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count);
	virtual void SetColorVBO(int type, int picked);


	// 陰関数値計算
	virtual double GetImplicit(double x, double y, double z);
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF);
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF);

	// SPH情報出力
	virtual void OutputSetting(string fn);

	// 描画関数
	virtual void DrawCell(int i, int j, int k);
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0);
	virtual void DrawObstacles(int drw);

protected:
	void setObjectToCell(RXREAL *p, RXREAL *v);


public:
	// SPH初期化
	void Initialize(const rxEnviroment &env);
	void Allocate(int max_particles);
	void Finalize(void);

	// 分割セルにパーティクルを格納
	virtual void SetParticlesToCell(void);
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h);
	virtual void SetPolygonsToCell(void);

	int  GetPolygonsInCell(int gi, int gj, int gk, vector<int> &polys){ return 0; }
	bool IsPolygonsInCell(int gi, int gj, int gk){ return false; }

	void CalMaxDensity(int k);

	// 人口圧力項
	bool& GetArtificialPressure(void){ return m_bArtificialPressure; }

	// 境界パーティクルの初期化
	virtual void InitBoundary(void);


protected:
	// rest densityの計算
	RXREAL calRestDensity(RXREAL h);

	// 分割セルの初期設定
	void setupCells(rxParticleCell &cell, uint3 &gridsize, double &cell_width, Vec3 vMin, Vec3 vMax, double h);

	// グリッドハッシュの計算
	uint calGridHash(uint x, uint y, uint z);
	uint calGridHash(Vec3 pos);

	// ポリゴンを分割セルに格納
	void setPolysToCell(RXREAL *vrts, int nv, int* tris, int nt);

	// ポリゴン頂点の調整
	bool fitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set);
};



#endif	// _SPH_H_

