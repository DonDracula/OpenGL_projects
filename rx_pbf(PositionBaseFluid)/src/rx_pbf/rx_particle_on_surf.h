/*!
  @file rx_particle_on_surf.h
	
  @brief 陰関数表面にパーティクルを配置
 
  @author Makoto Fujisawa
  @date   2013-06
*/


#ifndef _RX_PARTICLE_ON_SURFACE_H_
#define _RX_PARTICLE_ON_SURFACE_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <cmath>

// STL
#include <vector>
#include <string>

#include <iostream>

#include "rx_utility.h"
#include "rx_nnsearch.h"


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
using namespace std;
typedef unsigned int uint;

#ifndef RXREAL
	#define RXREAL float
#endif
#ifndef DIM
	#define DIM 4
#endif





//-----------------------------------------------------------------------------
// 陰関数等値面にパーティクルを配置するクラス
//  -参考: A. Witkin and P. Heckbert, "Using particles to sample and control implicit surfaces", SIGGRAPH1994.
//-----------------------------------------------------------------------------
class rxParticleOnSurf
{
	struct rxSurfParticle
	{
		Vec3 Vel;
		Vec3 Ep;
		RXREAL Sigma;
		int Flag;

		rxSurfParticle() : Vel(Vec3(0.0)), Ep(Vec3(0.0)), Sigma(0.0), Flag(0) {}
	};

protected:
	uint m_uNumParticles;

	RXREAL m_fParticleRadius;
	RXREAL m_fEffectiveRadius;		//!< 有効半径
	RXREAL m_fKernelRadius;			//!< カーネルの影響範囲

	vector<RXREAL> m_vPos;			//!< 近傍探索ルーチンに渡すために位置だけ別管理
	vector<rxSurfParticle> m_vPs;	//!< パーティクル情報

	// Repulsionパラメータ
	RXREAL m_fAlpha;	//!< repulsion amplitude
	RXREAL m_fEh;		//!< desired energy
	RXREAL m_fRho;		//!< パーティクルのenergyをm_fEhに保つための係数
	RXREAL m_fPhi;		//!< パーティクル位置を曲面上に保つための係数
	RXREAL m_fBeta;		//!< zero-divide防止用
	RXREAL m_fSh;		//!< desired repulsion radius (ユーザ指定)
	RXREAL m_fSmax;		//!< maximum repulsion radius

	RXREAL m_fGamma;	//!< equilibrium speed (σに対する倍数)
	RXREAL m_fNu;		//!< パーティクル分裂のための係数
	RXREAL m_fDelta;	//!< パーティクル削除のための係数

	void *m_pFuncPtr;
	Vec4 (*m_fpFunc)(void*, double, double, double);

	rxNNGrid *m_pNNGrid;			//!< 分割グリッドによる近傍探索
	vector< vector<rxNeigh> > m_vNeighs;	//!< 近傍パーティクル

public:
	//! コンストラクタ
	rxParticleOnSurf();

	//! デストラクタ
	~rxParticleOnSurf(){}

	// パーティクル初期化&破棄
	void Initialize(const vector<Vec3> &vrts, double rad, Vec3 minp, Vec3 maxp, 
					Vec4 (*func)(void*, double, double, double), void* func_ptr);
	void Finalize(void);

	//! パーティクル数
	int	GetNumParticles() const { return m_uNumParticles; }

	//! パーティクル半径
	float GetParticleRadius(void){ return m_fParticleRadius; }

	//! パーティクルデータの取得
	RXREAL* GetPositionArray(void){ return &m_vPos[0]; }

	//! パーティクル位置の更新
	void Update(double dt, int &num_iter, RXREAL &eps);

protected:
	//! σの更新
	void updateSigma(double dt);

	//! 反発力の計算
	void applyRepulsion(double dt);
	void applyRepulsion2(double dt);

	//! 表面に沿った移動
	void applyFloating(double dt, double &v_avg);

	//!< パーティクル分裂/破棄の判定
	void testFissionDeath(void);

public:
	// 近傍取得
	void GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h = -1.0);
	void GetNearestNeighbors(int idx, RXREAL *p, vector<rxNeigh> &neighs, RXREAL h = -1.0);

	// 分割セルにパーティクルを格納
	void SetParticlesToCell(void);
	void SetParticlesToCell(RXREAL *prts, int n, RXREAL h);

};


#endif // _RX_PARTICLE_ON_SURFACE_H_

