/*!
  @file rx_ps.h
	
  @brief パーティクルを扱うシミュレーションの基底クラス
 
  @author Makoto Fujisawa
  @date 2011-06
*/
// FILE --rx_ps.h--

#ifndef _RX_PS_H_
#define _RX_PS_H_


//-----------------------------------------------------------------------------
// MARK:インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph_commons.h"

#include "rx_cu_common.cuh"

#include "rx_sph_solid.h"

//#include <helper_functions.h>



//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
#ifndef DIM
	#define DIM 4
#endif

const int RX_MAX_STEPS = 10000;


//-----------------------------------------------------------------------------
// パーティクル流入ライン
//-----------------------------------------------------------------------------
struct rxInletLine
{
	Vec3 pos1, pos2;	//!< ラインの端点
	Vec3 vel;			//!< 追加するパーティクルの速度
	Vec3 up;			//!< パーティクル堆積方向
	int accum;			//!< パーティクル堆積数
	int span;			//!< 時間的なスパン
	double spacing;		//!< 空間的なスパン
};


//-----------------------------------------------------------------------------
// パーティクルを扱うシミュレーションの基底クラス
//-----------------------------------------------------------------------------
class rxParticleSystemBase
{
public:

	enum rxParticleConfig
	{
		RX_CONFIG_RANDOM,
		RX_CONFIG_GRID,
		RX_CONFIG_BOX,
		RX_CONFIG_NONE, 
		_NUM_CONFIGS
	};

	enum rxParticleArray
	{
		RX_POSITION = 0,
		RX_VELOCITY,
		RX_FORCE, 
		RX_DENSITY, 

		RX_PREDICT_POS, 
		RX_PREDICT_VEL, 
		RX_SCALING_FACTOR, 
		RX_CORRECTION, 

		RX_BOUNDARY_PARTICLE, 
		RX_BOUNDARY_PARTICLE_VOL, 

		RX_TEST, 

		RX_CONSTANT, 
		RX_RAMP, 
		RX_NONE, 

		RX_PSDATA_END, 
	};

protected:
	bool m_bInitialized;
	bool m_bUseOpenGL;

	uint m_uNumParticles;	//!< 現在のパーティクル数
	uint m_uMaxParticles;	//!< 最大パーティクル数

	uint m_uNumBParticles;	//!< 境界パーティクルの数

	uint m_uNumArdGrid;

	RXREAL m_fParticleRadius;	//!< パーティクル半径(有効半径ではない)
	Vec3   m_v3Gravity;			//!< 重力
	RXREAL m_fRestitution;		//!< 反発係数[0,1]

	RXREAL *m_hPos;		//!< パーティクル位置
	RXREAL *m_hVel;		//!< パーティクル速度

	RXREAL *m_hPosB;	//!< 境界パーティクル
	RXREAL *m_hVolB;	//!< 境界パーティクルの体積

	RXREAL *m_hSb;		//!< 境界パーティクルのScaling factor

	uint m_posVBO;		//!< パーティクル座標VBO
	uint m_colorVBO;	//!< カラーVBO

	RXREAL *m_hTmp;		//!< 一時的な値の格納場所
	RXREAL m_fTmpMax;

	Vec3 m_v3EnvMin;	//!< 環境のAABB最小座標
	Vec3 m_v3EnvMax;	//!< 環境のAABB最大座標
	
	int m_iColorType;

	RXREAL m_fTime;

	vector<rxInletLine> m_vInletLines;	//!< 流入ライン
	int m_iInletStart;		//!< パーティクル追加開始インデックス

public:	
	vector<RXREAL> m_vFuncB;
	uint m_uNumBParticles0;	//!< 描画しない境界パーティクルの数

protected:
	rxParticleSystemBase(){}

public:
	//! コンストラクタ
	rxParticleSystemBase(bool bUseOpenGL) : 
		m_bInitialized(false),
		m_bUseOpenGL(bUseOpenGL), 
		m_hPos(0),
		m_hVel(0), 
		m_hTmp(0), 
		m_hPosB(0), 
		m_hVolB(0)
	{
		m_v3Gravity = Vec3(0.0, -9.82, 0.0);
		m_fRestitution = 0.0;
		m_fParticleRadius = 0.1;
		m_fTime = 0.0;
		m_iInletStart = -1;
		m_fTmpMax = 1.0;
		m_uNumBParticles0 = 0;
	}

	//! デストラクタ
	virtual ~rxParticleSystemBase(){}


	// シミュレーション空間
	Vec3 GetMax(void) const { return m_v3EnvMax; }
	Vec3 GetMin(void) const { return m_v3EnvMin; }
	Vec3 GetDim(void) const { return m_v3EnvMax-m_v3EnvMin; }
	Vec3 GetCen(void) const { return 0.5*(m_v3EnvMax+m_v3EnvMin); }

	// パーティクル数
	int	GetNumParticles() const { return m_uNumParticles; }
	int	GetMaxParticles() const { return m_uMaxParticles; }
	int GetNumBoundaryParticles() const { return m_uNumBParticles; }

	// パーティクル半径
	float GetParticleRadius(){ return m_fParticleRadius; }

	// シミュレーション設定
	void SetGravity(RXREAL x){ m_v3Gravity = Vec3(0.0, x, 0.0); }	//!< 重力

	// パーティクルVBO
	unsigned int GetCurrentReadBuffer(void) const { return m_posVBO; }
	unsigned int GetColorBuffer(void) const { return m_colorVBO; }

	// 描画用カラー設定
	void SetColorType(int type){ m_iColorType = type; }
	int  GetColorType(void) const { return m_iColorType; }

public:
	// 純粋仮想関数
	virtual bool Update(RXREAL dt, int step = 0) = 0;

	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true, int num = -1) = 0;
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count) = 0;
	virtual void SetColorVBO(int type, int picked) = 0;

	virtual RXREAL* GetParticle(void) = 0;
	virtual RXREAL* GetParticleDevice(void) = 0;

public:
	// 仮想関数
	virtual void UnmapParticle(void){}

	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris, Vec3 vel){}
	virtual void SetPolygonObstacle(const string &filename, Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel){}
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel, int flg){}
	virtual void SetSphereObstacle(Vec3 cen, double rad, Vec3 vel, int flg){}

	virtual void SetParticlesToCell(void) = 0;
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h) = 0;

	virtual void SetPolygonsToCell(void){}

	// 陰関数値計算
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF){}
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF){}
	virtual double GetImplicit(double x, double y, double z){ return 0.0; }
	static RXREAL GetImplicit_s(double x, double y, double z, void* p)
	{
		return (RXREAL)(((rxParticleSystemBase*)p)->GetImplicit(x, y, z));
	}

	// 描画関数
	virtual void DrawCell(int i, int j, int k){}
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0){}

	virtual void DrawObstacles(int drw){}

	// シミュレーション設定の出力
	virtual void OutputSetting(string fn){}

	// 粒子情報の取得
	virtual string GetParticleInfo(int i){ return "particle "+RX_TO_STRING(i); }

	// 境界パーティクルの初期化
	virtual void InitBoundary(void){}

public:
	void Reset(rxParticleConfig config);
	bool Set(const vector<Vec3> &ppos, const vector<Vec3> &pvel);

	void AddSphere(int start, RXREAL *pos, RXREAL *vel, int r, RXREAL spacing);
	void AddBox(int start, Vec3 cen, Vec3 dim, Vec3 vel, RXREAL spacing);
	int  AddLine(rxInletLine line);

	RXREAL SetColorVBOFromArray(RXREAL *hVal, int d, bool use_max = true, RXREAL vmax = 1.0);
	void SetColorVBO(int picked = -1){ SetColorVBO(m_iColorType, picked); }

	int OutputParticles(string fn);
	int InputParticles(string fn);

protected:
	int  addParticles(int &start, rxInletLine line);

	uint createVBO(uint size)
	{
		GLuint vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		return vbo;
	}

};



#endif	// _PS_H_

