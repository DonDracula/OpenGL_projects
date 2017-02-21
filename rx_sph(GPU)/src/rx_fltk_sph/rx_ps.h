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
const int DIM = 4;
const int RX_MAX_STEPS = 100000;


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
		RX_NORMAL, 
		RX_FORCE, 
		RX_DENSITY, 
		RX_PRESSURE, 

		RX_SURFACE, 
		RX_ATTRIBUTE, 

		RX_TEST, 

		RX_UPDATED_POSITION, 
		RX_EIGEN_VALUE, 
		RX_ROTATION_MATRIX, 
		RX_TRANSFORMATION, 
		RX_SUBPOSITION,

		RX_CONSTANT, 
		RX_RAMP, 
		RX_NONE, 

		RX_PSDATA_END, 
	};

protected:
	bool m_bInitialized;
	bool m_bUseOpenGL;

	uint m_uNumParticles;
	uint m_uMaxParticles;
	uint m_uNumArdGrid;

	uint m_solverIterations;

	RXREAL m_fParticleRadius;
	Vec3   m_v3Gravity;
	RXREAL m_fDamping;
	RXREAL m_fRestitution;

	RXREAL *m_hPos;		//!< パーティクル位置
	RXREAL *m_hVel;		//!< パーティクル速度

	uint *m_hAttr;		//!< パーティクル属性

	uint m_posVBO;		//!< パーティクル座標VBO
	uint m_colorVBO;	//!< カラーVBO

	RXREAL *m_hTmp;		//!< 一時的な値の格納場所
	RXREAL m_fTmpMax;
	
	Vec3 m_v3EnvMin;	//!< 環境のAABB最小座標
	Vec3 m_v3EnvMax;	//!< 環境のAABB最大座標

	int m_iColorType;

	RXREAL m_fTime;

	bool m_bCalNormal;		//!< 法線計算フラグ

	vector<rxInletLine> m_vInletLines;	//!< 流入ライン
	int m_iInletStart;		//!< パーティクル追加開始インデックス


protected:
	rxParticleSystemBase(){}

public:
	//! コンストラクタ
	rxParticleSystemBase(bool bUseOpenGL) : 
		m_bInitialized(false),
		m_bUseOpenGL(bUseOpenGL), 
		m_hPos(0),
		m_hVel(0), 
		m_hAttr(0), 
		m_hTmp(0)
	{
		m_v3Gravity = Vec3(0.0, -9.82, 0.0);
		m_fDamping = 0.0;
		m_fRestitution = 0.0;
		m_fParticleRadius = 0.1;
		m_fTime = 0.0;
		m_bCalAnisotropic = false;
		m_iInletStart = -1;
		m_fTmpMax = 1.0;
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

	// シミュレーション反復回数
	void SetIterations(int i) { m_solverIterations = i; }
		
	// パーティクル半径
	float GetParticleRadius(){ return m_fParticleRadius; }

	// シミュレーション設定
	void SetDamping(RXREAL x){ m_fDamping = x; }	//!< 固体境界での反発
	void SetGravity(RXREAL x){ m_v3Gravity = Vec3(0.0, x, 0.0); }	//!< 重力

	// パーティクルVBO
	unsigned int GetCurrentReadBuffer() const { return m_posVBO; }
	unsigned int GetColorBuffer()	   const { return m_colorVBO; }

	// 描画用カラー設定
	void SetColorType(int type){ m_iColorType = type; }
	int  GetColorType(void) const { return m_iColorType; }

	// フラグ切替
	void ToggleNormalCalc(int t = -1){ RX_TOGGLE(m_bCalNormal, t); }		//!< パーティクル法線の計算
	bool IsNormalCalc(void) const { return m_bCalNormal; }

public:
	// 純粋仮想関数
	virtual bool Update(RXREAL dt, int step = 0) = 0;

	virtual RXREAL* GetArrayVBO(rxParticleArray type, bool d2h = true) = 0;
	virtual void SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count) = 0;
	virtual void SetColorVBO(int type) = 0;

	virtual RXREAL* GetParticle(void) = 0;
	virtual RXREAL* GetParticleDevice(void) = 0;

public:
	// 仮想関数
	virtual void UnmapParticle(void){}

	virtual void SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris){}
	virtual void SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, int flg){}
	virtual void SetSphereObstacle(Vec3 cen, double rad, int flg){}
	virtual void MoveSphereObstacle(int b, Vec3 disp){}
	virtual Vec3 GetSphereObstaclePos(int b = -1){ return Vec3(0.0); }

	virtual void SetParticlesToCell(void) = 0;
	virtual void SetParticlesToCell(RXREAL *prts, int n, RXREAL h) = 0;

	virtual void SetPolygonsToCell(void){}

	// 陰関数値計算
	virtual double GetImplicit(double x, double y, double z){ return 0.0; }
	virtual void CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF){}
	virtual void CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF){}

	// 描画関数
	virtual void DrawCell(int i, int j, int k){}
	virtual void DrawCells(Vec3 col, Vec3 col2, int sel = 0){}

	virtual void DrawObstacles(void){}

	// シミュレーション設定の出力
	virtual void OutputSetting(string fn){}

	// Anisotropic Kernel
	virtual void CalAnisotropicKernel(void){}
	bool m_bCalAnisotropic;

public:
	void Reset(rxParticleConfig config);
	bool Set(const vector<Vec3> &ppos, const vector<Vec3> &pvel);

	void AddSphere(int start, RXREAL *pos, RXREAL *vel, int r, RXREAL spacing, int attr = 0);
	void AddBox(int start, Vec3 cen, Vec3 dim, Vec3 vel, RXREAL spacing, int attr = 0);
	int  AddLine(rxInletLine line);

	RXREAL SetColorVBOFromArray(RXREAL *hVal, int d, bool use_max = true, RXREAL vmax = 1.0);
	void SetColorVBO(void){ SetColorVBO(m_iColorType); }

	int OutputParticles(string fn);
	int InputParticles(string fn);

protected:
	int  addParticles(int &start, rxInletLine line, int attr = 0);

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

