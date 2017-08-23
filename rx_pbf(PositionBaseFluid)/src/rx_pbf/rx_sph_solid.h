/*!
  @file rx_sph_solid.h
	
  @brief SPH用固体定義
 
  @author Makoto Fujisawa
  @date 2008-12
*/

#ifndef _RX_SPH_SOLID_H_
#define _RX_SPH_SOLID_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------


#include "rx_sph_commons.h"
#include "rx_model.h"
#include "rx_nnsearch.h"	// グリッド分割による近傍探索
#include "rx_mesh.h"

// SPHカーネル
#include "rx_kernel.h"


//-----------------------------------------------------------------------------
// 定義・定数
//-----------------------------------------------------------------------------

const int RX_DISPMAP_N = 128;


//-----------------------------------------------------------------------------
// MARK:rxCollisionInfoクラス
//-----------------------------------------------------------------------------
class rxCollisionInfo
{
protected:
	Vec3 m_vContactPoint;	//!< 衝突点
	Vec3 m_vNormal;			//!< (衝突点での)法線
	double m_fDepth;		//!< めり込み量

	Vec3 m_vVelocity;		//!< 衝突点の速度

public:
	//! デフォルトコンストラクタ
	rxCollisionInfo()
	  : m_vContactPoint(Vec3(0.0)), 
		m_vNormal(Vec3(0.0)), 
		m_fDepth(0.0), 
		m_vVelocity(Vec3(0.0))
	{
	}

	//! コンストラクタ
	rxCollisionInfo(const Vec3 &contact_point, 
					const Vec3 &normal = Vec3(0.0), 
					const double &depth = 0.0, 
					const Vec3 &veloc = Vec3(0.0))
	  : m_vContactPoint(contact_point), 
		m_vNormal(normal), 
		m_fDepth(depth), 
		m_vVelocity(veloc)
	{
	}

	//! デストラクタ
	~rxCollisionInfo(){}

public:
	const Vec3& Contact() const { return m_vContactPoint; }
	Vec3& Contact(){ return m_vContactPoint; }

	const Vec3& Normal() const { return m_vNormal; }
	Vec3& Normal(){ return m_vNormal; }

	const double& Penetration() const { return m_fDepth; }
	double& Penetration(){ return m_fDepth; }

	const Vec3& Velocity() const { return m_vVelocity; }
	Vec3& Velocity(){ return m_vVelocity; }
};




//-----------------------------------------------------------------------------
// MARK:rxSolid : 固体オブジェクト基底クラス
//-----------------------------------------------------------------------------
class rxSolid
{
public:
	enum{
		RXS_SPHERE, 
		RXS_AABB, 
		RXS_OPEN_BOX, 
		RXS_POLYGON, 
		RXS_IMPLICIT, 
		RXS_OTHER = -1, 
	};

protected:
	Vec3 m_vMassCenter;	//!< 重心座標
	Vec3 m_vVelocity;	//!< 固体速度
	rxMatrix4 m_matRot;	//!< 姿勢
	rxMatrix4 m_matRotInv;	//!< 姿勢

	int m_iName;

	RXREAL m_fOffset;

	bool m_bFix;		//!< 固定フラグ
	bool m_bSP;			//!< 固体パーティクル生成フラグ

	int m_iSgn;			//!< 箱:-1, オブジェクト:1

public:
	rxSolid() : m_iName(RXS_OTHER)
	{
		m_bFix = true;
		m_vMassCenter = Vec3(0.0);
		m_vVelocity = Vec3(0.0);
		m_iSgn = 1;
		m_fOffset = (RXREAL)(0.0);
		m_bSP = true;

		m_matRot.MakeIdentity();
	}

	//
	// 仮想関数
	//
	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col) = 0;	//!< 距離関数計算
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col) = 0;

	virtual bool GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col) = 0;	//!< 距離関数計算
	virtual bool GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col) = 0;

	virtual bool GetCurvature(const Vec3 &pos, double &k) = 0;	//!< 距離関数の曲率計算

	virtual void Draw(int drw) = 0;				//!< OpenGLでの描画
	virtual void SetGLMatrix(void) = 0;							//!< OpenGL変換行列の適用

	virtual Vec3 GetMin(void) = 0;
	virtual Vec3 GetMax(void) = 0;


	//
	// 表面上にパーティクルを配置
	//
	static Vec4 GetImplicitG_s(void* ptr, double x, double y, double z);
	inline Vec4 GetImplicitG(Vec3 pos);
	static RXREAL GetImplicit_s(void* ptr, double x, double y, double z);
	inline RXREAL GetImplicit(Vec3 pos);

	int GenerateParticlesOnSurf(RXREAL rad, RXREAL **ppos);

	static bool GetDistance_s(Vec3 pos, rxCollisionInfo &col, void* x);

	//
	// 取得・設定関数
	//
	inline Vec3 CalLocalCoord(const Vec3 &pos);			//!< グローバルから固体ローカルへの座標変換
	inline Vec3 CalGlobalCoord(const Vec3 &pos);		//!< 固体ローカルからグローバルへの座標変換

	inline Vec3 GetPosition(void);						//!< 固体重心位置の取得
	inline void SetPosition(const Vec3 &pos);			//!< 固体重心位置の設定

	inline rxMatrix4 GetMatrix(void);					//!< 回転行列の取得
	inline void SetMatrix(const rxMatrix4 &mat);		//!< 回転行列の設定
	inline void SetMatrix(double mat[16]);				//!< 回転行列の設定

	inline Vec3 GetVelocityAtGrobal(const Vec3 &pos);	//!< 体座標値の固体速度の取得
	inline void SetVelocity(const Vec3 &vec);			//!< 

	inline bool GetFix(void) const { return m_bFix; }		//!< 固定フラグの取得
	inline void SetFix(bool fix){ m_bFix = fix; }			//!< 固定フラグの設定

	inline bool& IsSolidParticles(void){ return m_bSP; }	//!< 固体パーティクルフラグの取得

	inline bool RigidSimulation(const double &dt);		//!< 剛体シミュレーション

	inline const int& Name() const { return m_iName; }
	inline int& Name(){ return m_iName; }
};



//-----------------------------------------------------------------------------
// MARK:rxSolidBox : 直方体
//-----------------------------------------------------------------------------
class rxSolidBox : public rxSolid
{
protected:
	Vec3 m_vMax, m_vMin;	//!< 最大座標，最小座標(中心からの相対値)
	Vec3 m_vC[2];

public:
	// コンストラクタ
	rxSolidBox(Vec3 minp, Vec3 maxp, int sgn)
	{
		Vec3 sl  = 0.5*(maxp-minp);
		Vec3 ctr = 0.5*(maxp+minp);

		m_vMin = -sl;
		m_vMax =  sl;
		m_vMassCenter = ctr;

		m_vC[0] = m_vMin;
		m_vC[1] = m_vMax;

		m_iSgn = sgn;

		m_iName = RXS_AABB;
	}

	virtual Vec3 GetMin(void){ return m_vMassCenter+m_vMin; }
	virtual Vec3 GetMax(void){ return m_vMassCenter+m_vMax; }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);

	virtual bool GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col){ return GetDistance(pos1, col); }
	virtual bool GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col){ return GetDistanceR(pos1, r, col); }

	virtual bool GetCurvature(const Vec3 &pos, double &k);
	virtual void Draw(int drw = 4);
	virtual void SetGLMatrix(void);

protected:
	bool aabb_point_dist(const Vec3 &spos, const double &h, const int &sgn, const Vec3 &vMin, const Vec3 &vMax, 
						 double &d, Vec3 &n, int &np, int &plist, double pdist[6]);

};

//-----------------------------------------------------------------------------
// rxSolidOpenBox : 直方体(開)
//-----------------------------------------------------------------------------
class rxSolidOpenBox : public rxSolid
{
protected:
	Vec3 m_vSLenIn, m_vSLenOut;

public:
	// コンストラクタ
	rxSolidOpenBox(Vec3 ctr, Vec3 sl_in, Vec3 sl_out, int sgn)
	{
		m_vSLenIn  = sl_in;
		m_vSLenOut = sl_out;
		m_vMassCenter = ctr;

		RXCOUT << "SLenIn  " << m_vSLenIn << endl;
		RXCOUT << "SLenOut " << m_vSLenOut << endl;

		m_iSgn = sgn;

		m_iName = RXS_OPEN_BOX;
	}

	Vec3 GetInMin(void) const { return -m_vSLenIn; }
	Vec3 GetInMax(void) const { return  m_vSLenIn; }
	Vec3 GetOutMin(void) const { return -m_vSLenOut; }
	Vec3 GetOutMax(void) const { return  m_vSLenOut; }
	Vec3 GetInSideLength(void) const { return m_vSLenIn; }
	Vec3 GetOutSideLength(void) const { return m_vSLenOut; }

	virtual Vec3 GetMin(void){ return -m_vSLenOut; }
	virtual Vec3 GetMax(void){ return  m_vSLenOut; }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);

	virtual bool GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col){ return GetDistance(pos1, col); }
	virtual bool GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col){ return GetDistanceR(pos1, r, col); }

	virtual bool GetCurvature(const Vec3 &pos, double &k);
	virtual void Draw(int drw = 4);
	virtual void SetGLMatrix(void);
};



//-----------------------------------------------------------------------------
// rxSolidSphere : 球
//-----------------------------------------------------------------------------
class rxSolidSphere : public rxSolid
{
protected:
	double m_fRadius;		//!< 半径
	double m_fRadiusSqr;	//!< 半径の自乗

public:
	// コンストラクタ
	rxSolidSphere(Vec3 ctr, double rad, int sgn)
		: m_fRadius(rad)
	{
		m_iSgn = sgn;
		m_vMassCenter = ctr;
		m_fRadiusSqr = rad*rad;

		m_iName = RXS_SPHERE;
	}

	Vec3 GetCenter(void) const { return m_vMassCenter; }
	double GetRadius(void) const { return m_fRadius; }

	virtual Vec3 GetMin(void){ return m_vMassCenter-Vec3(m_fRadius); }
	virtual Vec3 GetMax(void){ return m_vMassCenter+Vec3(m_fRadius); }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);

	virtual bool GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col){ return GetDistance(pos1, col); }
	virtual bool GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col){ return GetDistanceR(pos1, r, col); }

	virtual bool GetCurvature(const Vec3 &pos, double &k){ return false; }
	virtual void Draw(int drw = 4);
	virtual void SetGLMatrix(void);
};





//-----------------------------------------------------------------------------
// rxSolidPolygon : ポリゴン
//-----------------------------------------------------------------------------
class rxSolidPolygon : public rxSolid
{
protected:
	Vec3 m_vMax, m_vMin;	//!< 最大座標，最小座標(中心からの相対値)

	string m_strFilename;

	rxPolygons m_Poly;
	RXREAL *m_hVrts;				//!< 固体ポリゴンの頂点
	int m_iNumVrts;					//!< 固体ポリゴンの頂点数
	int *m_hTris;					//!< 固体ポリゴン
	int m_iNumTris;					//!< 固体ポリゴンの数

	double m_fMaxRad;
	rxNNGrid *m_pNNGrid;			//!< 分割グリッドによる近傍探索

public:
	// コンストラクタとデストラクタ
	rxSolidPolygon(const string &fn, Vec3 cen, Vec3 ext, Vec3 ang, double h, int aspect = 1);
	~rxSolidPolygon();

	virtual Vec3 GetMin(void){ return m_vMassCenter+m_vMin; }
	virtual Vec3 GetMax(void){ return m_vMassCenter+m_vMax; }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);

	virtual bool GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col);

	virtual bool GetCurvature(const Vec3 &pos, double &k);
	virtual void Draw(int drw);
	virtual void SetGLMatrix(void);

	// 分割セル
	int  GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys);
	bool IsPolygonsInCell(int gi, int gj, int gk);

protected:
	int getDistanceToPolygon(Vec3 x, int p, double &dist);

	int readFile(const string filename, rxPolygons &polys);
	void readOBJ(const string filename, rxPolygons &polys);

	bool fitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set, int aspect = 1);

	int intersectSegmentTriangle(Vec3 P0, Vec3 P1, Vec3 V0, Vec3 V1, Vec3 V2, Vec3 &I, Vec3 &n);

};




//-----------------------------------------------------------------------------
// MARK:rxSolidクラスの実装
//-----------------------------------------------------------------------------
/*!
 * グローバルから固体ローカルへの座標変換
 * @param[in] x,y,z グローバル座標系での位置
 * @return 固体ローカル座標系での位置
 */
inline Vec3 rxSolid::CalLocalCoord(const Vec3 &pos)
{
	// 流体座標から固体座標へと変換
	Vec3 rpos;
	rpos = pos-m_vMassCenter;
	//m_matRot.multMatrixVec(rpos);
	rpos[0] = rpos[0]*m_matRot(0,0)+rpos[1]*m_matRot(1,0)+rpos[2]*m_matRot(2,0);
	rpos[1] = rpos[0]*m_matRot(0,1)+rpos[1]*m_matRot(1,1)+rpos[2]*m_matRot(2,1);
	rpos[2] = rpos[0]*m_matRot(0,2)+rpos[1]*m_matRot(1,2)+rpos[2]*m_matRot(2,2);
	return rpos;
}
/*!
 * 固体ローカルからグローバルへの座標変換
 * @param[in] x,y,z 固体ローカル座標系での位置
 * @return グローバル座標系での位置
 */
inline Vec3 rxSolid::CalGlobalCoord(const Vec3 &pos)
{
	// 固体座標から流体座標へと変換
	Vec3 fpos = pos;
	//m_matRotInv.multMatrixVec(pos, fpos);
	fpos[0] = pos[0]*m_matRotInv(0,0)+pos[1]*m_matRotInv(1,0)+pos[2]*m_matRotInv(2,0);
	fpos[1] = pos[0]*m_matRotInv(0,1)+pos[1]*m_matRotInv(1,1)+pos[2]*m_matRotInv(2,1);
	fpos[2] = pos[0]*m_matRotInv(0,2)+pos[1]*m_matRotInv(1,2)+pos[2]*m_matRotInv(2,2);
	fpos = fpos+m_vMassCenter;
	return fpos;
}

/*!
 * 固体重心位置の取得
 * @return 固体重心座標(流体座標系)
 */
inline Vec3 rxSolid::GetPosition(void)
{
	return m_vMassCenter;
}

/*!
 * 固体重心位置の設定
 * @param[in] pos 固体重心座標(流体座標系)
 */
void rxSolid::SetPosition(const Vec3 &pos)
{
	m_vMassCenter = pos;
}

/*!
 * 固体重心位置の取得
 * @return 固体重心座標(流体座標系)
 */
inline rxMatrix4 rxSolid::GetMatrix(void)
{
	return m_matRot;
}


inline rxMatrix4 CalInverseMatrix(const rxMatrix4 &mat)
{
	real d = mat(0, 0)*mat(1, 1)*mat(2, 2)-mat(0, 0)*mat(2, 1)*mat(1, 2)+ 
			 mat(1, 0)*mat(2, 1)*mat(0, 2)-mat(1, 0)*mat(0, 1)*mat(2, 2)+ 
			 mat(2, 0)*mat(0, 1)*mat(1, 2)-mat(2, 0)*mat(1, 1)*mat(0, 2);

	if(d == 0) d = 1;

	return rxMatrix4( (mat(1, 1)*mat(2, 2)-mat(1, 2)*mat(2, 1))/d,
					 -(mat(0, 1)*mat(2, 2)-mat(0, 2)*mat(2, 1))/d,
					  (mat(0, 1)*mat(1, 2)-mat(0, 2)*mat(1, 1))/d,
					  0.0, 
					 -(mat(1, 0)*mat(2, 2)-mat(1, 2)*mat(2, 0))/d,
					  (mat(0, 0)*mat(2, 2)-mat(0, 2)*mat(2, 0))/d,
					 -(mat(0, 0)*mat(1, 2)-mat(0, 2)*mat(1, 0))/d,
					  0.0, 
					  (mat(1, 0)*mat(2, 1)-mat(1, 1)*mat(2, 0))/d,
					 -(mat(0, 0)*mat(2, 1)-mat(0, 1)*mat(2, 0))/d,
					  (mat(0, 0)*mat(1, 1)-mat(0, 1)*mat(1, 0))/d,
					  0.0, 
					  0.0, 0.0, 0.0, 1.0);
}


/*!
 * 固体重心位置の設定
 * @param[in] pos 固体重心座標(流体座標系)
 */
void rxSolid::SetMatrix(const rxMatrix4 &mat)
{
	//m_matRot = mat;
	for(int i = 0; i < 3; ++i){
		for(int j = 0; j < 3; ++j){
			m_matRot(i, j) = mat(i, j);
		}
	}

	//m_matRotInv = m_matRot.Inverse();
	m_matRotInv = CalInverseMatrix(m_matRot);
}

/*!
 * 固体重心位置の設定
 * @param[in] pos 固体重心座標(流体座標系)
 */
void rxSolid::SetMatrix(double mat[16])
{
	//m_matRot = mat;
	for(int i = 0; i < 4; ++i){
		for(int j = 0; j < 4; ++j){
			m_matRot(i, j) = mat[i+j*4];
		}
	}

	//m_matRotInv = m_matRot.Inverse();
	m_matRotInv = CalInverseMatrix(m_matRot);
}


/*!
 * グローバル座標値での固体速度の取得
 * @param[in] pos グローバル座標値
 * @return 固体速度
 */
inline Vec3 rxSolid::GetVelocityAtGrobal(const Vec3 &pos)
{
	return m_vVelocity;
}

/*!
 * 固体速度をセット
 * @param[in] vec 重心速度
 */
inline void rxSolid::SetVelocity(const Vec3 &vec)
{
	m_vVelocity = vec;
}


/*!
 * 剛体シミュレーション(fix=trueの時)
 * @param[in] dt タイムステップ幅
 */
inline bool rxSolid::RigidSimulation(const double &dt)
{
	m_vMassCenter += dt*m_vVelocity;
	return true;
}


	
	
//-----------------------------------------------------------------------------
// MARK:その他関数
//-----------------------------------------------------------------------------
inline bool GetImplicitPlane(const Vec3 &pos, double &d, Vec3 &n, Vec3 &v, const Vec3 &pn, const Vec3 &pq)
{
	d = dot(pq-pos, pn);
	n = pn;
	v = Vec3(0.0);

	return true;
}


/*!
 * 距離関数から曲率を計算
 * @param[in] pos 計算点
 * @param[out] k 曲率
 * @param[in] fpDist 距離関数
 */
inline bool CalCurvature(const Vec3 &pos, double &k, bool (fpDist)(Vec3, rxCollisionInfo&, void*), void* fp = 0)
{
	k = 0.0;

	double h = 0.005;
	double x0, y0, z0;
	double p[3][3][3];
	rxCollisionInfo col;

	x0 = pos[0]-0.5*h;
	y0 = pos[1]-0.5*h;
	z0 = pos[2]-0.5*h;

//	fpDist(Vec3(x0-h, y0-h, z0-h), col, fp); p[0][0][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0-h, z0  ), col, fp); p[0][0][1] = col.Penetration();
//	fpDist(Vec3(x0-h, y0-h, z0+h), col, fp); p[0][0][2] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0-h), col, fp); p[0][1][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0  ), col, fp); p[0][1][1] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0+h), col, fp); p[0][1][2] = col.Penetration();
//	fpDist(Vec3(x0-h, y0+h, z0-h), col, fp); p[0][2][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0+h, z0  ), col, fp); p[0][2][1] = col.Penetration();
//	fpDist(Vec3(x0-h, y0+h, z0+h), col, fp); p[0][2][2] = col.Penetration();

	fpDist(Vec3(x0  , y0-h, z0-h), col, fp); p[1][0][0] = col.Penetration();
	fpDist(Vec3(x0  , y0-h, z0  ), col, fp); p[1][0][1] = col.Penetration();
	fpDist(Vec3(x0  , y0-h, z0+h), col, fp); p[1][0][2] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0-h), col, fp); p[1][1][0] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0  ), col, fp); p[1][1][1] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0+h), col, fp); p[1][1][2] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0-h), col, fp); p[1][2][0] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0  ), col, fp); p[1][2][1] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0+h), col, fp); p[1][2][2] = col.Penetration();

//	fpDist(Vec3(x0+h, y0-h, z0-h), col, fp); p[2][0][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0-h, z0  ), col, fp); p[2][0][1] = col.Penetration();
//	fpDist(Vec3(x0+h, y0-h, z0+h), col, fp); p[2][0][2] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0-h), col, fp); p[2][1][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0  ), col, fp); p[2][1][1] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0+h), col, fp); p[2][1][2] = col.Penetration();
//	fpDist(Vec3(x0+h, y0+h, z0-h), col, fp); p[2][2][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0+h, z0  ), col, fp); p[2][2][1] = col.Penetration();
//	fpDist(Vec3(x0+h, y0+h, z0+h), col, fp); p[2][2][2] = col.Penetration();

	double px, py, pz, pxx, pyy, pzz, pxy, pyz, pxz, np;
	px = (p[2][1][1]-p[0][1][1])/(2.0*h);
	py = (p[1][2][1]-p[1][0][1])/(2.0*h);
	pz = (p[1][1][2]-p[1][1][0])/(2.0*h);

	pxx = (p[2][1][1]-2.0*p[1][1][1]+p[0][1][1])/(h*h);
	pyy = (p[1][2][1]-2.0*p[1][1][1]+p[1][0][1])/(h*h);
	pzz = (p[1][1][2]-2.0*p[1][1][1]+p[1][1][0])/(h*h);

	pxy = (p[0][0][1]+p[2][2][1]-p[0][2][1]-p[2][0][1])/(4.0*h*h);
	pxz = (p[0][1][0]+p[2][1][2]-p[0][1][2]-p[2][1][0])/(4.0*h*h);
	pyz = (p[1][0][0]+p[1][2][2]-p[1][0][2]-p[1][2][0])/(4.0*h*h);

	np = px*px+py*py+pz*pz;
	if(np > RX_FEQ_EPS){
		np = sqrt(np);

		// 曲率の計算
		k = (px*px*pyy-2.0*px*py*pxy+py*py*pxx+px*px*pzz-2.0*px*pz*pxz+pz*pz*pxx+py*py*pzz-2.0*py*pz*pyz+pz*pz*pyy)/(np*np*np);
	}

	k = -k;

	return true;
}

/*!
 * Poly6カーネルの値を計算
 * @param[in] r カーネル中心までの距離
 * @param[in] h 有効半径
 * @param[in] ptr 関数呼び出しポインタ
 * @return カーネル関数値
 */
static inline double CalKernelPoly6(double r, double h, void *ptr = 0)
{
	static double a = 0.0;
	if(a == 0.0) a = KernelCoefPoly6(h, 3, 1);	
	return KernelPoly6(r, h, a);
}

/*!
 * Poly6カーネルの値を計算(最大値が1になるように正規化)
 * @param[in] r カーネル中心までの距離
 * @param[in] h 有効半径
 * @param[in] ptr 関数呼び出しポインタ
 * @return カーネル関数値
 */
static inline double CalKernelPoly6r(double r, double h, void *ptr = 0)
{
	static double a = 0.0;
	static double b = 0.0;
	if(a == 0.0) a = KernelCoefPoly6(h, 3, 1);	
	if(b == 0.0) b = KernelPoly6(0.0, h, a);
	return KernelPoly6(r, h, a)/b;
}


/*!
 * 光線(レイ,半直線)と球の交差判定
 * @param[in] p,d レイの原点と方向
 * @param[in] c,r 球の中心と半径
 * @param[out] t1,t2 pから交点までの距離
 * @return 交点数
 */
inline int ray_sphere(const Vec3 &p, const Vec3 &d, const Vec3 &sc, const double r, double &t1, double &t2)
{
	Vec3 q = p-sc;	// 球中心座標系での光線原点座標

	double a = norm2(d);
	double b = 2*dot(q, d);
	double c = norm2(q)-r*r;

	// 判別式
	double D = b*b-4*a*c;

	if(D < 0.0){ // 交差なし
		return 0;
	}
	else if(D < RX_FEQ_EPS){ // 交点数1
		t1 = -b/(2*a);
		t2 = -1;
		return 1;
	}
	else{ // 交点数2
		double sqrtD = sqrt(D);
		t1 = (-b-sqrtD)/(2*a);
		t2 = (-b+sqrtD)/(2*a);
		return 2;
	}

}

/*!
 * 三角形と球の交差判定
 * @param[in] v0,v1,v2	三角形の頂点
 * @param[in] n			三角形の法線
 * @param[in] p			最近傍点
 * @return 
 */
inline bool triangle_sphere(const Vec3 &v0, const Vec3 &v1, const Vec3 &v2, const Vec3 &n, 
							const Vec3 &c, const double &r, double &dist, Vec3 &ipoint)
{
	// ポリゴンを含む平面と球中心の距離
	double d = dot(v0, n);
	double l = dot(n, c)-d;

	dist = l;
	if(l > r) return false;

	// 平面との最近傍点座標
	Vec3 p = c-l*n;

	// 近傍点が三角形内かどうかの判定
	Vec3 n1 = cross((v0-c), (v1-c));
	Vec3 n2 = cross((v1-c), (v2-c));
	Vec3 n3 = cross((v2-c), (v0-c));

	ipoint = p;
	dist = l;
	if(dot(n1, n2) > 0 && dot(n2, n3) > 0){		// 三角形内
		return true;
	}
	else{		// 三角形外
		// 三角形の各エッジと球の衝突判定
		for(int e = 0; e < 3; ++e){
			Vec3 va0 = (e == 0 ? v0 : (e == 1 ? v1 : v2));
			Vec3 va1 = (e == 0 ? v1 : (e == 1 ? v2 : v0));

			double t1, t2;
			int n = ray_sphere(va0, Unit(va1-va0), c, r, t1, t2);

			if(n){
				double le2 = norm2(va1-va0);
				if((t1 >= 0.0 && t1*t1 < le2) || (t2 >= 0.0 && t2*t2 < le2)){
					return true;
				}
			}
		}
		return false;
	}
}

/*!
 * 線分(を含む直線)と点の距離
 * @param[in] v0,v1 線分の両端点座標
 * @param[in] p 点の座標
 * @return 距離
 */
inline double segment_point_dist(const Vec3 &v0, const Vec3 &v1, const Vec3 &p)
{
	Vec3 v = Unit(v1-v0);
	Vec3 vp = p-v0;
	Vec3 vh = dot(vp, v)*v;
	return norm(vp-vh);
}

/*!
 * 線分と球の交差判定
 * @param[in] s0,s1	線分の端点
 * @param[in] sc,r   球の中心座標と半径
 * @param[out] d2 線分との距離の二乗
 * @return 交差ありでtrue
 */
inline bool segment_sphere(const Vec3 &s0, const Vec3 &s1, const Vec3 &sc, const double &r, double &d2)
{
	Vec3 v = s1-s0;
	Vec3 c = sc-s0;

	double vc = dot(v, c);
	if(vc < 0){		// 球の中心が線分の始点s0の外にある
		d2 = norm2(c);
		return (d2 < r*r);	// 球中心と始点s0の距離で交差判定
	}
	else{
		double v2 = norm2(v);
		if(vc > v2){	// 球の中心が線分の終点s1の外にある
			d2 = norm2(s1-sc);
			return (d2 < r*r);	// 球中心と終点s1の距離で交差判定
		}
		else{			// 球がs0とs1の間にある
			d2 = norm2((vc*v)/norm2(v)-c);
			return (d2 < r*r);	// 直線と球中心の距離で交差判定
		}
	}

	return false;
}





#endif	// _RX_SPH_SOLID_H_
