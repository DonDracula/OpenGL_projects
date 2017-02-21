/*!
  @file rx_sph_solid.h
	
  @brief SPH用固体定義
*/

#ifndef _RX_SPH_SOLID_H_
#define _RX_SPH_SOLID_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph_commons.h"

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
protected:
	Vec3 m_vMassCenter;	//!< 重心座標
	Vec3 m_vVelocity;	//!< 固体速度
	rxMatrix4 m_matRot;	//!< 姿勢
	rxMatrix4 m_matRotInv;	//!< 姿勢

	bool m_bFix;		//!< 固定フラグ

	int m_iSgn;			//!< 箱:-1, オブジェクト:1

public:
	rxSolid()
	{
		m_bFix = true;
		m_vMassCenter = Vec3(0.0);
		m_vVelocity = Vec3(0.0);
		m_iSgn = 1;

		m_matRot.MakeIdentity();
	}

	//
	// 純粋仮想関数
	//
	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col) = 0;	//!< 距離関数計算
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col) = 0;
	virtual bool GetCurvature(const Vec3 &pos, double &k) = 0;	//!< 距離関数の曲率計算
	virtual void Draw(const bool wire = true) = 0;				//!< OpenGLでの描画
	virtual void SetGLMatrix(void) = 0;							//!< OpenGL変換行列の適用

	virtual Vec3 GetMin(void) = 0;
	virtual Vec3 GetMax(void) = 0;


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

	inline bool RigidSimulation(const double &dt);		//!< 剛体シミュレーション

};



//-----------------------------------------------------------------------------
// MARK:rxSolidBox : 直方体
//-----------------------------------------------------------------------------
class rxSolidBox : public rxSolid
{
protected:
	Vec3 m_vMax, m_vMin;	//!< 最大座標，最小座標(中心からの相対値)

public:
	// コンストラクタ
	rxSolidBox(Vec3 minp, Vec3 maxp, int sgn)
	{
		Vec3 sl  = 0.5*(maxp-minp);
		Vec3 ctr = 0.5*(maxp+minp);

		m_vMin = -sl;
		m_vMax =  sl;
		m_vMassCenter = ctr;

		m_iSgn = sgn;
	}

	virtual Vec3 GetMin(void){ return m_vMassCenter+m_vMin; }
	virtual Vec3 GetMax(void){ return m_vMassCenter+m_vMax; }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);
	virtual bool GetCurvature(const Vec3 &pos, double &k);
	virtual void Draw(const bool wire = true);
	virtual void SetGLMatrix(void);
};

//-----------------------------------------------------------------------------
// MARK:rxSolidOpenBox : 直方体(開)
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
	virtual bool GetCurvature(const Vec3 &pos, double &k);
	virtual void Draw(const bool wire = true);
	virtual void SetGLMatrix(void);
};



//-----------------------------------------------------------------------------
// MARK:rxSolidSphere : 球
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
	}

	Vec3 GetCenter(void) const { return m_vMassCenter; }
	double GetRadius(void) const { return m_fRadius; }

	virtual Vec3 GetMin(void){ return m_vMassCenter-Vec3(m_fRadius); }
	virtual Vec3 GetMax(void){ return m_vMassCenter+Vec3(m_fRadius); }

	virtual bool GetDistance(const Vec3 &pos, rxCollisionInfo &col);
	virtual bool GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col);
	virtual bool GetCurvature(const Vec3 &pos, double &k){ return false; }
	virtual void Draw(const bool wire = true);
	virtual void SetGLMatrix(void);
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
inline bool CalCurvature(const Vec3 &pos, double &k, boost::function<bool (Vec3, rxCollisionInfo&)> fpDist)
{
	k = 0.0;

	double h = 0.005;
	double x0, y0, z0;
	double p[3][3][3];
	rxCollisionInfo col;

	x0 = pos[0]-0.5*h;
	y0 = pos[1]-0.5*h;
	z0 = pos[2]-0.5*h;

//	fpDist(Vec3(x0-h, y0-h, z0-h), col); p[0][0][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0-h, z0  ), col); p[0][0][1] = col.Penetration();
//	fpDist(Vec3(x0-h, y0-h, z0+h), col); p[0][0][2] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0-h), col); p[0][1][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0  ), col); p[0][1][1] = col.Penetration();
	fpDist(Vec3(x0-h, y0  , z0+h), col); p[0][1][2] = col.Penetration();
//	fpDist(Vec3(x0-h, y0+h, z0-h), col); p[0][2][0] = col.Penetration();
	fpDist(Vec3(x0-h, y0+h, z0  ), col); p[0][2][1] = col.Penetration();
//	fpDist(Vec3(x0-h, y0+h, z0+h), col); p[0][2][2] = col.Penetration();

	fpDist(Vec3(x0  , y0-h, z0-h), col); p[1][0][0] = col.Penetration();
	fpDist(Vec3(x0  , y0-h, z0  ), col); p[1][0][1] = col.Penetration();
	fpDist(Vec3(x0  , y0-h, z0+h), col); p[1][0][2] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0-h), col); p[1][1][0] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0  ), col); p[1][1][1] = col.Penetration();
	fpDist(Vec3(x0  , y0  , z0+h), col); p[1][1][2] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0-h), col); p[1][2][0] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0  ), col); p[1][2][1] = col.Penetration();
	fpDist(Vec3(x0  , y0+h, z0+h), col); p[1][2][2] = col.Penetration();

//	fpDist(Vec3(x0+h, y0-h, z0-h), col); p[2][0][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0-h, z0  ), col); p[2][0][1] = col.Penetration();
//	fpDist(Vec3(x0+h, y0-h, z0+h), col); p[2][0][2] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0-h), col); p[2][1][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0  ), col); p[2][1][1] = col.Penetration();
	fpDist(Vec3(x0+h, y0  , z0+h), col); p[2][1][2] = col.Penetration();
//	fpDist(Vec3(x0+h, y0+h, z0-h), col); p[2][2][0] = col.Penetration();
	fpDist(Vec3(x0+h, y0+h, z0  ), col); p[2][2][1] = col.Penetration();
//	fpDist(Vec3(x0+h, y0+h, z0+h), col); p[2][2][2] = col.Penetration();

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




#endif	// _RX_SPH_SOLID_H_
