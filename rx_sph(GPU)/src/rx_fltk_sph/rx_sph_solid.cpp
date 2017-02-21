/*!
  @file rx_sph_solid.cpp
	
  @brief SPH用固体定義
*/


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph_solid.h"


//-----------------------------------------------------------------------------
// 定数・変数
//-----------------------------------------------------------------------------
const Vec3 RX_AABB_NORMALS[6] = { Vec3( 1.0,  0.0,  0.0), 
								  Vec3(-1.0,  0.0,  0.0), 
								  Vec3( 0.0,  1.0,  0.0), 
								  Vec3( 0.0, -1.0,  0.0), 
								  Vec3( 0.0,  0.0,  1.0), 
								  Vec3( 0.0,  0.0, -1.0) };


//-----------------------------------------------------------------------------
// 距離計算関数
//-----------------------------------------------------------------------------
/*!
 * 立方体と点の距離
 * @param[in] spos 立方体の中心を原点とした相対座標値
 * @param[in] r    半径(球の場合)
 * @param[in] sgn  立方体の内で距離が正:1,外で正:-1
 * @param[in] vMin 立方体の最小座標値(相対座標)
 * @param[in] vMax 立方体の最大座標値(相対座標)
 * @param[out] d   符号付距離値
 * @param[out] n   最近傍点の法線方向
 */
bool AABB_point_dist(const Vec3 &spos, const double &r, const int &sgn, 
					 const Vec3 &vMin, const Vec3 &vMax, 
					 double &d, Vec3 &n)
{
	bitset<6> bout;
	bout.reset();
	double d0[6];
	int idx0 = -1;

	// 各軸ごとに最小と最大境界外になっていないか調べる
	for(int i = 0; i < 3; ++i){
		int idx = 2*i;
		if((d0[idx] = (spos[i]-r)-vMin[i]) < 0.0){
			bout[idx] = true;
			idx0 = idx;
		}
		idx = 2*i+1;
		if((d0[idx] = vMax[i]-(spos[i]+r)) < 0.0){
			bout[idx] = true;
			idx0 = idx;
		}
	}

	// 立方体内(全軸で境界内)
	if(bout.none()){
		double min_d = 1e10;
		int idx1 = -1;
		for(int i = 0; i < 6; ++i){
			if(d0[i] < min_d){
				min_d = d0[i];
				idx1 = i;
			}
		}

		d = sgn*min_d;
		n = (idx1 != -1) ? sgn*RX_AABB_NORMALS[idx1] : Vec3(0.0);
		return true;
	}


	Vec3 x(0.0);
	for(int i = 0; i < 3; ++i){
		if(bout[2*i]){
			x[i] = d0[2*i];
		}
		else if(bout[2*i+1]){
			x[i] = -d0[2*i+1];
		}
	}

	// sgn = 1:箱，-1:オブジェクト
	int c = (int)bout.count();
	if(c == 1){
		// 平面近傍
		d = sgn*d0[idx0];
		n = sgn*RX_AABB_NORMALS[idx0];
	}
	else{
		// エッジ/コーナー近傍
		d = -sgn*norm(x);
		n = sgn*(-Unit(x));
	}

	return false;
}


//-----------------------------------------------------------------------------
// MARK:rxSolidBoxクラスの実装
//-----------------------------------------------------------------------------
/*!
 * 距離値計算
 * @param[in] pos グローバル座標での位置
 * @param[out] d 固体境界近傍点までの符号付距離
 * @param[out] n 法線方向
 * @param[out] v 近傍点での速度
 * @return 
 */
bool rxSolidBox::GetDistance(const Vec3 &pos, rxCollisionInfo &col)
{
	return GetDistanceR(pos, 0.0, col);
}

/*!
 * 距離値計算(球体)
 * @param[in] pos グローバル座標での位置
 * @param[out] d 固体境界近傍点までの符号付距離
 * @param[out] n 法線方向
 * @param[out] v 近傍点での速度
 * @return 
 */
bool rxSolidBox::GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col)
{
	Vec3 spos;

	spos = CalLocalCoord(pos);

	col.Velocity() = GetVelocityAtGrobal(pos);//Vec3(0.0);
	int sgn = -m_iSgn;

	double d;
	Vec3 n;
	AABB_point_dist(spos, r, sgn, m_vMin, m_vMax, d, n);

	col.Penetration() = d;
	col.Normal() = n;
	col.Contact() = pos+n*fabs(d);

	return (col.Penetration() <= 0.0);
}

bool rxSolidBox::GetCurvature(const Vec3 &pos, double &k)
{
	//return 0.0;
	return CalCurvature(pos, k, boost::bind(&rxSolidBox::GetDistance, this, _1, _2));
}

void rxSolidBox::SetGLMatrix(void)
{
	glTranslatef(m_vMassCenter[0], m_vMassCenter[1], m_vMassCenter[2]);
	glMultMatrixd(m_matRot.GetValue());
}

void rxSolidBox::Draw(const bool wire)
{
	glPushMatrix();

	SetGLMatrix();

	Vec3 sl = 0.5*(m_vMax-m_vMin);
	sl = RXFunc::Fabs(sl);
	glScalef(2.0*sl[0], 2.0*sl[1], 2.0*sl[2]);

	if(wire){
		glutWireCube(1.0);
	}
	else{
		glutSolidCube(1.0);
	}

	glPopMatrix();
}



//-----------------------------------------------------------------------------
// MARK:rxSolidOpenBoxクラスの実装
//-----------------------------------------------------------------------------
/*!
 * 距離値計算
 * @param[in] pos グローバル座標での位置
 * @param[out] d 固体境界近傍点までの符号付距離
 * @param[out] n 法線方向
 * @param[out] v 近傍点での速度
 * @return 
 */
bool rxSolidOpenBox::GetDistance(const Vec3 &pos, rxCollisionInfo &col)
{
	return GetDistanceR(pos, 0.0, col);
}

/*!
 * 距離値計算(球体)
 * @param[in] pos グローバル座標での位置
 * @param[out] d 固体境界近傍点までの符号付距離
 * @param[out] n 法線方向
 * @param[out] v 近傍点での速度
 * @return 
 */
bool rxSolidOpenBox::GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col)
{
	Vec3 spos;

	spos = CalLocalCoord(pos);

	col.Velocity() = GetVelocityAtGrobal(pos);//Vec3(0.0);
	int sgn = -m_iSgn;
	int t = 2;
	double d = RX_FEQ_INF;
	Vec3 n;

	double td;
	Vec3 tn;

	// 底
	Vec3 m0, m1;
	m0 = -m_vSLenOut;
	m1 =  m_vSLenOut;
	m1[t] = m_vSLenIn[t];
	AABB_point_dist(spos, r, sgn, m0, m1, td, tn);
	if(td < d){
		d = td;
		n = tn;
	}

	// 側面 -x
	m0 = Vec3(-m_vSLenOut[0], -m_vSLenOut[1], -m_vSLenIn[2]);
	m1 = Vec3(-m_vSLenIn[0],   m_vSLenOut[1],  m_vSLenOut[2]);
	AABB_point_dist(spos, r, sgn, m0, m1, td, tn);
	if(td < d){
		d = td;
		n = tn;
	}

	// 側面 +x
	m0 = Vec3( m_vSLenIn[0],  -m_vSLenOut[1], -m_vSLenIn[2]);
	m1 = Vec3( m_vSLenOut[0],  m_vSLenOut[1],  m_vSLenOut[2]);
	AABB_point_dist(spos, r, sgn, m0, m1, td, tn);
	if(td < d){
		d = td;
		n = tn;
	}

	// 側面 -y
	m0 = Vec3(-m_vSLenIn[0], -m_vSLenOut[1], -m_vSLenIn[2]);
	m1 = Vec3( m_vSLenIn[0], -m_vSLenIn[1],   m_vSLenOut[2]);
	AABB_point_dist(spos, r, sgn, m0, m1, td, tn);
	if(td < d){
		d = td;
		n = tn;
	}

	// 側面 +y
	m0 = Vec3(-m_vSLenIn[0],  m_vSLenIn[1],  -m_vSLenIn[2]);
	m1 = Vec3( m_vSLenIn[0],  m_vSLenOut[1],  m_vSLenOut[2]);
	AABB_point_dist(spos, r, sgn, m0, m1, td, tn);
	if(td < d){
		d = td;
		n = tn;
	}

	col.Penetration() = d;
	col.Normal() = n;
	col.Contact() = pos+n*fabs(d);

	return (col.Penetration() <= 0.0);
}

bool rxSolidOpenBox::GetCurvature(const Vec3 &pos, double &k)
{
	return CalCurvature(pos, k, boost::bind(&rxSolidOpenBox::GetDistance, this, _1, _2));
}

void rxSolidOpenBox::SetGLMatrix(void)
{
	glTranslatef(m_vMassCenter[0], m_vMassCenter[1], m_vMassCenter[2]);
	glMultMatrixd(m_matRot.GetValue());
}

inline void SetVerticesCube(const Vec3 &cn, const Vec3 &sl, Vec3 v[8])
{
	v[0] = cn+Vec3(-sl[0], -sl[1], -sl[2]);
	v[1] = cn+Vec3(-sl[0],  sl[1], -sl[2]);
	v[2] = cn+Vec3(-sl[0],  sl[1],  sl[2]);
	v[3] = cn+Vec3(-sl[0], -sl[1],  sl[2]);

	v[4] = cn+Vec3( sl[0], -sl[1], -sl[2]);
	v[5] = cn+Vec3( sl[0],  sl[1], -sl[2]);
	v[6] = cn+Vec3( sl[0],  sl[1],  sl[2]);
	v[7] = cn+Vec3( sl[0], -sl[1],  sl[2]);
}

inline void CreateBoxPolygon(const Vec3 &sl0, const Vec3 &sl1, const int &d, 
							 vector<Vec3> &vrts, vector< vector<int> > &idxs)
{
	if(d < 0 || d > 2) return;

	double h = sl1[d]-sl0[d];
	Vec3 cn(0.0);
	

	vrts.resize(16);

	// 外側の頂点
	SetVerticesCube(cn, sl1, &vrts[0]);

	// 内側の頂点
	cn[d] += h;
	SetVerticesCube(cn, sl0, &vrts[8]);


	int idxs0[5][4] = { {0, 3, 2, 1}, 
						{1, 2, 6, 5}, 
						{5, 6, 7, 4}, 
						{4, 7, 3, 0}, 
						{0, 1, 5, 4} };
	
	int idxs1[4][4] = { {2, 3, 11, 10}, 
						{3, 7, 15, 11}, 
						{7, 6, 14, 15}, 
						{6, 2, 10, 14} };
	
	// 三角形作成
	idxs.resize(28);
	for(int i = 0; i < 28; ++i) idxs[i].resize(3);

	int c = 0;

	// 外側の箱
	for(int i = 0; i < 5; ++i){
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs0[i][j];
		}
		c++;
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs0[i][((j+2 > 3) ? 0 : j+2)];
		}
		c++;
	}

	// 内側の箱
	for(int i = 0; i < 5; ++i){
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs0[i][2-j]+8;
		}
		c++;
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs0[i][(((2-j)+2 > 3) ? 0 : (2-j)+2)]+8;
		}
		c++;
	}

	// 上部
	for(int i = 0; i < 4; ++i){
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs1[i][j];
		}
		c++;
		for(int j = 0; j < 3; ++j){
			idxs[c][j] = idxs1[i][((j+2 > 3) ? 0 : j+2)];
		}
		c++;
	}

}

void rxSolidOpenBox::Draw(const bool wire)
{
	glPushMatrix();

	SetGLMatrix();

	Vec3 len0 = 2.0*m_vSLenIn;
	Vec3 len1 = 2.0*m_vSLenOut;
	double d = 0.5*(len1[2]-len0[2]);

	glTranslatef(0.0, 0.0, 0.5*len1[2]);
	if(wire){
		glPushMatrix();
		glTranslatef(0.0, 0.0, d);
		glScalef(len0[0], len0[1], len0[2]);
		glutWireCube(1.0);
		glPopMatrix();

		glPushMatrix();
		glScalef(len1[0], len1[1], len1[2]);
		glutWireCube(1.0);
		glPopMatrix();
	}
	else{
		vector<Vec3> vrts;
		vector< vector<int> > idxs;
		CreateBoxPolygon(m_vSLenIn, m_vSLenOut, 2, vrts, idxs);

		// インデックスを1始まりに
		int n = (int)idxs.size();
		for(int i = 0; i < n; ++i){
			for(int j = 0; j < 3; ++j){
				idxs[i][j]++;
			}
		}

		// ワイヤーフレーム描画
		glDisable(GL_LIGHTING);
		glPushMatrix();
		glTranslatef(0.0, 0.0, d);
		glScalef(len0[0], len0[1], len0[2]);
		glutWireCube(1.0);
		glPopMatrix();

		glPushMatrix();
		glScalef(len1[0], len1[1], len1[2]);
		glutWireCube(1.0);
		glPopMatrix();

		// 面描画
		glEnable(GL_LIGHTING);
		rxMaterial mat;
		mat.SetGL();
		glColor3f(0.0, 0.0, 1.0);
		for(int i = 0; i < (int)idxs.size(); ++i){
			glBegin(GL_POLYGON);
			for(int j = 0; j < (int)idxs[i].size(); ++j){
				glVertex3dv(vrts[idxs[i][j]-1].data);
			}
			glEnd();
		}
	}


	glPopMatrix();
}



//-----------------------------------------------------------------------------
// MARK:rxSolidSphereクラスの実装
//-----------------------------------------------------------------------------
bool rxSolidSphere::GetDistance(const Vec3 &pos, rxCollisionInfo &col)
{
	return GetDistanceR(pos, 0.0, col);
}

bool rxSolidSphere::GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col)
{
	Vec3 rpos = pos-m_vMassCenter;
	double d = m_iSgn*(norm(rpos)-m_fRadius);
	if(d < r){
		Vec3 n = Unit(rpos);

		col.Penetration() = d-r;
		col.Contact() = m_vMassCenter+n*(m_fRadius+m_iSgn*r);
		col.Normal() = m_iSgn*n;

		col.Velocity() = GetVelocityAtGrobal(pos);
	}
	else{
		return false;
	}

	return (col.Penetration() <= 0.0);
}

void rxSolidSphere::SetGLMatrix(void)
{
	glTranslatef(m_vMassCenter[0], m_vMassCenter[1], m_vMassCenter[2]);
	glMultMatrixd(m_matRot.GetValue());
}

/*!
 * 原点中心の円のワイヤーフレーム描画
 * @param rad 円の半径
 * @param n 分割数
 */
static void DrawWireCircle(const double &rad, const int &n)
{
	double t = 0.0;
	double dt = 2.0*RX_PI/(double)n;
 
	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), rad*sin(t), 0.0);
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();
}
 
/*!
 * 原点中心の円のワイヤーフレーム描画(XZ平面)
 * @param rad 円の半径
 * @param n 分割数
 */
static void DrawWireCircleXZ(const double &rad, const int &n)
{
	double t = 0.0;
	double dt = 2.0*RX_PI/(double)n;
 
	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), 0.0, rad*sin(t));
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();
}
 
/*!
 * 球のワイヤーフレーム描画
 * @param cen 球の中心
 * @param rad 球の半径
 * @param col 描画色
 */
void DrawWireSphere(const Vec3 &cen, const float &rad, const Vec3 &col)
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glTranslatef(cen[0], cen[1], cen[2]);
	glRotatef(90, 1.0, 0.0, 0.0);
	glColor3f(col[0], col[1], col[2]);
 
	// 緯度(x-y平面に平行)
	float z, dz;
	dz = 2.0*rad/8.0f;
	z = -(rad-dz);
	do{
		glPushMatrix();
		glTranslatef(0.0, 0.0, z);
		DrawWireCircle(sqrt(rad*rad-z*z), 32);
		glPopMatrix();
		z += dz;
	}while(z < rad);
 
	// 経度(z軸まわりに回転)
	float t, dt;
	t = 0.0f;
	dt = 180.0/8.0;
	do{
		glPushMatrix();
		glRotatef(t,  0.0, 0.0, 1.0);
		DrawWireCircleXZ(rad, 32);
		glPopMatrix();
 
		t += dt;
	}while(t < 180);
 
	//glutWireSphere(rad, 10, 5);
	glPopMatrix();
}

void rxSolidSphere::Draw(const bool wire)
{
	glPushMatrix();
	SetGLMatrix();

	if(wire){
		DrawWireSphere(Vec3(0.0), m_fRadius, Vec3(0.0, 1.0, 0.0));
	}
	else{
		glPushMatrix();
		glRotated(90, 1.0, 0.0, 0.0);
		glutSolidSphere(m_fRadius, 20, 10);
		glPopMatrix();
	}

	glPopMatrix();
}



