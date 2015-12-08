//include files
#include "rx_sbd_2d.h"
#include "utils.h"

#include "rx_mesh_e.h"

#include "rx_sampler.h"
#include "rx_delaunay.h"

// SBD2d class
rxSBD2D ::rxSBD2D(int n)
{
	// size of the HeightField
	m_iNx = n;
	m_iNy = n;

	m_iModifiedConstraint = 1;
	m_fGravity = 9.81;
	m_fMass = 0.05;

	Init();
}

rxSBD2D::~rxSBD2D()
{

}

// clculate the Material coordinate of the triangle
//param: owrld pos, textrue pos, material pos
void calMaterialCoord(Vec2  x[3], Vec2 u[3], Vec2 c[2])
{
	Vec2 ub[2], xb[2];
	ub[0] = u[1]-u[0];
	ub[1] = u[2]-u[0];
	double du = ub[0][0]*ub[1][1]-ub[0][1]*ub[1][0];

	xb[0] = x[1]-x[0];
	xb[1] = x[2]-x[0];

	Vec2 tu, tv;
	tu[0] = ( xb[0][0]*ub[1][1]-xb[1][0]*ub[0][1])/du;
	tu[1] = ( xb[0][1]*ub[1][1]-xb[1][1]*ub[0][1])/du;
	tv[0] = (-xb[0][0]*ub[1][0]+xb[1][0]*ub[0][0])/du;
	tv[1] = (-xb[0][1]*ub[1][0]+xb[1][1]*ub[0][0])/du;

	Vec2 n[2];
	n[0] = Unit(tu);
	n[1] = Unit(tv);

	c[0][0] = dot(n[0], xb[0]);
	c[0][1] = dot(n[1], xb[0]);
	c[1][0] = dot(n[0], xb[1]);
	c[1][1] = dot(n[1], xb[1]);
}

//init
void rxSBD2D::Init(void)
{
	//create the mesh
	Vec2 c1(-1.0,-1.0);
	Vec2 c2(1.0,1.0);
	generateRandomMesh(c1,c2,0.15,300);

	//set the fix vector
	m_vFix.resize(m_iNv,0);
	int idx;
	idx = SearchNearest(Vec2(c1[0], c2[1]));
	SetFix(idx, m_vX[idx]);
	idx = SearchNearest(Vec2(c2[0], c2[1]));
	SetFix(idx, m_vX[idx]);

	//calculate Q
	m_vQ.resize(m_iNt*4,0.0);
	m_vInvQ.resize(m_iNt*4,0.0);
	for(int i = 0; i<m_iNt; ++i){
		Vec2 v[3];
		for(int j=0; j<3; ++j) v[j] = m_vX[m_vTri[i][j]];

		Vec2 c[2];
		calMaterialCoord(v,v,c);
		int idx = 4*i;
		m_vQ[idx+0] = c[0][0];
		m_vQ[idx+1] = c[1][0];
		m_vQ[idx+2] = c[0][1];
		m_vQ[idx+3] = c[1][1];

		//inverse matrix of Q
		double *q = &m_vQ[idx];
		double invq[4];
		if(CalInvMat2x2(q,invq)){
			for(int j = 0; j < 4; ++j) m_vInvQ[idx+j] = invq[j];
		}
		else{
			for(int j= 0; j<2; ++j){
				for(int k = 0; k <2; ++k){
					m_vInvQ[idx+2*j+k] = (j == k) ?1:0;
				}
			}
		}
	}

	m_fTime = 0.0;
	m_iStep = 0;
}

/*cal the strain tensor
 * @param[in] v 三角形の頂点位置
 * @param[in] invq 元の頂点位置ベクトルを並べた行列Qの逆行列{00,01,10,11}
 * @param[out] f 三角形の変形勾配テンソルF=PQ^-1 (縦ベクトルx2で行列を表現)
 * @param[out] S 右Cauchy-Green変形テンソルF^T F (縦ベクトルx2で行列を表現)
 * @param[out] dS Sの各頂点pkについての勾配∇pk
 */
void rxSBD2D::calStrainTensor(Vec2 * v, double *invq, Vec2 f[2], Vec2 S[2], Vec2 dS[3][2][2])
{
	Vec2 p[2], c[2];

	//cal matrix P
	Vec2 p0[2];
	calMaterialCoord(v,v,p0);
	p[0][0] = p0[0][0];
	p[0][1] = p0[1][0];
	p[1][0] = p0[0][1];
	p[1][1] = p0[1][1];

	//inverse matrix of Q
	//  Q^-1 = | c[0][0] c[1][0] |
	//		   | c[0][1] c[1][1] |
	c[0][0] = invq[0];
	c[0][1] = invq[2];
	c[1][0] = invq[1];
	c[1][1] = invq[3];

	// cal F = PQ^-1
	//  F = | f[0][0] f[1][0] | = | p[0][0] p[0][1] | * | c[0][0] c[1][0] |
	//      | f[0][1] f[1][1] |	  | p[1][0] p[1][1] |   | c[0][1] c[1][1] |
	f[0][0] = dot(p[0], c[0]);
	f[0][1] = dot(p[1], c[0]);
	f[1][0] = dot(p[0], c[1]);
	f[1][1] = dot(p[1], c[1]);

	//cal S and ∇S(Sij=fi・fj=(P ci)・(P cj), ∇Sij=fj ci^T + fi cj^T)
	for(int i= 0; i<2; ++i){
		for(int j = 0; j<2; ++j){
			S[i][j] = dot(f[i], f[j]);
			dS[1][i][j] = f[j]*c[i][0]+f[i]*c[j][0];
			dS[2][i][j] = f[j]*c[i][1]+f[i]*c[j][1];
			dS[0][i][j] = Vec2(0.0);
			for(int l = 1; l < 3; ++l){
				dS[0][i][j] += -dS[l][i][j];
		}
	}
}
}
/*
	cal the correct vertex pos
 * @param[in] v 三角形の頂点位置
 * @param[in] invq 元の頂点位置ベクトルを並べた行列Qの逆行列{00,01,10,11}
 * @param[in] invm 頂点質量mの逆数1/m
 * @param[out] dp 位置修正ベクトル(各頂点ごと)
*/

void rxSBD2D::calPositionCorrectionStrain(Vec2 *v, double *invq, double *invm, Vec2 dp[3])
{
	Vec2 f[2], S[2], dS[3][2][2];

	//cal strain tensor
	calStrainTensor(v, invq,f,S,dS);

	if(m_iModifiedConstraint)
	{
		S[0][0] = sqrt(S[0][0]);
		S[1][1] = sqrt(S[1][1]);
		S[0][1] /= norm(f[0])*norm(f[1]);
	}

	double si = 1.0;
	double lambda[4];		
		for(int i = 0; i < 2; ++i){
		for(int j = 0; j < 2; ++j){
			// 分母項
			double d = 0.0;
			for(int l = 0; l < 3; ++l){
				double wk = invm[l];
				d += wk*norm2(dS[l][i][j]);
			}
			if(i == j){
				// 式(33),(34)
				lambda[i] = (S[i][i]-si)/d;
			}
			else if(i < j){
				// 式(35)
				lambda[2] = S[i][j]/d;
			}
		}
	}
		
	// 変形テンソルによる位置修正量の計算
	for(int i = 0; i < 3; ++i){
		double wk = invm[i];
		dp[i] = Vec2(0.0);
		dp[i] += -lambda[0]*wk*dS[i][0][0];
		dp[i] += -lambda[1]*wk*dS[i][1][1];
		dp[i] += -lambda[2]*wk*dS[i][0][1];
	}

}

/*!
 * cal the correct area
 * @param[in] v 三角形の頂点位置
 * @param[in] invq 元の頂点位置ベクトルを並べた行列Qの逆行列{00,01,10,11}
 * @param[in] invm 頂点質量mの逆数1/m
 * @param[out] dp 位置修正ベクトル(各頂点ごと)
 */
void rxSBD2D::calPositionCorrectionArea(Vec2 *v, Vec2 *q, double *invm, Vec2 dp[3])
{
	Vec2 p[2];
	calMaterialCoord(v,v,p);

	Vec2 p12 = cross(p[0], p[1]);
	Vec2 p21 = cross(p[1], p[0]);
	Vec2 q12 = cross(q[0], q[1]);

	// 面積保存拘束条件とその勾配
	double C = norm2(p12)-norm2(q12);
	Vec2 dC[3];
	dC[1] = 2*cross(p[1], p12);
	dC[2] = 2*cross(p[0], p21);
	dC[0] = -dC[1]-dC[2];

		double d = 0.0;
	for(int i = 0; i < 3; ++i){
		double wk = invm[i];
		d += wk*norm2(dC[i]);
	}
	double lambda = C/d;

		for(int i = 0; i < 3; ++i){
		double wk = invm[i];
		dp[i] = -lambda*wk*dC[i];
	}
}

//update
int rxSBD2D::Update(double dt)
{
	//cal velocity
	for(int i = 0; i < m_iNv; ++i){
		if(m_vFix[i]) continue;
		Vec2 acc = Vec2(0.0, -m_fGravity);
		m_vV[i] += acc*dt;
	}

	//update pos
	for(int i = 0; i < m_iNv; ++i){
		if(m_vFix[i]) continue;
		m_vP[i] = m_vX[i]+m_vV[i]*dt;
	}

	//cal strain based constraint
	for(int ig = 0; ig < 50; ++ig){
		double total_dp = 0.0;
		for(int k = 0; k < m_iNt; ++k){
			Vec2 v[3], q[2];
			double invm[3], *invq = &m_vInvQ[4*k];
			for(int i = 0; i < 3; ++i){
				v[i] = m_vP[m_vTri[k][i]];
				invm[i] = 1.0/m_vM[m_vTri[k][i]];
			}
			q[0] = Vec2(m_vQ[4*k], m_vQ[4*k+2]);
			q[1] = Vec2(m_vQ[4*k+1], m_vQ[4*k+3]);

			// 変形テンソルSによる位置修正量の計算
			Vec2 dps[3];
			calPositionCorrectionStrain(v, invq, invm, dps);

			// 面積保存条件による位置修正量の計算
			Vec2 dpa[3];
			calPositionCorrectionArea(v, q, invq, dpa);
		
			// 各Constraintによる位置変動
			double ck = 0.5;	// stiffness coefficient
			for(int i = 0; i < 3; ++i){
				Vec2 dp(0.0);
				dp += ck*(dps[i]);
				//dp += ck*(dps[i]+dpa[i]);
				m_vP[m_vTri[k][i]] += dp;
				total_dp += norm2(dp);
			}
		}

		if(total_dp/m_iNt < 1e-5) break;
	}

	//update vec and pos
	for(int i = 0; i < m_iNv; ++i){
		if(m_vFix[i]) continue;
		m_vV[i] = (m_vP[i]-m_vX[i])/dt;
		m_vX[i] = m_vP[i];
	}

	m_fTime -= dt;
	m_iStep++;
	return m_iStep;
}

int rxSBD2D::Search(Vec2 pos,double h)
{
	int idx = -1;

	double h2 = h*h;
	for(int i= 0; i<m_iNv; ++i){
		if(norm2(m_vX[i]-pos) < h2){
			idx = i;
			break;
		}
	}
	return idx;
}

//search the neares vertex
int rxSBD2D::SearchNearest(Vec2 pos)
{
	int idx = -1;
	double min_d2 = RX_FEQ_INF;
	double d2;
	for(int i = 0; i<m_iNv; ++i){
		if ((d2 = norm2(m_vX[i]-pos)) < min_d2)
		{
			min_d2 = d2;
			idx = i;
		}
	}
	return idx;
}

//set fix vector
void rxSBD2D::SetFix(int idx, Vec2 pos)
{
	m_vFix[idx] = 1;
	m_vM[idx] = 100;
	m_vX[idx] = pos;
	m_vP[idx] = pos;
}

//unset fixed  vector
void rxSBD2D::UnsetFix(int idx)
{
	m_vFix[idx] = 0;
	m_vM[idx] = m_fMass;
}

//draw function 
void rxSBD2D::Draw(int drw, Vec3 line_col, Vec3 vertex_col)
{
	if(drw & 0x02){
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.0,1.0);
	}

	if(drw & 0x04){
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

		//draw all of the polygon
		for(int i= 0; i < m_iNt; ++i){
			vector<int> &face = m_vTri[i];
			int n = (int)face.size();

			//draw polygon
			glBegin(GL_POLYGON);
			for(int j = 0; j<n; ++j){
				int idx = face[j];
				if(idx >=0 && idx < m_iNv){
					glVertex2dv((m_vX[idx]).data);
				}
			}
			glEnd();
		}
	}

		// 頂点描画
	if(drw & 0x01){
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.0, 1.0);

		glDisable(GL_LIGHTING);
		glColor3dv(vertex_col.data);
		for(int i = 0; i < m_iNv; ++i){
			if(m_vFix[i]){
				glPointSize(10.0);
			}
			else{
				glPointSize(5.0);
			}
			glBegin(GL_POINTS);
			glVertex2dv((m_vX[i]).data);
			glEnd();
		}
	}

	// エッジ描画
	if(drw & 0x02){
		glDisable(GL_LIGHTING);
		glColor3dv(line_col.data);
		glLineWidth(1.0);
		for(int i = 0; i < m_iNt; ++i){
			vector<int> &face = m_vTri[i];
			int n = (int)face.size();
			
			glBegin(GL_LINE_LOOP);
			for(int j = 0; j < n; ++j){
				int idx = face[j];
				if(idx >= 0 && idx < m_iNv){
					glVertex2dv((m_vX[idx]).data);
				}
			}
			glEnd();
		}
	}
}

void rxSBD2D::generateMesh(Vec2 c1, Vec2 c2, int nx, int ny)
{
	if(!m_vX.empty()){
		m_vX.clear();
		m_vP.clear();
		m_vV.clear();
		m_vM.clear();
		m_vTri.clear();
	}

	// 頂点座標生成
	double dx = (c2[0]-c1[0])/(nx-1);
	double dz = (c2[1]-c1[1])/(ny-1);
	m_iNv = nx*ny;
	m_vX.resize(m_iNv);
	m_vP.resize(m_iNv);
	m_vV.resize(m_iNv);
	m_vM.resize(m_iNv);
	for(int j = 0; j < ny; ++j){
		for(int i = 0; i < nx; ++i){
			Vec2 pos;
			pos[0] = c1[0]+i*dx;
			pos[1] = c1[1]+j*dz;

			int idx = IDX(i, j, nx);
			m_vX[idx] = m_vP[idx] = pos;
			m_vV[idx] = Vec2(0.0);
			m_vM[idx] = m_fMass;
		}
	}

	// メッシュ作成
	for(int j = 0; j < ny-1; ++j){
		for(int i = 0; i < nx-1; ++i){
			vector<int> face;
			face.resize(3);

			face[0] = IDX(i, j, nx);
			face[1] = IDX(i+1, j+1, nx);
			face[2] = IDX(i+1, j, nx);
			m_vTri.push_back(face);

			face[0] = IDX(i, j, nx);
			face[1] = IDX(i, j+1, nx);
			face[2] = IDX(i+1, j+1, nx);
			m_vTri.push_back(face);
		}
	}

	m_iNt = (int)m_vTri.size();
}

/*!
 * cal the distance between point and segment
 * @param[in] v0,v1 線分の両端点座標
 * @param[in] p 点の座標
 * @return 距離
 */
inline double segment_point_dist(const Vec2 &v0, const Vec2 &v1, const Vec2 &p, Vec2 &ip)
{
	Vec2 v = Unit(v1-v0);
	Vec2 vp = p-v0;
	Vec2 vh = dot(vp, v)*v;
	ip = v0+vh;
	return norm(vp-vh);
}

/*!
 * n×nの頂点を持つメッシュ生成(x-z平面)
 * @param[in] c1,c2 2端点座標
 */
void rxSBD2D::generateRandomMesh(Vec2 c1, Vec2 c2, double min_dist, int n)
{
	if(!m_vX.empty()){
		m_vX.clear();
		m_vP.clear();
		m_vV.clear();
		m_vM.clear();
		m_vTri.clear();
	}

	Vec2 minp = c1, maxp = c2;

	vector<Vec2> c(4);
	c[0] = Vec2(minp[0], minp[1]);
	c[1] = Vec2(maxp[0], minp[1]);
	c[2] = Vec2(maxp[0], maxp[1]);
	c[3] = Vec2(minp[0], maxp[1]);

	// 4隅の点を追加
	vector<Vec2> points;
	for(int i = 0; i < 4; ++i) points.push_back(c[i]);

	// 境界エッジ上にmin_distを基準に点を追加
	double d = 0.0;
	for(int j = 0; j < 4; ++j){
		Vec2 v0 = c[j];
		Vec2 v1 = c[(j == 3 ? 0 : j+1)];
		Vec2 edir = v1-v0;
		double len = normalize(edir);
		while(d < len){
			double t = min_dist*RXFunc::Rand(1.4, 1.0);
			d += t;
			if(len-d < 0.3*min_dist) break;
			points.push_back(v0+d*edir);
		}
		d = 0;
	}

	// ポアソンディスクサンプリングで内部に点を生成
	rxUniformPoissonDiskSampler sampler(minp, maxp, n, 10, min_dist);
	sampler.Generate(points);

	// ドロネー三角形分割で三角形を生成
	vector< vector<int> > tris;
	CreateDelaunayTriangles(points, tris);
	
	m_iNv = (int)points.size();
	m_iNt = (int)tris.size();

	m_vX = m_vP = points;
	m_vTri = tris;
	m_vV.resize(m_iNv);
	m_vM.resize(m_iNv);
	for(int i = 0; i < m_iNv; ++i){
		m_vV[i] = Vec2(0.0);
		m_vM[i] = m_fMass;
	}
}

