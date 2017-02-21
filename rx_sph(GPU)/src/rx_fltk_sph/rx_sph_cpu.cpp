/*!
  @file rx_sph_cpu.cpp
	
  @brief SPH法(GPU)の実装
 
*/
// FILE --rx_sph_cpu.cpp--

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph.h"

#include "rx_cu_funcs.cuh"
#include "rx_cu_common.cuh"

#include "rx_pcube.h"


//-----------------------------------------------------------------------------
// rxSPHクラスの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 * @param[in] use_opengl VBO使用フラグ
 */
rxSPH::rxSPH(bool use_opengl) : 
	rxParticleSystemBase(use_opengl), 
	m_hNrm(0), 
	m_hFrc(0),
	m_hDens(0), 
	m_hPres(0), 
	m_hSurf(0), 
	m_hUpPos(0), 
	m_hPosW(0), 
	m_hEigen(0), 
	m_hRMatrix(0), 
	m_hG(0), 
	m_hVrts(0), 
	m_hTris(0), 
	m_pBoundary(0)
{
	m_v3Gravity = Vec3(0.0, -9.82, 0.0);

	m_fViscosity = 6.0f;
	m_fGasStiffness = 3.0f;
	m_fBuoyancy = 0.0f;

	m_fDamping = 0.0;
	m_fRestitution = 0.0;

	m_bCalNormal = false;
	m_bCalAnisotropic = false;

	// 近傍探索セル
	m_pNNGrid = new rxNNGrid(DIM);


	m_uNumParticles = 0;
	m_iNumTris = 0;

	m_iColorType = RX_RAMP;
}

/*!
 * デストラクタ
 */
rxSPH::~rxSPH()
{
	Finalize();
}


/*!
 * シミュレーションの初期化
 * @param[in] max_particles 最大パーティクル数
 * @param[in] boundary_ext 境界の大きさ(各辺の長さの1/2)
 * @param[in] dens 初期密度
 * @param[in] mass パーティクルの質量
 * @param[in] kernel_particle 有効半径h以内のパーティクル数
 */
void rxSPH::Initialize(const rxSPHEnviroment &env)
{
	// MARK:Initialize
	RXCOUT << "[rxSPH::Initialize]" << endl;

	m_fInitDens        = env.dens;
	m_fMass            = env.mass;
	m_iKernelParticles = env.kernel_particles;

	RXREAL volume = env.max_particles*m_fMass/m_fInitDens;

	m_fEffectiveRadius = pow(((3.0*m_iKernelParticles*volume)/(4.0*env.max_particles*RX_PI)), 1.0/3.0);
	m_fParticleRadius = 0.5f*m_fEffectiveRadius;

	RXCOUT << "particle : " << endl;
	RXCOUT << " n_max = " << env.max_particles << endl;
	RXCOUT << " h = " << m_fEffectiveRadius << endl;
	RXCOUT << " r = " << m_fParticleRadius << endl;
	RXCOUT << " dens = " << m_fInitDens << endl;
	RXCOUT << " mass = " << m_fMass << endl;
	RXCOUT << " kernel_particles = " << m_iKernelParticles << endl;
	RXCOUT << " volume = " << volume << endl;

	RXREAL h = m_fEffectiveRadius;
	RXREAL r = m_fParticleRadius;

	//
	// 境界設定
	//
	m_pBoundary = new rxSolidBox(env.boundary_cen-env.boundary_ext, env.boundary_cen+env.boundary_ext, -1);
	//m_pBoundary = new rxSolidSphere(Vec3(0.0, 0.1, 0.0), 0.25, -1);
	
	// シミュレーション環境の大きさ
	m_v3EnvMin = m_pBoundary->GetMin();
	m_v3EnvMax = m_pBoundary->GetMax();
	RXCOUT << "simlation range : " << m_v3EnvMin << " - " << m_v3EnvMax << endl;

	Vec3 world_size = m_v3EnvMax-m_v3EnvMin;
	Vec3 world_origin = m_v3EnvMin;
	
	double expansion = 0.01;
	world_origin -= 0.5*expansion*world_size;
	world_size *= (1.0+expansion); // シミュレーション環境全体を覆うように設定

	m_v3EnvMin = world_origin;
	m_v3EnvMax = world_origin+world_size;

	// カーネル関数の定数
	m_fWpoly6	=  315.0/(64.0*RX_PI*pow(h, (RXREAL)9.0));
	m_fGWpoly6	= -945.0/(32.0*RX_PI*pow(h, (RXREAL)9.0));
	m_fLWpoly6	= -945.0/(32.0*RX_PI*pow(h, (RXREAL)9.0));

	m_fWspiky	=  15.0/(RX_PI*pow(h, (RXREAL)6.0));
	m_fGWspiky	= -45.0/(RX_PI*pow(h, (RXREAL)6.0));
	m_fLWspiky	= -90.0/(RX_PI*pow(h, (RXREAL)6.0));

	m_fWvisc	= 15.0/(2.0*RX_PI*pow(h, (RXREAL)3.0));
	m_fGWvisc	= 15.0/(2.0*RX_PI*pow(h, (RXREAL)3.0));
	m_fLWvisc	= 45.0/(RX_PI*pow(h, (RXREAL)6.0));

	m_uNumParticles = 0;

	Allocate(env.max_particles);
}

/*!
 * メモリの確保
 *  - 最大パーティクル数で確保
 * @param[in] max_particles 最大パーティクル数
 */
void rxSPH::Allocate(int max_particles)
{
	// MARK:Allocate
	assert(!m_bInitialized);

	//m_uNumParticles = max_particles;
	m_uMaxParticles = max_particles;
	
	unsigned int size  = m_uMaxParticles*DIM;
	unsigned int size1 = m_uMaxParticles;
	unsigned int mem_size  = sizeof(RXREAL)*size;
	unsigned int mem_size1 = sizeof(RXREAL)*size1;

	//
	// メモリ確保
	//
	m_hPos = new RXREAL[size];
	m_hVel = new RXREAL[size];
	m_hNrm = new RXREAL[size];
	m_hFrc = new RXREAL[size];
	memset(m_hPos, 0, mem_size);
	memset(m_hVel, 0, mem_size);
	memset(m_hNrm, 0, mem_size);
	memset(m_hFrc, 0, mem_size);

	m_hDens = new RXREAL[size1];
	m_hPres = new RXREAL[size1];
	memset(m_hDens, 0, mem_size1);
	memset(m_hPres, 0, mem_size1);

	m_hSurf = new uint[m_uMaxParticles];
	memset(m_hSurf, 0, sizeof(uint)*m_uMaxParticles);

	m_hAttr = new uint[size1];
	memset(m_hAttr, 0, size1*sizeof(uint));
	m_hTmp = new RXREAL[m_uMaxParticles];
	memset(m_hTmp, 0, sizeof(RXREAL)*m_uMaxParticles);

	// Anisotropic kernel
	m_hUpPos = new RXREAL[size];
	m_hPosW = new RXREAL[size];
	m_hEigen = new RXREAL[m_uMaxParticles*3];
	m_hRMatrix = new RXREAL[m_uMaxParticles*9];
	m_hG = new RXREAL[m_uMaxParticles*9];

	m_vNeighs.resize(m_uMaxParticles);

	if(m_bUseOpenGL){
		m_posVBO = createVBO(mem_size);	
		m_colorVBO = createVBO(m_uMaxParticles*4*sizeof(RXREAL));

		SetColorVBO(RX_RAMP);
	}

	// 分割セル設定
	m_pNNGrid->Setup(m_v3EnvMin, m_v3EnvMax, m_fEffectiveRadius, m_uMaxParticles);
	m_vNeighs.resize(m_uMaxParticles);

	m_bInitialized = true;
}

/*!
 * 確保したメモリの解放
 */
void rxSPH::Finalize(void)
{
	assert(m_bInitialized);

	// メモリ解放
	if(m_hPos) delete [] m_hPos;
	if(m_hVel) delete [] m_hVel;
	if(m_hNrm) delete [] m_hNrm;
	if(m_hFrc) delete [] m_hFrc;

	if(m_hDens) delete [] m_hDens;
	if(m_hPres) delete [] m_hPres;

	if(m_hSurf) delete [] m_hSurf;
	if(m_hAttr) delete [] m_hAttr;
	if(m_hTmp) delete [] m_hTmp;

	// Anisotoropic kernel
	if(m_hUpPos) delete [] m_hUpPos;
	if(m_hPosW) delete [] m_hPosW;
	if(m_hEigen) delete [] m_hEigen;
	if(m_hRMatrix) delete [] m_hRMatrix;
	if(m_hG) delete [] m_hG;

	m_vNeighs.clear();

	if(m_bUseOpenGL){
		glDeleteBuffers(1, (const GLuint*)&m_posVBO);
		glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
	}

	if(m_pNNGrid) delete m_pNNGrid;
	m_vNeighs.clear();

	if(m_hVrts) delete [] m_hVrts;
	if(m_hTris) delete [] m_hTris;

	if(m_pBoundary) delete m_pBoundary;

	int num_solid = (int)m_vSolids.size();
	for(int i = 0; i < num_solid; ++i){
		delete m_vSolids[i];
	}
	m_vSolids.clear();	

	m_uNumParticles = 0;
	m_uMaxParticles = 0;
}


/*!
 * SPHを1ステップ進める
 * @param[in] dt 時間ステップ幅
 * @retval ture  計算完了
 * @retval false 最大ステップ数を超えています
 */
bool rxSPH::Update(RXREAL dt, int step)
{
	// 流入パーティクルを追加
	if(!m_vInletLines.empty()){
		int start = (m_iInletStart == -1 ? 0 : m_iInletStart);
		int num = 0;
		vector<rxInletLine>::iterator itr = m_vInletLines.begin();
		for(; itr != m_vInletLines.end(); ++itr){
			rxInletLine iline = *itr;
			if(iline.span > 0 && step%(iline.span) == 0){
				int count = addParticles(m_iInletStart, iline);
				num += count;
			}
		}
		SetArrayVBO(RX_POSITION, m_hPos, start, num);
		SetArrayVBO(RX_VELOCITY, m_hVel, start, num);
	}


	assert(m_bInitialized);

	Vec3 v_half, acc_f;

	static bool init = false;

	for(uint j = 0; j < m_solverIterations; ++j){
		// 近傍粒子探索
		SetParticlesToCell();

		RXTIMER("cell construction");

		// 密度計算
		calDensity();

		// 圧力項,粘性項の計算
		calForce();

		RXTIMER("force calculation");

		// 位置・速度の更新
		integrate(m_hPos, m_hVel, m_hFrc, m_hPos, m_hVel, dt);

		RXTIMER("update position");
	}


	SetArrayVBO(RX_POSITION, m_hPos, 0, m_uNumParticles);

	SetColorVBO(m_iColorType);

	RXTIMER("color(vbo)");

	return true;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxSPH::GetParticle(void)
{
	return m_hPos;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxSPH::GetParticleDevice(void)
{
	if(!m_uNumParticles) return 0;

	RXREAL *dPos = 0;
	CuAllocateArray((void**)&dPos, m_uNumParticles*4*sizeof(RXREAL));

	CuCopyArrayToDevice(dPos, m_hPos, 0, m_uNumParticles*4*sizeof(RXREAL));

	return dPos;
}


/*!
 * カラー値用VBOの編集
 * @param[in] type 色のベースとする物性値
 */
void rxSPH::SetColorVBO(int type)
{
	switch(type){
	case RX_DENSITY:
		SetColorVBOFromArray(m_hDens, 1, false, m_fInitDens);
		break;

	case RX_PRESSURE:
		SetColorVBOFromArray(m_hPres, 1);
		break;

	case RX_TEST:
		SetColorVBOFromArray(m_hTmp, 1, false, m_fTmpMax);
		break;
		
	case RX_SURFACE:	// 表面パーティクルだけ色を変えて描画
		if(m_bUseOpenGL){
			// カラーバッファに値を設定
			glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
			RXREAL *data = (RXREAL*)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			RXREAL *ptr = data;
			for(uint i = 0; i < m_uNumParticles; ++i){
				uint s = m_hSurf[i];

				*ptr++ = (RXREAL)s;
				*ptr++ = 0.0f;
				*ptr++ = 0.0f;
				*ptr++ = 1.0f;
			}
			glUnmapBufferARB(GL_ARRAY_BUFFER);
		}
		break;

	case RX_CONSTANT:
		if(m_bUseOpenGL){
			// カラーバッファに値を設定
			glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
			RXREAL *data = (RXREAL*)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			RXREAL *ptr = data;
			for(uint i = 0; i < m_uNumParticles; ++i){
				RXREAL t = i/(RXREAL)m_uNumParticles;
				*ptr++ = 0.15f;
				*ptr++ = 0.15f;
				*ptr++ = 0.95f;
				*ptr++ = 1.0f;
			}
			glUnmapBufferARB(GL_ARRAY_BUFFER);
		}
		break;

	case RX_RAMP:
		if(m_bUseOpenGL){
			// カラーバッファに値を設定
			glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
			RXREAL *data = (RXREAL*)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			RXREAL *ptr = data;
			for(uint i = 0; i < m_uNumParticles; ++i){
				RXREAL t = i/(RXREAL)m_uNumParticles;
#if 0
				*ptr++ = rand()/(RXREAL)RAND_MAX;
				*ptr++ = rand()/(RXREAL)RAND_MAX;
				*ptr++ = rand()/(RXREAL)RAND_MAX;
#else
				RX_COLOR_RAMP(t, ptr);
				ptr += 3;
#endif
				*ptr++ = 1.0f;
			}
			glUnmapBufferARB(GL_ARRAY_BUFFER);
		}
		break;

	default:
		break;
	}

}



/*!
 * 密度を計算
 */
void rxSPH::calDensity(void)
{
	// MRK:calDensity
	RXREAL r0 = m_fInitDens;
	RXREAL h = m_fEffectiveRadius;

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = m_hPos[4*i+0];
		pos0[1] = m_hPos[4*i+1];
		pos0[2] = m_hPos[4*i+2];

		m_hDens[i] = 0.0;

		// 近傍粒子
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPos[4*j+0];
			pos1[1] = m_hPos[4*j+1];
			pos1[2] = m_hPos[4*j+2];

			RXREAL r = sqrt(itr->Dist2);

			RXREAL q = h*h-r*r;

			m_hDens[i] += m_fMass*m_fWpoly6*q*q*q;
		}

		m_hPres[i] = m_fGasStiffness*(m_hDens[i]-r0);
	}

}


/*!
 * 圧力項，粘性項の計算
 */
void rxSPH::calForce(void)
{
	Vec3 rij, vji;
	RXREAL h = m_fEffectiveRadius;
	RXREAL r0 = m_fInitDens;
	
	for(uint i = 0; i < m_uNumParticles; ++i){
		m_hFrc[4*i+0] = 0.0;
		m_hFrc[4*i+1] = 0.0;
		m_hFrc[4*i+2] = 0.0;
		m_hFrc[4*i+3] = 0.0;

		m_hPres[i] = m_fGasStiffness*(m_hDens[i]-r0);
	}


	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0, vel0;
		pos0 = Vec3(m_hPos[4*i+0], m_hPos[4*i+1], m_hPos[4*i+2]);
		vel0 = Vec3(m_hVel[4*i+0], m_hVel[4*i+1], m_hVel[4*i+2]);

		RXREAL prsi = m_hPres[i]/(m_hDens[i]*m_hDens[i]);

		Vec3 pfrc(0.0), vfrc(0.0);
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1, vel1;
			pos1 = Vec3(m_hPos[4*j+0], m_hPos[4*j+1], m_hPos[4*j+2]);
			vel1 = Vec3(m_hVel[4*j+0], m_hVel[4*j+1], m_hVel[4*j+2]);

			rij = pos0-pos1;
			vji = vel1-vel0;

			RXREAL r = norm(rij);//sqrt(itr->Dist2);
			RXREAL q = h-r;

			RXREAL prsj = m_hPres[j]/(m_hDens[j]*m_hDens[j]);

			Vec3 kernel;
			if(r <= 0.0){
				Vec3 rnd;
				rnd = RXFunc::Rand(Vec3(-0.0001), Vec3(0.0001));
				kernel = m_fGWspiky*rnd;
			}
			else{
				kernel = m_fGWspiky*q*q*rij/r;
			}

			// 圧力
			pfrc += m_fMass*(prsi+prsj)*kernel;

			// 粘性
			vfrc += m_fMass*(vji/m_hDens[j])*m_fLWvisc*q;
		}

		Vec3 force(0.0);
		force += -m_hDens[i]*pfrc;
		force += m_fViscosity*vfrc;

		// 外力項
		if(m_fBuoyancy > 0){
			force += m_v3Gravity*m_fBuoyancy*(m_hDens[i]-r0);
		}
		else{
			force += m_v3Gravity*m_hDens[i];
		}

		m_hFrc[4*i+0] += force[0];
		m_hFrc[4*i+1] += force[1];
		m_hFrc[4*i+2] += force[2];
	}
}

/*!
 * 法線を計算
 */
void rxSPH::calNormal(void)
{
	RXREAL h = m_fEffectiveRadius;

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = m_hPos[4*i+0];
		pos0[1] = m_hPos[4*i+1];
		pos0[2] = m_hPos[4*i+2];

		Vec3 nrm(0.0);

		// 近傍粒子
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1;
			pos1[0] = m_hPos[4*j+0];
			pos1[1] = m_hPos[4*j+1];
			pos1[2] = m_hPos[4*j+2];

			Vec3 rij = pos0-pos1;

			RXREAL r = sqrt(itr->Dist2);
			RXREAL q = h*h-r*r;

			nrm += (m_fMass/m_hDens[j])*m_fGWpoly6*q*q*rij;
		}

		//normalize(nrm);
		//nrm *= -1;

		m_hNrm[4*i+0] = nrm[0];
		m_hNrm[4*i+1] = nrm[1];
		m_hNrm[4*i+2] = nrm[2];
		m_hNrm[4*i+3] = 0.0;
	}

}


/*!
 * レイ/線分と三角形の交差
 * @param[in] P0,P1 レイ/線分の端点orレイ上の点
 * @param[in] V0,V1,V2 三角形の頂点座標
 * @param[out] I 交点座標
 * @retval 1 交点Iで交差 
 * @retval 0 交点なし
 * @retval 2 三角形の平面内
 * @retval -1 三角形が"degenerate"である(面積が0，つまり，線分か点になっている)
 */
static int intersectSegmentTriangle(Vec3 P0, Vec3 P1,			// Segment
									Vec3 V0, Vec3 V1, Vec3 V2,	// Triangle
									Vec3 &I, Vec3 &n, float rp)			// Intersection point (return)
{
	// 三角形のエッジベクトルと法線
	Vec3 u = V1-V0;
	Vec3 v = V2-V0;
	n = Unit(cross(u, v));
	if(RXFunc::IsZeroVec(n)){
		return -1;	// 三角形が"degenerate"である(面積が0)
	}

	// 線分
	Vec3 dir = P1-P0;
	double a = dot(n, P0-V0);
	double b = dot(n, dir);
	if(fabs(b) < 1e-10){	// 線分と三角形平面が平行
		//if(fabs(a+b) < rp){
		//	I = P1+n*(rp-fabs(a+b));
		//	return 1;
		//}
		//else if(fabs(a) < rp){
		//	I = P1+n*(rp-fabs(a));
		//	return 1;
		//}
		//else{
			if(a == 0){
				return 2;	// 線分が平面上
			}
			else{
				return 0;	// 交点なし
			}
		//}
	}

	// 交点計算

	// 2端点がそれぞれ異なる面にあるかどうかを判定
	float r = -a/b;
	Vec3 offset = Vec3(0.0);
	float dn = 0;
	float sign_n = 1;
	//if(r < 0.0 || fabs(a) > fabs(b) || b > 0){
	//	return 0;
	//}
	if(a < 0){
		return 0;
	}

	if(r < 0.0){
		//if(fabs(a) < rp){
		//	r = 0.0;
		//	offset = -fabs(a)*n;

		//	dn = fabs(a);
		//}
		//else{
			return 0;
		//}
	}
	else{
		if(fabs(a) > fabs(b)){
			//if(fabs(a+b) < rp){
			//	r = 1.0;
			//	offset = -fabs(a+b)*n;

			//	dn = fabs(a+b);
			//}
			//else{
				return 0;
			//}
		}
		else{
			if(b > 0){
				return 0;
			}
		}
	}

	// 線分と平面の交点
	I = P0+r*dir;//+offset;

	// 交点が三角形内にあるかどうかの判定
	double uu, uv, vv, wu, wv, D;
	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	Vec3 w = I-V0;
	wu = dot(w, u);
	wv = dot(w, v);
	D = uv*uv-uu*vv;

	double s, t;
	s = (uv*wv-vv*wu)/D;
	if(s < 0.0 || s > 1.0){
		return 0;
	}
	
	t = (uv*wu-uu*wv)/D;
	if(t < 0.0 || (s+t) > 1.0){
		return 0;
	}

	//I -= offset;

	// 半径分だけ戻す
	//Vec3 back = -dir*((rp-dn)/(dot(-dir, sign_n*n)));
	//I += back;

	return 1;
}

/*!
 * ポリゴンオブジェクトとの衝突判定，衝突応答
 * @param[in] grid_hash 調査するグリッドのハッシュ
 * @param[in] pos0 前ステップの位置
 * @param[inout] pos1 新しい位置
 * @param[inout] vel 速度
 * @param[in] dt タイムステップ幅
 * @return 衝突オブジェクトの数
 */
int rxSPH::calCollisionPolygon(uint grid_hash, Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt)
{
	vector<int> polys_in_cell;

	int c = 0;
	if(m_pNNGrid->GetPolygonsInCell(grid_hash, polys_in_cell)){
		for(int j = 0; j < (int)polys_in_cell.size(); ++j){
			int pidx = polys_in_cell[j];

			int vidx[3];
			vidx[0] = m_hTris[3*pidx+0];
			vidx[1] = m_hTris[3*pidx+1];
			vidx[2] = m_hTris[3*pidx+2];

			Vec3 vrts[3];
			vrts[0] = Vec3(m_hVrts[3*vidx[0]], m_hVrts[3*vidx[0]+1], m_hVrts[3*vidx[0]+2]);
			vrts[1] = Vec3(m_hVrts[3*vidx[1]], m_hVrts[3*vidx[1]+1], m_hVrts[3*vidx[1]+2]);
			vrts[2] = Vec3(m_hVrts[3*vidx[2]], m_hVrts[3*vidx[2]+1], m_hVrts[3*vidx[2]+2]);

			Vec3 cp, n;
			if(intersectSegmentTriangle(pos0, pos1, vrts[0], vrts[1], vrts[2], cp, n, m_fParticleRadius) == 1){
				double d = length(pos1-cp);
				n = Unit(n);

				RXREAL res = m_fRestitution;
				res = (res > 0) ? (res*fabs(d)/(dt*length(vel))) : 0.0f;
				Vec3 vr = -(1.0+res)*n*dot(n, vel);

				double l = norm(pos1-pos0);
				pos1 = cp+vr*(dt*d/l);
				vel += vr;

				c++;
			}
		}
	}

	return c;
}


/*!
 * 固体オブジェクトとの衝突判定，衝突応答
 * @param[in] pos0 前ステップの位置
 * @param[inout] pos1 新しい位置
 * @param[inout] vel 速度
 * @param[in] dt タイムステップ幅
 * @return 衝突オブジェクトの数
 */
int rxSPH::calCollisionSolid(Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt)
{
	int c = 0;
	rxCollisionInfo coli;

	// 固体オブジェクトとの衝突処理
	for(vector<rxSolid*>::iterator i = m_vSolids.begin(); i != m_vSolids.end(); ++i){
		if((*i)->GetDistanceR(pos1, m_fParticleRadius, coli)){
			RXREAL res = m_fDamping;
			res = (res > 0) ? (res*fabs(coli.Penetration())/(dt*norm(vel))) : 0.0f;
			vel -= (1+res)*dot(vel, coli.Normal())*coli.Normal();
			pos1 = coli.Contact();
		}
	}

	// シミュレーション空間境界との衝突処理
	if(m_pBoundary->GetDistanceR(pos1, m_fParticleRadius, coli)){
		RXREAL res = m_fDamping;
		res = (res > 0) ? (res*fabs(coli.Penetration())/(dt*norm(vel))) : 0.0f;
		vel -= (1+res)*dot(vel, coli.Normal())*coli.Normal();
		pos1 = coli.Contact();
	}

	return c;
}

/*!
 * 位置・速度の更新
 * @param[in] pos パーティクル位置
 * @param[in] vel パーティクル速度
 * @param[in] frc パーティクルにかかる力
 * @param[out] pos_new 更新パーティクル位置
 * @param[out] vel_new 更新パーティクル速度
 * @param[in] dt タイムステップ幅
 */
void rxSPH::integrate(const RXREAL *pos, const RXREAL *vel, const RXREAL *frc, 
					  RXREAL *pos_new, RXREAL *vel_new, RXREAL dt)
{
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 x, x_old, v, f, v_old;
		for(int k = 0; k < 3; ++k){
			x[k] = pos[DIM*i+k];
			v[k] = vel[DIM*i+k];
			f[k] = frc[DIM*i+k];
		}
		x_old = x;

		// 新しい速度と位置
		v += dt*f/m_hDens[i];
		x += dt*v;

		// ポリゴンオブジェクトとの交差判定
		if(m_iNumTris != 0){
			uint grid_hash0 = m_pNNGrid->CalGridHash(x_old);
			calCollisionPolygon(grid_hash0, x_old, x, v, dt);

			uint grid_hash1 = m_pNNGrid->CalGridHash(x);
			if(grid_hash1 != grid_hash0){
				calCollisionPolygon(grid_hash1, x_old, x, v, dt);
			}
		}

		// 境界との衝突判定
		calCollisionSolid(x_old, x, v, dt);

		// 新しい速度と位置で更新
		for(int k = 0; k < 3; ++k){
			pos_new[DIM*i+k] = x[k];
			vel_new[DIM*i+k] = v[k];
		}

	}
}



//-----------------------------------------------------------------------------
// 表面パーティクルの検出
//-----------------------------------------------------------------------------
/*!
 * 表面パーティクル検出
 *  - B. Solenthaler, Y. Zhang and R. Pajarola, "Efficient Refinement of Dynamic Point Data",
 *    Proceedings Eurographics/IEEE VGTC Symposium on Point-Based Graphics, 2007.
 *  - 3.1 Surface Particle Detection の 式(2) 
 *  - d_i,cm が閾値以上で表面パーティクルと判定と書かれているが，
 *    d_i,cm は近傍パーティクルの重心へのベクトルで実際にはそのベクトルの長さ |d_i,cm| を使って判定
 */
void rxSPH::DetectSurfaceParticles(void)
{
	RXREAL h = m_fEffectiveRadius;
	RXREAL r = m_fParticleRadius;

	//m_hPos = GetArrayVBO(RX_POSITION);

	// 近傍粒子探索
	SetParticlesToCell();

	for(uint i = 0; i < m_uNumParticles; ++i){
		double d = CalDistToNormalizedMassCenter(i);
		int nn_num = (int)m_vNeighs[i].size();

		// デバッグ用
		//m_hTmp[i] = (RXREAL)d;
		//m_fTmpMax = r;

		//m_hSurf[i] = 0;
		if(nn_num <= 3){	// 近傍パーティクル数が3以下ならば表面
			m_hSurf[i] = 1;
		}
		else{				// 3より大きい場合は近傍重心までの距離で判断
			if(m_hSurf[i]){
				// 前ステップで表面だったら小さい閾値で判断
				if(d < g_fSurfThr[0]*r) m_hSurf[i] = 0;
			}
			else{
				if(d > g_fSurfThr[1]*r || d < 0.0) m_hSurf[i] = 1;
			}
		}

		// 密度を使って判断する場合
		//if(m_hDens[2*i] < 0.7*m_fInitMaxDens){
		//	m_hSurf[i] = 1;
		//}
	}
}

/*!
 * 近傍パーティクルの正規化重心までの距離を計算
 * @param[in] i パーティクルインデックス
 */
double rxSPH::CalDistToNormalizedMassCenter(const int i)
{
	Vec3 sum_pos(0.0);
	double sum_mass = 0.0;

	Vec3 pos0 = Vec3(m_hPos[DIM*i+0], m_hPos[DIM*i+1], m_hPos[DIM*i+2]);

	for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
		int j = itr->Idx;
		if(j < 0 && i == j) continue;

		//RXREAL r = sqrt(itr->Dist2);
		Vec3 pos1 = Vec3(m_hPos[DIM*j+0], m_hPos[DIM*j+1], m_hPos[DIM*j+2]);

		sum_pos  += (pos0-pos1)*m_fMass;
		sum_mass += m_fMass;
	}

	double dis = DBL_MIN;
	if(sum_mass != 0.0){
		sum_pos /= sum_mass;
		dis     = norm(sum_pos);
	}

	return dis;
}

/*!
 * 表面パーティクル情報の取得
 *  パーティクル数と同じ大きさのuint配列で表面パーティクルならば1, そうでなければ0が格納されている
 */
uint* rxSPH::GetArraySurf(void)
{
	return m_hSurf;
}

/*!
 * 指定した座標の近傍の表面パーティクル情報を取得する
 * @param[in] pos 探索中心座標
 * @param[in] h 探索半径(0以下の値を設定したら有効半径を用いる)
 * @param[out] sp 表面パーティクル
 * @return 見つかったパーティクル数
 */
int rxSPH::GetSurfaceParticles(const Vec3 pos, RXREAL h, vector<rxSurfaceParticle> &sps)
{
	int num = 0;
	sps.clear();

	DetectSurfaceParticles();

	if(pos[0] < m_v3EnvMin[0]) return 0;
	if(pos[0] > m_v3EnvMax[0]) return 0;
	if(pos[1] < m_v3EnvMin[1]) return 0;
	if(pos[1] > m_v3EnvMax[1]) return 0;
	if(pos[2] < m_v3EnvMin[2]) return 0;
	if(pos[2] > m_v3EnvMax[2]) return 0;

	if(h <= 0.0) h = m_fEffectiveRadius;

	vector<rxNeigh> ne;
	m_pNNGrid->GetNN(pos, m_hPos, m_uNumParticles, ne, h);

	// 近傍粒子
	for(vector<rxNeigh>::iterator itr = ne.begin(); itr != ne.end(); ++itr){
		int j = itr->Idx;
		if(j < 0) continue;

		int surf = m_hSurf[j];

		if(surf){
			rxSurfaceParticle sp;
			sp.pos = Vec3(m_hPos[DIM*j+0], m_hPos[DIM*j+1], m_hPos[DIM*j+2]);
			sp.vel = Vec3(m_hVel[DIM*j+0], m_hVel[DIM*j+1], m_hVel[DIM*j+2]);
			sp.nrm = Vec3(m_hNrm[DIM*j+0], m_hNrm[DIM*j+1], m_hNrm[DIM*j+2]);
			sp.idx = j;
			sp.d = sqrt(itr->Dist2);
			sps.push_back(sp);
			num++;
		}
	}

	return num;
}



//-----------------------------------------------------------------------------
// 法線計算
//-----------------------------------------------------------------------------
/*!
 * 法線を計算
 *  - 越川純弥，"粒子法による流体シミュレーションの高品質レンダリング"，静岡大学卒業論文，2007.
 *  - p.24, 3.4.3 法線の修正 の 式(3.8) 
 *  - 式(3.8) では Σw(v-r)(v-r) となっているが Σw(v-r) の間違い
 */
void rxSPH::CalNormal(void)
{
	RXREAL h = m_fEffectiveRadius;

	// 近傍粒子探索
	SetParticlesToCell();

	//CalNormalFromDensity();

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0 = Vec3(m_hPos[DIM*i+0], m_hPos[DIM*i+1], m_hPos[DIM*i+2]);
		Vec3 nrm(0.0);

		// 近傍粒子
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1 = Vec3(m_hPos[DIM*j+0], m_hPos[DIM*j+1], m_hPos[DIM*j+2]);
			Vec3 rij = pos0-pos1;

			RXREAL rr = sqrt(itr->Dist2)/h;

			RXREAL w = (rr < 1.0 ? 1.0/rr-1.0 : 0.0);

			nrm += w*rij;
		}

		//normalize(nrm);
		//nrm *= -1;

		m_hNrm[DIM*i+0] = nrm[0];
		m_hNrm[DIM*i+1] = nrm[1];
	}
}

/*!
 * 密度分布から法線を計算
 */
void rxSPH::CalNormalFromDensity(void)
{
	RXREAL h = m_fEffectiveRadius;

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = m_hPos[DIM*i+0];
		pos0[1] = m_hPos[DIM*i+1];
		pos0[2] = m_hPos[DIM*i+2];

		Vec3 nrm(0.0);

		// 近傍粒子
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1;
			pos1[0] = m_hPos[DIM*j+0];
			pos1[1] = m_hPos[DIM*j+1];
			pos1[2] = m_hPos[DIM*j+2];

			Vec3 rij = pos0-pos1;

			RXREAL r = sqrt(itr->Dist2);
			RXREAL q = h*h-r*r;

			nrm += (m_fMass/m_hDens[j])*m_fGWpoly6*q*q*rij;
		}

		//normalize(nrm);
		//nrm *= -1;

		m_hNrm[DIM*i+0] = nrm[0];
		m_hNrm[DIM*i+1] = nrm[1];
	}
}




//-----------------------------------------------------------------------------
// 近傍探索
//-----------------------------------------------------------------------------
/*!
 * 近傍粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[in] prts パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
void rxSPH::GetNearestNeighbors(int idx, RXREAL *prts, vector<rxNeigh> &neighs, RXREAL h)
{
	if(idx < 0 || idx >= (int)m_uNumParticles) return;

	Vec3 pos;
	pos[0] = prts[DIM*idx+0];
	pos[1] = prts[DIM*idx+1];
	pos[2] = prts[DIM*idx+2];

	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGrid->GetNN(pos, prts, m_uNumParticles, neighs, h);
	//m_pNNGrid->GetNN_Direct(pos, prts, m_uNumParticles, neighs, h);	// グリッドを使わない総当たり
}

/*!
 * 近傍粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
void rxSPH::GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h)
{
	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGrid->GetNN(pos, m_hPos, m_uNumParticles, neighs, h);
	//m_pNNGrid->GetNN_Direct(pos, prts, m_uNumParticles, neighs, h);	// グリッドを使わない総当たり
}


/*!
 * 全パーティクルを分割セルに格納
 */
void rxSPH::SetParticlesToCell(RXREAL *prts, int n, RXREAL h)
{
	// 分割セルに粒子を登録
	m_pNNGrid->SetObjectToCell(prts, n);

	// 近傍粒子探索
	if(h < 0.0) h = m_fEffectiveRadius;
	for(int i = 0; i < (int)m_uNumParticles; i++){
		m_vNeighs[i].clear();
		GetNearestNeighbors(i, prts, m_vNeighs[i], h);
	}
}
void rxSPH::SetParticlesToCell(void)
{
	SetParticlesToCell(m_hPos, m_uNumParticles, m_fEffectiveRadius);
}


/*!
 * 分割セルに格納されたポリゴン情報を取得
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] polys ポリゴン
 * @return 格納ポリゴン数
 */
int rxSPH::GetPolygonsInCell(int gi, int gj, int gk, vector<int> &polys)
{
	return m_pNNGrid->GetPolygonsInCell(gi, gj, gk, polys);
}
/*!
 * 分割セル内のポリゴンの有無を調べる
 * @param[in] gi,gj,gk 対象分割セル
 * @return ポリゴンが格納されていればtrue
 */
bool rxSPH::IsPolygonsInCell(int gi, int gj, int gk)
{
	return m_pNNGrid->IsPolygonsInCell(gi, gj, gk);
}


/*!
 * ポリゴンを分割セルに格納
 */
void rxSPH::SetPolygonsToCell(void)
{
	m_pNNGrid->SetPolygonsToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);
}



/*!
 * 探索用セルの描画
 * @param[in] i,j,k グリッド上のインデックス
 */
void rxSPH::DrawCell(int i, int j, int k)
{
	if(m_pNNGrid) m_pNNGrid->DrawCell(i, j, k);
}

/*!
 * 探索用グリッドの描画
 * @param[in] col パーティクルが含まれるセルの色
 * @param[in] col2 ポリゴンが含まれるセルの色
 * @param[in] sel ランダムに選択されたセルのみ描画(1で新しいセルを選択，2ですでに選択されているセルを描画，0ですべてのセルを描画)
 */
void rxSPH::DrawCells(Vec3 col, Vec3 col2, int sel)
{
	if(m_pNNGrid) m_pNNGrid->DrawCells(col, col2, sel, m_hPos);

}

/*!
 * 固体障害物の描画
 */
void rxSPH::DrawObstacles(void)
{
	for(vector<rxSolid*>::iterator i = m_vSolids.begin(); i != m_vSolids.end(); ++i){
		(*i)->Draw();
	}
}




/*!
 * 三角形ポリゴンによる障害物
 * @param[in] vrts 頂点
 * @param[in] tris メッシュ
 */
void rxSPH::SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris)
{
	int vn = (int)vrts.size();
	int n = (int)tris.size();

	if(m_hVrts) delete [] m_hVrts;
	if(m_hTris) delete [] m_hTris;
	m_hVrts = new RXREAL[vn*3];
	m_hTris = new int[n*3];

	for(int i = 0; i < vn; ++i){
		for(int j = 0; j < 3; ++j){
			m_hVrts[3*i+j] = vrts[i][j];
		}
	}

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < 3; ++j){
			m_hTris[3*i+j] = tris[i][j];
		}
	}

	m_iNumVrts = vn;
	m_iNumTris = n;
	RXCOUT << "the number of triangles : " << m_iNumTris << endl;

	m_pNNGrid->SetPolygonsToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);

	//setPolysToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);
}

/*!
 * ボックス型障害物
 * @param[in] cen ボックス中心座標
 * @param[in] ext ボックスの大きさ(辺の長さの1/2)
 * @param[in] ang ボックスの角度(オイラー角)
 * @param[in] flg 有効/無効フラグ
 */
void rxSPH::SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, int flg)
{
	rxSolidBox *box = new rxSolidBox(cen-ext, cen+ext, 1);
	double m[16];
	EulerToMatrix(m, ang[0], ang[1], ang[2]);
	box->SetMatrix(m);

	m_vSolids.push_back(box);

}

/*!
 * 球型障害物
 * @param[in] cen 球体中心座標
 * @param[in] rad 球体の半径
 * @param[in] flg 有効/無効フラグ
 */
void rxSPH::SetSphereObstacle(Vec3 cen, double rad, int flg)
{
	rxSolidSphere *sphere = new rxSolidSphere(cen, rad, 1);
	m_vSolids.push_back(sphere);
}

/*!
 * 球型障害物を動かす
 * @param[in] b 物体番号
 * @param[in] disp 移動量
 */
void rxSPH::MoveSphereObstacle(int b, Vec3 disp)
{
	if(m_vSolids.empty()) return;
	m_vSolids[b]->SetPosition(m_vSolids[b]->GetPosition()+disp);
}

/*!
 * 球型障害物の位置を取得
 * @param[in] b 物体番号
 */
Vec3 rxSPH::GetSphereObstaclePos(int b)
{
	if(m_vSolids.empty()) return Vec3(0.0);
	return m_vSolids[b]->GetPosition();
}


/*!
 * VBOからホストメモリへデータを転送，取得
 * @param[in] type データの種類
 * @return ホストメモリ上のデータ
 */
RXREAL* rxSPH::GetArrayVBO(rxParticleArray type, bool d2h)
{
	assert(m_bInitialized);
 
	RXREAL* hdata = 0;

	unsigned int vbo = 0;

	switch(type){
	default:
	case RX_POSITION:
		hdata = m_hPos;
		vbo = m_posVBO;
		break;

	case RX_VELOCITY:
		hdata = m_hVel;
		break;

	case RX_DENSITY:
		hdata = m_hDens;
		break;

	case RX_PRESSURE:
		hdata = m_hPres;
		break;

	case RX_NORMAL:
		hdata = m_hNrm;
		break;

	case RX_FORCE:
		hdata = m_hFrc;
		break;

	case RX_UPDATED_POSITION:
		hdata = m_hUpPos;
		break;

	case RX_EIGEN_VALUE:
		hdata = m_hEigen;
		break;

	case RX_ROTATION_MATRIX:
		hdata = m_hRMatrix;
		break;

	case RX_TRANSFORMATION:
		hdata = m_hG;
		break;
	}

	return hdata;
}

/*!
 * ホストメモリからVBOメモリへデータを転送
 * @param[in] type データの種類
 * @param[in] data ホストメモリ上のデータ
 * @param[in] start データの開始インデックス
 * @param[in] count 追加数
 */
void rxSPH::SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count)
{
	assert(m_bInitialized);
 
	switch(type){
	default:
	case RX_POSITION:
		{
			if(m_bUseOpenGL){
				glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
				glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(RXREAL), count*4*sizeof(RXREAL), data);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}
		}
		break;

	case RX_VELOCITY:
		//CuCopyArrayToDevice(m_dVel, data, start*DIM*sizeof(RXREAL), count*DIM*sizeof(RXREAL));
		break;

	case RX_NORMAL:
		//CuCopyArrayToDevice(m_dNrm, data, start*DIM*sizeof(RXREAL), count*DIM*sizeof(RXREAL));
		break;
	}	   
}




//-----------------------------------------------------------------------------
// MARK:陰関数値
//-----------------------------------------------------------------------------
double rxSPH::GetImplicit(double x, double y, double z)
{
	return CalColorField(x, y, z);
}

/*!
 * パーティクルからグリッドの陰関数値を計算
 * @param[in] n グリッド数
 * @param[in] minp グリッドの最小座標
 * @param[in] d グリッド幅
 * @param[out] hF 陰関数値(nx×ny×nzの配列)
 */
void rxSPH::CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF)
{
	int slice0 = n[0];
	int slice1 = n[0]*n[1];

	for(int k = 0; k < n[2]; ++k){
		for(int j = 0; j < n[1]; ++j){
			for(int i = 0; i < n[0]; ++i){
				int idx = k*slice1+j*slice0+i;
				Vec3 pos = minp+Vec3(i, j, k)*d;
				hF[idx] = GetImplicit(pos[0], pos[1], pos[2]);
			}
		}
	}
}

/*!
 * パーティクルからグリッドの陰関数値を計算
 * @param[in] pnx,pny,pnz グリッド数の指数 nx=2^pnx
 * @param[in] minp グリッドの最小座標
 * @param[in] d グリッド幅
 * @param[out] hF 陰関数値(nx×ny×nzの配列)
 */
void rxSPH::CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF)
{
	RXREAL *hF = new RXREAL[n[0]*n[1]*n[2]];

	CalImplicitField(n, minp, d, hF);
	CuCopyArrayToDevice(dF, hF, 0, n[0]*n[1]*n[2]*sizeof(RXREAL));

	delete [] hF;
}


/*!
 * カラーフィールド値計算
 * @param[in] pos 計算位置
 * @return カラーフィールド値
 */
double rxSPH::CalColorField(double x, double y, double z)
{
	// MRK:CalColorField
	RXREAL c = 0.0;
	Vec3 pos(x, y, z);

	if(pos[0] < m_v3EnvMin[0]) return c;
	if(pos[0] > m_v3EnvMax[0]) return c;
	if(pos[1] < m_v3EnvMin[1]) return c;
	if(pos[1] > m_v3EnvMax[1]) return c;
	if(pos[2] < m_v3EnvMin[2]) return c;
	if(pos[2] > m_v3EnvMax[2]) return c;

	RXREAL h = m_fEffectiveRadius;

	vector<rxNeigh> ne;
	m_pNNGrid->GetNN(pos, m_hPos, m_uNumParticles, ne, h);

	// 近傍粒子
	for(vector<rxNeigh>::iterator itr = ne.begin(); itr != ne.end(); ++itr){
		int j = itr->Idx;
		if(j < 0) continue;

		Vec3 pos1;
		pos1[0] = m_hPos[DIM*j+0];
		pos1[1] = m_hPos[DIM*j+1];
		pos1[2] = m_hPos[DIM*j+2];

		RXREAL r = sqrt(itr->Dist2);

		RXREAL q = h*h-r*r;
		c += m_fMass*m_fWpoly6*q*q*q;

		//RXREAL q = r/h;
		//if(q <= 1.0 && q >= 0.0){
		//	c += m_fMass*(1.0-q)*(1.0-q);
		//}
	}

	return c;
}


//-----------------------------------------------------------------------------
// 異方性カーネル
//-----------------------------------------------------------------------------
inline float RxPythag(const float a, const float b)
{
	float absa = abs(a), absb = abs(b);
	return (absa > absb ? absa*(float)sqrt((double)(1.0+(absb/absa)*(absb/absa))) :
		   (absb == 0.0 ? 0.0 : absb*(float)sqrt((double)(1.0+(absa/absb)*(absa/absb)))));
}

void RxSVDecomp3(float w[3], float u[9], float v[9], float eps)
{
	bool flag;
	int i, its, j, jj, k, l, nm;
	float anorm, c, f, g, h, s, scale, x, y, z;
	float rv1[3];
	g = scale = anorm = 0.0;
	for(i = 0; i < 3; ++i){
		l = i+2;
		rv1[i] = scale*g;
		g = s = scale = 0.0;
		for(k = i; k < 3; ++k) scale += abs(u[k*3+i]);
		if(scale != 0.0){
			for(k = i; k < 3; ++k){
				u[k*3+i] /= scale;
				s += u[k*3+i]*u[k*3+i];
			}
			f = u[i*3+i];
			g = -RX_SIGN2(sqrt(s), f);
			h = f*g-s;
			u[i*3+i] = f-g;
			for(j = l-1; j < 3; ++j){
				for(s = 0.0, k = i; k < 3; ++k) s += u[k*3+i]*u[k*3+j];
				f = s/h;
				for(k = i; k < 3; ++k) u[k*3+j] += f*u[k*3+i];
			}
			for(k = i; k < 3; ++k) u[k*3+i] *= scale;
		}

		w[i] = scale*g;
		g = s = scale = 0.0;
		if(i+1 <= 3 && i+1 != 3){
			for(k = l-1; k < 3; ++k) scale += abs(u[i*3+k]);
			if(scale != 0.0){
				for(k = l-1; k < 3; ++k){
					u[i*3+k] /= scale;
					s += u[i*3+k]*u[i*3+k];
				}
				f = u[i*3+l-1];
				g = -RX_SIGN2(sqrt(s), f);
				h = f*g-s;
				u[i*3+l-1] = f-g;
				for(k = l-1; k < 3; ++k) rv1[k] = u[i*3+k]/h;
				for(j = l-1; j < 3; ++j){
					for(s = 0.0,k = l-1; k < 3; ++k) s += u[j*3+k]*u[i*3+k];
					for(k = l-1; k < 3; ++k) u[j*3+k] += s*rv1[k];
				}
				for(k = l-1; k < 3; ++k) u[i*3+k] *= scale;
			}
		}
		anorm = RX_MAX(anorm, (abs(w[i])+abs(rv1[i])));
	}
	for(i = 2; i >= 0; --i){
		if(i < 2){
			if(g != 0.0){
				for(j = l; j < 3; ++j){
					v[j*3+i] = (u[i*3+j]/u[i*3+l])/g;
				}
				for(j = l; j < 3; ++j){
					for(s = 0.0, k = l; k < 3; ++k) s += u[i*3+k]*v[k*3+j];
					for(k = l; k < 3; ++k) v[k*3+j] += s*v[k*3+i];
				}
			}
			for(j = l; j < 3; ++j) v[i*3+j] = v[j*3+i] = 0.0;
		}
		v[i*3+i] = 1.0;
		g = rv1[i];
		l = i;
	}
	for(i = 2; i >= 0; --i){
		l = i+1;
		g = w[i];
		for(j = l; j < 3; ++j) u[i*3+j] = 0.0;
		if(g != 0.0){
			g = 1.0/g;
			for(j = l; j < 3; ++j){
				for(s = 0.0, k = l; k < 3; ++k) s += u[k*3+i]*u[k*3+j];
				f = (s/u[i*3+i])*g;
				for(k = i; k < 3; ++k) u[k*3+j] += f*u[k*3+i];
			}
			for(j = i; j < 3; ++j) u[j*3+i] *= g;
		}
		else{
			for(j = i; j < 3; ++j) u[j*3+i] = 0.0;
		}
		++u[i*3+i];
	}
	for(k = 2; k >= 0; --k){
		for(its = 0; its < 30; ++its){
			flag = true;
			for(l = k; l >= 0; --l){
				nm = l-1;
				if(l == 0 || abs(rv1[l]) <= eps*anorm){
					flag = false;
					break;
				}
				if(abs(w[nm]) <= eps*anorm) break;
			}
			if(flag){
				c = 0.0;
				s = 1.0;
				for(i = l; i < k+1; ++i){
					f = s*rv1[i];
					rv1[i] = c*rv1[i];
					if(abs(f) <= eps*anorm) break;
					g = w[i];
					h = RxPythag(f, g);
					w[i] = h;
					h = 1.0/h;
					c = g*h;
					s = -f*h;
					for(j = 0; j < 3; ++j){
						y = u[j*3+nm];
						z = u[j*3+i];
						u[j*3+nm] = y*c+z*s;
						u[j*3+i] = z*c-y*s;
					}
				}
			}
			z = w[k];
			if(l == k){
				if(z < 0.0){
					w[k] = -z;
					for(j = 0; j < 3; ++j) v[j*3+k] = -v[j*3+k];
				}
				break;
			}
			if(its == 29){
				printf("no convergence in 30 svdcmp iterations");
				return;
			}
			x = w[l];
			nm = k-1;
			y = w[nm];
			g = rv1[nm];
			h = rv1[k];
			f = ((y-z)*(y+z)+(g-h)*(g+h))/(2.0*h*y);
			g = RxPythag(f, 1.0f);
			f = ((x-z)*(x+z)+h*((y/(f+RX_SIGN2(g, f)))-h))/x;
			c = s = 1.0;
			for(j = l; j <= nm; ++j){
				i = j+1;
				g = rv1[i];
				y = w[i];
				h = s*g;
				g = c*g;
				z = RxPythag(f, h);
				rv1[j] = z;
				c = f/z;
				s = h/z;
				f = x*c+g*s;
				g = g*c-x*s;
				h = y*s;
				y *= c;
				for(jj = 0; jj < 3; ++jj){
					x = v[jj*3+j];
					z = v[jj*3+i];
					v[jj*3+j] = x*c+z*s;
					v[jj*3+i] = z*c-x*s;
				}
				z = RxPythag(f, h);
				w[j] = z;
				if(z){
					z = 1.0/z;
					c = f*z;
					s = h*z;
				}
				f = c*g+s*y;
				x = c*y-s*g;
				for(jj = 0; jj < 3; ++jj){
					y = u[jj*3+j];
					z = u[jj*3+i];
					u[jj*3+j] = y*c+z*s;
					u[jj*3+i] = z*c-y*s;
				}
			}
			rv1[l] = 0.0;
			rv1[k] = f;
			w[k] = x;
		}
	}

	// reorder
	int inc = 1;
	float sw;
	float su[3], sv[3];

	do{
		inc *= 3;
		inc++; 
	}while(inc <= 3);

	do{
		inc /= 3;
		for(i = inc; i < 3; ++i){
			sw = w[i];
			for(k = 0; k < 3; ++k) su[k] = u[k*3+i];
			for(k = 0; k < 3; ++k) sv[k] = v[k*3+i];
			j = i;
			while (w[j-inc] < sw){
				w[j] = w[j-inc];
				for(k = 0; k < 3; ++k) u[k*3+j] = u[k*3+j-inc];
				for(k = 0; k < 3; ++k) v[k*3+j] = v[k*3+j-inc];
				j -= inc;
				if (j < inc) break;
			}
			w[j] = sw;
			for(k = 0; k < 3; ++k) u[k*3+j] = su[k];
			for(k = 0; k < 3; ++k) v[k*3+j] = sv[k];

		}
	}while(inc > 1);

	for(k = 0; k < 3; ++k){
		s = 0;
		for(i = 0; i < 3; ++i) if(u[i*3+k] < 0.) s++;
		for(j = 0; j < 3; ++j) if(v[j*3+k] < 0.) s++;
		if(s > 3){
			for(i = 0; i < 3; ++i) u[i*3+k] = -u[i*3+k];
			for(j = 0; j < 3; ++j) v[j*3+k] = -v[j*3+k];
		}
	}
}

/*!
 * Anisotropic kernelの計算
 *  - J. Yu and G. Turk, Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels, SCA2010. 
 */
void rxSPH::CalAnisotropicKernel(void)
{
	// MARK:CalAnisotropicKernel
	if(m_uNumParticles == 0) return;

	// VBOからメモリへ
	RXREAL r0 = m_fInitDens;
	RXREAL h0 = m_fEffectiveRadius;
	RXREAL h = 2.0*h0;

	// 2倍の探索半径で近傍探索
	SetParticlesToCell(m_hPos, m_uNumParticles, h);

	RXREAL lambda = 0.9;

	// 更新位置の計算
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = m_hPos[4*i+0];
		pos0[1] = m_hPos[4*i+1];
		pos0[2] = m_hPos[4*i+2];

		// 近傍粒子
		Vec3 posw(0.0);
		RXREAL sumw = 0.0f;
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPos[4*j+0];
			pos1[1] = m_hPos[4*j+1];
			pos1[2] = m_hPos[4*j+2];

			RXREAL r = sqrt(itr->Dist2);
			
			if(r < h){
				RXREAL q = 1-r/h;
				RXREAL wij = q*q*q;
				posw += pos1*wij;
				sumw += wij;
			}
		}

		m_hPosW[DIM*i+0] = posw[0]/sumw;
		m_hPosW[DIM*i+1] = posw[1]/sumw;
		m_hPosW[DIM*i+2] = posw[2]/sumw;
		m_hPosW[DIM*i+3] = 0.0f;

		m_hUpPos[DIM*i+0] = (1-lambda)*m_hPos[DIM*i+0]+lambda*m_hPosW[DIM*i+0];
		m_hUpPos[DIM*i+1] = (1-lambda)*m_hPos[DIM*i+1]+lambda*m_hPosW[DIM*i+1];
		m_hUpPos[DIM*i+2] = (1-lambda)*m_hPos[DIM*i+2]+lambda*m_hPosW[DIM*i+2];

		//m_hUpPos[DIM*i+0] = m_hPos[DIM*i+0];
		//m_hUpPos[DIM*i+1] = m_hPos[DIM*i+1];
		//m_hUpPos[DIM*i+2] = m_hPos[DIM*i+2];
	}

	SetParticlesToCell(m_hUpPos, m_uNumParticles, h);

	// covariance matrixの計算
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 xi;
		xi[0] = m_hUpPos[4*i+0];
		xi[1] = m_hUpPos[4*i+1];
		xi[2] = m_hUpPos[4*i+2];

		Vec3 xiw(0.0);
		RXREAL sumw = 0.0f;
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 xj;
			xj[0] = m_hUpPos[4*j+0];
			xj[1] = m_hUpPos[4*j+1];
			xj[2] = m_hUpPos[4*j+2];

			RXREAL r = sqrt(itr->Dist2);
			
			if(r < h){
				RXREAL q = 1-r/h;
				RXREAL wij = q*q*q;
				xiw += xj*wij;
				sumw += wij;
			}
		}

		xiw /= sumw;

		matrix3x3 c;
		for(int k = 0; k < 3; ++k){
			c.e[k].x = 0.0;
			c.e[k].y = 0.0;
			c.e[k].z = 0.0;
		}

		int n = 0;
		sumw = 0.0f;
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 xj;
			xj[0] = m_hUpPos[4*j+0];
			xj[1] = m_hUpPos[4*j+1];
			xj[2] = m_hUpPos[4*j+2];

			RXREAL r = sqrt(itr->Dist2);

			if(r < h){
				RXREAL q = 1-r/h;
				RXREAL wij = q*q*q;

				Vec3 dxj = xj-xiw;

				for(int k = 0; k < 3; ++k){
					c.e[k].x += wij*dxj[k]*dxj[0];
					c.e[k].y += wij*dxj[k]*dxj[1];
					c.e[k].z += wij*dxj[k]*dxj[2];
				}

				sumw += wij;

				n++;
			}
		}

		for(int k = 0; k < 3; ++k){
			c.e[k].x /= sumw;
			c.e[k].y /= sumw;
			c.e[k].z /= sumw;
		}

		// 特異値分解
		float w[3], u[9], v[9];
		for(int k = 0; k < 3; ++k){
			u[k*3+0] = c.e[k].x;
			u[k*3+1] = c.e[k].y;
			u[k*3+2] = c.e[k].z;
		}

		RxSVDecomp3(w, u, v, 1.0e-10);
		
		// 固有値Σ
		Vec3 sigma;
		for(int j = 0; j < 3; ++j){
			sigma[j] = w[j];
		}

		// 固有ベクトル(回転行列R)
		rxMatrix3 R;
		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				R(k, j) = u[k*3+j];
			}
		}

		// デバッグ用に値を待避
		for(int j = 0; j < 3; ++j){
			m_hEigen[3*i+j] = sigma[j];
		}
		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				m_hRMatrix[9*i+3*j+k] = R(j, k);
			}
		}

		
		int ne = 10;//m_params.KernelParticles*0.8;
		RXREAL ks = 1400;
		RXREAL kn = 0.5;
		RXREAL kr = 4.0;
		if(n > ne){
			for(int j = 1; j < 3; ++j){
				sigma[j] = RX_MAX(sigma[j], sigma[0]/kr);
			}
			sigma *= ks;
		}
		else{
			for(int j = 0; j < 3; ++j){
				sigma[j] = kn*1.0;
			}
		}


		// カーネル変形行列G
		rxMatrix3 G;
		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				RXREAL x = 0;
				for(int l = 0; l < 3; ++l){
					//x += m_hRMatrix[9*i+3*j+l]*m_hRMatrix[9*i+3*k+l]/m_hEigen[3*i+l];
					x += R(j, l)*R(k, l)/sigma[l];
				}
				
				G(j, k) = x;
				//m_hG[9*i+3*j+k] = x;
			}
		}

		double max_diag = -1.0e10;
		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				if(G(j, k) > max_diag) max_diag = G(j, k);
			}
		}

		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				G(j, k) /= max_diag;
			}
		}
				
		G = G.Inverse();


		for(int j = 0; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				m_hG[9*i+3*j+k] = G(j, k);
			}
		}

	}
}



//-----------------------------------------------------------------------------
// MARK:シミュデータの出力
//-----------------------------------------------------------------------------
/*!
 * シミュレーション設定(パーティクル数，範囲，密度，質量など)
 * @param[in] fn 出力ファイル名
 */
void rxSPH::OutputSetting(string fn)
{
	ofstream fout;
	fout.open(fn.c_str());
	if(!fout){
		RXCOUT << fn << " couldn't open." << endl;
		return;
	}

	fout << m_uNumParticles << endl;
	fout << m_v3EnvMin[0] << " " << m_v3EnvMin[1] << " " << m_v3EnvMin[2] << endl;
	fout << m_v3EnvMax[0] << " " << m_v3EnvMax[1] << " " << m_v3EnvMax[2] << endl;
	fout << m_fInitDens << endl;
	fout << m_fMass << endl;
	fout << m_iKernelParticles << endl;

	fout.close();
}

