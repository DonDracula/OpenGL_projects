/*!
  @file rx_sph_pbd.cpp
	
  @brief Position based fluidの実装
 
  @author Makoto Fujisawa
  @date   2013-06
*/

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph.h"

#include "rx_cu_funcs.cuh"
#include "rx_cu_common.cuh"

#include "rx_pcube.h"


extern int g_iIterations;			//!< 修正反復回数
extern double g_fEta;				//!< 密度変動率


//-----------------------------------------------------------------------------
// rxPBDSPHクラスの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 * @param[in] use_opengl VBO使用フラグ
 */
rxPBDSPH::rxPBDSPH(bool use_opengl) : 
	rxParticleSystemBase(use_opengl), 
	m_hFrc(0),
	m_hDens(0), 
	m_hS(0), 
	m_hDp(0), 
	m_hSb(0), 
	m_hPredictPos(0), 
	m_hPredictVel(0), 
	m_hVrts(0), 
	m_hTris(0), 
	m_pBoundary(0)
{
	m_v3Gravity = Vec3(0.0, -9.82, 0.0);

	m_fViscosity = 0.0;

	m_fRestitution = 0.0;

	m_fEpsilon = 0.01;

	m_bArtificialPressure = true;

	// 近傍探索セル
	m_pNNGrid = new rxNNGrid(DIM);
	m_pNNGridB = new rxNNGrid(DIM);

	m_uNumParticles = 0;
	m_uNumBParticles = 0;
	m_iNumTris = 0;

	m_iColorType = RX_RAMP;
}

/*!
 * デストラクタ
 */
rxPBDSPH::~rxPBDSPH()
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
void rxPBDSPH::Initialize(const rxEnviroment &env)
{
	RXCOUT << "[rxPBDSPH::Initialize]" << endl;

	m_fRestDens        = env.dens;
	m_fMass            = env.mass;
	m_iKernelParticles = env.kernel_particles;

	RXREAL volume = env.max_particles*m_fMass/m_fRestDens;

	m_fEffectiveRadius = pow(((3.0*m_iKernelParticles*volume)/(4.0*env.max_particles*RX_PI)), 1.0/3.0);
	m_fParticleRadius = pow((RX_PI/(6.0*m_iKernelParticles)), 1.0/3.0)*m_fEffectiveRadius;

	m_fKernelRadius = m_fEffectiveRadius;
	m_fViscosity = env.viscosity;

	RXREAL h = m_fEffectiveRadius;
	RXREAL r = m_fParticleRadius;

	// カーネル関数の定数
	m_fAw = KernelCoefPoly6(h, 3, 1);
	m_fAg = KernelCoefSpiky(h, 3, 2);
	m_fAl = KernelCoefVisc(h, 3, 3);

	// カーネル関数
	m_fpW  = KernelPoly6;
	m_fpGW = KernelSpikyG<Vec3>;
	m_fpLW = KernelViscL;

	// 初期密度の計算
	m_fRestDens = calRestDensity(h);

	RXCOUT << "particle : " << endl;
	RXCOUT << " n_max = " << env.max_particles << endl;
	RXCOUT << " h = " << m_fEffectiveRadius << endl;
	RXCOUT << " r = " << m_fParticleRadius << endl;
	RXCOUT << " dens = " << m_fRestDens << endl;
	RXCOUT << " mass = " << m_fMass << endl;
	RXCOUT << " kernel_particles = " << m_iKernelParticles << endl;
	RXCOUT << " volume = " << volume << endl << endl;
	RXCOUT << " viscosity = " << m_fViscosity << endl;

	// PBD用パラメータ
	m_fEpsilon = env.epsilon;
	m_fEta = env.eta;
	m_iMinIterations = env.min_iter;
	m_iMaxIterations = env.max_iter;

	// 人工圧力のための係数
	m_bArtificialPressure = (env.use_ap ? true : false);
	m_fApK = env.ap_k;
	m_fApN = env.ap_n;
	m_fApQ = env.ap_q;
	RXREAL dq = m_fApQ*h;
	RXREAL wq = m_fpW(dq, h, m_fAw);

	RXCOUT << " epsilon = " << m_fEpsilon << endl;
	RXCOUT << " eta = " << m_fEta << endl;
	RXCOUT << " min_iter = " << m_iMinIterations << endl;
	RXCOUT << " max_iter = " << m_iMaxIterations << endl;
	RXCOUT << " artificial pressure : " << (m_bArtificialPressure ? "on" : "off") << endl;
	RXCOUT << "  k = " << m_fApK << endl;
	RXCOUT << "  n = " << m_fApN << endl;
	RXCOUT << "  dq = " << m_fApQ << endl;
	RXCOUT << "  wq = " << wq << endl;

	//
	// 境界設定
	//
	m_pBoundary = new rxSolidBox(env.boundary_cen-env.boundary_ext, env.boundary_cen+env.boundary_ext, -1);
	//m_pBoundary = new rxSolidSphere(Vec3(0.0, 0.1, 0.0), 0.25, -1);

	// 境界パーティクル生成
	m_uNumBParticles = m_pBoundary->GenerateParticlesOnSurf(0.75*m_fParticleRadius, &m_hPosB);

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

	m_uNumParticles = 0;

	Allocate(env.max_particles);
}

/*!
 * メモリの確保
 *  - 最大パーティクル数で確保
 * @param[in] max_particles 最大パーティクル数
 */
void rxPBDSPH::Allocate(int max_particles)
{
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
	m_hFrc = new RXREAL[size];
	memset(m_hPos, 0, mem_size);
	memset(m_hVel, 0, mem_size);
	memset(m_hFrc, 0, mem_size);

	m_hDens = new RXREAL[size1];
	memset(m_hDens, 0, mem_size1);

	m_hS = new RXREAL[size1];
	memset(m_hS, 0, mem_size1);

	m_hDp = new RXREAL[size];
	memset(m_hDp, 0, mem_size);

	m_hPredictPos = new RXREAL[size];
	memset(m_hPredictPos, 0, mem_size);
	m_hPredictVel = new RXREAL[size];
	memset(m_hPredictVel, 0, mem_size);

	m_hTmp = new RXREAL[m_uMaxParticles];
	memset(m_hTmp, 0, sizeof(RXREAL)*m_uMaxParticles);

	m_vNeighs.resize(m_uMaxParticles);

	if(m_bUseOpenGL){
		m_posVBO = createVBO(mem_size);	
		m_colorVBO = createVBO(m_uMaxParticles*DIM*sizeof(RXREAL));

		SetColorVBO(RX_RAMP, -1);
	}

	// 分割セル設定
	m_pNNGrid->Setup(m_v3EnvMin, m_v3EnvMax, m_fEffectiveRadius, m_uMaxParticles);
	m_vNeighs.resize(m_uMaxParticles);

	if(m_uNumBParticles){
		Vec3 minp = m_pBoundary->GetMin()-Vec3(4.0*m_fParticleRadius);
		Vec3 maxp = m_pBoundary->GetMax()+Vec3(4.0*m_fParticleRadius);
		m_pNNGridB->Setup(minp, maxp, m_fEffectiveRadius, m_uNumBParticles);

		// 分割セルに粒子を登録
		m_pNNGridB->SetObjectToCell(m_hPosB, m_uNumBParticles);

		// 境界パーティクルの体積
		m_hVolB = new RXREAL[m_uNumBParticles];
		memset(m_hVolB, 0, sizeof(RXREAL)*m_uNumBParticles);
		calBoundaryVolumes(m_hPosB, m_hVolB, m_fMass, m_uNumBParticles, m_fEffectiveRadius);

		// 境界パーティクルのスケーリングファクタ
		m_hSb = new RXREAL[m_uNumBParticles];
		memset(m_hSb, 0, sizeof(RXREAL)*m_uNumBParticles);

		//Dump<RXREAL>("_volb.txt", m_hVolB, m_uNumBParticles, 1);
	}

	m_bInitialized = true;
}

/*!
 * 確保したメモリの解放
 */
void rxPBDSPH::Finalize(void)
{
	assert(m_bInitialized);

	// メモリ解放
	if(m_hPos) delete [] m_hPos;
	if(m_hVel) delete [] m_hVel;
	if(m_hFrc) delete [] m_hFrc;

	if(m_hDens) delete [] m_hDens;

	if(m_hPosB) delete [] m_hPosB;
	if(m_hVolB) delete [] m_hVolB;

	if(m_hS)  delete [] m_hS;
	if(m_hDp) delete [] m_hDp;

	if(m_hSb) delete [] m_hSb;

	if(m_hPredictPos) delete [] m_hPredictPos;
	if(m_hPredictVel) delete [] m_hPredictVel;

	if(m_hTmp) delete [] m_hTmp;

	m_vNeighs.clear();
	m_vNeighsB.clear();

	if(m_bUseOpenGL){
		glDeleteBuffers(1, (const GLuint*)&m_posVBO);
		glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
	}

	if(m_pNNGrid) delete m_pNNGrid;
	m_vNeighs.clear();

	if(m_pNNGridB) delete m_pNNGridB;

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
bool rxPBDSPH::Update(RXREAL dt, int step)
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
	RXREAL h = m_fEffectiveRadius;
	static bool init = true;

	// 近傍粒子探索用セルに粒子を格納
	SetParticlesToCell();

	// 密度計算
	calDensity(m_hPos, m_hDens, h);

	// 外力項,粘性項による力場の計算
	calForceExtAndVisc(m_hPos, m_hVel, m_hDens, m_hFrc, h);

	// 予測位置，速度の計算
	integrate(m_hPos, m_hVel, m_hDens, m_hFrc, m_hPredictPos, m_hPredictVel, dt);

	RXTIMER("force calculation");

	// 反復
	int iter = 0;	// 反復回数
	RXREAL dens_var = 1.0;	// 密度の分散
	while(((dens_var > m_fEta) || (iter < m_iMinIterations)) && (iter < m_iMaxIterations)){
		// 近傍粒子探索用セルに粒子を格納
		SetParticlesToCell(m_hPredictPos, m_uNumParticles, m_fKernelRadius);

		// 拘束条件Cとスケーリングファクタsの計算
		calScalingFactor(m_hPredictPos, m_hDens, m_hS, h, dt);

		// 位置修正量Δpの計算
		calPositionCorrection(m_hPredictPos, m_hS, m_hDp, h, dt);	

		// パーティクル位置修正
		for(uint i = 0; i < m_uNumParticles; ++i){
			for(int k = 0; k < DIM; ++k){
				m_hPredictPos[DIM*i+k] += m_hDp[DIM*i+k];
				m_hPredictVel[DIM*i+k] = 0.0; // integrateで衝突処理だけにするために0で初期化
				m_hFrc[DIM*i+k] = 0.0;
			}
		}

		// 衝突処理
		integrate2(m_hPos, m_hVel, m_hDens, m_hFrc, m_hPredictPos, m_hPredictVel, dt);

		// 平均密度変動の計算
		dens_var = 0.0;
		for(uint i = 0; i < m_uNumParticles; ++i){
			RXREAL err = fabs(m_hDens[i]-m_fRestDens)/m_fRestDens;
			//if(err > dens_var) dens_var = err;
			dens_var += err;
		}
		dens_var /= (double)m_uNumParticles;

		if(dens_var <= m_fEta && iter > m_iMinIterations) break;

		iter++;
	}

	g_iIterations = iter;
	g_fEta = dens_var;

	// 速度・位置更新
	for(uint i = 0; i < m_uNumParticles; ++i){
		for(int k = 0; k < DIM; ++k){
			int idx = DIM*i+k;
			m_hVel[idx] = (m_hPredictPos[idx]-m_hPos[idx])/dt;
			m_hPos[idx] = m_hPredictPos[idx];
		}
	}

	RXTIMER("update position");


	SetArrayVBO(RX_POSITION, m_hPos, 0, m_uNumParticles);

	SetColorVBO(m_iColorType, -1);

	RXTIMER("color(vbo)");

	init = false;
	return true;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxPBDSPH::GetParticle(void)
{
	return m_hPos;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxPBDSPH::GetParticleDevice(void)
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
void rxPBDSPH::SetColorVBO(int type, int picked)
{
	switch(type){
	case RX_DENSITY:
		SetColorVBOFromArray(m_hDens, 1, false, m_fRestDens);
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
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[in] h 有効半径
 */
void rxPBDSPH::calDensity(const RXREAL *ppos, RXREAL *pdens, RXREAL h)
{
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = ppos[DIM*i+0];
		pos0[1] = ppos[DIM*i+1];
		pos0[2] = ppos[DIM*i+2];

		pdens[i] = 0.0;

		// 近傍粒子から密度を計算
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*j+0];
			pos1[1] = ppos[DIM*j+1];
			pos1[2] = ppos[DIM*j+2];

			RXREAL r = sqrt(itr->Dist2);

			// Poly6カーネルで密度を計算 (rho = Σ m Wij)
			pdens[i] += m_fMass*m_fpW(r, h, m_fAw);
		}

		// 近傍境界粒子探索
		vector<rxNeigh> neigh;
		GetNearestNeighborsB(pos0, neigh, h);

		// 境界粒子の密度への影響を計算([Akinci et al.,SIG2012]の式(6)の右辺第二項)
		RXREAL brho = 0.0;
		for(vector<rxNeigh>::iterator itr = neigh.begin() ; itr != neigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*j+0];
			pos1[1] = m_hPosB[DIM*j+1];
			pos1[2] = m_hPosB[DIM*j+2];

			RXREAL r = norm(pos1-pos0);

			// 流体の場合と違って境界の仮想体積と初期密度から複数層境界粒子があった場合の仮想質量Φ=ρ0*Vbを求めて使う 
			brho += m_fRestDens*m_hVolB[j]*m_fpW(r, h, m_fAw);
		}
		pdens[i] += brho;
	}
}

/*!
 * 圧力項，粘性項の計算
 *  - 粘性項への境界パーティクルの影響は考慮していない(slip境界条件)
 * @param[in] ppos パーティクル中心座標
 * @param[in] pvel パーティクル速度
 * @param[in] pdens パーティクル密度
 * @param[out] pfrc パーティクルにかかる力
 * @param[in] h 有効半径
 */
void rxPBDSPH::calForceExtAndVisc(const RXREAL *ppos, const RXREAL *pvel, const RXREAL *pdens, RXREAL *pfrc, RXREAL h)
{
	Vec3 rij, vji;
	RXREAL r0 = m_fRestDens;
	
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0, vel0;
		pos0 = Vec3(ppos[4*i+0], ppos[4*i+1], ppos[4*i+2]);
		vel0 = Vec3(pvel[4*i+0], pvel[4*i+1], pvel[4*i+2]);

		Vec3 Fev(0.0);
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1, vel1;
			pos1 = Vec3(ppos[4*j+0], ppos[4*j+1], ppos[4*j+2]);
			vel1 = Vec3(pvel[4*j+0], pvel[4*j+1], pvel[4*j+2]);

			rij = pos0-pos1;
			vji = vel1-vel0;

			RXREAL r = norm(rij);//sqrt(itr->Dist2);

			// 粘性
			Fev += m_fMass*(vji/m_hDens[i])*m_fpLW(r, h, m_fAl, 3);
		}

		Vec3 force(0.0);
		force += m_fViscosity*Fev;

		// 外力項
		force += m_v3Gravity;

		pfrc[4*i+0] += force[0];
		pfrc[4*i+1] += force[1];
		pfrc[4*i+2] += force[2];
	}
}


/*!
 * スケーリングファクタの計算
 * @param[in] ppos パーティクル中心座標
 * @param[out] pdens パーティクル密度
 * @param[out] pscl スケーリングファクタ
 * @param[in] h 有効半径
 * @param[in] dt 時間ステップ幅
 */
void rxPBDSPH::calScalingFactor(const RXREAL *ppos, RXREAL *pdens, RXREAL *pscl, RXREAL h, RXREAL dt)
{
	RXREAL r0 = m_fRestDens;

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = ppos[DIM*i+0];
		pos0[1] = ppos[DIM*i+1];
		pos0[2] = ppos[DIM*i+2];

		pdens[i] = 0.0;

		// 近傍粒子から密度を計算
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*j+0];
			pos1[1] = ppos[DIM*j+1];
			pos1[2] = ppos[DIM*j+2];

			Vec3 rij = pos0-pos1;
			RXREAL r = norm(rij);

			// Poly6カーネルで密度を計算 (rho = Σ m Wij)
			pdens[i] += m_fMass*m_fpW(r, h, m_fAw);
		}

		// 近傍境界粒子探索
		vector<rxNeigh> neigh;
		GetNearestNeighborsB(pos0, neigh, h);

		// 境界粒子の密度への影響を計算([Akinci et al.,SIG2012]の式(6)の右辺第二項)
		RXREAL brho = 0.0;
		for(vector<rxNeigh>::iterator itr = neigh.begin() ; itr != neigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*j+0];
			pos1[1] = m_hPosB[DIM*j+1];
			pos1[2] = m_hPosB[DIM*j+2];

			RXREAL r = norm(pos1-pos0);

			// 流体の場合と違って境界の仮想体積と初期密度から複数層境界粒子があった場合の仮想質量Φ=ρ0*Vbを求めて使う 
			brho += m_fRestDens*m_hVolB[j]*m_fpW(r, h, m_fAw);
		}
		pdens[i] += brho;

		// 密度拘束条件(式(1))
		RXREAL C = pdens[i]/r0-1;

		// スケーリングファクタの分母項計算
		RXREAL sd = 0.0;
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int k = itr->Idx;
			if(k < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*k+0];
			pos1[1] = ppos[DIM*k+1];
			pos1[2] = ppos[DIM*k+2];

			Vec3 rik = pos0-pos1;
			RXREAL r = norm(rik);
			
			// k == i とその他で処理を分ける(式(8))
			Vec3 dp(0.0);
			if(k == i){
				for(vector<rxNeigh>::iterator jtr = m_vNeighs[k].begin() ; jtr != m_vNeighs[k].end(); ++jtr){
					int j = jtr->Idx;
					if(j < 0) continue;

					Vec3 pos2;
					pos2[0] = ppos[DIM*j+0];
					pos2[1] = ppos[DIM*j+1];
					pos2[2] = ppos[DIM*j+2];

					Vec3 rij = pos0-pos2;
					RXREAL ri = norm(rij);

					dp += m_fpGW(ri, h, m_fAg, rij)/r0;
				}
			}
			else{
				dp = -m_fpGW(r, h, m_fAg, rik)/r0;
			}

			sd += norm2(dp);
		}

		// 境界粒子のスケーリングファクタへの影響
		Vec3 dpb(0.0);
		for(vector<rxNeigh>::iterator itr = neigh.begin() ; itr != neigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*j+0];
			pos1[1] = m_hPosB[DIM*j+1];
			pos1[2] = m_hPosB[DIM*j+2];

			Vec3 rij = pos0-pos1;
			RXREAL r = norm(rij);

			// 仮想質量Φ=ρ0*Vbと流体質量の比で単層しかない境界粒子の影響を制御
			dpb = (m_fRestDens*m_hVolB[j]/m_fMass)*m_fpGW(r, h, m_fAg, rij)/r0;

			sd += norm2(dpb);
		}

		// スケーリングファクタの計算(式(11))
		pscl[i] = -C/(sd+m_fEpsilon);
	}

	// 境界粒子のスケーリングファクター(流体粒子の変位量を計算するときに使う)
	for(uint i = 0; i < m_uNumBParticles; ++i){
		Vec3 pos0;
		pos0[0] = m_hPosB[DIM*i+0];
		pos0[1] = m_hPosB[DIM*i+1];
		pos0[2] = m_hPosB[DIM*i+2];

		RXREAL brho = 0.0;
		vector<rxNeigh> fneigh, bneigh;

		// 近傍粒子探索(流体パーティクル)
		GetNearestNeighbors(pos0, fneigh, h);

		// 近傍粒子探索(境界パーティクル)
		GetNearestNeighborsB(pos0, bneigh, h);

		// 近傍粒子から密度を計算
		for(vector<rxNeigh>::iterator itr = fneigh.begin() ; itr != fneigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*j+0];
			pos1[1] = ppos[DIM*j+1];
			pos1[2] = ppos[DIM*j+2];

			RXREAL r = norm(pos1-pos0);

			// Poly6カーネルで密度を計算 (rho = Σ m Wij)
			brho += m_fMass*m_fpW(r, h, m_fAw);
		}
		for(vector<rxNeigh>::iterator itr = bneigh.begin() ; itr != bneigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*j+0];
			pos1[1] = m_hPosB[DIM*j+1];
			pos1[2] = m_hPosB[DIM*j+2];

			RXREAL r = norm(pos1-pos0);

			brho += m_fRestDens*m_hVolB[j]*m_fpW(r, h, m_fAw);
		}

		// 密度拘束条件(式(1))
		RXREAL C = brho/r0-1;

		// スケーリングファクタの分母項計算
		RXREAL sd = 0.0;
		for(vector<rxNeigh>::iterator itr = fneigh.begin() ; itr != fneigh.end(); ++itr){
			int k = itr->Idx;
			if(k < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*k+0];
			pos1[1] = ppos[DIM*k+1];
			pos1[2] = ppos[DIM*k+2];

			Vec3 rik = pos0-pos1;
			RXREAL r = norm(rik);

			// 近傍流体粒子の場合は k == i となり得ないので場合分けの必要なし
			Vec3 dp = m_fpGW(r, h, m_fAg, rik)/r0;

			sd += norm2(dp);
		}

		for(vector<rxNeigh>::iterator itr = bneigh.begin() ; itr != bneigh.end(); ++itr){
			int k = itr->Idx;
			if(k < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*k+0];
			pos1[1] = m_hPosB[DIM*k+1];
			pos1[2] = m_hPosB[DIM*k+2];

			Vec3 rik = pos0-pos1;
			RXREAL r = norm(rik);

			// k == i とその他で処理を分ける(式(8))
			Vec3 dp(0.0);
			if(k == i){
				for(vector<rxNeigh>::iterator jtr = m_vNeighsB[k].begin() ; jtr != m_vNeighsB[k].end(); ++jtr){
					int j = jtr->Idx;
					if(j < 0) continue;

					Vec3 pos2;
					pos2[0] = ppos[DIM*j+0];
					pos2[1] = ppos[DIM*j+1];
					pos2[2] = ppos[DIM*j+2];

					Vec3 rij = pos0-pos2;
					RXREAL ri = norm(rij);

					dp += (m_fRestDens*m_hVolB[k]/m_fMass)*m_fpGW(ri, h, m_fAg, rij)/r0;
				}
			}
			else{
				dp = -(m_fRestDens*m_hVolB[k]/m_fMass)*m_fpGW(r, h, m_fAg, rik)/r0;
			}

			sd += norm2(dp);
		}

		// スケーリングファクタの計算(式(11))
		m_hSb[i] = -C/(sd+m_fEpsilon);
	}
}


/*!
 * スケーリングファクタによるパーティクル位置修正量の計算
 * @param[in] ppos パーティクル中心座標
 * @param[in] pscl スケーリングファクタ
 * @param[out] pdp パーティクル位置修正量
 * @param[in] h 有効半径
 * @param[in] dt 時間ステップ幅
 */
void rxPBDSPH::calPositionCorrection(const RXREAL *ppos, const RXREAL *pscl, RXREAL *pdp, RXREAL h, RXREAL dt)
{
	RXREAL r0 = m_fRestDens;

	// 人工圧力用パラメータ
	RXREAL k = m_fApK;
	RXREAL n = m_fApN;
	RXREAL dq = m_fApQ*h;
	RXREAL wq = m_fpW(dq, h, m_fAw);

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0;
		pos0[0] = ppos[DIM*i+0];
		pos0[1] = ppos[DIM*i+1];
		pos0[2] = ppos[DIM*i+2];

		pdp[DIM*i+0] = 0.0;
		pdp[DIM*i+1] = 0.0;
		pdp[DIM*i+2] = 0.0;

		// 近傍粒子から位置修正量を計算
		Vec3 dpij(0.0);
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = ppos[DIM*j+0];
			pos1[1] = ppos[DIM*j+1];
			pos1[2] = ppos[DIM*j+2];

			Vec3 rij = pos0-pos1;
			RXREAL r = norm(rij);
			
			if(r > h) continue;

			RXREAL scorr = 0.0;

			if(m_bArtificialPressure){
				// クラスタリングを防ぐためのスケーリングファクタ
				RXREAL ww = m_fpW(r, h, m_fAw)/wq;

				// [Macklin&Muller2003]だとdt*dtはないが，
				// [Monaghan2000]にあるようにk*(ww/wq)^nは加速度[m/s^2]となるので，
				// 座標値[m]にするためにdt^2を掛けている
				scorr = -k*pow(ww, n)*dt*dt; 
			}

			// Spikyカーネルで位置修正量を計算
			dpij += (pscl[i]+pscl[j]+scorr)*m_fpGW(r, h, m_fAg, rij)/r0;
		}

		// 境界パーティクルの影響による位置修正
		// 近傍粒子探索
		vector<rxNeigh> bneigh;
		GetNearestNeighborsB(pos0, bneigh, h);

		Vec3 dpbij(0.0);
		for(vector<rxNeigh>::iterator itr = bneigh.begin() ; itr != bneigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = m_hPosB[DIM*j+0];
			pos1[1] = m_hPosB[DIM*j+1];
			pos1[2] = m_hPosB[DIM*j+2];

			Vec3 rij = pos0-pos1;
			RXREAL r = norm(rij);

			if(r > h) continue;

			RXREAL scorr = 0.0;

			if(m_bArtificialPressure){
				// クラスタリングを防ぐためのスケーリングファクタ
				RXREAL ww = (m_fRestDens*m_hVolB[j]/m_fMass)*m_fpW(r, h, m_fAw)/wq;

				// [Macklin&Muller2003]だとdt*dtはないが，
				// [Monaghan2000]にあるようにk*(ww/wq)^nは加速度[m/s^2]となるので，
				// 座標値[m]にするためにdt^2を掛けている
				scorr = -k*pow(ww, n)*dt*dt; 
			}

			// Spikyカーネルで位置修正量を計算
			dpbij += (pscl[i]+m_hSb[j]+scorr)*m_fpGW(r, h, m_fAg, rij)/r0;
		}

		dpij += dpbij;

		pdp[DIM*i+0] = dpij[0];
		pdp[DIM*i+1] = dpij[1];
		pdp[DIM*i+2] = dpij[2];
	}
}


/*!
 * rest densityの計算
 *  - 近傍にパーティクルが敷き詰められているとして密度を計算する
 * @param[in] h 有効半径
 * @return rest density
 */
RXREAL rxPBDSPH::calRestDensity(RXREAL h)
{
	RXREAL r0 = 0.0;
	RXREAL l = 2*GetParticleRadius();
	int n = (int)ceil(m_fKernelRadius/l)+1;
	for(int x = -n; x <= n; ++x){
		for(int y = -n; y <= n; ++y){
			for(int z = -n; z <= n; ++z){
				Vec3 rij = Vec3(x*l, y*l, z*l);
				r0 += m_fMass*m_fpW(norm(rij), h, m_fAw);
			}
		}
	}
	return r0;
}


/*!
 * 境界パーティクルの体積を計算
 *  - "Versatile Rigid-Fluid Coupling for Incompressible SPH", 2.2 式(3)の上
 * @param[in] bpos 境界パーティクルの位置
 * @param[out] bvol 境界パーティクルの体積
 * @param[in] mass パーティクル質量
 * @param[in] n 境界パーティクル数
 * @param[in] h 有効半径
 */
void rxPBDSPH::calBoundaryVolumes(const RXREAL *bpos, RXREAL *bvol, RXREAL mass, uint n, RXREAL h)
{
	m_vNeighsB.resize(n);
	for(uint i = 0; i < n; ++i){
		Vec3 pos0;
		pos0[0] = bpos[DIM*i+0];
		pos0[1] = bpos[DIM*i+1];
		pos0[2] = bpos[DIM*i+2];

		// 近傍粒子
		vector<rxNeigh> neigh;
		GetNearestNeighborsB(pos0, neigh, h);
		m_vNeighsB[i] = neigh;

		RXREAL mw = 0.0;
		for(vector<rxNeigh>::iterator itr = neigh.begin() ; itr != neigh.end(); ++itr){
			int j = itr->Idx;
			if(j < 0) continue;

			Vec3 pos1;
			pos1[0] = bpos[DIM*j+0];
			pos1[1] = bpos[DIM*j+1];
			pos1[2] = bpos[DIM*j+2];

			RXREAL r = norm(pos1-pos0);

			mw += mass*m_fpW(r, h, m_fAw);
		}

		bvol[i] = mass/mw;
	}
}

/*!
 * 時間ステップ幅の修正
 *  - Ihmsen et al., "Boundary Handling and Adaptive Time-stepping for PCISPH", Proc. VRIPHYS, pp.79-88, 2010.
 * @param[in] dt 現在のタイプステップ
 * @param[in] eta_avg 密度変動の平均(ユーザ指定)
 * @param[in] pfrc 圧力項による力場(実際には加速度)
 * @param[in] pvel パーティクル速度
 * @param[in] pdens パーティクル密度
 * @return 修正されたタイプステップ幅
 */
RXREAL rxPBDSPH::calTimeStep(RXREAL &dt, RXREAL eta_avg, const RXREAL *pfrc, const RXREAL *pvel, const RXREAL *pdens)
{
	RXREAL h = m_fEffectiveRadius;
	RXREAL r0 = m_fRestDens;
	RXREAL new_dt = dt;

	// 最大力，最大速度，密度偏差の平均と最大を算出
	RXREAL ft_max = 0.0;
	RXREAL vt_max = 0.0;
	RXREAL rerr_max = 0.0, rerr_avg = 0.0;
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 f, v;
		for(int k = 0; k < 3; ++k){
			f[k] = pfrc[DIM*i+k];
			v[k] = pvel[DIM*i+k];
		}
		RXREAL ft = norm(f);
		RXREAL vt = norm(v);

		if(ft > ft_max) ft_max = ft;
		if(vt > vt_max) vt_max = vt;

		RXREAL rerr = (pdens[i]-r0)/r0;
		if(rerr > rerr_max) rerr_max = rerr;
		rerr_avg += rerr;
	}
	rerr_avg /= (RXREAL)m_uNumParticles;

	ft_max = sqrt(h/ft_max);
	vt_max = h/vt_max;

	int inc = 0;

	if(0.19*ft_max > dt && rerr_max < 4.5*eta_avg && rerr_avg < 0.9*eta_avg && 0.39*vt_max > dt){
		inc = 1;
	}
	else{
		if(0.2*ft_max < dt && rerr_max > 5.5*eta_avg && rerr_avg >= eta_avg && 0.4*vt_max <= dt){
			inc = -1;
		}
	}

	new_dt += 0.002*inc*dt;

	return new_dt;
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
static int intersectSegmentTriangle(Vec3 P0, Vec3 P1,
									Vec3 V0, Vec3 V1, Vec3 V2,
									Vec3 &I, Vec3 &n, float rp)
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
		if(a == 0){
			return 2;	// 線分が平面上
		}
		else{
			return 0;	// 交点なし
		}
	}

	// 交点計算

	// 2端点がそれぞれ異なる面にあるかどうかを判定
	float r = -a/b;
	Vec3 offset = Vec3(0.0);
	float dn = 0;
	float sign_n = 1;
	if(a < 0){
		return 0;
	}

	if(r < 0.0){
		return 0;
	}
	else{
		if(fabs(a) > fabs(b)){
			return 0;
		}
		else{
			if(b > 0){
				return 0;
			}
		}
	}

	// 線分と平面の交点
	I = P0+r*dir;

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
int rxPBDSPH::calCollisionPolygon(uint grid_hash, Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt)
{
	set<int> polys_in_cell;

	int c = 0;
	if(m_pNNGrid->GetPolygonsInCell(grid_hash, polys_in_cell)){
		set<int>::iterator p = polys_in_cell.begin();
		for(; p != polys_in_cell.end(); ++p){
			int pidx = *p;

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
int rxPBDSPH::calCollisionSolid(Vec3 &pos0, Vec3 &pos1, Vec3 &vel, RXREAL dt)
{
	int c = 0;
	rxCollisionInfo coli;

	// 固体オブジェクトとの衝突処理
	for(vector<rxSolid*>::iterator i = m_vSolids.begin(); i != m_vSolids.end(); ++i){
		if((*i)->GetDistanceR(pos1, m_fParticleRadius, coli)){
			RXREAL res = m_fRestitution;
			res = (res > 0) ? (res*fabs(coli.Penetration())/(dt*norm(vel))) : 0.0f;
			//vel -= (1+res)*dot(vel, coli.Normal())*coli.Normal();
			pos1 = coli.Contact();
		}
	}

	// シミュレーション空間境界との衝突処理
	if(m_pBoundary->GetDistanceR(pos1, m_fParticleRadius, coli)){
		RXREAL res = m_fRestitution;
		res = (res > 0) ? (res*fabs(coli.Penetration())/(dt*norm(vel))) : 0.0f;
		//vel -= (1+res)*dot(vel, coli.Normal())*coli.Normal();
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
void rxPBDSPH::integrate(const RXREAL *pos, const RXREAL *vel, const RXREAL *dens, const RXREAL *acc, 
						  RXREAL *pos_new, RXREAL *vel_new, RXREAL dt)
{
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 x, x_old, v, a, v_old;
		for(int k = 0; k < 3; ++k){
			x[k] = pos[DIM*i+k];
			v[k] = vel[DIM*i+k];
			a[k] = acc[DIM*i+k];
		}
		x_old = x;

		// 新しい速度と位置
		v += dt*a;
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


/*!
 * 位置・速度の更新
 * @param[in] pos パーティクル位置
 * @param[in] vel パーティクル速度
 * @param[in] frc パーティクルにかかる力
 * @param[out] pos_new 更新パーティクル位置
 * @param[out] vel_new 更新パーティクル速度
 * @param[in] dt タイムステップ幅
 */
void rxPBDSPH::integrate2(const RXREAL *pos, const RXREAL *vel, const RXREAL *dens, const RXREAL *acc, 
						 RXREAL *pos_new, RXREAL *vel_new, RXREAL dt)
{
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 x, x_old, v, a, v_old;
		for(int k = 0; k < 3; ++k){
			x[k] = pos_new[DIM*i+k];
			x_old[k] = pos[DIM*i+k];
			v[k] = vel_new[DIM*i+k];
			a[k] = acc[DIM*i+k];
		}

		// 新しい速度と位置
		v += dt*a;
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
// 近傍探索
//-----------------------------------------------------------------------------
/*!
 * 近傍粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[in] prts パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 探索半径
 */
void rxPBDSPH::GetNearestNeighbors(int idx, RXREAL *prts, vector<rxNeigh> &neighs, RXREAL h)
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
 * @param[in] pos 探索中心位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 探索半径
 */
void rxPBDSPH::GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h)
{
	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGrid->GetNN(pos, m_hPos, m_uNumParticles, neighs, h);
	//m_pNNGrid->GetNN_Direct(pos, prts, m_uNumParticles, neighs, h);	// グリッドを使わない総当たり
}

/*!
 * 近傍境界粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 探索半径
 */
void rxPBDSPH::GetNearestNeighborsB(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h)
{
	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGridB->GetNN(pos, m_hPosB, m_uNumBParticles, neighs, h);
}


/*!
 * 全パーティクルを分割セルに格納
 */
void rxPBDSPH::SetParticlesToCell(RXREAL *prts, int n, RXREAL h)
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
void rxPBDSPH::SetParticlesToCell(void)
{
	SetParticlesToCell(m_hPos, m_uNumParticles, m_fEffectiveRadius);
}


/*!
 * 分割セルに格納されたポリゴン情報を取得
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] polys ポリゴン
 * @return 格納ポリゴン数
 */
int rxPBDSPH::GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys)
{
	return m_pNNGrid->GetPolygonsInCell(gi, gj, gk, polys);
}
/*!
 * 分割セル内のポリゴンの有無を調べる
 * @param[in] gi,gj,gk 対象分割セル
 * @return ポリゴンが格納されていればtrue
 */
bool rxPBDSPH::IsPolygonsInCell(int gi, int gj, int gk)
{
	return m_pNNGrid->IsPolygonsInCell(gi, gj, gk);
}


/*!
 * ポリゴンを分割セルに格納
 */
void rxPBDSPH::SetPolygonsToCell(void)
{
	m_pNNGrid->SetPolygonsToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);
}



/*!
 * 探索用セルの描画
 * @param[in] i,j,k グリッド上のインデックス
 */
void rxPBDSPH::DrawCell(int i, int j, int k)
{
	if(m_pNNGrid) m_pNNGrid->DrawCell(i, j, k);
}

/*!
 * 探索用グリッドの描画
 * @param[in] col パーティクルが含まれるセルの色
 * @param[in] col2 ポリゴンが含まれるセルの色
 * @param[in] sel ランダムに選択されたセルのみ描画(1で新しいセルを選択，2ですでに選択されているセルを描画，0ですべてのセルを描画)
 */
void rxPBDSPH::DrawCells(Vec3 col, Vec3 col2, int sel)
{
	if(m_pNNGrid) m_pNNGrid->DrawCells(col, col2, sel, m_hPos);

}

/*!
 * 固体障害物の描画
 */
void rxPBDSPH::DrawObstacles(int drw)
{
	for(vector<rxSolid*>::iterator i = m_vSolids.begin(); i != m_vSolids.end(); ++i){
		(*i)->Draw(drw);
	}
}




/*!
 * 三角形ポリゴンによる障害物
 * @param[in] vrts 頂点
 * @param[in] nrms 頂点法線
 * @param[in] tris メッシュ
 * @param[in] vel 初期速度
 */
void rxPBDSPH::SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris, Vec3 vel)
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

	// ポリゴンを近傍探索グリッドに登録
	m_pNNGrid->SetPolygonsToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);
}

/*!
 * ボックス型障害物
 * @param[in] cen ボックス中心座標
 * @param[in] ext ボックスの大きさ(辺の長さの1/2)
 * @param[in] ang ボックスの角度(オイラー角)
 * @param[in] vel 初期速度
 * @param[in] flg 有効/無効フラグ
 */
void rxPBDSPH::SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, Vec3 vel, int flg)
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
 * @param[in] vel 初期速度
 * @param[in] flg 有効/無効フラグ
 */
void rxPBDSPH::SetSphereObstacle(Vec3 cen, double rad, Vec3 vel, int flg)
{
	rxSolidSphere *sphere = new rxSolidSphere(cen, rad, 1);
	m_vSolids.push_back(sphere);
}

/*!
 * VBOからホストメモリへデータを転送，取得
 * @param[in] type データの種類
 * @return ホストメモリ上のデータ
 */
RXREAL* rxPBDSPH::GetArrayVBO(rxParticleArray type, bool d2h, int num)
{
	assert(m_bInitialized);
 
	if(num == -1) num = m_uNumParticles;
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

	case RX_FORCE:
		hdata = m_hFrc;
		break;

	case RX_BOUNDARY_PARTICLE:
		hdata = m_hPosB;
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
void rxPBDSPH::SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count)
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
	}	   
}




//-----------------------------------------------------------------------------
// MARK:陰関数値
//-----------------------------------------------------------------------------
double rxPBDSPH::GetImplicit(double x, double y, double z)
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
void rxPBDSPH::CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF)
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
void rxPBDSPH::CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF)
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
double rxPBDSPH::CalColorField(double x, double y, double z)
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

		c += m_fMass*m_fpW(r, h, m_fAw);
	}

	return c;
}



//-----------------------------------------------------------------------------
// MARK:シミュデータの出力
//-----------------------------------------------------------------------------
/*!
 * シミュレーション設定(パーティクル数，範囲，密度，質量など)
 * @param[in] fn 出力ファイル名
 */
void rxPBDSPH::OutputSetting(string fn)
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
	fout << m_fRestDens << endl;
	fout << m_fMass << endl;
	fout << m_iKernelParticles << endl;

	fout.close();
}

