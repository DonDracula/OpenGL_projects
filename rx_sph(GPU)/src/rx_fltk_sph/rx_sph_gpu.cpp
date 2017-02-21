/*!
  @file rx_sph_gpu.cpp
	
  @brief SPH法(GPU)の実装
*/
// FILE --rx_sph_gpu.cpp--

//#pragma comment(lib, "gsl.lib")
//#pragma comment(lib, "gslcblas.lib")


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_sph.h"

#include "rx_cu_funcs.cuh"
#include <cuda_runtime.h>

#include "rx_pcube.h"


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



//-----------------------------------------------------------------------------
// rxSPH_GPUクラスの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 * @param[in] use_opengl VBO使用フラグ
 */
rxSPH_GPU::rxSPH_GPU(bool use_opengl) : 
	rxParticleSystemBase(use_opengl), 
	m_hNrm(0), 
	m_hFrc(0),
	m_hDens(0), 
	m_hPres(0), 
	m_hSurf(0), 
	m_hUpPos(0), 
	m_hPosW(0), 
	m_hCMatrix(0), 
	m_hEigen(0), 
	m_hRMatrix(0), 
	m_hG(0), 
	m_dPos(0),
	m_dVel(0),
	m_dNrm(0), 
	m_dFrc(0), 
	m_dDens(0), 
	m_dPres(0), 
	m_dAttr(0), 
	m_dUpPos(0), 
	m_dPosW(0), 
	m_dCMatrix(0), 
	m_dEigen(0), 
	m_dRMatrix(0), 
	m_dG(0), 
	m_dVrts(0), 
	m_dTris(0)
{
	m_params.Gravity = make_float3(m_v3Gravity[0], m_v3Gravity[1], m_v3Gravity[2]);

	m_params.InitDensity = 1.0f;
	m_params.Dt = 0.01f;

	m_params.Pressure = 101325.0f;

	m_params.Tension = 0.0728f;
	m_params.Viscosity = 3.0f;
	m_params.GasStiffness = 3.0f;

	m_params.Restitution = 0.0f;

	m_params.VorticityConfinement = 1.0f;
	m_params.Threshold = 1.0f;
	m_params.Buoyancy = 0.0f;

	m_params.BoxNum = 0;
	m_params.SphereNum = 0;

	m_fRestitution = 0.0f;
	m_bCalNormal = false;
	m_bCalAnisotropic = false;

	m_fEigenMax = 1.0;

	m_dCellData.dSortedPolyIdx = 0;
	m_dCellData.dGridPolyHash = 0;
	m_dCellData.dPolyCellStart = 0;
	m_dCellData.dPolyCellEnd = 0;
	m_dCellData.uNumPolyHash = 0;

	m_fTime = 0.0f;
	m_uNumParticles = 0;
	m_iNumVrts = 0;
	m_iNumTris = 0;

	m_iColorType = RX_RAMP;
}

/*!
 * デストラクタ
 */
rxSPH_GPU::~rxSPH_GPU()
{
	Finalize();
	CuClearData();
}



/*!
 * シミュレーションの初期化
 * @param[in] max_particles 最大パーティクル数
 * @param[in] boundary_ext 境界の大きさ(各辺の長さの1/2)
 * @param[in] dens 初期密度
 * @param[in] mass パーティクルの質量
 * @param[in] kernel_particle 有効半径h以内のパーティクル数
 */
void rxSPH_GPU::Initialize(const rxSPHEnviroment &env)
{
	// MARK:Initialize
	RXCOUT << "[rxSPH_GPU::Initialize]" << endl;

	m_params.Density = env.dens;
	m_params.Mass    = env.mass;
	m_params.KernelParticles = env.kernel_particles;

	m_params.Volume = env.kernel_particles*m_params.Mass/m_params.Density;

	m_params.EffectiveRadius = pow(((3.0*m_params.Volume)/(4.0*RX_PI)), 1.0/3.0);
	m_fParticleRadius = 0.5f*m_params.EffectiveRadius;
	m_params.ParticleRadius = m_fParticleRadius;

	RXCOUT << "particle : " << endl;
	RXCOUT << " n_max = " << env.max_particles << endl;
	RXCOUT << " h = " << m_params.EffectiveRadius << endl;
	RXCOUT << " r = " << m_params.ParticleRadius << endl;
	RXCOUT << " dens = " << m_params.Density << endl;
	RXCOUT << " mass = " << m_params.Mass << endl;
	RXCOUT << " kernel_particles = " << m_params.KernelParticles << endl;
	RXCOUT << " volume = " << m_params.Volume << endl;

	RXREAL h = m_params.EffectiveRadius;
	RXREAL r = m_fParticleRadius;

	//
	// 境界設定
	//
	// 境界の範囲
	// シミュレーション環境の大きさ
	m_v3EnvMin = env.boundary_cen-env.boundary_ext;
	m_v3EnvMax = env.boundary_cen+env.boundary_ext;
	RXCOUT << "simlation range : " << m_v3EnvMin << " - " << m_v3EnvMax << endl;

	m_params.BoundaryMin = MAKE_FLOAT3V(m_v3EnvMin);
	m_params.BoundaryMax = MAKE_FLOAT3V(m_v3EnvMax);

	Vec3 world_size = m_v3EnvMax-m_v3EnvMin;
	Vec3 world_origin = m_v3EnvMin;

	double expansion = 0.01;
	world_origin -= 0.5*expansion*world_size;
	world_size *= (1.0+expansion); // シミュレーション環境全体を覆うように設定

	m_v3EnvMin = world_origin;
	m_v3EnvMax = world_origin+world_size;

	// 分割セル設定
	setupCells(m_v3EnvMin, m_v3EnvMax, h);

	m_gridSortBits = 24;	// グリッド数が多い時はこの値を増やす

	m_params.GridSize = m_gridSize;
	m_params.NumCells = m_dCellData.uNumCells;
	m_params.NumBodies = m_uNumParticles;

	m_params.WorldOrigin = MAKE_FLOAT3(world_origin[0], world_origin[1], world_origin[2]);
	m_params.WorldMax = MAKE_FLOAT3(m_v3EnvMax[0], m_v3EnvMax[1], m_v3EnvMax[2] );

	RXCOUT << "grid for nn search : " << endl;
	RXCOUT << "  size   : " << m_params.GridSize << endl;
	RXCOUT << "  num    : " << m_params.NumCells << endl;
	RXCOUT << "  origin : " << m_params.WorldOrigin << endl;
	RXCOUT << "  width  : " << m_params.CellWidth << endl;

	m_uNumArdGrid = (int)(m_params.EffectiveRadius/m_params.CellWidth.x)+1;
	RXCOUT << "  numArdGrid : " << m_uNumArdGrid << endl;

	// GPUに渡すための分割セル情報
	m_dCellData.uNumArdGrid = m_uNumArdGrid;
	m_dCellData.uNumCells = m_dCellData.uNumCells;
	m_dCellData.uNumParticles = m_uNumParticles;

	// カーネル関数の定数
	m_params.Wpoly6  =  315.0/(64.0*RX_PI*pow(h, (RXREAL)9.0));
	m_params.GWpoly6 = -945.0/(32.0*RX_PI*pow(h, (RXREAL)9.0));
	m_params.LWpoly6 = -945.0/(32.0*RX_PI*pow(h, (RXREAL)9.0));

	m_params.Wspiky  =  15.0/(RX_PI*pow(h, (RXREAL)6.0));
	m_params.GWspiky = -45.0/(RX_PI*pow(h, (RXREAL)6.0));
	m_params.LWspiky = -90.0/(RX_PI*pow(h, (RXREAL)6.0));

	m_params.Wvisc   = 15.0/(2.0*RX_PI*pow(h, (RXREAL)3.0));
	m_params.GWvisc  = 15.0/(2.0*RX_PI*pow(h, (RXREAL)3.0));
	m_params.LWvisc  = 45.0/(RX_PI*pow(h, (RXREAL)6.0));

	m_uNumParticles = 0;

	Allocate(env.max_particles);
}

inline uint calUintPow(uint x, uint y)
{
	uint x_y = 1;
	for(uint i=0; i < y;i++) x_y *= x;
	return x_y;
}

/*!
 * メモリの確保
 *  - 最大パーティクル数で確保
 * @param[in] max_particles 最大パーティクル数
 */
void rxSPH_GPU::Allocate(int maxParticles)
{
	// MARK:Allocate
	assert(!m_bInitialized);

	//m_uNumParticles = maxParticles;
	m_uMaxParticles = maxParticles;

	unsigned int size  = m_uMaxParticles*DIM;
	unsigned int size1 = m_uMaxParticles;
	unsigned int mem_size  = sizeof(RXREAL)*size;
	unsigned int mem_size1 = sizeof(RXREAL)*size1;


	//
	// CPU側メモリ確保
	//
	// GPUとのデータ転送が多いデータはページロックメモリに確保
	cudaMallocHost((void**)&m_hPos, mem_size);
	cudaMallocHost((void**)&m_hVel, mem_size);
	cudaMallocHost((void**)&m_hAttr, sizeof(uint)*size1);

	// 通常のメモリ確保
	m_hNrm = new RXREAL[size];
	m_hFrc = new RXREAL[size];
	m_hDens = new RXREAL[size1];
	m_hPres = new RXREAL[size1];
	memset(m_hNrm, 0, mem_size);
	memset(m_hFrc, 0, mem_size);
	memset(m_hDens, 0, mem_size1);
	memset(m_hPres, 0, mem_size1);

	m_hSurf = new uint[m_uMaxParticles];
	memset(m_hSurf, 0, sizeof(uint)*m_uMaxParticles);
	m_hTmp = new RXREAL[m_uMaxParticles];
	memset(m_hTmp, 0, sizeof(RXREAL)*m_uMaxParticles);

	// Anisotropic kernel
	m_hUpPos   = new RXREAL[size];
	m_hPosW    = new RXREAL[size];
	m_hCMatrix = new RXREAL[m_uMaxParticles*9];
	m_hEigen   = new RXREAL[m_uMaxParticles*3];
	m_hRMatrix = new RXREAL[m_uMaxParticles*9];
	m_hG       = new RXREAL[m_uMaxParticles*9];


	//
	// GPU側メモリ確保
	//
	if(m_bUseOpenGL){
		m_posVBO = createVBO(mem_size);	
		CuRegisterGLBufferObject(m_posVBO, &m_pPosResource);

		m_colorVBO = createVBO(sizeof(RXREAL)*4*m_uMaxParticles);
		SetColorVBO(RX_RAMP);
	}
	else{
		CuAllocateArray((void**)&m_dPos, mem_size);
	}

	CuAllocateArray((void**)&m_dVel,    mem_size);
	CuAllocateArray((void**)&m_dNrm,    mem_size);
	CuAllocateArray((void**)&m_dFrc,    mem_size);
	CuAllocateArray((void**)&m_dDens,   mem_size1);
	CuAllocateArray((void**)&m_dPres,   mem_size1);
	CuAllocateArray((void**)&m_dAttr,   size1*sizeof(uint));

	// Anisotropic kernel
	CuAllocateArray((void**)&m_dUpPos,      mem_size);
	CuAllocateArray((void**)&m_dPosW,       mem_size);
	CuAllocateArray((void**)&m_dCMatrix,	m_uMaxParticles*9*sizeof(RXREAL));
	CuAllocateArray((void**)&m_dEigen,		m_uMaxParticles*3*sizeof(RXREAL));
	CuAllocateArray((void**)&m_dRMatrix,	m_uMaxParticles*9*sizeof(RXREAL));
	CuAllocateArray((void**)&m_dG,          m_uMaxParticles*9*sizeof(RXREAL));

	// パーティクルグリッドデータ
	CuAllocateArray((void**)&m_dCellData.dSortedPos, mem_size);
	CuAllocateArray((void**)&m_dCellData.dSortedVel, mem_size);
	CuAllocateArray((void**)&m_dCellData.dGridParticleHash,  m_uMaxParticles*sizeof(uint));
	CuAllocateArray((void**)&m_dCellData.dSortedIndex, m_uMaxParticles*sizeof(uint));
	CuAllocateArray((void**)&m_dCellData.dCellStart, m_dCellData.uNumCells*sizeof(uint));
	CuAllocateArray((void**)&m_dCellData.dCellEnd, m_dCellData.uNumCells*sizeof(uint));

	CuSetParameters(&m_params);

	m_bInitialized = true;
}

/*!
 * 確保したメモリの解放
 */
void rxSPH_GPU::Finalize(void)
{
	assert(m_bInitialized);

	// ページロックメモリ解放
	cudaFreeHost(m_hPos);
	cudaFreeHost(m_hVel);
	cudaFreeHost(m_hAttr);

	// 通常メモリ解放
	if(m_hNrm) delete [] m_hNrm;
	if(m_hFrc) delete [] m_hFrc;
	if(m_hDens) delete [] m_hDens;
	if(m_hPres) delete [] m_hPres;

	if(m_hSurf) delete [] m_hSurf;
	if(m_hTmp) delete [] m_hTmp;

	// Anisotoropic kernel
	if(m_hUpPos) delete [] m_hUpPos;
	if(m_hPosW) delete [] m_hPosW;
	if(m_hCMatrix) delete [] m_hCMatrix;
	if(m_hEigen) delete [] m_hEigen;
	if(m_hRMatrix) delete [] m_hRMatrix;
	if(m_hG) delete [] m_hG;


	// GPUメモリ解放
	CuFreeArray(m_dVel);
	CuFreeArray(m_dNrm);
	CuFreeArray(m_dFrc);
	CuFreeArray(m_dDens);
	CuFreeArray(m_dPres);
	CuFreeArray(m_dAttr);

	CuFreeArray(m_dUpPos);
	CuFreeArray(m_dPosW);
	CuFreeArray(m_dCMatrix);
	CuFreeArray(m_dEigen);
	CuFreeArray(m_dRMatrix);
	CuFreeArray(m_dG);

	CuFreeArray(m_dCellData.dSortedPos);
	CuFreeArray(m_dCellData.dSortedVel);
	CuFreeArray(m_dCellData.dGridParticleHash);
	CuFreeArray(m_dCellData.dSortedIndex);
	CuFreeArray(m_dCellData.dCellStart);
	CuFreeArray(m_dCellData.dCellEnd);

	if(m_dVrts) CuFreeArray(m_dVrts);
	if(m_dTris) CuFreeArray(m_dTris);

	if(m_dCellData.dSortedPolyIdx) CuFreeArray(m_dCellData.dSortedPolyIdx);
	if(m_dCellData.dGridPolyHash) CuFreeArray(m_dCellData.dGridPolyHash);
	if(m_dCellData.dPolyCellStart) CuFreeArray(m_dCellData.dPolyCellStart);
	if(m_dCellData.dPolyCellEnd) CuFreeArray(m_dCellData.dPolyCellEnd);


	if(m_bUseOpenGL){
		CuUnregisterGLBufferObject(m_pPosResource);
		glDeleteBuffers(1, (const GLuint*)&m_posVBO);
		glDeleteBuffers(1, (const GLuint*)&m_colorVBO);
	}
	else{
		CuFreeArray(m_dPos);
	}

	m_uNumParticles = 0;
	m_uMaxParticles = 0;
}



#if MAX_BOX_NUM
void rxSPH_GPU::ToggleSolidFlg(int t)
{
	if(!m_params.BoxNum) return;

	RX_TOGGLE(m_params.BoxFlg[m_params.BoxNum-1], t);
	CuSetParameters(&m_params);
}
#endif


/*!
 * GPU側のパラメータファイルを更新
 */
void rxSPH_GPU::UpdateParams(void)
{
	m_params.ParticleRadius = m_fParticleRadius;
	m_params.Gravity = make_float3(m_v3Gravity[0], m_v3Gravity[1], m_v3Gravity[2]);
	m_params.GlobalDamping = m_fDamping;

	m_params.Restitution = m_fRestitution;
	CuSetParameters(&m_params);
}

/*!
 * SPHを1ステップ進める
 * @param[in] dt 時間ステップ幅
 * @retval ture  計算完了
 * @retval false 最大ステップ数を超えています
 */
bool rxSPH_GPU::Update(RXREAL dt, int step)
{
	//RXTIMER_RESET;

	// 流入パーティクルを追加
	if(!m_vInletLines.empty()){
		int start = (m_iInletStart == -1 ? 0 : m_iInletStart);
		int num = 0;
		int attr = 0;
		vector<rxInletLine>::iterator itr = m_vInletLines.begin();
		for(; itr != m_vInletLines.end(); ++itr){
			rxInletLine iline = *itr;
			if(iline.span > 0 && step%(iline.span) == 0){
				int count = addParticles(m_iInletStart, iline, attr);
				num += count;
			}
			attr++;
		}
		SetArrayVBO(RX_POSITION, m_hPos, start, num);
		SetArrayVBO(RX_VELOCITY, m_hVel, start, num);
	}

	if(!m_uNumParticles) return false;

	assert(m_bInitialized);

	static bool init = true;

	m_params.Dt = dt;
	UpdateParams();

	// GPU用変数にパーティクル数を設定
	m_params.NumBodies = m_uNumParticles;
	m_dCellData.uNumParticles = m_uNumParticles;

	// パーティクル座標配列をマッピング
	RXREAL *dPos;
	if(m_bUseOpenGL){
		dPos = (RXREAL*)CuMapGLBufferObject(&m_pPosResource);
	}
	else{
		dPos = (RXREAL*)m_dPos;
	}

	for(uint j = 0; j < m_solverIterations; ++j){
		// 近傍探索高速化用グリッドデータの作成
		// 分割セルのハッシュを計算
		CuCalcHash(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, dPos, m_uNumParticles);

		// ハッシュに基づきパーティクルをソート
		CuSort(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, m_uNumParticles);

		// パーティクル配列をソートされた順番に並び替え，
		// 各セルの始まりと終わりのインデックスを検索
		CuReorderDataAndFindCellStart(m_dCellData, dPos, m_dVel);

		//RXTIMER("cell construction");

		// 密度計算
		CuSphDensity(m_dDens, m_dPres, m_dCellData);

		if(m_bCalNormal){
			// 法線計算
			CuSphNormal(m_dNrm, m_dDens, m_dCellData);

			m_hNrm = GetArrayVBO(RX_NORMAL);
		}

		// パーティクルにかかる力(外力，圧力，粘性拡散力など)を計算
		CuSphForces(m_dDens, m_dPres, m_dFrc, m_dCellData, dt);
		RXTIMER("force calculation");

		// 位置，速度の更新
		if(m_iNumTris == 0){
			CuSphIntegrate(dPos, m_dVel, m_dFrc, m_dDens, dt, m_uNumParticles);
		}
		else{	// ポリゴンによる固体境界有り
			CuSphIntegrateWithPolygon(dPos, m_dVel, m_dFrc, m_dDens,  
									  m_dVrts, m_dTris, m_iNumTris, dt, m_dCellData);
		}

		RXTIMER("update position");

		init = false;
	}

	if(m_bUseOpenGL){
		CuUnmapGLBufferObject(m_pPosResource);
	}

	m_fTime += dt;

	// VBOからメモリへ
	//m_hPos = GetArrayVBO(RX_POSITION);
	//m_hVel = GetArrayVBO(RX_VELOCITY);

	SetColorVBO(m_iColorType);

	//RXTIMER("color(vbo)");

	m_bCalAnisotropic = false;

	return true;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxSPH_GPU::GetParticle(void)
{
	return m_hPos;
}

/*!
 * パーティクルデータの取得
 * @return パーティクルデータのデバイスメモリポインタ
 */
RXREAL* rxSPH_GPU::GetParticleDevice(void)
{
	return (m_bUseOpenGL ? (RXREAL*)CuMapGLBufferObject(&m_pPosResource) : (RXREAL*)m_dPos);
}
/*!
 * パーティクルデータのアンマップ(VBO)
 */
void rxSPH_GPU::UnmapParticle(void)
{
	if(m_bUseOpenGL) CuUnmapGLBufferObject(m_pPosResource);
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
void rxSPH_GPU::DetectSurfaceParticles(void)
{
}

/*!
 * 近傍パーティクルの正規化重心までの距離を計算
 * @param[in] i パーティクルインデックス
 */
double rxSPH_GPU::CalDistToNormalizedMassCenter(const int i)
{
	double dis = DBL_MIN;

	return dis;
}

/*!
 * 表面パーティクル情報の取得
 *  パーティクル数と同じ大きさのuint配列で表面パーティクルならば1, そうでなければ0が格納されている
 */
uint* rxSPH_GPU::GetArraySurf(void)
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
int rxSPH_GPU::GetSurfaceParticles(const Vec3 pos, RXREAL h, vector<rxSurfaceParticle> &sps)
{
	int num = 0;
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
void rxSPH_GPU::CalNormal(void)
{
}

/*!
 * 密度分布から法線を計算
 */
void rxSPH_GPU::CalNormalFromDensity(void)
{
	// 法線計算
	CuSphNormal(m_dNrm, m_dDens, m_dCellData);

	m_hNrm = GetArrayVBO(RX_NORMAL);
}



//-----------------------------------------------------------------------------
// Anisotropic Kernel
//-----------------------------------------------------------------------------
/*!
 * Anisotropic kernelの計算
 *  - J. Yu and G. Turk, Reconstructing Surfaces of Particle-Based Fluids Using Anisotropic Kernels, SCA2010. 
 */
void rxSPH_GPU::CalAnisotropicKernel(void)
{
	if(m_uNumParticles == 0) return;

	RXREAL r0 = m_params.Density;
	RXREAL h0 = m_params.EffectiveRadius;
	RXREAL h = 2.0*h0;

	//SetParticlesToCell();

	UpdateParams();
	m_params.NumBodies = m_uNumParticles;
	m_dCellData.uNumParticles = m_uNumParticles;

	RXREAL *dPos;
	if(m_bUseOpenGL){
		dPos = (RXREAL*)CuMapGLBufferObject(&m_pPosResource);
	}
	else{
		dPos = (RXREAL*)m_dPos;
	}

	// 近傍探索高速化用グリッドデータの作成
	// 分割セルのハッシュを計算
	CuCalcHash(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, dPos, m_uNumParticles);

	// ハッシュに基づきパーティクルをソート
	CuSort(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, m_uNumParticles);

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	CuReorderDataAndFindCellStart(m_dCellData, dPos, m_dVel);

	if(m_bUseOpenGL){
		CuUnmapGLBufferObject(m_pPosResource);
	}


	// 平滑化カーネル中心(式6)と重み付き平均位置(式10)の計算
	RXREAL lambda = 0.9;
	CuSphCalUpdatedPosition(m_dUpPos, m_dPosW, lambda, h, m_dCellData);

	// 平滑化位置で近傍探索セルを更新
	setObjectToCell(m_dUpPos);

	// 平滑化位置での重み付き平均位置の再計算とcovariance matrixの計算(式9)
	CuSphCalCovarianceMatrix(m_dPosW, m_dCMatrix, h, m_dCellData);

	RXTIMER("aniso : cmatrix");

	// 特異値分解により特異値,特異ベクトルを計算
	CuSphSVDecomposition(m_dCMatrix, m_dPosW, m_dEigen, m_dRMatrix, m_uNumParticles);

	RXTIMER("aniso : svd");
	
	// 変形行列Gの計算
	CuSphCalTransformMatrix(m_dEigen, m_dRMatrix, m_dG, m_uNumParticles);

	//CuCopyArrayFromDevice(m_hG, m_dG, 0, m_uNumParticles*9*sizeof(RXREAL));

	CuCopyArrayFromDevice(m_hEigen, m_dEigen, 0, m_uNumParticles*3*sizeof(RXREAL));
	m_fEigenMax = 0.0;
	for(uint i = 0; i < m_uNumParticles; ++i){
		for(int j = 0; j < 3; ++j){
			if(m_hEigen[3*i+j] > m_fEigenMax) m_fEigenMax = m_hEigen[3*i+j];
		}
	}
	//RXCOUT << "eigen_max = " << m_fEigenMax << endl;

	m_bCalAnisotropic = true;

	RXTIMER("aniso : G");

}


/*!
 * カラー値用VBOの編集
 * @param[in] type 色のベースとする物性値
 */
void rxSPH_GPU::SetColorVBO(int type)
{
	// MRK:SetColorVBO

	switch(type){
	case RX_DENSITY:
		CuCopyArrayFromDevice(m_hDens, m_dDens, 0, sizeof(RXREAL)*m_uNumParticles);
		SetColorVBOFromArray(m_hDens, 1, false, m_params.Density*3);
		break;

	case RX_PRESSURE:
		CuCopyArrayFromDevice(m_hPres, m_dPres, 0, sizeof(RXREAL)*m_uNumParticles);
		SetColorVBOFromArray(m_hPres, 1);
		break;

	case RX_CONSTANT:
		if(m_bUseOpenGL){
			// カラーバッファに値を設定
			glBindBufferARB(GL_ARRAY_BUFFER, m_colorVBO);
			RXREAL *data = (RXREAL*)glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			RXREAL *ptr = data;
			for(uint i = 0; i < m_uNumParticles; ++i){
				RXREAL t = i/(RXREAL)m_uNumParticles;
				RX_COLOR_RAMP<RXREAL>(m_hAttr[i]/3.0, ptr);
				ptr += 3;

				//*ptr++ = 0.15f;
				//*ptr++ = 0.15f;
				//*ptr++ = 0.95f;
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


void rxSPH_GPU::CalMaxDensity(int k)
{
	CuCopyArrayFromDevice(m_hDens, m_dDens, 0, sizeof(RXREAL)*2*m_uNumParticles);

	RXREAL max_dens = 0.0;
	for(uint i = 0; i < m_uNumParticles; ++i){
		if(m_hDens[2*i+k] > max_dens) max_dens = m_hDens[2*i+k];
	}

	RXCOUT << "Density  : " << max_dens << endl;
}


/*!
 * グリッドハッシュ値の計算
 * @param[in] x,y,z グリッド位置
 * @return グリッドハッシュ値
 */
uint rxSPH_GPU::calGridHash(uint x, uint y, uint z)
{
	x = (x < 0 ? 0 : (x >= m_params.GridSize.x ? m_params.GridSize.x-1 : x));
	y = (y < 0 ? 0 : (y >= m_params.GridSize.y ? m_params.GridSize.y-1 : y));
	z = (z < 0 ? 0 : (z >= m_params.GridSize.z ? m_params.GridSize.z-1 : z));
	return z*m_params.GridSize.y*m_params.GridSize.x+y*m_params.GridSize.x+x;
}
/*!
 * グリッドハッシュ値の計算
 * @param[in] pos パーティクル座標
 * @return グリッドハッシュ値
 */
uint rxSPH_GPU::calGridHash(Vec3 pos)
{
	pos -= m_v3EnvMin;

	// 分割セルインデックスの算出
	uint x = pos[0]/m_params.CellWidth.x;
	uint y = pos[1]/m_params.CellWidth.y;
	uint z = pos[2]/m_params.CellWidth.z;
	return calGridHash(x, y, z);
}

/*!
 * 空間分割法の準備
 * @param[in] vMin 環境の最小座標
 * @param[in] vMax 環境の最大座標
 * @param[in] h 影響半径
 */
void rxSPH_GPU::setupCells(const Vec3 &vMin, const Vec3 &vMax, const double &h)
{
	if(h < RX_EPS) return;

	Vec3 world_size = vMax-vMin;
	Vec3 world_origin = vMin;

	double max_axis = RXFunc::Max3(world_size);

	int d = (int)(log(max_axis/h)/log(2.0)+0.5);
	int n = (int)(pow(2.0, (double)d)+0.5);
	double cell_width = max_axis/n;

	d = (int)(log(world_size[0]/cell_width)/log(2.0)+0.5);
	m_gridSize.x = (int)(pow(2.0, (double)d)+0.5);
	d = (int)(log(world_size[1]/cell_width)/log(2.0)+0.5);
	m_gridSize.y = (int)(pow(2.0, (double)d)+0.5);;
	d = (int)(log(world_size[2]/cell_width)/log(2.0)+0.5);
	m_gridSize.z = (int)(pow(2.0, (double)d)+0.5);;

	m_dCellData.uNumCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
	m_params.CellWidth = MAKE_FLOAT3(cell_width, cell_width, cell_width);
}



/*!
 * 全パーティクルを分割セルに格納
 *  - 各パーティクルの属するグリッドハッシュを計算して格納する
 * @param[in] p 全パーティクルの座標を格納した配列
 */
void rxSPH_GPU::setObjectToCell(RXREAL *p)
{
	// 近傍探索高速化用グリッドデータの作成
	// 分割セルのハッシュを計算
	CuCalcHash(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, p, m_uNumParticles);

	// ハッシュに基づきパーティクルをソート
	CuSort(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, m_uNumParticles);

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	CuReorderDataAndFindCellStart(m_dCellData, p, m_dVel);
}

/*!
 * 全パーティクルを分割セルに格納
 */
void rxSPH_GPU::SetParticlesToCell(RXREAL *prts, int n, RXREAL h)
{
}
void rxSPH_GPU::SetParticlesToCell(void)
{
	SetParticlesToCell(m_hPos, m_uNumParticles, m_params.EffectiveRadius);
}

/*!
 * ポリゴンを分割セルに格納
 * @param[in] vrts ポリゴン頂点
 * @param[in] nv 頂点数
 * @param[in] tris メッシュ
 * @param[in] nt メッシュ数
 */
void rxSPH_GPU::setPolysToCell(RXREAL *vrts, int nv, int* tris, int nt)
{
	uint *hPolyCellStart = new uint[m_dCellData.uNumCells];
	uint *hPolyCellEnd = new uint[m_dCellData.uNumCells];

	int mem_size2 = m_dCellData.uNumCells*sizeof(uint);
	memset(hPolyCellStart, 0xffffffff, mem_size2);
	memset(hPolyCellEnd,   0,          mem_size2);

	int num_hash = 0;

	// 各パーティクルのグリッドハッシュの計算
	vector<uint> tri_hash, tri_idx;
	vector<Vec3> tri_vrts, tri_vrts_c;
	tri_vrts.resize(3);
	tri_vrts_c.resize(3);
	for(int i = 0; i < nt; i++){
		for(int j = 0; j < 3; ++j){
			Vec3 pos;
			pos[0] = vrts[3*tris[3*i+j]+0];
			pos[1] = vrts[3*tris[3*i+j]+1];
			pos[2] = vrts[3*tris[3*i+j]+2];
			tri_vrts[j] = pos;
		}

		Vec3 nrm = Unit(cross(tri_vrts[1]-tri_vrts[0], tri_vrts[2]-tri_vrts[0]));

		// ポリゴンのBBox
		Vec3 bmin, bmax;
		bmin = tri_vrts[0];
		bmax = tri_vrts[0];
		for(int j = 1; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				if(tri_vrts[j][k] < bmin[k]) bmin[k] = tri_vrts[j][k];
				if(tri_vrts[j][k] > bmax[k]) bmax[k] = tri_vrts[j][k];
			}
		}

		// BBoxと重なるセル
		bmin -= m_v3EnvMin;
		bmax -= m_v3EnvMin;

		// 分割セルインデックスの算出
		int bmin_gidx[3], bmax_gidx[3];
		bmin_gidx[0] = bmin[0]/m_params.CellWidth.x;
		bmin_gidx[1] = bmin[1]/m_params.CellWidth.y;
		bmin_gidx[2] = bmin[2]/m_params.CellWidth.z;
		bmax_gidx[0] = bmax[0]/m_params.CellWidth.x;
		bmax_gidx[1] = bmax[1]/m_params.CellWidth.y;
		bmax_gidx[2] = bmax[2]/m_params.CellWidth.z;

		bmin_gidx[0] = RX_CLAMP(bmin_gidx[0], 0, (int)m_params.GridSize.x-1);
		bmin_gidx[1] = RX_CLAMP(bmin_gidx[1], 0, (int)m_params.GridSize.y-1);
		bmin_gidx[2] = RX_CLAMP(bmin_gidx[2], 0, (int)m_params.GridSize.z-1);
		bmax_gidx[0] = RX_CLAMP(bmax_gidx[0], 0, (int)m_params.GridSize.x-1);
		bmax_gidx[1] = RX_CLAMP(bmax_gidx[1], 0, (int)m_params.GridSize.y-1);
		bmax_gidx[2] = RX_CLAMP(bmax_gidx[2], 0, (int)m_params.GridSize.z-1);

		// 各セルにポリゴンが含まれるかをチェック
		Vec3 len = Vec3(m_params.CellWidth.x, m_params.CellWidth.y, m_params.CellWidth.z);
		Vec3 cen(0.0);
		for(int x = bmin_gidx[0]; x <= bmax_gidx[0]; ++x){
			for(int y = bmin_gidx[1]; y <= bmax_gidx[1]; ++y){
				for(int z = bmin_gidx[2]; z <= bmax_gidx[2]; ++z){
					cen = m_v3EnvMin+Vec3(x+0.5, y+0.5, z+0.5)*len;

					for(int j = 0; j < 3; ++j){
						tri_vrts_c[j] = (tri_vrts[j]-cen)/len;
					}

					if(RXFunc::polygon_intersects_cube(tri_vrts_c, nrm, 0)){
						// ハッシュ値計算
						uint hash = calGridHash(x, y, z);

						tri_idx.push_back((uint)i);
						tri_hash.push_back(hash);

						num_hash++;
					}
				}
			}
		}
	}

	RXCOUT << "polygon hash : " << num_hash << endl;

	m_dCellData.uNumPolyHash = (uint)num_hash;

	if(num_hash){
		int mem_size1 = m_dCellData.uNumPolyHash*sizeof(uint);
		uint *hSortedPolyIdx = new uint[m_dCellData.uNumPolyHash];
		uint *hGridPolyHash  = new uint[m_dCellData.uNumPolyHash];
		memcpy(hSortedPolyIdx, &tri_idx[0], mem_size1);
		memcpy(hGridPolyHash, &tri_hash[0], mem_size1);

		// グリッドハッシュでソート
		if(m_dCellData.dSortedPolyIdx) CuFreeArray(m_dCellData.dSortedPolyIdx);
		if(m_dCellData.dGridPolyHash) CuFreeArray(m_dCellData.dGridPolyHash);
		CuAllocateArray((void**)&m_dCellData.dSortedPolyIdx, mem_size1);
		CuAllocateArray((void**)&m_dCellData.dGridPolyHash,  mem_size1);

		CuCopyArrayToDevice(m_dCellData.dSortedPolyIdx, &tri_idx[0],  0, mem_size1);
		CuCopyArrayToDevice(m_dCellData.dGridPolyHash,  &tri_hash[0], 0, mem_size1);

		CuSort(m_dCellData.dGridPolyHash, m_dCellData.dSortedPolyIdx, m_dCellData.uNumPolyHash);

		CuCopyArrayFromDevice(hSortedPolyIdx, m_dCellData.dSortedPolyIdx, 0, mem_size1);
		CuCopyArrayFromDevice(hGridPolyHash,  m_dCellData.dGridPolyHash,  0, mem_size1);

		// パーティクル配列をソートされた順番に並び替え，
		// 各セルの始まりと終わりのインデックスを検索
		for(uint i = 0; i < m_dCellData.uNumPolyHash; ++i){
			uint hash = hGridPolyHash[i];

			if(i == 0){
				hPolyCellStart[hash] = i;
			}
			else{
				uint prev_hash = hGridPolyHash[i-1];

				if(i == 0 || hash != prev_hash){
					hPolyCellStart[hash] = i;
					if(i > 0){
						hPolyCellEnd[prev_hash] = i;
					}
				}

				if(i == m_uNumParticles-1){
					hPolyCellEnd[hash] = i+1;
				}
			}
		}

		if(m_dCellData.dPolyCellStart) CuFreeArray(m_dCellData.dPolyCellStart);
		if(m_dCellData.dPolyCellEnd) CuFreeArray(m_dCellData.dPolyCellEnd);

		CuAllocateArray((void**)&m_dCellData.dPolyCellStart, mem_size2);
		CuAllocateArray((void**)&m_dCellData.dPolyCellEnd,   mem_size2);

		CuCopyArrayToDevice(m_dCellData.dPolyCellStart,  hPolyCellStart, 0, mem_size2);
		CuCopyArrayToDevice(m_dCellData.dPolyCellEnd,    hPolyCellEnd,   0, mem_size2);

		delete [] hGridPolyHash;
		delete [] hSortedPolyIdx;
	}

	delete [] hPolyCellStart;
	delete [] hPolyCellEnd;
}

/*!
 * ポリゴンを分割セルに格納
 */
void rxSPH_GPU::SetPolygonsToCell(void)
{
	setPolysToCell(&m_vVrts[0], m_iNumVrts, &m_vTris[0], m_iNumTris);
}


/*!
 * 探索用セルの描画
 * @param[in] i,j,k グリッド上のインデックス
 */
void rxSPH_GPU::DrawCell(int i, int j, int k)
{
	glPushMatrix();
	glTranslated(m_v3EnvMin[0], m_v3EnvMin[1], m_v3EnvMin[2]);
	glTranslatef((i+0.5)*m_params.CellWidth.x, (j+0.5)*m_params.CellWidth.y, (k+0.5)*m_params.CellWidth.z);
	glutWireCube(m_params.CellWidth.x);
	glPopMatrix();
}


/*!
 * 探索用グリッドの描画
 * @param[in] col パーティクルが含まれるセルの色
 * @param[in] col2 ポリゴンが含まれるセルの色
 * @param[in] sel ランダムに選択されたセルのみ描画(1で新しいセルを選択，2ですでに選択されているセルを描画，0ですべてのセルを描画)
 */
void rxSPH_GPU::DrawCells(Vec3 col, Vec3 col2, int sel)
{
	glPushMatrix();

	uint *hCellStart     = new uint[m_dCellData.uNumCells];
	uint *hPolyCellStart = new uint[m_dCellData.uNumCells];
	CuCopyArrayFromDevice(hCellStart,     m_dCellData.dCellStart,     0, m_dCellData.uNumCells*sizeof(uint));
	CuCopyArrayFromDevice(hPolyCellStart, m_dCellData.dPolyCellStart, 0, m_dCellData.uNumCells*sizeof(uint));

	if(sel){
		uint *hCellEnd     = new uint[m_dCellData.uNumCells];
		uint *hSortedIndex = new uint[m_uNumParticles];
		CuCopyArrayFromDevice(hCellEnd,     m_dCellData.dCellEnd,     0, m_dCellData.uNumCells*sizeof(uint));
		CuCopyArrayFromDevice(hSortedIndex, m_dCellData.dSortedIndex, 0, m_uNumParticles*sizeof(uint));

		// ランダムに選んだセルとその中のパーティクルのみ描画
		static int grid_hash = 0;
		static uint start_index = 0xffffffff;
		if(sel == 1){
			do{
				grid_hash = RXFunc::Nrand(m_dCellData.uNumCells-1);
				start_index = hCellStart[grid_hash];
			}while(start_index == 0xffffffff);
		}

		uint w = grid_hash%(m_params.GridSize.x*m_params.GridSize.y);
		DrawCell(w%m_params.GridSize.x, w/m_params.GridSize.x, grid_hash/(m_params.GridSize.x*m_params.GridSize.y));

		glColor3d(1.0, 0.0, 0.0);
		glPointSize(10.0);
		glBegin(GL_POINTS);

		int c = 0;
		uint end_index = hCellEnd[grid_hash];
		for(uint j = start_index; j < end_index; ++j){
			uint idx = hSortedIndex[j];
			Vec3 pos;
			pos[0] = m_hPos[4*idx+0];
			pos[1] = m_hPos[4*idx+1];
			pos[2] = m_hPos[4*idx+2];
			
			glVertex3dv(pos);

			c++;
		}
		glEnd();
		cout << "cell(" << grid_hash << ") : " << c << endl;

		delete [] hCellEnd;
		delete [] hSortedIndex;
	}
	else{
		// パーティクル or ポリゴンを含む全セルの描画
		RXFOR3(0, (int)m_params.GridSize.x, 0, (int)m_params.GridSize.y, 0, (int)m_params.GridSize.z){
			bool disp = false;
			uint grid_hash = calGridHash(i, j, k);
			uint start_index = hCellStart[grid_hash];
			uint start_index_poly = 0xffffffff;
		
			if(m_dCellData.uNumPolyHash) start_index_poly = hPolyCellStart[grid_hash];

			if(start_index != 0xffffffff){
				glColor3dv(col2.data);
				disp = true;
			}
			if(start_index_poly != 0xffffffff){
				glColor3dv(col.data);
				disp = true;
			}

			if(disp){
				DrawCell(i, j, k);
			}
		}
	}

	delete [] hCellStart;
	delete [] hPolyCellStart;

	glPopMatrix();
}

/*!
 * 固体障害物の描画
 */
void rxSPH_GPU::DrawObstacles(void)
{
#if MAX_BOX_NUM
	for(int i = 0; i < (int)m_params.BoxNum; ++i){
		if(!m_params.BoxFlg[i]) continue;

		Vec3 bcen = Vec3(m_params.BoxCen[i].x, m_params.BoxCen[i].y, m_params.BoxCen[i].z);
		Vec3 bext = Vec3(m_params.BoxExt[i].x, m_params.BoxExt[i].y, m_params.BoxExt[i].z);
		float bmat[16];
		GetGLMatrix(m_params.BoxRot[i], bmat);

		glPushMatrix();
		glTranslated(bcen[0], bcen[1], bcen[2]);
		glMultMatrixf(bmat);
		glScalef(2.0*bext[0], 2.0*bext[1], 2.0*bext[2]);
		//glRotated(brot, 0, 0, 1);
		glutWireCube(1.0);
		glPopMatrix();
	}
#endif

#if MAX_SPHERE_NUM
	for(int i = 0; i < (int)m_params.SphereNum; ++i){
		if(!m_params.SphereFlg[i]) continue;

		Vec3 scen  = Vec3(m_params.SphereCen[i].x, m_params.SphereCen[i].y, m_params.SphereCen[i].z);
		float srad = m_params.SphereRad[i];

		glPushMatrix();
		glTranslated(scen[0], scen[1], scen[2]);
		glutWireSphere(srad, 32, 32);
		glPopMatrix();
	}
#endif

}





/*!
 * 三角形ポリゴンによる障害物
 * @param[in] vrts 頂点
 * @param[in] tris メッシュ
 */
void rxSPH_GPU::SetPolygonObstacle(const vector<Vec3> &vrts, const vector<Vec3> &nrms, const vector< vector<int> > &tris)
{
	int vn = (int)vrts.size();
	int n = (int)tris.size();
	m_iNumVrts += vn;
	m_iNumTris += n;

	//if(nrms.empty()){
		for(int i = 0; i < vn; ++i){
			for(int j = 0; j < 3; ++j){
				m_vVrts.push_back(vrts[i][j]);
			}
		}
	//}
	//else{
	//	for(int i = 0; i < vn; ++i){
	//		Vec3 v = vrts[i]+nrms[i]*m_fParticleRadius;
	//		for(int j = 0; j < 3; ++j){
	//			m_vVrts.push_back(v[j]);
	//		}
	//	}
	//}

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < 3; ++j){
			m_vTris.push_back(tris[i][j]);
		}
	}

	// GPUメモリの確保と転送
	if(m_dVrts) CuFreeArray(m_dVrts);
	if(m_dTris) CuFreeArray(m_dTris);
	m_dVrts = 0;
	m_dTris = 0;

	CuAllocateArray((void**)&m_dVrts, m_iNumVrts*3*sizeof(RXREAL));
	CuAllocateArray((void**)&m_dTris, m_iNumTris*3*sizeof(int));

	CuCopyArrayToDevice(m_dVrts, &m_vVrts[0], 0, m_iNumVrts*3*sizeof(RXREAL));
	CuCopyArrayToDevice(m_dTris, &m_vTris[0], 0, m_iNumTris*3*sizeof(int));

	RXCOUT << "the number of triangles : " << m_iNumTris << endl;

	//SetParticlesToCell();
	setPolysToCell(&m_vVrts[0], m_iNumVrts, &m_vTris[0], m_iNumTris);
}

/*!
 * ボックス型障害物
 * @param[in] cen ボックス中心座標
 * @param[in] ext ボックスの大きさ(辺の長さの1/2)
 * @param[in] ang ボックスの角度(オイラー角)
 * @param[in] flg 有効/無効フラグ
 */
void rxSPH_GPU::SetBoxObstacle(Vec3 cen, Vec3 ext, Vec3 ang, int flg)
{
	// 障害物
	int b = m_params.BoxNum;

	if(b < MAX_BOX_NUM){
		m_params.BoxCen[b] = MAKE_FLOAT3(cen[0], cen[1], cen[2]);
		m_params.BoxExt[b] = MAKE_FLOAT3(ext[0], ext[1], ext[2]);
		m_params.BoxRot[b] = EulerToMatrix(ang[0], ang[1], ang[2]);
		m_params.BoxInvRot[b] = Inverse(m_params.BoxRot[b]);
		m_params.BoxFlg[b] = flg;
		b++;

		m_params.BoxNum = b;
	}
}

/*!
 * 球型障害物
 * @param[in] cen 球体中心座標
 * @param[in] rad 球体の半径
 * @param[in] flg 有効/無効フラグ
 */
void rxSPH_GPU::SetSphereObstacle(Vec3 cen, double rad, int flg)
{
	// 障害物
	int b = m_params.SphereNum;

	if(b < MAX_SPHERE_NUM){
		m_params.SphereCen[b] = MAKE_FLOAT3(cen[0], cen[1], cen[2]);
		m_params.SphereRad[b] = (RXREAL)rad;
		m_params.SphereFlg[b] = flg;
		b++;

		m_params.SphereNum = b;
	}
}

/*!
 * 球型障害物を動かす
 * @param[in] b 物体番号
 * @param[in] disp 移動量
 */
void rxSPH_GPU::MoveSphereObstacle(int b, Vec3 disp)
{
	// 障害物
	if(b >= (int)m_params.SphereNum) return;

	if(b < MAX_SPHERE_NUM){
		m_params.SphereCen[b].x += disp[0];
		m_params.SphereCen[b].y += disp[1];
		m_params.SphereCen[b].z += disp[2];
	}
}

/*!
 * 球型障害物の位置を取得
 * @param[in] b 物体番号
 */
Vec3 rxSPH_GPU::GetSphereObstaclePos(int b)
{
	if(b >= (int)m_params.SphereNum || (int)m_params.SphereNum == 0) return Vec3(0.0);
	if(b < 0) b = (int)m_params.SphereNum-1;
	return Vec3(m_params.SphereCen[b].x, m_params.SphereCen[b].y, m_params.SphereCen[b].z);
}


/*!
 * VBO,デバイスメモリからホストメモリへデータを転送，取得
 * @param[in] type データの種類
 * @return ホストメモリ上のデータ
 */
RXREAL* rxSPH_GPU::GetArrayVBO(rxParticleArray type, bool d2h)
{
	assert(m_bInitialized);
 
	RXREAL* hdata = 0;
	RXREAL* ddata = 0;

	cudaGraphicsResource **graphics_resource = 0;
	int d = DIM;

	switch(type){
	default:
	case RX_POSITION:
		hdata = m_hPos;
		ddata = m_dPos;
		graphics_resource = &m_pPosResource;
		break;

	case RX_VELOCITY:
		hdata = m_hVel;
		ddata = m_dVel;
		break;

	case RX_DENSITY:
		hdata = m_hDens;
		ddata = m_dDens;
		d = 1;
		break;

	case RX_PRESSURE:
		hdata = m_hPres;
		ddata = m_dPres;
		d = 1;
		break;

	case RX_NORMAL:
		hdata = m_hNrm;
		ddata = m_dNrm;
		break;

	case RX_FORCE:
		hdata = m_hFrc;
		ddata = m_dFrc;
		break;

	case RX_UPDATED_POSITION:
		hdata = m_hUpPos;
		ddata = m_dUpPos;
		d2h = true;
		break;

	case RX_EIGEN_VALUE:
		hdata = m_hEigen;
		ddata = m_dEigen;
		d = 3;
		d2h = true;
		break;

	case RX_ROTATION_MATRIX:
		hdata = m_hRMatrix;
		ddata = m_dRMatrix;
		d = 9;
		d2h = true;
		break;

	case RX_TRANSFORMATION:
		hdata = m_hG;
		ddata = m_dG;
		d = 9;
		d2h = true;
		break;
	}

	if(d2h) CuCopyArrayFromDevice(hdata, ddata, graphics_resource, m_uNumParticles*d*sizeof(RXREAL));

	return hdata;
}

/*!
 * ホストメモリからVBO,デバイスメモリへデータを転送
 * @param[in] type データの種類
 * @param[in] data ホストメモリ上のデータ
 * @param[in] start データの開始インデックス
 * @param[in] count 追加数
 */
void rxSPH_GPU::SetArrayVBO(rxParticleArray type, const RXREAL* data, int start, int count)
{
	assert(m_bInitialized);
 
	switch(type){
	default:
	case RX_POSITION:
		{
			if(m_bUseOpenGL){
				CuUnregisterGLBufferObject(m_pPosResource);
				glBindBuffer(GL_ARRAY_BUFFER, m_posVBO);
				glBufferSubData(GL_ARRAY_BUFFER, start*4*sizeof(RXREAL), count*4*sizeof(RXREAL), data);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				CuRegisterGLBufferObject(m_posVBO, &m_pPosResource);
			}
		}
		break;

	case RX_VELOCITY:
		CuCopyArrayToDevice(m_dVel, data, start*DIM*sizeof(RXREAL), count*DIM*sizeof(RXREAL));
		break;

	case RX_NORMAL:
		CuCopyArrayToDevice(m_dNrm, data, start*DIM*sizeof(RXREAL), count*DIM*sizeof(RXREAL));
		break;
	}	   
}



//-----------------------------------------------------------------------------
// MARK:陰関数値
//-----------------------------------------------------------------------------
double rxSPH_GPU::GetImplicit(double x, double y, double z)
{
	return 0.0;
}

/*!
 * パーティクルからグリッドの陰関数値を計算
 * @param[in] pnx,pny,pnz グリッド数の指数 nx=2^pnx
 * @param[in] minp グリッドの最小座標
 * @param[in] d グリッド幅
 * @param[out] hF 陰関数値(nx×ny×nzの配列)
 */
void rxSPH_GPU::CalImplicitField(int n[3], Vec3 minp, Vec3 d, RXREAL *hF)
{
	unsigned int memSize = sizeof(RXREAL)*n[0]*n[1]*n[2];
	//RXCOUT << memSize/sizeof(RXREAL) << endl;
	
	RXREAL *dF = 0;
	CuAllocateArray((void**)&dF, memSize);

	CuSphGridDensity(dF, m_dCellData, n[0], n[1], n[2], minp[0], minp[1], minp[2], d[0], d[1], d[2]);

	CuCopyArrayFromDevice(hF, dF, 0, memSize);

	if(dF) CuFreeArray(dF);
}


/*!
 * パーティクルからグリッドの陰関数値を計算
 * @param[in] n[3] グリッド数
 * @param[in] minp グリッドの最小座標
 * @param[in] d グリッド幅
 * @param[out] hF 陰関数値(nx×ny×nzの配列)
 */
void rxSPH_GPU::CalImplicitFieldDevice(int n[3], Vec3 minp, Vec3 d, RXREAL *dF)
{
	RXREAL *dPos;
	if(m_bUseOpenGL){
		dPos = (RXREAL*)CuMapGLBufferObject(&m_pPosResource);
	}
	else{
		dPos = (RXREAL*)m_dPos;
	}

	// 分割セルのハッシュを計算
	CuCalcHash(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, dPos, m_uNumParticles);

	// ハッシュに基づきパーティクルをソート
	CuSort(m_dCellData.dGridParticleHash, m_dCellData.dSortedIndex, m_uNumParticles);

	// パーティクル配列をソートされた順番に並び替え，
	// 各セルの始まりと終わりのインデックスを検索
	CuReorderDataAndFindCellStart(m_dCellData, dPos, m_dVel);

	if(m_bCalAnisotropic){
		// パーティクル密度を用いたボリュームデータ(異方性カーネル)
		CuSphGridDensityAniso(dF, m_dG, m_fEigenMax, m_dCellData, n[0], n[1], n[2], minp[0], minp[1], minp[2], d[0], d[1], d[2]);
	}
	else{
		// パーティクル密度を用いたボリュームデータ
		CuSphGridDensity(dF, m_dCellData, n[0], n[1], n[2], minp[0], minp[1], minp[2], d[0], d[1], d[2]);
	}

	if(m_bUseOpenGL){
		CuUnmapGLBufferObject(m_pPosResource);
	}
}





//-----------------------------------------------------------------------------
// MARK:シミュデータの出力
//-----------------------------------------------------------------------------
/*!
 * シミュレーション設定(パーティクル数，範囲，密度，質量など)
 * @param[in] fn 出力ファイル名
 */
void rxSPH_GPU::OutputSetting(string fn)
{
	ofstream fout;
	fout.open(fn.c_str());
	if(!fout){
		RXCOUT << fn << " couldn't open." << endl;
		return;
	}

	fout << m_uNumParticles << endl;
	fout << m_params.BoundaryMin.x << " " << m_params.BoundaryMin.y << " " << m_params.BoundaryMin.z << endl;
	fout << m_params.BoundaryMax.x << " " << m_params.BoundaryMax.y << " " << m_params.BoundaryMax.z << endl;
	fout << m_params.Density << endl;
	fout << m_params.Mass    << endl;
	fout << m_params.KernelParticles << endl;

	fout.close();
}
