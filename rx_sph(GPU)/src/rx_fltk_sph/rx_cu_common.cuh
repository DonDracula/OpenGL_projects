/*! 
  @file rx_cu_common.cuh
	
  @brief CUDA共通ヘッダ
 
*/
// FILE --rx_cu_common.cuh--

#ifndef _RX_CU_COMMON_CUH_
#define _RX_CU_COMMON_CUH_

#pragma warning (disable: 4819)

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "vector_types.h"
#include "vector_functions.h"


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
#define RX_CUCHECK checkCudaErrors
#define RX_CUERROR getLastCudaError


typedef unsigned int uint;
typedef unsigned char uchar;

#define FLOAT float
#define FLOAT3 float3

#define MAKE_FLOAT3 make_float3

#define RX_CUMC_USE_GEOMETRY

#define RX_USE_ATOMIC_FUNC // 要Compute capability 1.1以上(-arch sm_11)


// テクスチャメモリの使用フラグ
#ifndef __DEVICE_EMULATION__
#define USE_TEX 0
#endif

#if USE_TEX
#define FETCH(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCH(t, i) t[i]
#endif

#if USE_TEX
#define FETCHC(t, i) tex1Dfetch(t##Tex, i)
#else
#define FETCHC(t, i) cell.t[i]
#endif


// 1ブロックあたりのスレッド数(/次元)
#define BLOCK_SIZE 16

// 1ブロックあたりの最大スレッド数
#define THREAD_NUM 256

// サンプルボリュームを用いる場合1, 陰関数を用いる場合は0
#define SAMPLE_VOLUME 1

// Shared Memoryの使用フラグ
#define USE_SHARED 1

// Shared Memoryを用いる場合のスレッド数の制限
#define NTHREADS 32

#define SKIP_EMPTY_VOXELS 1

// CWT
#define M_PI 3.141592653589793238462643383279502884197
#define M_2PI 6.283185307179586476925286766559005768394		// 2*PI
#define M_SQRTPI 1.772453850905516027298167483341145182797	// sqrt(PI)
#define M_SQRT2PI 2.506628274631000502415765284811045253006	// sqrt(2*PI)

#define MEXICAN_HAT_FC (1.0/M_PI)
#define MEXICAN_HAT_C 0.8673250705840776f // c =  2/(sqrt(3)*pi^(1/4))
#define MEXICAN_HAT_R 5.0f

#define RX_MAX_FILTER_SIZE 10
#define RX_BINOMIALS_SIZE (RX_MAX_FILTER_SIZE+1)*(RX_MAX_FILTER_SIZE+1)

//サブパーティクルに関する定数
#define MAX_SUB_LEVEL (3)
#define MAX_SUB_NUM (15) //2^(MAX_SUB_LEVEL+1)-1
#define RATIO_AXIS_ANGLE (0.50) //軸のぶれに関する比例定数
#define POW_2_M1D3 (0.793700526) //pow(2.0,-(1.0/3.0))
#define POW_2_M5D3 (0.314980262) //pow(2.0,-(5.0/3.0))
#define POW_2_M5D6 (0.561231024) //pow(2.0,-(5.0/6.0))
#define POW_2_M5D9 (0.680395000) //pow(2.0,-(5.0/9.0))
#define INV_LOG_2_M5D9 (-5.979470571) // 1/log(2^(-5/9));
//#define FLT_MAX         3.402823466e+38F        // max value 


// メルセンヌツイスター
#define      DCMT_SEED 4172
#define  MT_RNG_PERIOD 607

typedef struct{
	unsigned int matrix_a;
	unsigned int mask_b;
	unsigned int mask_c;
	unsigned int seed;
} mt_struct_stripped;

#define   MT_RNG_COUNT 4096
#define          MT_MM 9
#define          MT_NN 19
#define       MT_WMASK 0xFFFFFFFFU
#define       MT_UMASK 0xFFFFFFFEU
#define       MT_LMASK 0x1U
#define      MT_SHIFT0 12
#define      MT_SHIFTB 7
#define      MT_SHIFTC 15
#define      MT_SHIFT1 18

// 行列
struct matrix3x3
{
	float3 e[3];
};

struct matrix4x4
{
	float4 e[4];
};


#define MAX_BOX_NUM 10
#define MAX_SPHERE_NUM 10


//-----------------------------------------------------------------------------
//! SPHシミュレーションパラメータ
//-----------------------------------------------------------------------------
struct rxSimParams
{
	FLOAT3 Gravity;
	FLOAT GlobalDamping;
	FLOAT ParticleRadius;

	float3 BoundaryMax;
	float3 BoundaryMin;

	uint3 GridSize;
	uint NumCells;
	FLOAT3 WorldOrigin;
	FLOAT3 WorldMax;
	FLOAT3 CellWidth;

	uint NumBodies;
	uint MaxParticlesPerCell;

	FLOAT EffectiveRadius;
	FLOAT Mass;			// パーティクル質量[kg]
	FLOAT VorticityConfinement;

	FLOAT Buoyancy;

	FLOAT Density;		// 密度[kg/m^3]
	FLOAT Pressure;     // [Pa = N/m^2 = kg/m.s^2]

	FLOAT Tension;		// [N/m = kg/s^2]
	FLOAT Viscosity;	// [Pa.s = N.s/m^2 = kg/m.s]
	FLOAT GasStiffness;	// [J = N.m = kg.m^2/s^2]  // used for DC96 symmetric pressure force

	FLOAT Volume;
	FLOAT KernelParticles;
	FLOAT Restitution;

	FLOAT Threshold;

	FLOAT InitDensity;

	FLOAT Dt;

	FLOAT Wpoly6;		//!< Pory6カーネルの定数係数
	FLOAT GWpoly6;		//!< Pory6カーネルの勾配の定数係数
	FLOAT LWpoly6;		//!< Pory6カーネルのラプラシアンの定数係数
	FLOAT Wspiky;		//!< Spikyカーネルの定数係数
	FLOAT GWspiky;		//!< Spikyカーネルの勾配の定数係数
	FLOAT LWspiky;		//!< Spikyカーネルのラプラシアンの定数係数
	FLOAT Wvisc;		//!< Viscosityカーネルの定数係数
	FLOAT GWvisc;		//!< Viscosityカーネルの勾配の定数係数
	FLOAT LWvisc;		//!< Viscosityカーネルのラプラシアンの定数係数

	FLOAT Wd2;		//!< 
	FLOAT Wd3;		//!< 
	FLOAT GWd2;		//!< 
	FLOAT GWd3;		//!< 
	FLOAT Wd1;		//!< 

	FLOAT NoiseScl;		//!< ノイズ付加時のスケール
	FLOAT NoiseEthr;	//!< ノイズ付加時のエネルギースペクトル閾値
	FLOAT NoiseMag;		//!< ノイズ付加時のエネルギースペクトル係数

	uint   BoxNum;
#if MAX_BOX_NUM
	FLOAT3 BoxCen[MAX_BOX_NUM];
	FLOAT3 BoxExt[MAX_BOX_NUM];
	matrix3x3 BoxRot[MAX_BOX_NUM];
	matrix3x3 BoxInvRot[MAX_BOX_NUM];
	uint   BoxFlg[MAX_BOX_NUM];
#endif

	uint SphereNum;
#if MAX_SPHERE_NUM
	FLOAT3 SphereCen[MAX_SPHERE_NUM];
	FLOAT  SphereRad[MAX_SPHERE_NUM];
	uint   SphereFlg[MAX_SPHERE_NUM];
#endif
};

struct rxParticleCell
{
	float4* dSortedPos;			//!< ソート済みパーティクル座標
	float4* dSortedVel;			//!< ソート済みパーティクル速度

	uint* dSortedIndex;			//!< ソート済みパーティクルインデックス
	uint* dGridParticleHash;	//!< 各パーティクルのグリッドハッシュ値(ソート用キー)
	uint* dCellStart;			//!< ソートリスト内の各セルのスタートインデックス
	uint* dCellEnd;				//!< ソートリスト内の各セルのエンドインデックス
	uint  uNumParticles;		//!< 総パーティクル数
	uint  uNumCells;			//!< 総セル数
	uint  uNumArdGrid;			//!< 近傍探索時参照グリッド範囲

	uint* dSortedPolyIdx;		//!< ソート済みポリゴンインデックス
	uint* dGridPolyHash;		//!< ポリゴンのグリッドハッシュ値(ソート用キー)
	uint* dPolyCellStart;		//!< ソートリスト内の各セルのスタートインデックス
	uint* dPolyCellEnd;			//!< ソートリスト内の各セルのエンドインデックス
	uint  uNumPolyHash;
};


//-----------------------------------------------------------------------------
//! サブパーティクル
//-----------------------------------------------------------------------------
struct rxSubParticleCell
{
	float4* dSubUnsortPos;	//!< サブパーティクル座標
	float*	dSubUnsortRad;	//!< サブパーティクル半径
	float*	dSubUnsortRat;	//!< サブパーティクル影響係数
	float4* dSubSortedPos;	//!< サブパーティクル座標(グリッドハッシュでソート)
	float*	dSubSortedRad;	//!< サブパーティクル半径(グリッドハッシュでソート)
	float*	dSubSortedRat;	//!< サブパーティクル影響係数(グリッドハッシュでソート)

	uint*   dSubOcc;		//!< サブパーティクル有効/無効
	uint*   dSubOccScan;	//!< サブパーティクル有効/無効のScan

	uint*	dSubSortedIndex;
	uint*	dSubGridParticleHash;	//!< 各サブパーティクルのグリッドハッシュ値
	uint*	dSubCellStart;
	uint*	dSubCellEnd;

	uint	uSubNumAllParticles;	//!< 表面生成に必要なサブパーティクル数
	uint	uSubNumMCParticles;		//!< 表面生成に必要なサブパーティクル数
	uint    uSubNumValidParticles;	//!< レンダリング時に有効なサブパーティクル数
	uint	uSubNumCells;
	uint	uSubNumArdGrid;			//!< 近傍探索時参照グリッド範囲
	uint	uSubMaxLevel;
	uint	uNumParticles;		//!< レベル0のパーティクル数
	uint	uMaxParticles;		//!< レベル0の最大パーティクル数
	uint	uSubHeadIndex[MAX_SUB_LEVEL+1];
	//uint	uSubNumLevel[MAX_SUB_LEVEL+1];
	uint	uSubNumEach[MAX_SUB_LEVEL+1];
	float	fSubRad[MAX_SUB_LEVEL+1];
	float	fSubEffectiveRadius[MAX_SUB_LEVEL+1];
	float	fSubEffectiveFactor;
	float	fEtcri;

	//レベル別表面生成用
	/*float*	dSubSelectRat[MAX_SUB_LEVEL+1];
	float4* dSubSelectPos[MAX_SUB_LEVEL+1];
	uint*	dSubSelectSortedIndex[MAX_SUB_LEVEL+1];
	uint*	dSubSelectCellStart[MAX_SUB_LEVEL+1];
	uint*	dSubSelectCellEnd[MAX_SUB_LEVEL+1];
	uint	uSubSelectNum[MAX_SUB_LEVEL+1];*/
};



//-----------------------------------------------------------------------------
// SSM
//-----------------------------------------------------------------------------
struct rxSsmParams
{
	float PrtRad;		//!< パーティクルの半径
	float4 PMV[4];		//!< 透視投影行列Pとモデルビュー行列MVを掛けた行列
	float3 Tr;			//!< 半径の投影変換
	uint W, H;			//!< 描画領域解像度
	float Spacing;		//!< デプスマップのサンプリング間隔
	float Zmax;			//!< 輪郭となるデプス差の閾値
	int Nfilter;		//!< デプス値平滑化のフィルタサイズ
	int Niters;			//!< 輪郭平滑化の反復回数
	int Ngx, Ngy;		//!< メッシュ生成用グリッドの解像度
};

//! グリッドエッジ
struct rxSSEdgeG
{
	float3 x0, x1;		//!< 端点座標とデプス値
	float depth;		//!< エッジデプス値
	int front_vertex;	//!< エッジ頂点のインデックス
	float dx;			//!< デプス値が小さい端点からエッジ頂点までの距離
	int silhouette;
};


//! メッシュ生成用グリッド
struct rxSSGridG
{
	int i, j;
	int node_vrts[4];	//!< ノード頂点インデックス
	int num_nv;			//!< ノード頂点数
	int edge_vrts[4];	//!< エッジ頂点(front vertex)
	int num_ev;			//!< エッジ頂点数(front vertex)
	int back_vrts[6];	//!< エッジ頂点(back vertex, back-2 vertex)
	int num_bv;			//!< エッジ頂点数(back vertex)

	float node_depth[4];	//!< ノードのデプス値
	int vrot;

	int table_index0;	//!< デバッグ用:メッシュ化のためのインデックス値
	int table_index1;	//!< デバッグ用:メッシュ化のためのインデックス値
	int mesh_num;		//!< デバッグ用:メッシュ数
	int mesh[6];		//!< デバッグ用:メッシュインデックス
	int back2;			//!< デバッグ用
	int v[14];
};



//-----------------------------------------------------------------------------
//! パックするためのデータ構造 
//-----------------------------------------------------------------------------
template<class T> 
struct rxVPack
{
	T* dPos;
	T* dCompactedPos;
	uint* dOcc;
	uint* dOccScan;

	rxVPack()
	{
		dPos = 0;
		dCompactedPos = 0;
		dOcc = 0;
		dOccScan = 0;
	}
};

typedef rxVPack<float> rxVPackf;
typedef rxVPack<rxSSEdgeG> rxVPacke;

struct rxVrtAdd
{
	int num;
	int layer;
	int edge[2];
	float3 vrts[2];
};



#endif // _RX_CU_COMMON_CUH_
