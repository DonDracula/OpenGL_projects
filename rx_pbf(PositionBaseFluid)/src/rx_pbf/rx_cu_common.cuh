/*! 
  @file rx_cu_common.cuh
	
  @brief CUDA共通ヘッダ
 
  @author Makoto Fujisawa
  @date 2009-08, 2011-06
*/
// FILE --rx_cu_common.cuh--

#ifndef _RX_CU_COMMON_CUH_
#define _RX_CU_COMMON_CUH_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "vector_types.h"
#include "vector_functions.h"


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------


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
#define THREAD_NUM 512

// サンプルボリュームを用いる場合1, 陰関数を用いる場合は0
#define SAMPLE_VOLUME 1

// Shared Memoryの使用フラグ
#define USE_SHARED 1

// Shared Memoryを用いる場合のスレッド数の制限
#define NTHREADS 32

#define SKIP_EMPTY_VOXELS 1



// 行列
struct matrix3x3
{
	float3 e[3];
};

struct matrix4x4
{
	float4 e[4];
};


#define MAX_POLY_NUM 10
#define MAX_BOX_NUM 10
#define MAX_SPHERE_NUM 10

#define DEBUG_N 17


//-----------------------------------------------------------------------------
//! SPHシミュレーションパラメータ
//-----------------------------------------------------------------------------
struct rxSimParams
{
	FLOAT3 Gravity;
	FLOAT ParticleRadius;

	float3 Boundary[2];

	uint3 GridSize;
	uint NumCells;
	FLOAT3 WorldOrigin;
	FLOAT3 WorldMax;
	FLOAT3 CellWidth;

	uint NumBodies;
	uint MaxParticlesPerCell;

	uint3 GridSizeB;
	uint NumCellsB;
	FLOAT3 WorldOriginB;
	FLOAT3 WorldMaxB;
	FLOAT3 CellWidthB;
	uint NumBodiesB;


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

	FLOAT Wpoly6r;		//!< Pory6カーネルの定数係数(最大値が1になるように正規化した係数)

	int AP;
	FLOAT AP_K;
	FLOAT AP_N;
	FLOAT AP_Q;
	FLOAT AP_WQ;

	uint   PolyNum;
	FLOAT3 PolyVel[MAX_POLY_NUM];

	uint   BoxNum;
#if MAX_BOX_NUM
	FLOAT3 BoxCen[MAX_BOX_NUM];
	FLOAT3 BoxExt[MAX_BOX_NUM];
	//FLOAT3 BoxVel[MAX_BOX_NUM];
	matrix3x3 BoxRot[MAX_BOX_NUM];
	matrix3x3 BoxInvRot[MAX_BOX_NUM];
	uint   BoxFlg[MAX_BOX_NUM];
#endif

	uint SphereNum;
#if MAX_SPHERE_NUM
	FLOAT3 SphereCen[MAX_SPHERE_NUM];
	FLOAT  SphereRad[MAX_SPHERE_NUM];
	//FLOAT3 SphereVel[MAX_SPHERE_NUM];
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





#endif // _RX_CU_COMMON_CUH_
