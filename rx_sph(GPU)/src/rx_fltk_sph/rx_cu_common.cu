/*! 
  @file rx_cu_common.cu
	
  @brief CUDA共通デバイス関数
 
*/
// FILE --rx_cu_common.cu--

#ifndef _RX_CU_COMMON_CU_
#define _RX_CU_COMMON_CU_

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>

#include "helper_math.h"
#include <math_constants.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include "rx_cu_common.cuh"



//-----------------------------------------------------------------------------
// 関数
//-----------------------------------------------------------------------------
__device__ __host__
inline uint calUintPow(uint x, uint y)
{
	uint x_y = 1;
	for(uint i=0; i < y;i++) x_y *= x;
	return x_y;
}

/*!
 * a/bの計算結果を切り上げ
 * @param[in] a,b a/b
 * @return 切り上げた除算結果
 */
__device__ __host__
inline uint DivCeil(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}


/*!
 * [a,b]にクランプ
 * @param[in] x クランプしたい数値
 * @param[in] a,b クランプ境界
 * @return クランプされた数値
 */
__device__
inline float CuClamp(float x, float a, float b)
{
	return max(a, min(b, x));
}
__device__
inline int CuClamp(int x, int a, int b)
{
	return max(a, min(b, x));
}

/*!
 * ゼロ判定 for float3
 * @param[in] v 値
 */
__device__
inline int CuIsZero(float3 v)
{
	if(fabsf(v.x) < 1.0e-10 && fabsf(v.y) < 1.0e-10 && fabsf(v.z) < 1.0e-10){
		return 1;
	}
	else{
		return 0;
	}
}

/*!
 * 行列とベクトルの積
 * @param[in] m 3x3行列
 * @param[in] v 3Dベクトル
 * @return 積の結果
 */
__device__
inline float3 CuMulMV(matrix3x3 m, float3 v)
{
	return make_float3(dot(m.e[0], v), dot(m.e[1], v), dot(m.e[2], v));
}



//-----------------------------------------------------------------------------
// アトミック関数
//-----------------------------------------------------------------------------
#ifdef RX_USE_ATOMIC_FUNC

/*!
 * float版atomicAdd
 */
__device__ 
inline void atomicFloatAdd(float *address, float val)
{
	int i_val = __float_as_int(val);
	int tmp0 = 0;
	int tmp1;
 
	while( (tmp1 = atomicCAS((int *)address, tmp0, i_val)) != tmp0)
	{
		tmp0 = tmp1;
		i_val = __float_as_int(val + __int_as_float(tmp1));
	}
}
/*!
 * double版atomicAdd
 */
__device__ 
inline double atomicDoubleAdd(double *address, double val)
{
	unsigned long long int *address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do{
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
	}while(assumed != old);
	return __longlong_as_double(old);
}
/*!
 * float版atomicMin
 */
__device__ 
inline float atomicFloatMin(float *address, float val)
{
	int *address_as_int = (int*)address;
	int old = atomicMin(address_as_int, __float_as_int(val));
	return __int_as_float(old);
}

/*!
 * float版atomicMax
 */
__device__ 
inline float atomicFloatMax(float *address, float val)
{
	int *address_as_int = (int*)address;
	int old = atomicMax(address_as_int, __float_as_int(val));
	return __int_as_float(old);
}

#endif // #ifdef RX_USE_ATOMIC_FUNC


//-----------------------------------------------------------------------------
// グリッド
//-----------------------------------------------------------------------------
/*!
 * 1Dインデックスから3Dインデックスへの変換(グリッド数は任意)
 * @param[in] i 1Dインデックス
 * @param[in] gridSize グリッド数
 * @return 3Dインデックス
 */
__device__
inline uint3 calcGridPosU(uint i, uint3 ngrid)
{
	uint3 gridPos;
	uint w = i%(ngrid.x*ngrid.y);
	gridPos.x = w%ngrid.x;
	gridPos.y = w/ngrid.x;
	gridPos.z = i/(ngrid.x*ngrid.y);
	return gridPos;
}
/*!
 * 3Dインデックスから1Dインデックスへの変換(グリッド数は任意)
 * @param[in] p 3Dインデックス
 * @param[in] gridSize グリッド数
 * @return 1Dインデックス
 */
__device__
inline uint calcGridPos3(uint3 p, uint3 ngrid)
{
	p.x = min(p.x, ngrid.x-1);
	p.y = min(p.y, ngrid.y-1);
	p.z = min(p.z, ngrid.z-1);
	return (p.z*ngrid.x*ngrid.y)+(p.y*ngrid.x)+p.x;
}



//-----------------------------------------------------------------------------
// CWTデバイス関数
//-----------------------------------------------------------------------------
/*!
 * メキシカンハット
 * @param[in] t 座標
 * @return ウェーブレット母関数値
 */
__device__
inline float MexicanHat(float t)
{
	t = t*t;
	return MEXICAN_HAT_C*(1.0-t)*exp(-t/2.0);
}
__device__
inline float MexicanHatIm(float t)
{
	return 0.0f;
}

/*!
 * メキシカンハット(波数空間)
 * @param[in] w 波数
 * @return ウェーブレット母関数値
 */
__device__
inline float MexicanHatWave(float w)
{
	w = w*w;
	return MEXICAN_HAT_C*M_SQRT2PI*w*exp(-w/2.0);
}
inline float MexicanHatWaveIm(float w)
{
	return 0.0f;
}

/*!
 * メキシカンハット(2D)
 * @param[in] x,y 座標
 * @return ウェーブレット母関数値
 */
__device__
inline float MexicanHat2D(float x, float y)
{
	x = x*x;
	y = y*y;
	return MEXICAN_HAT_C*(x+y-2)*exp(-(x+y)/2.0);
}
__device__
inline float MexicanHat2DIm(float x, float y)
{
	return 0.0f;
}

/*!
 * メキシカンハット(3D)
 * @param[in] x,y 座標
 * @return ウェーブレット母関数値
 */
__device__ __host__
inline float MexicanHat3D(float x, float y, float z)
{
	x = x*x;
	y = y*y;
	z = z*z;
	return MEXICAN_HAT_C*(x+y+z-3.0f)*exp(-(x+y+z)/2.0f);
}
__device__ __host__
inline float MexicanHat3DIm(float x, float y)
{
	return 0.0f;
}

__device__
inline int Mod(int x, int n)
{
	int m = (int)fmodf((float)x, (float)n); 
	return ((m < 0) ? m+n : m);
}


//-----------------------------------------------------------------------------
// 乱数
//-----------------------------------------------------------------------------
__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];

/*!
 * Mersenne Twister による乱数生成 (CUDAサンプルより)
 * @param[out] d_Random 乱数生成結果
 * @param[in] NPerRng 生成数
 */
__global__
static void RandomGPU(float *d_Random, int NPerRng)
{
	const int	  tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int THREAD_N = blockDim.x * gridDim.x;

	int iState, iState1, iStateM, iOut;
	unsigned int mti, mti1, mtiM, x;
	unsigned int mt[MT_NN];

	for(int iRng = tid; iRng < MT_RNG_COUNT; iRng += THREAD_N){
		//Load bit-vector Mersenne Twister parameters
		mt_struct_stripped config = ds_MT[iRng];

		//Initialize current state
		mt[0] = config.seed;
		for(iState = 1; iState < MT_NN; iState++)
			mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

		iState = 0;
		mti1 = mt[0];
		for(iOut = 0; iOut < NPerRng; iOut++){
			//iState1 = (iState +	 1) % MT_NN
			//iStateM = (iState + MT_MM) % MT_NN
			iState1 = iState + 1;
			iStateM = iState + MT_MM;
			if(iState1 >= MT_NN) iState1 -= MT_NN;
			if(iStateM >= MT_NN) iStateM -= MT_NN;
			mti  = mti1;
			mti1 = mt[iState1];
			mtiM = mt[iStateM];

			x	= (mti & MT_UMASK) | (mti1 & MT_LMASK);
			x	=  mtiM ^ (x >> 1) ^ ((x & 1) ? config.matrix_a : 0);
			mt[iState] = x;
			iState = iState1;

			//Tempering transformation
			x ^= (x >> MT_SHIFT0);
			x ^= (x << MT_SHIFTB) & config.mask_b;
			x ^= (x << MT_SHIFTC) & config.mask_c;
			x ^= (x >> MT_SHIFT1);

			//Convert to (0, 1] float and write to global memory
			d_Random[iRng + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
		}
	}
}


// 線型合同法による乱数生成(C言語などと同じ)
__device__ static unsigned int randx = 1;

__device__
inline void Srand(unsigned int s)
{
	randx = s;
}

__device__
inline unsigned int Rand()
{
	randx = randx*1103515245+12345;
	return randx&2147483647;
}

__device__
inline unsigned int Rand2(unsigned int x)
{
	x = x*1103515245+12345;
	return x&2147483647;
}

#define RAND2_MAX (2147483647)



// XORShiftによる乱数
__device__ static unsigned long xors_x = 123456789;
__device__ static unsigned long xors_y = 362436069;
__device__ static unsigned long xors_z = 521288629;
__device__ static unsigned long xors_w = 88675123;

/*!
 * G. Marsaglia, "Xorshift RNGs", Journal of Statistical Software, Vol. 8(14), pp.1-6, 2003. 
 *  - http://www.jstatsoft.org/v08/i14/
 * @param[in] 
 * @return 
 */
__device__
inline unsigned long Xorshift128()
{ 
	unsigned long t; 
	t = (xors_x^(xors_x<<11));
	xors_x = xors_y; xors_y = xors_z; xors_z = xors_w; 
	return ( xors_w = (xors_w^(xors_w>>19))^(t^(t>>8)) ); 
}
__device__
inline long Xorshift128(long l, long h)
{ 
	unsigned long t; 
	t = (xors_x^(xors_x<<11));
	xors_x = xors_y; xors_y = xors_z; xors_z = xors_w; 
	xors_w = (xors_w^(xors_w>>19))^(t^(t>>8));
	return l+(xors_w%(h-l));
}


__device__
inline float XorFrand(float l, float h)
{
	return l+(h-l)*(Xorshift128(0, 1000000)/1000000.0f);
}

__device__
inline void Random(float2 &x, float a, float b)
{
	x.x = XorFrand(a, b);
	x.y = XorFrand(a, b);
}

__device__
inline void Random(float3 &x, float a, float b)
{
	x.x = XorFrand(a, b);
	x.y = XorFrand(a, b);
	x.z = XorFrand(a, b);
}

// ガウスノイズ
__device__
inline float GaussianNoise(void)
{
	float x1, x2;
	float ret;
	float r2;

	do {
		x1 = 2.0 * XorFrand(0.0, 1.0-(1e-10)) - 1.0;	/* [-1, 1) */
		x2 = 2.0 * XorFrand(0.0, 1.0-(1e-10)) - 1.0;

		r2 = x1*x1 + x2*x2;

	} while ((r2 == 0) || (r2 > 1.0));

	ret = x1 * sqrtf((-2.0 * logf(r2))/r2);
	ret *= 0.25;		// Possibility of ( N(0, 1) < 4.0 ) = 100%

	if (ret < -1.0) ret = -1.0; /* Account for loss of precision. */
	if (ret >  1.0) ret = 1.0;

	return ret;
}




//-----------------------------------------------------------------------------
// 交差判定
//-----------------------------------------------------------------------------

/*!
 * 線分と円の交差判定(2D, Aに)
 * @param[in] A,B 線分の両端点座標
 * @param[in] C 円の中心
 * @param[in] r 円の半径
 * @param[out] P 交点座標
 * @return 交点数
 */
__device__ 
static int CuLineCircleIntersection(float2 A, float2 B, float2 C, float r, float2 P[2], float t[2])
{
	float rr = r*r;
	float2 AC = C-A;
	float2 BC = C-B;

	float2 v = B-A;
	float l = length(v);
	v /= l;

	float td = dot(v, AC);
	float2 D = A+td*v;
	float dd = dot(D-C, D-C);

	if(dd < rr){
		float dt = sqrtf(rr-dd);

		float da = rr-dot(AC, AC);
		float db = rr-dot(BC, BC);

		int inter = 0;
		float t1 = td-dt;
		float t2 = td+dt;
		if(t1 >= 0 && t1 <= l){
			P[inter] = A+t1*v;
			t[inter] = t1;
			inter++;
		}
		if(t2 >= 0 && t2 <= l){
			P[inter] = A+t2*v;
			t[inter] = t2;
			inter++;
		}

		return inter;
	}
	else{
		return 0;
	}
}


/*!
 * AABBと球の距離
 * @param[in] spos 球中心
 * @param[in] r 球半径
 * @param[in] sgn
 * @param[in] box_min,box_max AABB最小，最大座標値
 * @param[out] cp AABB表面の最近傍点
 * @param[out] d 旧都とAABBの距離
 * @param[out] n 交点における単位法線ベクトル
 */
__device__
inline int collisionSphereAABB(float3 spos, float r, int sgn, float3 box_min, float3 box_max, float3 &cp, float &d, float3 &n)
{
	float3 dist_min;	// box_minとの距離
	float3 dist_max;	// box_maxとの距離
	float d0 = 0.0f;
	float3 n0 = make_float3(0.0f, 0.0f, 0.0f);
	int bout = 0;
	int count = 0;

	// 各軸ごとに最小と最大境界外になっていないか調べる
	if((dist_min.x = (spos.x-r)-box_min.x) < 0.0){ bout |= 0x0001; count++; d0 = dist_min.x; n0 = make_float3( 1.0,  0.0,  0.0);}
	if((dist_min.y = (spos.y-r)-box_min.y) < 0.0){ bout |= 0x0002; count++; d0 = dist_min.y; n0 = make_float3( 0.0,  1.0,  0.0);}
	if((dist_min.z = (spos.z-r)-box_min.z) < 0.0){ bout |= 0x0004; count++; d0 = dist_min.z; n0 = make_float3( 0.0,  0.0,  1.0);}
	if((dist_max.x = box_max.x-(spos.x+r)) < 0.0){ bout |= 0x0008; count++; d0 = dist_max.x; n0 = make_float3(-1.0,  0.0,  0.0);}
	if((dist_max.y = box_max.y-(spos.y+r)) < 0.0){ bout |= 0x0010; count++; d0 = dist_max.y; n0 = make_float3( 0.0, -1.0,  0.0);}
	if((dist_max.z = box_max.z-(spos.z+r)) < 0.0){ bout |= 0x0020; count++; d0 = dist_max.z; n0 = make_float3( 0.0,  0.0, -1.0);}

	// 立方体内(全軸で境界内)
	if(bout == 0){
		float min_d = 1e10;
		if(dist_min.x < min_d){ min_d = dist_min.x; n = make_float3( 1.0,  0.0,  0.0); }
		if(dist_min.y < min_d){ min_d = dist_min.y; n = make_float3( 0.0,  1.0,  0.0); }
		if(dist_min.z < min_d){ min_d = dist_min.z; n = make_float3( 0.0,  0.0,  1.0); }

		if(dist_max.x < min_d){ min_d = dist_max.x; n = make_float3(-1.0,  0.0,  0.0); }
		if(dist_max.y < min_d){ min_d = dist_max.y; n = make_float3( 0.0, -1.0,  0.0); }
		if(dist_max.z < min_d){ min_d = dist_max.z; n = make_float3( 0.0,  0.0, -1.0); }

		d = (float)sgn*min_d;
		n *= (float)sgn;
		cp = spos+n*fabs(d);
		return 1;
	}

	// 立方体外
	// sgn = 1:箱，-1:オブジェクト
	if(count == 1){
		// 平面近傍
		d = (float)sgn*d0;
		n = (float)sgn*n0;
		cp = spos+n*fabs(d);
	}
	else{
		// エッジ/コーナー近傍
		float3 x = make_float3(0.0f, 0.0f, 0.0f);
		if(bout & 0x0001) x.x =  dist_min.x;
		if(bout & 0x0002) x.y =  dist_min.y;
		if(bout & 0x0004) x.z =  dist_min.z;
		if(bout & 0x0008) x.x = -dist_max.x;
		if(bout & 0x0010) x.y = -dist_max.y;
		if(bout & 0x0020) x.z = -dist_max.z;

		d = length(x);
		n = normalize(x);

		d *= -(float)sgn;
		n *= -(float)sgn;

		cp = spos+n*fabs(d);

		float3 disp = make_float3(0.00001);
		//Random(disp, 0, 0.00001);
		disp = disp*n;
		cp += disp;
	}

	return 0;
}


/*!
 * AABBと点の距離
 * @param[in] p 点座標
 * @param[in] box_cen AABBの中心
 * @param[in] box_ext AABBの各辺の長さの1/2
 * @param[out] cp AABB表面の最近傍点
 * @param[out] d 旧都とAABBの距離
 * @param[out] n 交点における単位法線ベクトル
 */
__device__
inline int collisionPointAABB(float3 p, float3 box_cen, float3 box_ext, float3 &cp, float &d, float3 &n)
{
	cp = p-box_cen;

	float3 tmp = fabs(cp)-box_ext;
	float res = ((tmp.x > tmp.y && tmp.x > tmp.z) ? tmp.x : (tmp.y > tmp.z ? tmp.y : tmp.z));

	float sgn = (res > 0.0) ? -1.0 : 1.0;

	int coli = 0;
	n = make_float3(0.0f);

	if(cp.x > box_ext.x){
		cp.x = box_ext.x;
		n.x -= 1.0;
		coli++;
	}
	else if(cp.x < -box_ext.x){
		cp.x = -box_ext.x;
		n.x += 1.0;
		coli++;
	}

	if(cp.y > box_ext.y){
		cp.y = box_ext.y;
		n.y -= 1.0;
		coli++;
	}
	else if(cp.y < -box_ext.y){
		cp.y = -box_ext.y;
		n.y += 1.0;
		coli++;
	}

	if(cp.z > box_ext.z){
		cp.z = box_ext.z;
		n.z -= 1.0;
		coli++;
	}
	else if(cp.z < -box_ext.z){
		cp.z = -box_ext.z;
		n.z += 1.0;
		coli++;
	}

	n = normalize(n);

	//if(coli > 1){
	//	float3 disp;
	//	Random(disp, 0, 0.00001);
	//	disp = disp*n;
	//	cp += disp;
	//}

	cp += box_cen;
	d = sgn*length(cp-p);

	return 0;
}


/*!
 * 点とBOXの距離
 * @param[in] p 点座標
 * @param[in] box_cen BOXの中心
 * @param[in] box_ext BOXの各辺の長さの1/2
 * @param[in] box_rot BOXの方向行列(3x3回転行列)
 * @param[in] box_inv_rot BOXの方向行列の逆行列(3x3)
 * @param[out] cp BOX表面の最近傍点
 * @param[out] d 点とBOXの距離
 * @param[out] n 交点における単位法線ベクトル
 */
__device__
inline int collisionPointBox(float3 p, float3 box_cen, float3 box_ext, matrix3x3 box_rot, matrix3x3 box_inv_rot, float3 &cp, float &d, float3 &n)
{
	cp = p-box_cen;
	cp = CuMulMV(box_rot, cp);

	float3 tmp = fabs(cp)-box_ext;

	int coli = 0;
	n = make_float3(0.0f);

	if(tmp.x < 0.0 && tmp.y < 0.0 && tmp.z < 0.0){
		tmp = fabs(tmp);

		if(tmp.x <= tmp.y && tmp.x <= tmp.z){	// x平面に近い
			if(cp.x > 0){
				cp.x = box_ext.x;
				n.x += 1.0;
			}
			else{
				cp.x = -box_ext.x;
				n.x -= 1.0;
			}
		}
		else if(tmp.y <= tmp.x && tmp.y <= tmp.z){ // y平面に近い
			if(cp.y > 0){
				cp.y = box_ext.y;
				n.y += 1.0;
			}
			else{
				cp.y = -box_ext.y;
				n.y -= 1.0;
			}
		}
		else{ // z平面に近い
			if(cp.z > 0){
				cp.z = box_ext.z;
				n.z += 1.0;
			}
			else{
				cp.z = -box_ext.z;
				n.z -= 1.0;
			}
		}

		coli++;
	}

	cp = CuMulMV(box_inv_rot, cp);
	n  = CuMulMV(box_inv_rot, n);

	n = normalize(n);
	cp += box_cen;

	float sgn = (coli) ? -1.0 : 1.0;
	d = sgn*(length(cp-p));

	return 0;
}

/*!
 * 点と球の距離
 * @param[in] p 点座標
 * @param[in] sphere_cen 球の中心
 * @param[in] sphere_rad 球の半径
 * @param[out] cp 点と球中心を結ぶ線分と球の交点
 * @param[out] d 点と球表面の距離
 * @param[out] n 球中心から点への単位ベクトル
 */
__device__
inline int collisionPointSphere(float3 p, float3 sphere_cen, float sphere_rad, float3 &cp, float &d, float3 &n)
{
	n = make_float3(0.0f);

	float3 l = p-sphere_cen;
	float ll = length(l);

	d = ll-sphere_rad;
	if(d < 0.0){
		n = normalize(p-sphere_cen);
		cp = sphere_cen+n*sphere_rad;
	}

	return 0;
}

/*!
 * 点と平面の距離
 * @param[in] v  点の座標
 * @param[in] px 平面上の点
 * @param[in] pn 平面の法線
 * @return 距離
 */
__device__ 
inline float distPointPlane(float3 v, float3 px, float3 pn)
{
	return dot((v-px), pn)/length(pn);
}

/*!
 * 三角形と点の距離と最近傍点
 * @param[in] v0,v1,v2	三角形の頂点
 * @param[in] n			三角形の法線
 * @param[in] p			点
 * @return 
 */
__device__ 
inline int distPointTriangle(float3 v0, float3 v1, float3 v2, float3 n, float3 p, float &dist, float3 &p0)
{
	// ポリゴンを含む平面と点の距離
	float l = distPointPlane(p, v0, n);
	
	// 平面との最近傍点座標
	float3 np = p-l*n;

	// 近傍点が三角形内かどうかの判定
	float3 n1 = cross((v0-p), (v1-p));
	float3 n2 = cross((v1-p), (v2-p));
	float3 n3 = cross((v2-p), (v0-p));

	if(dot(n1, n2) > 0 && dot(n2, n3) > 0){
		// 三角形内
		dist = l;
		p0 = np;
		return 1;
	}
	else{
		// 三角形外
		return 0;
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
inline __device__ 
int intersectSegmentTriangle(float3 P0, float3 P1, 
							 float3 V0, float3 V1, float3 V2, 
							 float3 &I, float3 &n, float rp = 0.01)
{
	// 三角形のエッジベクトルと法線
	float3 u = V1-V0;		
	float3 v = V2-V0;			
	n = normalize(cross(u, v));
	if(CuIsZero(n)){
		return -1;	// 三角形が"degenerate"である(面積が0)
	}

	// 線分
	float3 dir = P1-P0;
	float a = dot(n, P0-V0);
	float b = dot(n, dir);
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
	if(r < 0.0 || fabs(a) > fabs(b) || b > 0){
		return 0;
	}
	//if(r < 0.0){
	//	return 0;
	//}
	//else{
	//	if(fabs(a) > fabs(b)){
	//		return 0;
	//	}
	//	else{
	//		if(b > 0){
	//			return 0;
	//		}
	//	}
	//}

	// 線分と平面の交点
	I = P0+r*dir;

	// 交点が三角形内にあるかどうかの判定
	float uu, uv, vv, wu, wv, D;
	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	float3 w = I-V0;
	wu = dot(w, u);
	wv = dot(w, v);
	D = uv*uv-uu*vv;

	float s, t;
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



#endif // #ifndef _RX_CU_COMMON_CU_



