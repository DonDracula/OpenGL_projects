/*! 
  @file rx_cu_common.cu
	
  @brief CUDA共通デバイス関数
 
  @author Makoto Fujisawa
  @date 2009-08, 2011-06
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

#include "rx_cuda_utils.h"

#include "rx_cu_common.cuh"


// シミュレーションパラメータ(コンスタントメモリ)
__constant__ rxSimParams params;


// AABBの法線方向(6*3)
__constant__ float RXG_AABB_NORMALS[18];


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



// グリッド内ブロック数，ブロック内スレッド数の計算
__device__ __host__
inline void computeGridSize(uint n, uint thread_per_block, uint &numBlocks, uint &numThreads)
{
	numThreads = min(thread_per_block, n);
	numBlocks = DivCeil(n, numThreads);
}


//-----------------------------------------------------------------------------
// グリッド
//-----------------------------------------------------------------------------
/*!
 * グリッド位置計算
 * @param[in] p 座標
 * @return グリッド座標
 */
__device__ 
inline int3 calcGridPos(float3 p)
{
	int3 gridPos;
	gridPos.x = floor((p.x-params.WorldOrigin.x)/params.CellWidth.x);
	gridPos.y = floor((p.y-params.WorldOrigin.y)/params.CellWidth.y);
	gridPos.z = floor((p.z-params.WorldOrigin.z)/params.CellWidth.z);

	gridPos.x = min(max(gridPos.x, 0), params.GridSize.x-1);
	gridPos.y = min(max(gridPos.y, 0), params.GridSize.y-1);
	gridPos.z = min(max(gridPos.z, 0), params.GridSize.z-1);

	return gridPos;
}

/*!
 * グリッド座標から1次元配列中での位置を計算
 * @param[in] gridPos グリッド座標
 * @return アドレス
 */
__device__ 
inline uint calcGridHash(int3 gridPos)
{
	return __umul24(__umul24(gridPos.z, params.GridSize.y), params.GridSize.x)+__umul24(gridPos.y, params.GridSize.x)+gridPos.x;
}

/*!
 * グリッド位置計算
 * @param[in] p 座標
 * @param[in] origin グリッドの最小座標
 * @param[in] cell_width 1グリッドセルの幅
 * @param[in] grid_size グリッド数
 * @return グリッド座標
 */
__device__ 
inline int3 calcGridPosB(float3 p, float3 origin, float3 cell_width, uint3 grid_size)
{
	int3 gridPos;
	gridPos.x = floor((p.x-origin.x)/cell_width.x);
	gridPos.y = floor((p.y-origin.y)/cell_width.y);
	gridPos.z = floor((p.z-origin.z)/cell_width.z);

	gridPos.x = min(max(gridPos.x, 0), grid_size.x-1);
	gridPos.y = min(max(gridPos.y, 0), grid_size.y-1);
	gridPos.z = min(max(gridPos.z, 0), grid_size.z-1);

	return gridPos;
}

/*!
 * グリッド座標から1次元配列中での位置を計算
 * @param[in] gridPos グリッド座標
 * @return アドレス
 */
__device__ 
inline uint calcGridHashB(int3 gridPos, uint3 grid_size)
{
	return __umul24(__umul24(gridPos.z, grid_size.y), grid_size.x)+__umul24(gridPos.y, grid_size.x)+gridPos.x;
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



/*!
 * AABBと点の距離(有効半径内)
 * @param[in] spos AABBの中心を原点とした相対座標値
 * @param[in] h    有効半径
 * @param[in] sgn  AABBの内で距離が正:1,外で正:-1
 * @param[in] vMin AABBの最小座標値(相対座標)
 * @param[in] vMax AABBの最大座標値(相対座標)
 * @param[out] d   符号付距離値
 * @param[out] n   最近傍点の法線方向
 * @param[out] p   有効半径内に含むAABB面のリスト(-x,+x,-y,+y,-z,+z面)
 */
__device__ 
inline bool aabb_point_dist(const float3 &spos, const float &r, const int &sgn, 
							const float3 &vMin, const float3 &vMax, 
							float &d, float3 &n)
{
	int bout = 0;
	float d0[6];
	int idx0 = -1;

	// 各軸ごとに最小と最大境界外になっていないか調べる
	int c = 0;
	int idx = 0;
	if((d0[idx] = (spos.x-r)-vMin.x) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 1;
	if((d0[idx] = vMax.x-(spos.x+r)) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 2;
	if((d0[idx] = (spos.y-r)-vMin.y) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 3;
	if((d0[idx] = vMax.y-(spos.y+r)) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 4;
	if((d0[idx] = (spos.z-r)-vMin.z) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 5;
	if((d0[idx] = vMax.z-(spos.z+r)) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}

	// AABB内(全軸で境界内)
	if(bout == 0){
		float min_d = 1e10;
		int idx1 = -1;
		for(int i = 0; i < 6; ++i){
			if(d0[i] <= min_d){
				min_d = d0[i];
				idx1 = i;
			}
		}

		d = sgn*min_d;
		n = (idx1 != -1) ? sgn*make_float3(RXG_AABB_NORMALS[3*idx1], RXG_AABB_NORMALS[3*idx1+1], RXG_AABB_NORMALS[3*idx1+2]) : make_float3(0.0);
		return true;
	}

	// AABB外
	float3 x = make_float3(0.0);
	idx = 0;
	if(bout & (1 << (2*idx))){
		x.x = d0[2*idx];
	}
	else if(bout & (1 << (2*idx+1))){
		x.x = -d0[2*idx+1];
	}
	idx = 1;
	if(bout & (1 << (2*idx))){
		x.y = d0[2*idx];
	}
	else if(bout & (1 << (2*idx+1))){
		x.y = -d0[2*idx+1];
	}
	idx = 2;
	if(bout & (1 << (2*idx))){
		x.z = d0[2*idx];
	}
	else if(bout & (1 << (2*idx+1))){
		x.z = -d0[2*idx+1];
	}

	// sgn = 1:箱，-1:オブジェクト
	if(c == 1){
		// 平面近傍
		d = sgn*d0[idx0];
		n = sgn*make_float3(RXG_AABB_NORMALS[3*idx0], RXG_AABB_NORMALS[3*idx0+1], RXG_AABB_NORMALS[3*idx0+2]);
	}
	else{
		// エッジ/コーナー近傍
		d = -sgn*length(x);
		n = sgn*(-normalize(x));
	}

	return false;
}


/*!
 * AABBと点の距離(有効半径内)
 * @param[in] spos AABBの中心を原点とした相対座標値
 * @param[in] h    有効半径
 * @param[in] sgn  AABBの内で距離が正:1,外で正:-1
 * @param[in] vMin AABBの最小座標値(相対座標)
 * @param[in] vMax AABBの最大座標値(相対座標)
 * @param[out] d   符号付距離値
 * @param[out] n   最近傍点の法線方向
 * @param[out] p   有効半径内に含むAABB面のリスト(-x,+x,-y,+y,-z,+z面)
 */
__device__ 
inline bool aabb_point_dist(const float3 &spos, const float &h, const int &sgn, 
							const float3 &vMin, const float3 &vMax, 
							float &d, float3 &n, int &np, int &plist, float pdist[6])
{
	int bout = 0;
	float d0[6];
	int idx0 = -1;
	plist = 0;

	// 各軸ごとに最小と最大境界外になっていないか調べる
	int c = 0;
	int idx = 0;
	if((d0[idx] = spos.x-vMin.x) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 1;
	if((d0[idx] = vMax.x-spos.x) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 2;
	if((d0[idx] = spos.y-vMin.y) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 3;
	if((d0[idx] = vMax.y-spos.y) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 4;
	if((d0[idx] = spos.z-vMin.z) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}
	idx = 5;
	if((d0[idx] = vMax.z-spos.z) < 0.0){
		bout |= (1 << idx); c++;
		idx0 = idx;
	}

	// AABB内(全軸で境界内)
	if(bout == 0){
		np = 0;
		float min_d = 1e10;
		int idx1 = -1;
		for(int i = 0; i < 6; ++i){
			if(d0[i] <= min_d){
				min_d = d0[i];
				idx1 = i;
			}
			if(d0[i] < h){
				plist |= (1 << i);
				pdist[i] = d0[i];
				np++;
			}
		}

		d = sgn*min_d;
		n = (idx1 != -1) ? sgn*make_float3(RXG_AABB_NORMALS[3*idx1], RXG_AABB_NORMALS[3*idx1+1], RXG_AABB_NORMALS[3*idx1+2]) : make_float3(0.0);
		return true;
	}

	// AABB外
	np = 0;
	float3 x = make_float3(0.0);
	idx = 0;
	if(bout & (1 << (2*idx))){
		x.x = d0[2*idx];
		if(-d0[2*idx] < h){
			plist |= (1 << (2*idx));
			pdist[2*idx] = d0[2*idx];
			np++;
		}
	}
	else if(bout & (1 << (2*idx+1))){
		x.x = -d0[2*idx+1];
		if(-d0[2*idx+1] < h){
			plist |= (1 << (2*idx+1));
			pdist[2*idx+1] = d0[2*idx+1];
			np++;
		}
	}
	idx = 1;
	if(bout & (1 << (2*idx))){
		x.y = d0[2*idx];
		if(-d0[2*idx] < h){
			plist |= (1 << (2*idx));
			pdist[2*idx] = d0[2*idx];
			np++;
		}
	}
	else if(bout & (1 << (2*idx+1))){
		x.y = -d0[2*idx+1];
		if(-d0[2*idx+1] < h){
			plist |= (1 << (2*idx+1));
			pdist[2*idx+1] = d0[2*idx+1];
			np++;
		}
	}
	idx = 2;
	if(bout & (1 << (2*idx))){
		x.z = d0[2*idx];
		if(-d0[2*idx] < h){
			plist |= (1 << (2*idx));
			pdist[2*idx] = d0[2*idx];
			np++;
		}
	}
	else if(bout & (1 << (2*idx+1))){
		x.z = -d0[2*idx+1];
		if(-d0[2*idx+1] < h){
			plist |= (1 << (2*idx+1));
			pdist[2*idx+1] = d0[2*idx+1];
			np++;
		}
	}

	// sgn = 1:箱，-1:オブジェクト
	if(c == 1){
		// 平面近傍
		d = sgn*d0[idx0];
		n = sgn*make_float3(RXG_AABB_NORMALS[3*idx0], RXG_AABB_NORMALS[3*idx0+1], RXG_AABB_NORMALS[3*idx0+2]);
	}
	else{
		// エッジ/コーナー近傍
		d = -sgn*length(x);
		n = sgn*(-normalize(x));
	}

	return false;
}




/*!
 * 線分と球の交差判定
 * @param[in] s0,s1	線分の端点
 * @param[in] sc,r   球の中心座標と半径
 * @param[out] d2 線分との距離の二乗
 * @return 交差ありでtrue
 */
__device__ 
inline bool segment_sphere(const float3 &s0, const float3 &s1, const float3 &sc, const float &r, float &d2)
{
	float3 v = s1-s0;
	float3 c = sc-s0;

	float vc = dot(v, c);
	if(vc < 0){		// 球の中心が線分の始点s0の外にある
		d2 = dot(c, c);
		return (d2 < r*r);	// 球中心と始点s0の距離で交差判定
	}
	else{
		float v2 = dot(v, v);
		if(vc > v2){	// 球の中心が線分の終点s1の外にある
			d2 = dot(s1-sc, s1-sc);
			return (d2 < r*r);	// 球中心と終点s1の距離で交差判定
		}
		else{			// 球がs0とs1の間にある
			float3 a = (vc*v)/dot(v, v)-c;
			d2 = dot(a, a);
			return (d2 < r*r);	// 直線と球中心の距離で交差判定
		}
	}
}

/*!
 * 線分(を含む直線)と点の距離
 * @param[in] v0,v1 線分の両端点座標
 * @param[in] p 点の座標
 * @return 距離
 */
__device__ 
inline double segment_point_dist(const float3 &v0, const float3 &v1, const float3 &p)
{
	float3 v = normalize(v1-v0);
	float3 vp = p-v0;
	float3 vh = dot(vp, v)*v;
	return length(vp-vh);
}


/*!
 * 光線(レイ,半直線)と球の交差判定
 * @param[in] p,d レイの原点と方向
 * @param[in] c,r 球の中心と半径
 * @param[out] t1,t2 pから交点までの距離
 * @return 交点数
 */
__device__
inline int ray_sphere(const float3 &p, const float3 &d, const float3 &sc, const float r, float &t1, float &t2)
{
	float3 q = p-sc;	// 球中心座標系での光線原点座標

	float a = dot(d, d);
	float b = 2*dot(q, d);
	float c = dot(q, q)-r*r;

	// 判別式
	float D = b*b-4*a*c;

	if(D < 0.0){ // 交差なし
		return 0;
	}
	else if(D < 1e-8){ // 交点数1
		t1 = -b/(2*a);
		t2 = -1;
		return 1;
	}
	else{ // 交点数2
		float sqrtD = sqrt(D);
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
__device__
inline bool triangle_sphere(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &n, 
							const float3 &c, const float &r, float &dist, float3 &ipoint)
{
	// ポリゴンを含む平面と球中心の距離
	float d = dot(v0, n);
	float l = dot(n, c)-d;

	dist = l;
	if(l > r) return false;

	// 平面との最近傍点座標
	float3 p = c-l*n;

	// 近傍点が三角形内かどうかの判定
	float3 n1 = cross((v0-c), (v1-c));
	float3 n2 = cross((v1-c), (v2-c));
	float3 n3 = cross((v2-c), (v0-c));

	ipoint = p;
	dist = l;
	if(dot(n1, n2) > 0 && dot(n2, n3) > 0){		// 三角形内
		return true;
	}
	else{		// 三角形外
		// 三角形の各エッジと球の衝突判定
		for(int e = 0; e < 3; ++e){
			float3 va0 = (e == 0 ? v0 : (e == 1 ? v1 : v2));
			float3 va1 = (e == 0 ? v1 : (e == 1 ? v2 : v0));

			float t1, t2;
			int n = ray_sphere(va0, normalize(va1-va0), c, r, t1, t2);

			if(n){
				float le2 = dot(va1-va0, va1-va0);
				if((t1 >= 0.0 && t1*t1 < le2) || (t2 >= 0.0 && t2*t2 < le2)){
					return true;
				}
			}
		}
		return false;
	}
}


#endif // #ifndef _RX_CU_COMMON_CU_



