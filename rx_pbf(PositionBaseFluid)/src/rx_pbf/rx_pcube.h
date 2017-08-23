#ifndef _RX_PCUBE_
#define _RX_PCUBE_

#include "rx_utility.h"

#include <vector>
using namespace std;


namespace RXFunc
{
	/*!
	 * 平面とAABBの交差判定
	 *  - Real-Time Rendering Third Edition, p755
	 * @param[in] n,d	平面 n・x+d=0
	 * @param[in] c,l   AABBの中心と辺の長さの半分
	 */
	inline bool plane_aabb(const Vec3 &n, const double &d, 
						   const Vec3 &c, const Vec3 &h)
	{
		double e = h[0]*fabs(n[0])+h[1]*fabs(n[1])+h[2]*fabs(n[2]);
		double s = dot(c, n)+d;
		if(s-e > 0) return false;	// 半空間外(法線の逆側)
		if(s+e < 0) return false;	// 半空間内(法線側)
		return true;
	}

	/*!
	 * 三角形とAABBの交差判定
	 *  - Real-Time Rendering Third Edition, p760
	 * @param[in] u0,u1,u2	三角形の頂点
	 * @param[in] n			三角形の法線
	 * @param[in] c,l   立方体の中心と辺の長さの半分
	 * @return 交差でtrue
	 */
	inline bool triangle_aabb(const Vec3 &u0, const Vec3 &u1, const Vec3 &u2, const Vec3 &n, 
							  const Vec3 &c, const Vec3 &h)
	{
		// 三角形の頂点をAABB中心を原点とした座標系に移動
		vector<Vec3> v(3);
		v[0] = u0-c;
		v[1] = u1-c;
		v[2] = u2-c;

		//
		// 1. 三角形のAABBとAABBの交差判定
		//
		Vec3 bmin = -h,  bmax = h;
		Vec3 tmin = v[0], tmax = v[0];
		for(int j = 1; j < 3; ++j){
			for(int k = 0; k < 3; ++k){
				if(v[j][k] < tmin[k]) tmin[k] = v[j][k];
				if(v[j][k] > tmax[k]) tmax[k] = v[j][k];
			}
		}
		// 各軸での重なり判定
		int overlap = 0;
		for(int k = 0; k < 3; ++k){
			if(!(tmax[k] < bmin[k] || tmin[k] > bmax[k])){
				overlap++;
			}
		}
		// 3軸すべてで重なっていたらAABB同士が重なっている
		if(overlap < 3){
			return false;
		}

		// 
		// 2. AABBと三角形を含む平面との交差判定
		//
		if(!plane_aabb(n, -dot(n, v[0]), Vec3(0.0), h)){
			return false;
		}

		// 
		// 3. AABBの各軸方向とエッジのテスト
		// 
		vector<Vec3> e(3, Vec3(0.0));	// AABBの各軸方向ベクトル
		vector<Vec3> f(3);				// 三角形のエッジベクトル
		for(int k = 0; k < 3; ++k){
			e[k][k] = 1.0;
			f[k] = v[(k == 2 ? 0 : k+1)]-v[k];
		}
		for(int i = 0; i < 3; ++i){
			for(int j = 0; j < 3; ++j){
				Vec3 a = cross(e[i], f[j]);
				Vec3 p = Vec3(dot(a, v[0]), dot(a, v[1]), dot(a, v[2]));
				double r = h[0]*fabs(a[0])+h[1]*fabs(a[1])+h[2]*fabs(a[2]);
				if(RX_MIN3(p[0], p[1], p[2]) > r || RX_MAX3(p[0], p[1], p[2]) < -r){
					return false;
				}
			}
		}

		return true;
	}

	/*!
	 * 原点中心，辺の長さ1のAABBとポリゴンの交差判定
	 * @param[in] verts 
	 * @param[out] 
	 * @return 
	 */
	static int polygon_intersects_cube(const vector<Vec3> &verts, const Vec3 &p_nrm)
	{
		bool intersect = false;
		int vn = (int)verts.size();
		for(int i = 0; i < vn-2; ++i){
			if(triangle_aabb(verts[0], verts[i+1], verts[i+2], p_nrm, Vec3(0.0), Vec3(0.5))){
				intersect = true;
				break;
			}
		}

		return intersect;
	}

}

#endif

