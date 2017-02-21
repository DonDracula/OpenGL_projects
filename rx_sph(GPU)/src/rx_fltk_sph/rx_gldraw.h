/*! @file rx_gldraw.h
	
	@brief OpenGL描画関数群

*/


#ifndef _RX_GLDRAW_H_
#define _RX_GLDRAW_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <GL/glew.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <GL/glut.h>

#include <boost/function.hpp>

#include "rx_utility.h"
#include "rx_texture.h"
#include "rx_material.h"

#include "rx_shaders.h"


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
using namespace std;

typedef boost::function<void (void)> RxFpDraw;


//-----------------------------------------------------------------------------
//! RxGLDraw名前空間
//-----------------------------------------------------------------------------
namespace RxGLDraw
{
	// 
	// OpenGL描画
	// 

	//
	// HACK:プリミティブの描画
	//
	void DrawChackerBoard(const int &n, const float &delta, const Vec3 &color);
	void DrawAxis3D(const float &length, const Vec3 &color_x, const Vec3 &color_y, const Vec3 &color_z);

	void DrawWireCube(const Vec3 &center, const float &s_length, const Vec3 &color);	//!< 立方体のワイヤーフレーム描画
	void DrawWireCuboid(const Vec3 &min, const Vec3 &max, const Vec3 &color);			//!< 直方体のワイヤーフレーム描画

	void DrawSolidCubeColor(const Vec3 &center, const float &s_length, const Vec3 color[6]);	//!< 立方体の描画(各面色分け)
	void DrawSolidCube(const Vec3 &center, const float &s_length, const Vec3 &color);			//!< 立方体の描画
	void DrawSolidCuboidColor(const Vec3 &min, const Vec3 &max, const Vec3 color[6], const int &sign = 1);			//!< 直方体の描画(各面色分け)

	void DrawArrow2D(const Vec3 &origin, const Vec3 &dir, const float &scale);	//!< 矢印の描画

	void DrawCircle(const Vec3 &cen, const float &rad);							//!< 円の描画
	void DrawWireCircle(const Vec3 &cen, const float &rad, const int &n);		//!< 円のワイヤーフレーム描画
	void DrawWireCircle(const float &rad, const int &n);						//!< 原点中心円のワイヤーフレーム描画
	void DrawWireCircleXZ(const float &rad, const int &n);						//!< 原点中心円のワイヤーフレーム描画(XZ平面)

	void DrawSphere(const Vec3 &cen, const float &rad, const Vec3 &col);		//!< 球の描画
	void DrawWireSphere(const Vec3 &cen, const float &rad, const Vec3 &col);	//!< 球のワイヤーフレーム描画
	
	void DrawString(const string &str, int w, int h);							//!< 文字列描画

	void DrawPoints(const vector<Vec3> &vrts, const double &size, const Vec3 &col);

	//
	// HACK:ポリゴンの描画
	//
	void DrawPolygon(const vector<Vec3> &vrts, const vector<int> &idx, const Vec3 &col);
	void DrawPolygon(const vector<Vec3> &vrts, const vector<int> &idxes, const Vec3 &nrm, const Vec3 &col);

	void DrawPolygonGouraud(const vector<Vec3> &vrts, const vector<int> &idxes, const vector<Vec3> &nrms, const Vec3 &col);

	void DrawLineLoop(const vector<Vec3> &vrts, const vector<int> &idx, const double &width, const Vec3 &col);

	//
	// HACK:複数ポリゴンの描画
	//
	void DrawPolygonsNormal(const vector<Vec3> &vrts, const vector< vector<int> > &idxes, const Vec3 &col, bool select = false);
	void DrawPolygonsGouraud(const vector<Vec3> &vrts, const vector< vector<int> > &idxes, const vector<Vec3> &nrms, const Vec3 &col, bool select = false);

	void DrawPolygons(const vector<Vec3> &vrts, const vector< vector<int> > &idxs, bool draw_nrm = false);
	void DrawPolygons(const float *vrts, const float *nrms, int vnum, bool draw_nrm = false);

	void DrawMesh(const vector<Vec3> &vrts, vector<int> &idxes, bool normal);
	void DrawTriangles(const vector<Vec3> &vrts, const vector<int> &idxes, const vector<Vec3> &vnrms, bool normal);
	void DrawTriangles(const Vec3 *vrts, const int *tris, const Vec3 *vnrms, int nvrts, int ntris, bool normal);

	void DrawMesh(const vector<Vec3> &vrts, vector<int> &idxes, const vector<Vec3> &vnrms, 
				  double col[4], double ncol0[4], double ncol1[4], bool normal);
	void DrawMeshV(const vector<Vec3> &vrts, const vector<Vec3> &vnms, bool normal);

	
	////! キューブマップテクスチャの読み込み
	//bool LoadCubeMap(rxCubeMapData &cube_map, string base, string ext);

	////! 環境マップ用のキューブマップテクスチャの読み込み
	//bool LoadCubeMapTexture(const string fn[6], rxCubeMapData &cube_map);

	////! キューブマップテクスチャを内部に貼り付けた立方体の描画
	//void DrawCubeMap(const rxCubeMapData &cube_map, double side);

	//-----------------------------------------------------------------------------
	// HACK:GLSL
	//-----------------------------------------------------------------------------
	//! GLSLコンパイル
	void InitGLSL(void);

	//! Fresnelの法則に基づく反射，屈折物体のレンダリング
	void DrawPolygonsRefrac(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
							float eta, float bias, float power, float scale, 
							const Vec3 &eye_pos, const rxCubeMapData &cube_map, bool select = false);
	void DrawPolygonsRefrac(RxFpDraw fpDraw, float eta, float bias, float power, float scale, 
							const Vec3 &eye_pos, const rxCubeMapData &cube_map);
	void DrawPolygonsRefracWithFoam(RxFpDraw fpDraw, float eta, float bias, float power, float scale, 
									const Vec3 &eye_pos, const rxCubeMapData &cube_map, 
									const Vec3 &foam_color, float noise_max, float noise_min);

	//! Cgを用いたフォンシェーディング
	void DrawPolygonsPhong(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
						   rxMaterial *mat, rxMaterial *light, Vec3 light_pos, const Vec3 &eye_pos, 
						   bool select = false);

	//! Cgを用いたトゥーンシェーディング
	void DrawPolygonsToon(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
						  rxMaterial *mat, Vec3 light_pos, const Vec3 &eye_pos, 
						  bool select = false);



};




#endif // _RX_GLDRAW_H_

