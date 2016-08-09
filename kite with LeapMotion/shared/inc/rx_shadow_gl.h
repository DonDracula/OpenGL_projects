/*! 
  @file rx_shadow_gl.h
	
  @brief シャドウマップ(GLSL無し版)
	     参考 : http://www.paulsprojects.net/tutorials/smt/smt.html
 
  @author Makoto Fujisawa
  @date 2013-
*/
// FILE --rx_shadowmap--

#ifndef _RX_SHADOWMAP_GL_H_
#define _RX_SHADOWMAP_GL_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"

using namespace std;


//-----------------------------------------------------------------------------
// 視錐台
//-----------------------------------------------------------------------------
struct rxFrustum
{
	double Near;
	double Far;
	double FOV;	// deg
	double W, H;
	Vec3 Origin;
	Vec3 LookAt;
	Vec3 Up;
};

const GLfloat RX_LIGHT_DIM[4]   = { 0.2f, 0.2f, 0.2f, 0.2f };
const GLfloat RX_LIGHT_WHITE[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
const GLfloat RX_LIGHT_BLACK[4] = { 0.0f, 0.0f, 0.0f, 0.0f };


//-----------------------------------------------------------------------------
// シャドウマッピングクラス
//-----------------------------------------------------------------------------
class rxShadowMap
{

	GLuint m_iTexShadowMap;		//!< シャドウマップテクスチャ
	int m_iShadowMapSize[2];	//!< シャドウマップテクスチャのサイズ

public:
	//! デフォルトコンストラクタ
	rxShadowMap()
	{
		m_iTexShadowMap = 0;
		m_iShadowMapSize[0] = m_iShadowMapSize[1] = 512;
	}

	//! デストラクタ
	~rxShadowMap(){}

	/*!
	 * 4x4行列同士のかけ算
	 * @param[out] m 結果行列
	 * @param[in] a,b 4x4行列
	 */
	inline void MulMat(float m[16], const float a[16], const float b[16])
	{
		m[0]  = a[0]*b[0]  + a[4]*b[1]  + a[8]*b[2]   + a[12]*b[3];
		m[1]  = a[1]*b[0]  + a[5]*b[1]  + a[9]*b[2]   + a[13]*b[3];
		m[2]  = a[2]*b[0]  + a[6]*b[1]  + a[10]*b[2]  + a[14]*b[3];
		m[3]  = a[3]*b[0]  + a[7]*b[1]  + a[11]*b[2]  + a[15]*b[3];
		m[4]  = a[0]*b[4]  + a[4]*b[5]  + a[8]*b[6]   + a[12]*b[7];
		m[5]  = a[1]*b[4]  + a[5]*b[5]  + a[9]*b[6]   + a[13]*b[7];
		m[6]  = a[2]*b[4]  + a[6]*b[5]  + a[10]*b[6]  + a[14]*b[7];
		m[7]  = a[3]*b[4]  + a[7]*b[5]  + a[11]*b[6]  + a[15]*b[7];
		m[8]  = a[0]*b[8]  + a[4]*b[9]  + a[8]*b[10]  + a[12]*b[11];
		m[9]  = a[1]*b[8]  + a[5]*b[9]  + a[9]*b[10]  + a[13]*b[11];
		m[10] = a[2]*b[8]  + a[6]*b[9]  + a[10]*b[10] + a[14]*b[11];
		m[11] = a[3]*b[8]  + a[7]*b[9]  + a[11]*b[10] + a[15]*b[11];
		m[12] = a[0]*b[12] + a[4]*b[13] + a[8]*b[14]  + a[12]*b[15];
		m[13] = a[1]*b[12] + a[5]*b[13] + a[9]*b[14]  + a[13]*b[15];
		m[14] = a[2]*b[12] + a[6]*b[13] + a[10]*b[14] + a[14]*b[15];
		m[15] = a[3]*b[12] + a[7]*b[13] + a[11]*b[14] + a[15]*b[15];
	}



	/*!
	 * シャドウマップ用テクスチャの初期化
	 * @param[in] w,h  シャドウマップの解像度
	 */
	int InitShadow(int w = 512, int h = 512)
	{
		// 
		glewInit();
		if(!glewIsSupported("GL_ARB_depth_texture "
							"GL_ARB_shadow "
							)){
			cout << "ERROR: Support for necessary OpenGL extensions missing." << endl;
			return 0;
		}

		m_iShadowMapSize[0] = w;
		m_iShadowMapSize[1] = h;

		// デプスマップの初期化と有効化
		glClearDepth(1.0f);
		glEnable(GL_DEPTH_TEST);

		// シャドウマップテクスチャの生成
		glGenTextures(1, &m_iTexShadowMap);
		glBindTexture(GL_TEXTURE_2D, m_iTexShadowMap);
		glTexImage2D(	GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_iShadowMapSize[0], m_iShadowMapSize[1], 0,
						GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);	// 投影範囲確認用
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		GLfloat border_color[4] = {1, 1, 1, 1};
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);

		glBindTexture(GL_TEXTURE_2D, 0);
	}

	/*!
	 * シャドウマップ(デプステクスチャ)の作成
	 * @param[in] light 光源
	 * @param[in] fpDraw 描画関数ポインタ
	 */
	void RenderSceneWithShadow(rxFrustum &light, void (*fpDraw)(void*), void* func_obj)
	{
		float light_proj[16], camera_proj[16];
		float light_modelview[16], camera_modelview[16];

		//
		// 現在の視点行列，光源行列を取得 or 作成
		//
		glMatrixMode(GL_PROJECTION);

		// 視点プロジェクション行列の取得
		glGetFloatv(GL_PROJECTION_MATRIX, camera_proj);
		glPushMatrix();	// 今のプロジェクション行列を退避させておく

		// 光源プロジェクション行列の生成と取得
		glLoadIdentity();
		gluPerspective(light.FOV, (double)light.W/(double)light.H, light.Near, light.Far);
		glGetFloatv(GL_PROJECTION_MATRIX, light_proj);
		
		glMatrixMode(GL_MODELVIEW);

		// 視点モデルビュー行列の取得
		glGetFloatv(GL_MODELVIEW_MATRIX, camera_modelview);
		glPushMatrix();	// 今のモデルビュー行列を退避させておく

		// 光源モデルビュー行列の生成と取得
		glLoadIdentity();
		gluLookAt(light.Origin[0], light.Origin[1], light.Origin[2], 
				  light.LookAt[0], light.LookAt[1], light.LookAt[2], 
				  light.Up[0], light.Up[1], light.Up[2]);
		glGetFloatv(GL_MODELVIEW_MATRIX, light_modelview);
	
		// 今のビューポート情報を後で戻すために確保
		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);


		//
		// 光源からレンダリングしてシャドウマップを生成
		//
		// カラー，デプスバッファのクリア
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// シャドウマップと同じサイズのビューポートを設定
		glViewport(0, 0, m_iShadowMapSize[0], m_iShadowMapSize[1]);

		// 光源を視点として設定
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(light_proj);
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(light_modelview);
		
		// 光源に対して裏の面に影をつけたくない場合はglCullFace(GL_FRONT)で表面をカリングするように設定
		glCullFace(GL_FRONT);

		// 高速化のためにフラットシェーディングを設定
		glShadeModel(GL_FLAT);

		// デプス値以外の色のレンダリングを無効に
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); 

		glPolygonOffset(1.1f, 4.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);

		// シーン描画
		fpDraw(func_obj);

		glDisable(GL_POLYGON_OFFSET_FILL);

		// デプスバッファをテクスチャに転送
		glBindTexture(GL_TEXTURE_2D, m_iTexShadowMap);
		glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, m_iShadowMapSize[0], m_iShadowMapSize[1]);

		glCullFace(GL_BACK);
		glShadeModel(GL_SMOOTH);

		// 無効にした色のレンダリングを有効にする
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 
		
		// 元のビューポート行列に戻す
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

		// 避させておいた視点行列を元に戻す
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();


		//
		// カメラから見たときのシーン描画：影領域
		// 
		glClear(GL_DEPTH_BUFFER_BIT);

		// 影エリア用の暗いライトの設定(なるべく他の光源設定と重複しないようにLIGHT7を使用)
		float light_pos[4] = { light.Origin[0], light.Origin[1], light.Origin[2], 1.0 };
		glLightfv(GL_LIGHT7, GL_POSITION, light_pos);
		glLightfv(GL_LIGHT7, GL_AMBIENT, RX_LIGHT_DIM);
		glLightfv(GL_LIGHT7, GL_DIFFUSE, RX_LIGHT_DIM);
		glLightfv(GL_LIGHT7, GL_SPECULAR, RX_LIGHT_BLACK);
		glDisable(GL_LIGHT0);
		glEnable(GL_LIGHT7);
		glEnable(GL_LIGHTING);

		// 描画
		fpDraw(func_obj);

		glDisable(GL_LIGHT7);


		//
		// カメラから見たときのシーン描画：日向領域
		// 
		glEnable(GL_LIGHT0);

		// テクスチャ行列用のバイアス行列		
		const float bias[16] = { 0.5, 0.0, 0.0, 0.0, 
								 0.0, 0.5, 0.0, 0.0,
								 0.0, 0.0, 0.5, 0.0,
								 0.5, 0.5, 0.5, 1.0 };

		// テクスチャ行列(視点空間からライト空間への変換行列)の計算
		float tex_mat[16], tmp[16];
		MulMat(tmp, bias, light_proj);
		MulMat(tex_mat, tmp, light_modelview);

		// テクスチャ座標系の設定
		float t[4];
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		t[0] = tex_mat[0]; t[1] = tex_mat[4]; t[2] = tex_mat[8]; t[3] = tex_mat[12];
		glTexGenfv(GL_S, GL_EYE_PLANE, t);
		glEnable(GL_TEXTURE_GEN_S);

		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		t[0] = tex_mat[1]; t[1] = tex_mat[5]; t[2] = tex_mat[9]; t[3] = tex_mat[13];
		glTexGenfv(GL_T, GL_EYE_PLANE, t);
		glEnable(GL_TEXTURE_GEN_T);

		glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		t[0] = tex_mat[2]; t[1] = tex_mat[6]; t[2] = tex_mat[10]; t[3] = tex_mat[14];
		glTexGenfv(GL_R, GL_EYE_PLANE, t);
		glEnable(GL_TEXTURE_GEN_R);

		glTexGeni(GL_Q, GL_TEXTURE_GEN_MODE, GL_EYE_LINEAR);
		t[0] = tex_mat[3]; t[1] = tex_mat[7]; t[2] = tex_mat[11]; t[3] = tex_mat[15];
		glTexGenfv(GL_Q, GL_EYE_PLANE, t);
		glEnable(GL_TEXTURE_GEN_Q);


		// シャドウマップテクスチャ
		glBindTexture(GL_TEXTURE_2D, m_iTexShadowMap);
		glEnable(GL_TEXTURE_2D);

		// テクスチャR座標とテクスチャを比較するように設定
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);

		// R <= テクスチャの値 ならばtrue(日向)にする
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

		// 比較結果は全体の色として反映(アルファ値だけ(GL_ALPHA)でもよい)
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

		// α値の比較関数を設定(第2引数はしきい値)
		glAlphaFunc(GL_GEQUAL, 0.99f);
		glEnable(GL_ALPHA_TEST);

		// 日向部分が元の図形に重ねて書かれるようにデプス比較関数を設定
		glDepthFunc(GL_LEQUAL);

		fpDraw(func_obj);

		glDisable(GL_TEXTURE_2D);

		glDisable(GL_TEXTURE_GEN_S);
		glDisable(GL_TEXTURE_GEN_T);
		glDisable(GL_TEXTURE_GEN_R);
		glDisable(GL_TEXTURE_GEN_Q);

		glDisable(GL_LIGHTING);
		glDisable(GL_ALPHA_TEST);
	}


	/*!
	 * デプスマップをテクスチャとして表示
	 * @param[in] w,h ウィンドウサイズ
	 */
	void DrawDepthTex(int w, int h)
	{
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, w, 0, h, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glDisable(GL_LIGHTING);
		glColor4f(1, 1, 1, 1);
		glBindTexture(GL_TEXTURE_2D, m_iTexShadowMap);
		glEnable(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

		glNormal3d(0, 0, -1);
		glBegin(GL_QUADS);
		glTexCoord2d(0, 0); glVertex3f(0.05*w,     0.05*h, 0);
		glTexCoord2d(1, 0); glVertex3f(0.05*w+100, 0.05*h, 0);
		glTexCoord2d(1, 1); glVertex3f(0.05*w+100, 0.05*h+100, 0);
		glTexCoord2d(0, 1); glVertex3f(0.05*w,     0.05*h+100, 0);
		glEnd();
		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
	}
};


/*!
 * 視錐台の生成
 * @param[in] fov_deg 視野角[deg]
 * @param[in] near_d,far_d 視線方向の範囲
 * @param[in] w,h ウィンドウサイズ
 * @param[in] pos 視点位置
 * @param[in] lookat 注視点位置
 * @param[in] up 上方向
 * @return 視錐台
 */
inline rxFrustum CalFrustum(double fov_deg, double near_d, double far_d, int w, int h, 
							Vec3 pos, Vec3 lookat = Vec3(0.0), Vec3 up = Vec3(0.0, 1.0, 0.0))
{
	rxFrustum f;
	f.Near = near_d;
	f.Far = far_d;
	f.FOV = fov_deg;
	f.W = w;
	f.H = h;
	f.Origin = pos;
	f.LookAt = lookat;
	f.Up = up;
	return f;
}

/*!
	* プロジェクション行列，視点位置の設定
	* @param[in] f 視錘台
	*/
inline void SetFrustum(const rxFrustum &f)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(f.FOV, (double)f.W/(double)f.H, f.Near, f.Far);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(f.Origin[0], f.Origin[1], f.Origin[2], f.LookAt[0], f.LookAt[1], f.LookAt[2], f.Up[0], f.Up[1], f.Up[2]);
}




#endif // #ifdef _RX_SHADOWMAP_H_