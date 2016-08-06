/*! 
  @file rx_shadowmap.h
	
  @brief 
 
  @author Makoto Fujisawa
  @date 2011-
*/
// FILE --rx_shadowmap--

#ifndef _RX_SHADOWMAP_H_
#define _RX_SHADOWMAP_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"
#include "rx_shaders.h"

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



//-----------------------------------------------------------------------------
// シャドウマッピングクラス
//-----------------------------------------------------------------------------
class rxShadowMap
{
	GLuint m_iFBODepth;		//!< 光源から見たときのデプスを格納するFramebuffer object
	GLuint m_iTexDepth;		//!< m_iFBODepthにattachするテクスチャ
	double m_fDepthSize[2];	//!< デプスを格納するテクスチャのサイズ

public:
	//! デフォルトコンストラクタ
	rxShadowMap()
	{
		m_iFBODepth = 0;
		m_iTexDepth = 0;
		m_fDepthSize[0] = m_fDepthSize[1] = 512;
	}

	//! デストラクタ
	~rxShadowMap(){}


	/*!
	 * シャドウマップ用FBOの初期化
	 * @param[in] w,h  シャドウマップの解像度
	 */
	void InitShadow(int w, int h)
	{
		m_fDepthSize[0] = w;
		m_fDepthSize[1] = h;
	
		// デプス値テクスチャ
		glActiveTexture(GL_TEXTURE0);
		glGenTextures(1, &m_iTexDepth);
		glBindTexture(GL_TEXTURE_2D, m_iTexDepth);

		// テクスチャパラメータの設定
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		GLfloat border_color[4] = {1, 1, 1, 1};
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, border_color);
	
		// テクスチャ領域の確保(GL_DEPTH_COMPONENTを用いる)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_fDepthSize[0], m_fDepthSize[1], 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	

		// FBO作成
		glGenFramebuffersEXT(1, &m_iFBODepth);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_iFBODepth);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
	
		// デプスマップテクスチャをFBOに接続
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_iTexDepth, 0);
	
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	/*!
	 * プロジェクション行列，視点位置の設定
	 * @param[in] f 視錘台
	 */
	void SetFrustum(const rxFrustum &f)
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(f.FOV, (double)f.W/(double)f.H, f.Near, f.Far);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		gluLookAt(f.Origin[0], f.Origin[1], f.Origin[2], f.LookAt[0], f.LookAt[1], f.LookAt[2], f.Up[0], f.Up[1], f.Up[2]);
	}

	/*!
	 * シャドウマップ(デプステクスチャ)の作成
	 * @param[in] light 光源
	 * @param[in] fpDraw 描画関数ポインタ
	 */
	void MakeShadowMap(rxFrustum &light, void (*fpDraw)(void*), void* func_obj, bool self_shading = false)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_iFBODepth);	// FBOにレンダリング
		glEnable(GL_TEXTURE_2D);	

		glUseProgram(0);

		// ビューポートをシャドウマップの大きさに変更
		glViewport(0, 0, m_fDepthSize[0], m_fDepthSize[1]);
	
		glClear(GL_DEPTH_BUFFER_BIT);
	
		// デプス値以外の色のレンダリングを無効にする
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); 
	
		double light_proj[16];
		double light_modelview[16];
		light.W = m_fDepthSize[0];
		light.H = m_fDepthSize[1];
		SetFrustum(light);

		// 光源視点のモデルビュー行列，プロジェクション行列を取得
		glMatrixMode(GL_PROJECTION);
		glGetDoublev(GL_PROJECTION_MATRIX, light_proj);

		glMatrixMode(GL_MODELVIEW);
		glGetDoublev(GL_MODELVIEW_MATRIX, light_modelview);

		glPolygonOffset(1.1f, 4.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);
	
		glDisable(GL_LIGHTING);
		if(self_shading){
			glDisable(GL_CULL_FACE);
		}
		else{
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
		}
		fpDraw(func_obj);

		glDisable(GL_POLYGON_OFFSET_FILL);
	
	
		const double bias[16] = { 0.5, 0.0, 0.0, 0.0, 
								  0.0, 0.5, 0.0, 0.0,
								  0.0, 0.0, 0.5, 0.0,
								  0.5, 0.5, 0.5, 1.0 };
	

	
		// テクスチャモードに移行
		glMatrixMode(GL_TEXTURE);
		glActiveTexture(GL_TEXTURE7);
	
		glLoadIdentity();
		glLoadMatrixd(bias);
	
		// 光源中心座標となるようにテクスチャ行列を設定
		// テクスチャ変換行列にモデルビュー，プロジェクションを設定
		glMultMatrixd(light_proj);
		glMultMatrixd(light_modelview);
	
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// 無効にした色のレンダリングを有効にする
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 
	}

	/*!
	 * 影付きでシーン描画
	 * @param[in] camera 視点
	 * @param[in] fpDraw 描画関数のポインタ
	 */
	void RenderSceneWithShadow(rxFrustum &camera, void (*fpDraw)(void*), void* func_obj)
	{
		// 視点設定
		SetFrustum(camera);

		glEnable(GL_TEXTURE_2D);

		// デプステクスチャを貼り付け
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, m_iTexDepth);
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);
		fpDraw(func_obj);

		glBindTexture(GL_TEXTURE_2D, 0);

	}

	/*!
	 * デプスマップをテクスチャとして表示
	 * @param[in] w,h ウィンドウサイズ
	 */
	void DrawDepthTex(int w, int h)
	{
		glUseProgram(0);
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		glOrtho(0, w, 0, h, -1, 1);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();

		glDisable(GL_LIGHTING);
		glColor4f(1, 1, 1, 1);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_iTexDepth);
		glEnable(GL_TEXTURE_2D);

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


#endif // #ifdef _RX_SHADOWMAP_H_