/*!	@file rx_shadow.h
	
	@brief シャドウマップ法による影付け
 
	@author Makoto Fujisawa
	@date   2009
*/
// FILE --rx_shadow.h--

#ifndef _RX_SHADOW_MAP_H_
#define _RX_SHADOW_MAP_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include "rx_matrix.h"
#include "rx_utility.h"

#include "rx_shaders.h"

#include <GL/glew.h>
#include <GL/glut.h>

#pragma comment (lib, "glew32.lib")


//-----------------------------------------------------------------------------
// 定義・定数
//-----------------------------------------------------------------------------
using namespace std;

#define RX_MAX_SPLITS 4
#define RX_LIGHT_FOV 45.0

#define RX_FAR_DIST 20.0f


//-----------------------------------------------------------------------------
// HACK:視錐台
//-----------------------------------------------------------------------------
struct rxFrustum
{
	double Near;
	double Far;
	double FOV;	// radian(OpenGLはdegreeなので変換すること)
	double Ratio;
	Vec3 Point[8];
};

//-----------------------------------------------------------------------------
// MARK:rxShadowMap
//-----------------------------------------------------------------------------
class rxShadowMap
{


	//
	// シャドウマップ変数
	//
	int m_iFrustumCutNum;	//!< 複数階層のシャドウマップを用いる場合の階層数
	GLuint m_iFBDepth;		//!< 光源から見たときのデプスを格納するFramebuffer object
	GLuint m_iTexDepth;		//!< m_iFBDepthにattachするテクスチャ
	GLuint m_iDepthSize;	//!< デプスを格納するテクスチャのサイズ

	vector<rxFrustum> m_vFrustums;	//!< 視錘台
	vector<rxMatrix4> m_vMatShadowCPM;

	int m_iShadowType;

	vector<rxGLSL> m_vShadGLSL;
	rxGLSL m_GLSLView;

	Vec3 m_v3EyePos, m_v3EyeDir;
	Vec3 m_v3LightPos;

	int m_iWidth, m_iHeight;

	boost::function<void (void)> m_fpDraw;

public:
	//! デフォルトコンストラクタ
	rxShadowMap()
	{
		m_iShadowType = 0;
	}

	//! デストラクタ
	~rxShadowMap(){}


	//
	// GLSL
	//
	// GLSLプログラムコンパイル・リンク
	//rxGLSL CreateGLSLFromFile(const string &vs, const string &fs, const string &name = "");
	//rxGLSL CreateGLSL(const char* vs, const char* fs, const string &name = "");

	void SetShadowGLSL(vector<rxGLSL> &gss);


	//
	// 影の生成方法
	//
	void SetShadowType(int n);
	string GetCurrentShadowType(void);


	//
	// 視錐台
	//
	// 視錐台の計算
	inline rxFrustum CalFrustum(int w, int h, double fov);
	void CalAllFrustums(int w, int h, double fov);

	// 視錐台の分割数の選択
	void SetFrustumCutNum(int n);
	int GetFrustumCutNum(void) const;
	void IncreaseFrustumCutNum(int d = 1);

	// 視錐台の8コーナー頂点を計算する
	void UpdateFrustumPoints(rxFrustum &f, const Vec3 &center, const Vec3 &view_dir);

	// カメラ視点空間での各視錐台スライスのnearとfarを計算する
	void UpdateSplitDist(vector<rxFrustum> &fs, float nd, float fd);

	// 光源からの平行投影行列の計算
	float ApplyCropMatrix(rxFrustum &f, const vector<Vec3> &bcns, const vector<Vec3> &bsls);

	void CameraInverse(float dst[16], float src[16]);


	//
	// シャドウマップ
	//
	// シャドウマップの初期化
	void InitShadowMap(int w, int h, double fov, int cut = 4, int size = 2048);

	// シャドウマップ(デプステクスチャ)の作成
	void MakeShadowMap(const Vec4 &light_dir, const Vec3 &eye_pos, const Vec3 &eye_dir, 
					   boost::function<void (void)> fpDraw, 
					   const vector<Vec3> &bcns, const vector<Vec3> &bsls);

	// 影付きでシーン描画
	void RenderSceneWithShadow(boost::function<void (void)> fpDraw, int w, int h);


	//
	// シャドウマップ確認描画
	//
	// カメラ視野角の描画
	void OverviewCam(boost::function<void (void)> fpDraw, int w, int h);

	// デプスマップの描画
	void DrawDepthTex(void);

};


/*!
 * シャドウマップの初期化
 * @param[in] w,h  ウィンドウサイズ
 * @param[in] fov  視野角
 * @param[in] cut 視錐台の分割数
 * @param[in] size シャドウマップの解像度
 */
inline void rxShadowMap::InitShadowMap(int w, int h, double fov, int cut, int size)
{
	// MARK:InitShadowMap
	m_iFrustumCutNum = cut;
	m_iDepthSize = size;

	glGenFramebuffersEXT(1, &m_iFBDepth);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_iFBDepth);
	glDrawBuffer(GL_NONE);
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glGenTextures(1, &m_iTexDepth);

	glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, m_iTexDepth);
	glTexImage3D(GL_TEXTURE_2D_ARRAY_EXT, 0, GL_DEPTH_COMPONENT24, m_iDepthSize, m_iDepthSize, RX_MAX_SPLITS, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);

	SetShadowGLSL(m_vShadGLSL);
	//m_GLSLView = CreateGLSLFromFile("shader/shadow_view.vs", "shader/shadow_view.fs", "view");
	m_GLSLView = CreateGLSL(shadow_view_vs, shadow_view_fs, "view");

	m_vFrustums.resize(RX_MAX_SPLITS);
	BOOST_FOREACH(rxFrustum &f, m_vFrustums){
		f = CalFrustum(w, h, fov);
	}

	m_vMatShadowCPM.resize(RX_MAX_SPLITS);
	BOOST_FOREACH(rxMatrix4 &m, m_vMatShadowCPM){
		m.MakeIdentity();
	}
}


/*!
 * シャドウマップ(デプステクスチャ)の作成
 * @param[in] light_pos 光源位置(点光源,指向性無し)
 * @param[in] eye_pos   視点位置
 * @param[in] eye_dir   視点方向
 * @param[in] fpDraw	描画関数のポインタ
 * @param[in] bcns      描画オブジェクトBBoxの中心座標
 * @param[in] bsls      描画オブジェクトBBoxの辺の長さの半分
 */
inline void rxShadowMap::MakeShadowMap(const Vec4 &light_dir, const Vec3 &eye_pos, const Vec3 &eye_dir, 
				   boost::function<void (void)> fpDraw, 
				   const vector<Vec3> &bcns, const vector<Vec3> &bsls)
{
	// MARK:MakeShadowMap
	if(!m_iFrustumCutNum){ return; }

	float shad_modelview[16];
	//glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	// glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	// 光源位置を視点に設定
	//gluLookAt(light_pos[0], light_pos[1], light_pos[2], 0, 0, 0, -1.0f, 0.0f, 0.0f);
	gluLookAt(0, 0, 0, -light_dir[0], -light_dir[1], -light_dir[2], -1.0f, 0.0f, 0.0f);
	glGetFloatv(GL_MODELVIEW_MATRIX, shad_modelview);

	// デプステクスチャ用FBO
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_iFBDepth);
	glPushAttrib(GL_VIEWPORT_BIT);					// スクリーンビューポートを一時待避
	glViewport(0, 0, m_iDepthSize, m_iDepthSize);	// シャドウマップのサイズに合わせる

	// Zファイティングを防ぐためにポリゴンオフセットを設定
	glPolygonOffset(0.0, 1.0);
	//glPolygonOffset(1.0, 4096.0);
	glEnable(GL_POLYGON_OFFSET_FILL);

	glDisable(GL_CULL_FACE);

	// 視錐台の分割の更新
	UpdateSplitDist(m_vFrustums, 0.01f, RX_FAR_DIST);

	//Vec3 eye_pos = g_viewObject.CalInverseTransform(Vec3(0.0));
	//Vec3 eye_dir = Unit(g_viewObject.CalInverseRotation(Vec3(0.0, 0.0, -1.0)));

	//eye_dir *= -1.0;

	//g_cbDraw << "eye_pos = " << eye_pos << "\n";
	//g_cbDraw << "eye_dir = " << eye_dir << "\n";
	m_v3EyePos = eye_pos;
	m_v3EyeDir = eye_dir;

	m_v3LightPos = Vec3(light_dir[0], light_dir[1], light_dir[2]);

	for(int i = 0; i < m_iFrustumCutNum; ++i){
		// ワールド座標での視錐台スライスの境界点を更新
		UpdateFrustumPoints(m_vFrustums[i], eye_pos, eye_dir);

		// 光源からの平行投影行列を視錐台スライスの境界点をもとに設定
		float minZ = ApplyCropMatrix(m_vFrustums[i], bcns, bsls);

		// 3Dテクスチャを用いたデプスマップ
		glFramebufferTextureLayerEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, m_iTexDepth, 0, i);

		glClear(GL_DEPTH_BUFFER_BIT);

		// 光源から見たシーンの描画
		glMatrixMode(GL_MODELVIEW);
		fpDraw();
		
		glMatrixMode(GL_PROJECTION);
		glMultMatrixf(shad_modelview);

		float shadow_cpm[16];
		glGetFloatv(GL_PROJECTION_MATRIX, shadow_cpm);
		m_vMatShadowCPM[i].SetValueT<float>(shadow_cpm);

		//glGetFloatv(GL_PROJECTION_MATRIX, shad_cpm[i]);
	}


	glDisable(GL_POLYGON_OFFSET_FILL);
	glPopAttrib(); 
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_CULL_FACE);

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
}

/*!
 * 影付きでシーン描画
 * @param[in] fpDraw 描画関数のポインタ
 * @param[in] w,h	ウィンドウサイズ
 */
inline void rxShadowMap::RenderSceneWithShadow(boost::function<void (void)> fpDraw, int w, int h)
{
	// MARK:RenderSceneWithShadow
	float cam_proj[16];
	float cam_modelview[16];
	float cam_inverse_modelview[16];
	float far_bound[RX_MAX_SPLITS];
	const float bias[16] = {	0.5f, 0.0f, 0.0f, 0.0f, 
								0.0f, 0.5f, 0.0f, 0.0f,
								0.0f, 0.0f, 0.5f, 0.0f,
								0.5f, 0.5f, 0.5f, 1.0f	};


	// 影計算のためにモデルビュー行列の逆行列を計算
	glGetFloatv(GL_MODELVIEW_MATRIX, cam_modelview);
	CameraInverse(cam_inverse_modelview, cam_modelview);

	//g_cbDraw << m_iTexDepth << "\n";

	if(!m_iFrustumCutNum){
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(45.0, (double)w/(double)h, m_vFrustums[0].Near, m_vFrustums[0].Far);

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		fpDraw();

		glPopMatrix();
	}
	else{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(45.0, (double)w/(double)h, m_vFrustums[0].Near, m_vFrustums[m_iFrustumCutNum-1].Far);
		glGetFloatv(GL_PROJECTION_MATRIX, cam_proj);

		//
		// デプスマップのバインド
		//
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, m_iTexDepth);
		if(m_iShadowType >= 4){
			glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
		}
		else{
			glTexParameteri(GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_MODE, GL_NONE);
		}

		for(int i = m_iFrustumCutNum; i < RX_MAX_SPLITS; ++i){
			far_bound[i] = 0;
		}

		// スライスごとのテクスチャをアクティブに
		for(int i = 0; i < m_iFrustumCutNum; ++i){
			float light_m[16];

			// カメラ同次座標系でのFarを計算(cam_proj*(0,0,Far,1)^tを計算後,[0,1]に正規化)
			far_bound[i] = (float)(0.5f*(-m_vFrustums[i].Far*cam_proj[10]+cam_proj[14])/m_vFrustums[i].Far+0.5f);

			// 保存しておいた光源座標系への変換行列を用いて，
			// 光源座標系に変換してテクスチャを貼り付け
			glActiveTexture(GL_TEXTURE0+(GLenum)i);
			glMatrixMode(GL_TEXTURE);
			glLoadMatrixf(bias);

			float shadow_cpm[16];
			m_vMatShadowCPM[i].GetValueT<float>(shadow_cpm);

			glMultMatrixf(shadow_cpm);
			glMultMatrixf(cam_inverse_modelview);
			
			// compute a normal matrix for the same thing (to transform the normals)
			// Basically, N = ((L)^-1)^-t
			glGetFloatv(GL_TEXTURE_MATRIX, light_m);
			rxMatrix4 nm;
			nm.SetValueT<float>(light_m);
			nm = nm.Inverse();
			nm = nm.Transpose();

			float m[16];
			nm.GetValueT<float>(m);
			glActiveTexture(GL_TEXTURE0 + (GLenum)(i+4));
			glMatrixMode(GL_TEXTURE);
			glLoadMatrixf(m);
		}

		// GLSL
		glUseProgram(m_vShadGLSL[m_iShadowType].Prog);
		glUniform1i(glGetUniformLocation(m_vShadGLSL[m_iShadowType].Prog, "stex"), 0);	// depth-maps
		glUniform1i(glGetUniformLocation(m_vShadGLSL[m_iShadowType].Prog, "tex"), 1);	// other tex
		glUniform4fv(glGetUniformLocation(m_vShadGLSL[m_iShadowType].Prog, "far_d"), 1, far_bound);
		if(m_iShadowType >= 4){
			glUniform2f(glGetUniformLocation(m_vShadGLSL[m_iShadowType].Prog, "texSize"), (float)m_iDepthSize, 1.0f/(float)m_iDepthSize);
		}

		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();

		fpDraw();

		glPopMatrix();

		glUseProgram(0);
	}
}

/*!
 * 影付きでシーン描画
 * @param[in] fpDraw 描画関数のポインタ
 * @param[in] w,h	ウィンドウサイズ
 */
inline void rxShadowMap::OverviewCam(boost::function<void (void)> fpDraw, int w, int h)
{
	// HACK:OverviewCam
	glPushAttrib(GL_VIEWPORT_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(m_iWidth-129, 0, 128, 128);
	glEnable(GL_DEPTH_TEST);
	glClear(GL_DEPTH_BUFFER_BIT);


	glActiveTexture(GL_TEXTURE0);
	glMatrixMode(GL_TEXTURE);
	glLoadIdentity();

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glPointSize(10);
	glColor3f(1.0f, 1.0f, 0.0f);
	gluLookAt(0, RX_FAR_DIST/2, 0, 0, 0, 0, 0, 0, 1.0f);

	glScalef(0.2f, 0.2f, 0.2f);
	glRotatef(20, 1, 0, 0);
	for(int i = 0; i < m_iFrustumCutNum; ++i){
		glBegin(GL_LINE_LOOP);
		for(int j=0; j<4; j++)
			glVertex3dv(m_vFrustums[i].Point[j].data);

		glEnd();
		glBegin(GL_LINE_LOOP);
		for(int j=4; j<8; j++)
			glVertex3dv(m_vFrustums[i].Point[j].data);
		glEnd();
	}

	for(int j = 0; j < 4; ++j){
		glBegin(GL_LINE_STRIP);
		glVertex3dv(m_v3EyePos.data);
		for(int i = 0; i < m_iFrustumCutNum; ++i){
			glVertex3dv(m_vFrustums[i].Point[j].data);
		}

		glVertex3dv(m_vFrustums[m_iFrustumCutNum-1].Point[j+4].data);
		glEnd();
	}

	glPushAttrib(GL_LIGHTING_BIT);

	GLfloat light_pos[4];
	for(int i = 0; i < 3; ++i) light_pos[i] = (GLfloat)m_v3LightPos[i];
	light_pos[3] = 1.0;
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
	glColor3f(0.9f, 0.9f, 1.0f);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	fpDraw();
	
	glPopAttrib();
	glPopAttrib();
}

/*!
 * デプスマップの描画
 */
inline void rxShadowMap::DrawDepthTex(void)
{
	int loc;
	glPushAttrib(GL_VIEWPORT_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glUseProgram(m_GLSLView.Prog);
	glUniform1i(glGetUniformLocation(m_GLSLView.Prog, "tex"), 0);
	loc = glGetUniformLocation(m_GLSLView.Prog, "layer");
	for(int i = 0; i < m_iFrustumCutNum; ++i){
		glViewport(130*i, 0, 128, 128);
		glBindTexture(GL_TEXTURE_2D_ARRAY_EXT, m_iTexDepth);
		glTexParameteri( GL_TEXTURE_2D_ARRAY_EXT, GL_TEXTURE_COMPARE_MODE, GL_NONE);
		glUniform1f(loc, (float)i);
		glBegin(GL_QUADS);
		glVertex3f(-1.0f, -1.0f, 0.0f);
		glVertex3f( 1.0f, -1.0f, 0.0f);
		glVertex3f( 1.0f,  1.0f, 0.0f);
		glVertex3f(-1.0f,  1.0f, 0.0f);
		glEnd();

	}
	glUseProgram(0);

	glEnable(GL_CULL_FACE);
	glPopAttrib();
}


/*!
 * 影の生成方法を選択
 * @param[in] n 生成方法番号
 */
inline void rxShadowMap::SetShadowType(int n)
{
	int size = (int)m_vShadGLSL.size();
	if(n >= size) n = size-1;
	if(n < 0) n = 0;

	m_iShadowType = n;
}

/*!
 * 設定されている影の生成方法を取得
 * @return 生成方法(文字列)
 */
inline string rxShadowMap::GetCurrentShadowType(void)
{
	return m_vShadGLSL[m_iShadowType].Name;
}



/*!
 * シャドウマップ用GLSLプログラムのセット
 * @param[out] gss GLSLプログラム
 */
inline void rxShadowMap::SetShadowGLSL(vector<rxGLSL> &gss)
{
	// MARK:SetShadowGLSL
	//string vert = "shader/shadow.vs";

	gss.clear();
	gss.push_back(CreateGLSL(shadow_vs, shadow_single_fs, 			"Normal Mode"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_single_hl_fs, 		"Show Splits"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_multi_fs, 			"Smooth shadows"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_multi_noleak_fs,		"Smooth shadows, no leak"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_pcf_fs,				"PCF"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_pcf_trilinear_fs,	"PCF w/ trilinear"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_pcf_4tap_fs,			"PCF w/ 4 taps"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_pcf_8tap_fs,			"PCF w/ 8 random taps"));
	gss.push_back(CreateGLSL(shadow_vs, shadow_pcf_gaussian_fs,		"PCF w/ gaussian blur"));

	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_single.fs", 			"Normal Mode"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_single_hl.fs", 		"Show Splits"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_multi.fs", 			"Smooth shadows"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_multi_noleak.fs",		"Smooth shadows, no leak"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_pcf.fs",				"PCF"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_pcf_trilinear.fs",	"PCF w/ trilinear"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_pcf_4tap.fs",			"PCF w/ 4 taps"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_pcf_8tap.fs",			"PCF w/ 8 random taps"));
	//gss.push_back(CreateGLSLFromFile(vert.c_str(), "shader/shadow_pcf_gaussian.fs",		"PCF w/ gaussian blur"));
}

/*!
 * カメラ行列の逆変換計算
 * @param[out] dst 逆変換行列
 * @param[in] src カメラ行列
 * @return 
 */
inline void rxShadowMap::CameraInverse(float dst[16], float src[16])
{
	dst[0] = src[0];
	dst[1] = src[4];
	dst[2] = src[8];
	dst[3] = 0.0f;
	dst[4] = src[1];
	dst[5] = src[5];
	dst[6]  = src[9];
	dst[7] = 0.0f;
	dst[8] = src[2];
	dst[9] = src[6];
	dst[10] = src[10];
	dst[11] = 0.0f;
	dst[12] = -(src[12] * src[0]) - (src[13] * src[1]) - (src[14] * src[2]);
	dst[13] = -(src[12] * src[4]) - (src[13] * src[5]) - (src[14] * src[6]);
	dst[14] = -(src[12] * src[8]) - (src[13] * src[9]) - (src[14] * src[10]);
	dst[15] = 1.0f;
}


/*!
 * 視錐台の計算
 * @param[in] w,h ウィンドウサイズ
 * @param[in] fov 視野角(deg)
 * @return 視錐台オブジェクト
 */
inline rxFrustum rxShadowMap::CalFrustum(int w, int h, double fov)
{
	rxFrustum fr;
	fr.FOV = RX_TO_RADIANS(fov)+0.2;
	fr.Ratio = (double)w/(double)h;
	return fr;
}

/*!
 * 全視錐台の計算
 * @param[in] w,h ウィンドウサイズ
 * @param[in] fov 視野角(deg)
 */
inline void rxShadowMap::CalAllFrustums(int w, int h, double fov)
{
	m_iWidth = w;
	m_iHeight = h;
	BOOST_FOREACH(rxFrustum &f, m_vFrustums){
		f = CalFrustum(w, h, fov);
	}
}

/*!
 * 視錐台の分割数を設定
 * @param[in] n 分割数
 */
inline void rxShadowMap::SetFrustumCutNum(int n)
{
	if(n > 4) n = 4;
	if(n < 0) n = 0;

	m_iFrustumCutNum = n;
}
/*!
 * 視錐台の分割数を取得
 * @return 分割数
 */
inline int rxShadowMap::GetFrustumCutNum(void) const { return m_iFrustumCutNum; }


/*!
 * 視錐台の分割数を増減
 * @param[in] d 増加数(負の値で減少数)
 */
inline void rxShadowMap::IncreaseFrustumCutNum(int d)
{
	m_iFrustumCutNum += d;

	if(m_iFrustumCutNum > 4) m_iFrustumCutNum = 4;
	if(m_iFrustumCutNum < 0) m_iFrustumCutNum = 0;
}

/*!
 * 視錐台の8コーナー頂点を計算する
 * @param[in] f 視錐台
 * @param[in] center 視点
 * @param[in] view_dir 視線方向
 */
inline void rxShadowMap::UpdateFrustumPoints(rxFrustum &f, const Vec3 &center, const Vec3 &view_dir)
{
	// MARK:UpdateFrustumPoints
	Vec3 up(0.0f, 1.0f, 0.0f);
	Vec3 right = cross(view_dir, up);

	Vec3 fc = center+view_dir*f.Far;
	Vec3 nc = center+view_dir*f.Near;

	right = Unit(right);
	up = Unit(cross(right, view_dir));

	// these heights and widths are half the heights and widths of
	// the near and far plane rectangles
	double near_height = tan(f.FOV/2.0f)*f.Near;
	double near_width  = near_height*f.Ratio;
	double far_height  = tan(f.FOV/2.0f)*f.Far;
	double far_width   = far_height*f.Ratio;

	f.Point[0] = nc-up*near_height-right*near_width;
	f.Point[1] = nc+up*near_height-right*near_width;
	f.Point[2] = nc+up*near_height+right*near_width;
	f.Point[3] = nc-up*near_height+right*near_width;

	f.Point[4] = fc-up*far_height-right*far_width;
	f.Point[5] = fc+up*far_height-right*far_width;
	f.Point[6] = fc+up*far_height+right*far_width;
	f.Point[7] = fc-up*far_height+right*far_width;
}

/*!
 * カメラ視点空間での各視錐台スライスのnearとfarを計算する
 * @param[in] fs 視錐台スライス
 * @param[in] nd 全体のnear
 * @param[in] fd 全体のfar
 */
inline void rxShadowMap::UpdateSplitDist(vector<rxFrustum> &fs, float nd, float fd)
{
	double lambda = 0.75;
	double ratio = fd/nd;
	fs[0].Near = nd;

	for(int i = 1; i < m_iFrustumCutNum; i++){
		double si = i/(double)m_iFrustumCutNum;

		fs[i].Near = lambda*(nd*powf(ratio, si))+(1-lambda)*(nd+(fd-nd)*si);
		fs[i-1].Far = fs[i].Near*1.05f;
	}

	if(m_iFrustumCutNum > 0){
		fs[m_iFrustumCutNum-1].Far = fd;
	}
}

/*!
 * 光源からの平行投影行列の計算
 *  - 最初に適切なzの範囲を計算し，平行投影をセットする．
 *  - 次に視錐台スライスにフィットするように平行移動/スケーリングする
 * @param[in] f 対応する視錐台スライス
 * @param[in] bcns 描画オブジェクトBBoxの中心座標
 * @param[in] bsls 描画オブジェクトBBoxの辺の長さの半分
 * @return zの最小値
 */
inline float rxShadowMap::ApplyCropMatrix(rxFrustum &f, const vector<Vec3> &bcns, const vector<Vec3> &bsls)
{
	// MARK:ApplyCropMatrix
	float shad_modelview[16];
	float shad_proj[16];
	float shad_crop[16];
	float shad_mvp[16];
	float maxX = -1000.0f;
	float maxY = -1000.0f;
	float maxZ;
	float minX =  1000.0f;
	float minY =  1000.0f;
	float minZ;

	rxMatrix4 mvp;
	Vec4 transf;	
	
	//
	// 現在の視錐台(光源を視点とした)のzの範囲を検索
	//
	// 光源視点のモデルビュー行列を取り出す
	glGetFloatv(GL_MODELVIEW_MATRIX, shad_modelview);
	mvp.SetValueT<float>(shad_modelview);

	// 視錐台スライスの8頂点にモデルビュー行列を掛け，最大・最小のz値を探索
	// (z要素のみ必要なので以下のように単純化も可能)
	 transf[2] = shad_modelview[2]*f.Point[0][0]+shad_modelview[6]*f.Point[0][1]+shad_modelview[10]*f.Point[0][1]+shad_modelview[14];
	mvp.multMatrixVec(Vec4(f.Point[0], 1.0f), transf);
	minZ = transf[2];
	maxZ = transf[2];
	for(int i = 1; i < 8; ++i){
		mvp.multMatrixVec(Vec4(f.Point[i], 1.0f), transf);

		if(transf[2] > maxZ) maxZ = transf[2];
		if(transf[2] < minZ) minZ = transf[2];
	}

	// 影をつけたいオブジェクトが確実に範囲内にあるようにz値を探索・変更
	for(int i = 0; i < (int)bcns.size(); ++i){
		Vec3 bcn = bcns[i];
		Vec3 bsl = bsls[i];
		mvp.multMatrixVec(Vec4(bcn[0], bcn[1], bcn[2], 1.0), transf);

		float diag = norm(bsl)/2.0f;
		if(transf[2]+diag > maxZ) maxZ = transf[2]+diag;
		//if(transf[2]-diag < minZ) minZ = transf[2]-diag;
	}


	//
	// 探索したzの範囲から投影を設定
	//
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//gluPerspective(RX_LIGHT_FOV, 1.0, minZ, maxZ); // 点光源の場合
	glOrtho(-1.0, 1.0, -1.0, 1.0, -maxZ, -minZ);

	glGetFloatv(GL_PROJECTION_MATRIX, shad_proj);
	glPushMatrix();

	glMultMatrixf(shad_modelview);
	glGetFloatv(GL_PROJECTION_MATRIX, shad_mvp);
	glPopMatrix();


	//
	// x,y方向の光源視錐台の大きさを調整
	//
	mvp.SetValueT<float>(shad_mvp);
	for(int i = 0; i < 8; ++i){
		mvp.multMatrixVec(Vec4(f.Point[i], 1.0f), transf);

		transf[0] /= transf[3];
		transf[1] /= transf[3];

		if(transf[0] > maxX) maxX = transf[0];
		if(transf[0] < minX) minX = transf[0];
		if(transf[1] > maxY) maxY = transf[1];
		if(transf[1] < minY) minY = transf[1];
	}

	float scaleX = 2.0f/(maxX-minX);
	float scaleY = 2.0f/(maxY-minY);
	float offsetX = -0.5f*(maxX+minX)*scaleX;
	float offsetY = -0.5f*(maxY+minY)*scaleY;

	// 光源視錐台をトリミングする行列をglOrthoから得た投影行列に適用
	mvp.MakeIdentity();
	mvp(0, 0) = scaleX;
	mvp(1, 1) = scaleY;
	mvp(0, 3) = offsetX;
	mvp(1, 3) = offsetY;
	//mvp = mvp.Transpose();
	mvp.GetValueT<float>(shad_crop);

	glLoadMatrixf(shad_crop);
	glMultMatrixf(shad_proj);

	return minZ;
}



#endif // #ifndef _RX_SHADOW_MAP_H_