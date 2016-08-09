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

rxFrustum CalFrustum(double fov_deg, double near_d, double far_d, int w, int h, 
					 Vec3 pos, Vec3 lookat = Vec3(0.0), Vec3 up = Vec3(0.0, 1.0, 0.0));
void SetFrustum(const rxFrustum &f);
bool CalInvMat4x4(const GLfloat m[16], GLfloat invm[16]);


//-----------------------------------------------------------------------------
// Shadow Map シェーダ
//-----------------------------------------------------------------------------
const char shadow1_vs[] = RXSTR(
#version 120 @

// フラグメントシェーダに値を渡すための変数
varying vec4 vPos;
varying vec3 vNrm;
varying vec4 vShadowCoord;	//!< シャドウデプスマップの参照用座標

void main(void)
{
	// フラグメントシェーダでの計算用(モデルビュー変換のみ)
	vPos = gl_ModelViewMatrix*gl_Vertex;			// 頂点位置
	vNrm = normalize(gl_NormalMatrix*gl_Normal);	// 頂点法線
	vShadowCoord = gl_TextureMatrix[7]*gl_ModelViewMatrix*gl_Vertex;	// 影用座標値(光源中心座標)

	// 描画用
	gl_Position = gl_ProjectionMatrix*vPos;	// 頂点位置
	gl_FrontColor = gl_Color;				// 頂点色
	gl_TexCoord[0] = gl_MultiTexCoord0;		// 頂点テクスチャ座標
}
);

const char shadow1_fs[] = RXSTR(
#version 120 @

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;
varying vec4 vShadowCoord;


// GLから設定される定数(uniform)
uniform sampler2D tex;			//!< 模様
uniform sampler2D depth_tex;	//!< デプス値テクスチャ

// 影の濃さ
uniform float shadow_ambient;

/*!
 * Phong反射モデルによるシェーディング
 * @return 表面反射色
 */
vec4 PhongShading(void)
{
	vec3 N = normalize(vNrm);			// 法線ベクトル
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// ライトベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	vec4 ambient = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float dcoef = max(dot(L, N), 0.0);
	float dcoef = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	vec4 diffuse = vec4(0.0);
	vec4 specular = vec4(0.0);
	if(dcoef > 0.0){
		vec3 V = normalize(-vPos.xyz);		// 視線ベクトル

		// 反射ベクトルの計算(フォン反射モデル)
		vec3 R = reflect(-L, N);
		//vec3 R = 2.0*dot(N, L)*N-L;	// reflect関数を用いない場合
		float specularLight = pow(max(dot(R, V), 0.0), gl_FrontMaterial.shininess);

		diffuse  = gl_FrontLightProduct[0].diffuse*dcoef;
		specular = gl_FrontLightProduct[0].specular*specularLight;
	}
	return ambient+diffuse+specular;
}

/*!
 * 影生成のための係数(影のあるところで1, それ以外で0)
 * @return 影係数(影のあるところで1, それ以外で0)
 */
float ShadowCoef(void)
{
	// 光源座標
	vec4 shadow_coord1 = vShadowCoord/vShadowCoord.w;

	// 光源からのデプス値(視点)
	float view_d = shadow_coord1.z;//-0.0001;
	
	// 格納された光源からの最小デプス値を取得
	float light_d = texture2D(depth_tex, shadow_coord1.xy).x;

	// 影で0,日向で1
	float shadow_coef = 1.0;
	if(vShadowCoord.w > 0.0){
		shadow_coef = light_d < view_d ? 0.0 : 1.0;
	}

	return shadow_coef;
}

void main(void)
{	
	vec4 light_col = PhongShading();	// 表面反射色
	float shadow_coef = ShadowCoef();	// 影影響係数

	// 出力
	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = light_col;
}
);


//-----------------------------------------------------------------------------
// シャドウマッピングクラス
//-----------------------------------------------------------------------------
class rxShadowMap
{
	GLuint m_iFBODepth;		//!< 光源から見たときのデプスを格納するFramebuffer object
	GLuint m_iTexDepth;		//!< m_iFBODepthにattachするテクスチャ
	double m_fDepthSize[2];	//!< デプスを格納するテクスチャのサイズ

	rxGLSL m_glslShading;	//!< GLSLシェーダ

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
		glewInit();
		if(!glewIsSupported("GL_ARB_depth_texture "
							"GL_ARB_shadow "
							)){
			cout << "ERROR: Support for necessary OpenGL extensions missing." << endl;
			return;
		}

		m_fDepthSize[0] = w;
		m_fDepthSize[1] = h;
	
		// デプス値テクスチャ
		glActiveTexture(GL_TEXTURE7);
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
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, m_fDepthSize[0], m_fDepthSize[1], 0, 
					 GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	

		// FBO作成
		glGenFramebuffersEXT(1, &m_iFBODepth);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, m_iFBODepth);

		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);
	
		// デプスマップテクスチャをFBOに接続
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_iTexDepth, 0);
	
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// GLSL
		m_glslShading = CreateGLSL(shadow1_vs, shadow1_fs, "shadow");
	}


	/*!
	 * 影付きでシーン描画
	 * @param[in] light 光源
	 * @param[in] fpDraw 描画関数のポインタ
	 */
	void RenderSceneWithShadow(rxFrustum &light, void (*fpDraw)(void*), void* func_obj, bool self_shading = false)
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
		glBindFramebuffer(GL_FRAMEBUFFER, m_iFBODepth);	// FBOにレンダリング

		// カラー，デプスバッファのクリア
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearDepth(1.0f);

		// ビューポートをシャドウマップの大きさに変更
		glViewport(0, 0, m_fDepthSize[0], m_fDepthSize[1]);
	
		// 光源を視点として設定
		glMatrixMode(GL_PROJECTION);
		glLoadMatrixf(light_proj);
		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(light_modelview);
	
		// デプス値以外の色のレンダリングを無効にする
		glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE); 
	
		glPolygonOffset(1.1f, 40.0f);
		glEnable(GL_POLYGON_OFFSET_FILL);

		glEnable(GL_TEXTURE_2D);	
	
		glDisable(GL_LIGHTING);
		if(self_shading){
			glDisable(GL_CULL_FACE);
		}
		else{
			glEnable(GL_CULL_FACE);
			glCullFace(GL_FRONT);
		}

		glUseProgram(0);
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
		glMultMatrixf(light_proj);
		glMultMatrixf(light_modelview);

		//// 現在のモデルビューの逆行列をかけておく
		GLfloat camera_modelview_inv[16];
		CalInvMat4x4(camera_modelview, camera_modelview_inv);
		glMultMatrixf(camera_modelview_inv);

		// 回転と平行移動で分けて計算する場合(こっちの方が高速)
		//GLfloat rot[16];
		//rot[0] = camera_modelview[0]; rot[1] = camera_modelview[1]; rot[2]  = camera_modelview[2];  rot[3]  = 0.0f;
		//rot[4] = camera_modelview[4]; rot[5] = camera_modelview[5]; rot[6]  = camera_modelview[6];  rot[7]  = 0.0f;
		//rot[8] = camera_modelview[8]; rot[9] = camera_modelview[9]; rot[10] = camera_modelview[10]; rot[11] = 0.0f;
		//rot[12] = 0.0f;               rot[13] = 0.0f;               rot[14] = 0.0f;                 rot[15] = 1.0f;
		//glMultTransposeMatrixf(rot);
		//glTranslatef(-camera_modelview[12], -camera_modelview[13], -camera_modelview[14]);
	
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		// 無効にした色のレンダリングを有効にする
		glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE); 

		// 元のビューポート行列に戻す
		glViewport(viewport[0], viewport[1], viewport[2], viewport[3]);

		// 退避させておいた視点行列を元に戻す
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();


		//
		// カメラから見たときのシーン描画
		// 
		glEnable(GL_TEXTURE_2D);

		// デプステクスチャを貼り付け
		glActiveTexture(GL_TEXTURE7);
		glBindTexture(GL_TEXTURE_2D, m_iTexDepth);
		
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		// GLSLシェーダをセット
		glUseProgram(m_glslShading.Prog);
		glUniform1i(glGetUniformLocation(m_glslShading.Prog, "depth_tex"), 7);
		glUniform1f(glGetUniformLocation(m_glslShading.Prog, "shadow_ambient"), 0.7f);

		fpDraw(func_obj);

		glUseProgram(0);
		glActiveTexture(GL_TEXTURE0);

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
							Vec3 pos, Vec3 lookat, Vec3 up)
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

/*!
 * 4x4行列の行列式の計算
 *  | m[0]  m[1]  m[2]  m[3]  |
 *  | m[4]  m[5]  m[6]  m[7]  |
 *  | m[8]  m[9]  m[10] m[11] |
 *  | m[12] m[13] m[14] m[15] |
 * @param[in] m 元の行列
 * @return 行列式の値
 */
inline double CalDetMat4x4(const GLfloat m[16])
{
	return m[0]*m[5]*m[10]*m[15]+m[0]*m[6]*m[11]*m[13]+m[0]*m[7]*m[9]*m[14]
		  +m[1]*m[4]*m[11]*m[14]+m[1]*m[6]*m[8]*m[15]+m[1]*m[7]*m[10]*m[12]
		  +m[2]*m[4]*m[9]*m[15]+m[2]*m[5]*m[11]*m[12]+m[2]*m[7]*m[8]*m[13]
		  +m[3]*m[4]*m[10]*m[13]+m[3]*m[5]*m[8]*m[14]+m[3]*m[6]*m[9]*m[12]
		  -m[0]*m[5]*m[11]*m[14]-m[0]*m[6]*m[9]*m[15]-m[0]*m[7]*m[10]*m[13]
		  -m[1]*m[4]*m[10]*m[15]-m[1]*m[6]*m[11]*m[12]-m[1]*m[7]*m[8]*m[14]
		  -m[2]*m[4]*m[11]*m[13]-m[2]*m[5]*m[8]*m[15]-m[2]*m[7]*m[9]*m[12]
		  -m[3]*m[4]*m[9]*m[14]-m[3]*m[5]*m[10]*m[12]-m[3]*m[6]*m[8]*m[13];
}
 
/*!
 * 4x4行列の行列式の計算
 *  | m[0]  m[1]  m[2]  m[3]  |
 *  | m[4]  m[5]  m[6]  m[7]  |
 *  | m[8]  m[9]  m[10] m[11] |
 *  | m[12] m[13] m[14] m[15] |
 * @param[in] m 元の行列
 * @param[out] invm 逆行列
 * @return 逆行列の存在
 */
inline bool CalInvMat4x4(const GLfloat m[16], GLfloat invm[16])
{
	GLfloat det = CalDetMat4x4(m);
	if(fabs(det) < RX_FEQ_EPS){
		return false;
	}
	else{
		GLfloat inv_det = 1.0/det;
 
		invm[0]  = inv_det*(m[5]*m[10]*m[15]+m[6]*m[11]*m[13]+m[7]*m[9]*m[14]-m[5]*m[11]*m[14]-m[6]*m[9]*m[15]-m[7]*m[10]*m[13]);
		invm[1]  = inv_det*(m[1]*m[11]*m[14]+m[2]*m[9]*m[15]+m[3]*m[10]*m[13]-m[1]*m[10]*m[15]-m[2]*m[11]*m[13]-m[3]*m[9]*m[14]);
		invm[2]  = inv_det*(m[1]*m[6]*m[15]+m[2]*m[7]*m[13]+m[3]*m[5]*m[14]-m[1]*m[7]*m[14]-m[2]*m[5]*m[15]-m[3]*m[6]*m[13]);
		invm[3]  = inv_det*(m[1]*m[7]*m[10]+m[2]*m[5]*m[11]+m[3]*m[6]*m[9]-m[1]*m[6]*m[11]-m[2]*m[7]*m[9]-m[3]*m[5]*m[10]);
 
		invm[4]  = inv_det*(m[4]*m[11]*m[14]+m[6]*m[8]*m[15]+m[7]*m[10]*m[12]-m[4]*m[10]*m[15]-m[6]*m[11]*m[12]-m[7]*m[8]*m[14]);
		invm[5]  = inv_det*(m[0]*m[10]*m[15]+m[2]*m[11]*m[12]+m[3]*m[8]*m[14]-m[0]*m[11]*m[14]-m[2]*m[8]*m[15]-m[3]*m[10]*m[12]);
		invm[6]  = inv_det*(m[0]*m[7]*m[14]+m[2]*m[4]*m[15]+m[3]*m[6]*m[12]-m[0]*m[6]*m[15]-m[2]*m[7]*m[12]-m[3]*m[4]*m[14]);
		invm[7]  = inv_det*(m[0]*m[6]*m[11]+m[2]*m[7]*m[8]+m[3]*m[4]*m[10]-m[0]*m[7]*m[10]-m[2]*m[4]*m[11]-m[3]*m[6]*m[8]);
 
		invm[8]  = inv_det*(m[4]*m[9]*m[15]+m[5]*m[11]*m[12]+m[7]*m[8]*m[13]-m[4]*m[11]*m[13]-m[5]*m[8]*m[15]-m[7]*m[9]*m[12]);
		invm[9]  = inv_det*(m[0]*m[11]*m[13]+m[1]*m[8]*m[15]+m[3]*m[9]*m[12]-m[0]*m[9]*m[15]-m[1]*m[11]*m[12]-m[3]*m[8]*m[13]);
		invm[10] = inv_det*(m[0]*m[5]*m[15]+m[1]*m[7]*m[12]+m[3]*m[4]*m[13]-m[0]*m[7]*m[13]-m[1]*m[4]*m[15]-m[3]*m[5]*m[12]);
		invm[11] = inv_det*(m[0]*m[7]*m[9]+m[1]*m[4]*m[11]+m[3]*m[5]*m[8]-m[0]*m[5]*m[11]-m[1]*m[7]*m[8]-m[3]*m[4]*m[9]);
 
		invm[12] = inv_det*(m[4]*m[10]*m[13]+m[5]*m[8]*m[14]+m[6]*m[9]*m[12]-m[4]*m[9]*m[14]-m[5]*m[10]*m[12]-m[6]*m[8]*m[13]);
		invm[13] = inv_det*(m[0]*m[9]*m[14]+m[1]*m[10]*m[12]+m[2]*m[8]*m[13]-m[0]*m[10]*m[13]-m[1]*m[8]*m[14]-m[2]*m[9]*m[12]);
		invm[14] = inv_det*(m[0]*m[6]*m[13]+m[1]*m[4]*m[14]+m[2]*m[5]*m[12]-m[0]*m[5]*m[14]-m[1]*m[6]*m[12]-m[2]*m[4]*m[13]);
		invm[15] = inv_det*(m[0]*m[5]*m[10]+m[1]*m[6]*m[8]+m[2]*m[4]*m[9]-m[0]*m[6]*m[9]-m[1]*m[4]*m[10]-m[2]*m[5]*m[8]);
 
		return true;
	}
}


#endif // #ifdef _RX_SHADOWMAP_H_