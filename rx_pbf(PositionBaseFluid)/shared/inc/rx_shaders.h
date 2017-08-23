/*! 
 @file rx_shader.h

 @brief GLSLシェーダー
 
 @author Makoto Fujisawa
 @date 2009-11
*/
// FILE --rx_shader.h--


#ifndef _RX_SHADERS_H_
#define _RX_SHADERS_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <iostream>
#include <vector>
#include <string>

#include <GL/glew.h>
#include <GL/glut.h>


#define RXSTR(A) #A

using namespace std;


//-----------------------------------------------------------------------------
// HACK:GLSLシェーダ
//-----------------------------------------------------------------------------
struct rxGLSL
{
	string VertProg;
	string FragProg;
	string Name;
	GLuint Prog;
};

//-----------------------------------------------------------------------------
// Volume Rendering シェーダ
//-----------------------------------------------------------------------------
const char volume_vs[] = RXSTR(
varying vec3 texcoord;	// ボリュームテクスチャ用のテクスチャ座標
void main(void)
{
	gl_Position = gl_ProjectionMatrix*gl_Vertex;
	texcoord = (gl_ModelViewMatrixInverse*gl_Vertex).xyz;
	gl_ClipVertex = gl_Vertex;
}
);

const char volume_fs[] = RXSTR(
uniform sampler3D volume_tex;
uniform float thickness;	// スライス間距離(=スライス厚さ)
uniform float opacity;		// 透明度
uniform vec3 dens_col;		// ボリュームの色
varying vec3 texcoord;
void main(void)
{
	float d = texture3D(volume_tex, texcoord).x;
	gl_FragColor = vec4(dens_col.r, dens_col.g, dens_col.b, d*opacity*thickness);
	//gl_FragColor = vec4(d, 1.0, 1.0, 0.5);
}
);


//-----------------------------------------------------------------------------
// Shadow Map シェーダ
//-----------------------------------------------------------------------------
//MARK:Shadow Map vs
const char shadow_vs[] = RXSTR(
varying vec4 vPos;
varying vec3 vNrm;
void main(void)
{
	vPos = gl_ModelViewMatrix*gl_Vertex;
	vNrm = normalize(gl_NormalMatrix*gl_Normal);
	//vNrm = gl_NormalMatrix*gl_Normal;

	gl_Position = gl_ProjectionMatrix*vPos;
	//gl_FrontColor = gl_Color*gl_LightSource[0].diffuse*vec4(max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0));
	gl_FrontColor = gl_Color;
	gl_TexCoord[0] = gl_MultiTexCoord0;
	//gl_TexCoord[0] = gl_TextureMatrix[0] * gl_MultiTexCoord0;

}
);

// MARK:Normal Mode fs
const char shadow_single_fs[] = RXSTR( 
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;			//!< 模様
uniform sampler2DArray stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;				//!< 視錐台遠距離
varying vec4 vPos;
varying vec3 vNrm;

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	float light_d = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 格納された光源からの最小デプス値を取得
	float shadow_d = texture2DArray(stex, shadow_coord.xyz).x;
	
	// 光源からのデプス値と遮蔽を考慮したデプス値の差を求める
	float diff = shadow_d-light_d;

	// 影で0,日向で1を返す
	return clamp(diff*250.0+1.0, 0.0, 1.0);
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	light_col += color_tex;

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*color_tex;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
	//gl_FragColor = shadow_ambient*shadow_coef*light_col*color_tex+(1.0-shadow_ambient)*color_tex;

}
);


//! 複数のシャドウサンプルを使用(リーク抑制版)
// MARK:Show Splits fs
const char shadow_single_hl_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;			//!< 模様
uniform sampler2DArray stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;				//!< 視錐台遠距離
uniform vec4 color[4] = vec4[4](vec4(1.0, 0.5, 0.5, 1.0),
								vec4(0.5, 1.0, 0.5, 1.0),
								vec4(0.5, 0.5, 1.0, 1.0),
								vec4(1.0, 1.0, 0.5, 1.0));

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

vec4 ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	float light_d = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 格納された光源からの最小デプス値を取得
	float shadow_d = texture2DArray(stex, shadow_coord.xyz).x;
	
	// 光源からのデプス値と遮蔽を考慮したデプス値の差を求める
	float diff = shadow_d-light_d;

	// 影で0,日向で1を返す
	return clamp(diff*250.0+1.0, 0.0, 1.0)*color[index];
}

void main(void)
{
	const float shadow_ambient = 0.9;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	vec4 shadow_coef = ShadowCoef();
	gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


//! 複数のシャドウサンプルを使用
// MARK:Smooth shadows fs
const char shadow_multi_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;			//!< 模様
uniform sampler2DArray stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;				//!< 視錐台遠距離

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

// 周囲の影係数参照用
const int nsamples = 8;
uniform vec4 offset[nsamples] = vec4[nsamples](vec4(0.000000, 0.000000, 0.0, 0.0),
											   vec4(0.079821, 0.165750, 0.0, 0.0),
											   vec4(-0.331500, 0.159642, 0.0, 0.0),
											   vec4(-0.239463, -0.497250, 0.0, 0.0),
											   vec4(0.662999, -0.319284, 0.0, 0.0),
											   vec4(0.399104, 0.828749, 0.0, 0.0),
											   vec4(-0.994499, 0.478925, 0.0, 0.0),
											   vec4(-0.558746, -1.160249, 0.0, 0.0));

/*!
 * 影係数の計算
 * @param[in] shadow_coord シャドウマップ参照用テクスチャ座標
 * @param[in] light_d 参照位置での光源からのデプス値
 * @return 影係数(影のあるところで1, それ以外で0)
 */
float GetOccCoef(vec4 shadow_coord, float light_d)
{
	// 格納された光源からの最小デプス値を取得
	float shadow_d = texture2DArray(stex, shadow_coord.xyz).x;
	
	// 光源からのデプス値と遮蔽を考慮したデプス値の差を求める
	float diff = shadow_d-light_d;
	
	// 影で0,日向で1を返す
	return clamp(diff*250.0+1.0, 0.0, 1.0);
}

/*!
 * 影生成のための係数(影のあるところで1, それ以外で0)
 * @return 影係数(影のあるところで1, それ以外で0)
 */
float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	const float scale = 2.0/4096.0;

	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;
	
	// 光源からのデプス値(遮蔽無し)を待避
	float light_d = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 周囲の影係数も含めて取得
	float shadow_coef = 0.0;
	for(int i = 0; i < nsamples; ++i){
		shadow_coef += GetOccCoef(shadow_coord+scale*offset[i], light_d);
	}
	shadow_coef /= nsamples;
	
	return shadow_coef;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


//! 複数のシャドウサンプルを使用(リーク抑制版)
// MARK:Smooth shadows, no leak fs
const char shadow_multi_noleak_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;			//!< 模様
uniform sampler2DArray stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;				//!< 視錐台遠距離

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

// 周囲の影係数参照用
const int nsamples = 8;
uniform vec4 offset[nsamples] = vec4[nsamples](vec4(0.000000, 0.000000, 0.0, 0.0),
											   vec4(0.079821, 0.165750, 0.0, 0.0),
											   vec4(-0.331500, 0.159642, 0.0, 0.0),
											   vec4(-0.239463, -0.497250, 0.0, 0.0),
											   vec4(0.662999, -0.319284, 0.0, 0.0),
											   vec4(0.399104, 0.828749, 0.0, 0.0),
											   vec4(-0.994499, 0.478925, 0.0, 0.0),
											   vec4(-0.558746, -1.160249, 0.0, 0.0));

/*!
 * 影係数の計算
 * @param[in] shadow_coord シャドウマップ参照用テクスチャ座標
 * @param[in] light_d 参照位置での光源からのデプス値
 * @return 影係数(影のあるところで1, それ以外で0)
 */
float GetOccCoef(vec4 shadow_coord, float light_d)
{
	// 格納された光源からの最小デプス値を取得
	float shadow_d = texture2DArray(stex, shadow_coord.xyz).x;
	
	// 光源からのデプス値と遮蔽を考慮したデプス値の差を求める
	float diff = shadow_d-light_d;
	
	// 影で0,日向で1を返す
	return clamp(diff*250.0+1.0, 0.0, 1.0);
}

/*!
 * 影生成のための係数(影のあるところで1, それ以外で0)
 * @return 影係数(影のあるところで1, それ以外で0)
 */
float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	const float scale = 2.0/4096.0;

	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	vec4 light_normal4 = gl_TextureMatrix[index+4]*vec4(vNrm, 0.0);
	vec3 light_normal = normalize(light_normal4.xyz);
	
	float d = -dot(light_normal, shadow_coord.xyz);

	// 光源からのデプス値(遮蔽無し)を待避
	float light_d = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 周囲の影係数も含めて取得
	float shadow_coef = GetOccCoef(shadow_coord, light_d);
	for(int i = 1; i < nsamples; ++i){
		vec4 shadow_lookup = shadow_coord+scale*offset[i];

		float lookup_z = -(light_normal.x*shadow_lookup.x + light_normal.y*shadow_lookup.y + d)/light_normal.z;

		shadow_coef += GetOccCoef(shadow_lookup, lookup_z);
	}
	shadow_coef /= nsamples;
	
	return shadow_coef;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);

	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.9;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	//light_col += color_tex;

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


// sampler2DArrayShadowを使用
// MARK:PCF fs
const char shadow_pcf_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;				//!< 模様
uniform sampler2DArrayShadow stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;					//!< 視錐台遠距離
uniform vec2 texSize;				//!< x - size, y - 1/size

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	shadow_coord.w = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 影で0,日向で1を返す
	return shadow2DArray(stex, shadow_coord).x;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


// sampler2DArrayShadowを使用
// MARK:PCF w/ trilinear fs
const char shadow_pcf_trilinear_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;				//!< 模様
uniform sampler2DArrayShadow stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;					//!< 視錐台遠距離
uniform vec2 texSize;				//!< x - size, y - 1/size

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	float blend = 0.0;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
		blend = clamp( (gl_FragCoord.z-far_d.x*0.995)*200.0, 0.0, 1.0); 
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
		blend = clamp( (gl_FragCoord.z-far_d.y*0.995)*200.0, 0.0, 1.0); 
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
		blend = clamp( (gl_FragCoord.z-far_d.z*0.995)*200.0, 0.0, 1.0); 
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	shadow_coord.w = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);
	
	// 影係数の取得
	float ret = shadow2DArray(stex, shadow_coord).x;
	
	if(blend > 0.0){
		shadow_coord = gl_TextureMatrix[index+1]*vPos;
	
		shadow_coord.w = shadow_coord.z;
		shadow_coord.z = float(index+1);
		
		ret = ret*(1.0-blend) + shadow2DArray(stex, shadow_coord).x*blend; 
	}
	
	return ret;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


//! 
// MARK:PCF w/ 4 taps fs
const char shadow_pcf_4tap_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;				//!< 模様
uniform sampler2DArrayShadow stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;					//!< 視錐台遠距離
uniform vec2 texSize;				//!< x - size, y - 1/size

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	shadow_coord.w = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);

	// 重み付き4-tapバイリニアフィルタ
	vec2 pos = mod(shadow_coord.xy*texSize.x, 1.0);
	vec2 offset = (0.5-step(0.5, pos))*texSize.y;
	float ret = 0.0;
	ret += shadow2DArray(stex, shadow_coord+vec4( offset.x,  offset.y, 0, 0)).x * (pos.x) * (pos.y);
	ret += shadow2DArray(stex, shadow_coord+vec4( offset.x, -offset.y, 0, 0)).x * (pos.x) * (1-pos.y);
	ret += shadow2DArray(stex, shadow_coord+vec4(-offset.x,  offset.y, 0, 0)).x * (1-pos.x) * (pos.y);
	ret += shadow2DArray(stex, shadow_coord+vec4(-offset.x, -offset.y, 0, 0)).x * (1-pos.x) * (1-pos.y);
	
	return ret;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


//! 
// MARK:PCF w/ 8 random taps fs
const char shadow_pcf_8tap_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;				//!< 模様
uniform sampler2DArrayShadow stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;					//!< 視錐台遠距離
uniform vec2 texSize;				//!< x - size, y - 1/size

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

// 周囲の影係数参照用
const int nsamples = 8;
uniform vec4 offset[nsamples] = vec4[nsamples](vec4(0.000000, 0.000000, 0.0, 0.0),
											   vec4(0.079821, 0.165750, 0.0, 0.0),
											   vec4(-0.331500, 0.159642, 0.0, 0.0),
											   vec4(-0.239463, -0.497250, 0.0, 0.0),
											   vec4(0.662999, -0.319284, 0.0, 0.0),
											   vec4(0.399104, 0.828749, 0.0, 0.0),
											   vec4(-0.994499, 0.478925, 0.0, 0.0),
											   vec4(-0.558746, -1.160249, 0.0, 0.0));

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	shadow_coord.w = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);

	// 重み付き8-tapランダムフィルタ
	float ret = 0.0;
	for(int i = 0; i < nsamples; ++i){
		vec4 shadow_lookup = shadow_coord+texSize.y*offset[i]*2.0; //scale the offsets to the texture size, and make them twice as large to cover a larger radius
		ret += shadow2DArray(stex, shadow_lookup).x*0.125;
	}
	
	return ret;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);


//! 
// MARK:PCF w/ gaussian blur fs
const char shadow_pcf_gaussian_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform sampler2D tex;				//!< 模様
uniform sampler2DArrayShadow stex;	//!< デプス値テクスチャ(×視錐台分割数)
uniform vec4 far_d;					//!< 視錐台遠距離
uniform vec2 texSize;				//!< x - size, y - 1/size

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;

float ShadowCoef(void)
{
	// 画素位置gl_FragCoord.xy におけるデプス値 gl_FragCoord.zから適切なデプスマップを検索
	// 分割数以上の部分には0を代入しておくこと(例えば分割数2なら,far_d.z,far_d.wは0)
	int index = 3;
	if(gl_FragCoord.z < far_d.x){
		index = 0;
	}
	else if(gl_FragCoord.z < far_d.y){
		index = 1;
	}
	else if(gl_FragCoord.z < far_d.z){
		index = 2;
	}
	
	// 視点座標系の位置を光源を視点とした座標系のものに変換
	vec4 shadow_coord = gl_TextureMatrix[index]*vPos;

	// 光源からのデプス値(遮蔽無し)を待避
	shadow_coord.w = shadow_coord.z;
	
	// どの分割を用いるか
	shadow_coord.z = float(index);

	// Gaussian 3x3 filter
	float ret = shadow2DArray(stex, shadow_coord).x * 0.25;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( -1, -1)).x * 0.0625;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( -1, 0)).x * 0.125;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( -1, 1)).x * 0.0625;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( 0, -1)).x * 0.125;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( 0, 1)).x * 0.125;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( 1, -1)).x * 0.0625;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( 1, 0)).x * 0.125;
	ret += shadow2DArrayOffset(stex, shadow_coord, ivec2( 1, 1)).x * 0.0625;
	
	return ret;
}

void main(void)
{
	vec4 light_col;
	vec3 N = normalize(vNrm);
	//vec3 L = gl_LightSource[0].position.xyz;
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// 環境光の計算
	//  - OpenGLが計算した光源強度と反射係数の積(gl_FrontLightProduct)を用いる．
	light_col = gl_FrontLightProduct[0].ambient;

	// 拡散反射の計算
	//float diff = max(dot(L, N), 0.0);
	float diff = max(dot(vNrm, gl_LightSource[0].position.xyz), 0.0);

	// 鏡面反射の計算
	if(diff > 0.0){
		// 反射ベクトルの計算
		vec3 V = normalize(-vPos.xyz);
		//vec3 R = 2.0*dot(N, L)*N-L;
		vec3 R = reflect(-L, N);
		float spec = pow(abs(dot(R, V)), gl_FrontMaterial.shininess);

		light_col += gl_FrontLightProduct[0].diffuse*diff+
					 gl_FrontLightProduct[0].specular*spec;
	}

	const float shadow_ambient = 0.8;
	vec4 color_tex = texture2D(tex, gl_TexCoord[0].st);
	float shadow_coef = ShadowCoef();

	gl_FragColor = shadow_ambient*shadow_coef*light_col+(1.0-shadow_ambient)*light_col;
	//gl_FragColor = shadow_ambient*shadow_coef*gl_Color+(1.0-shadow_ambient)*gl_Color;
}
);










//-----------------------------------------------------------------------------
// Shadow View シェーダ
//-----------------------------------------------------------------------------
const char shadow_view_vs[] = RXSTR(
void main(void)
{
	gl_TexCoord[0] = vec4(0.5)*gl_Vertex + vec4(0.5);
	gl_Position = gl_Vertex;
}
);

const char shadow_view_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform sampler2DArray tex;
uniform float layer;

void main(void)
{
	vec4 tex_coord = vec4(gl_TexCoord[0].x, gl_TexCoord[0].y, layer, 1.0);
	gl_FragColor = texture2DArray(tex, tex_coord.xyz);
//	gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
);


//-----------------------------------------------------------------------------
// MARK:Kajiya-kai シェーダ
//-----------------------------------------------------------------------------

//! Kajiya-kaiモデル 頂点シェーダ
const char kajiyakai_vs[] = RXSTR(
varying vec4 vPos;
varying vec3 vNrm;
void main(void)
{
	vPos = gl_ModelViewMatrix*gl_Vertex;
	vNrm = normalize(gl_NormalMatrix*gl_Normal);

	gl_Position = ftransform();
	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_ModelViewMatrix * gl_Vertex;

	// 視点->頂点ベクトル
	vec3 E = normalize(vPos.xyz);
	vec3 L = normalize(gl_LightSource[0].position.xyz-vPos.xyz);	// 光源ベクトル

	// Kajiya-kayモデル
	vec3 nl = cross(vNrm, L);		// N×L
	float nnl = sqrt(dot(nl, nl));	// |N×L| = sinθ

	vec3 ne = cross(vNrm, E);		// N×E
	float nne = sqrt(dot(ne, ne));	// |N×E| = sinφ

	float dnl = dot(vNrm, L);
	float dne = dot(vNrm, E);

	float spec = dne*dnl+nne*nnl;
	//spec *= spec;	// 2乗
	//spec *= spec;	// 4乗
	//spec *= spec;	// 8乗
	spec = pow(max(spec, 0.0), 80.0f);
	
	// Kajiya-kay拡散項
	vec3 Ad = gl_FrontMaterial.diffuse.xyz*nnl;		// Kd sinθ = Kd |n×l|
	vec3 As = gl_FrontMaterial.specular.xyz*spec;	// Ks (cosγ)^n = Ks(cosφcosθ-sinφsinθ)^n

	// ランバート拡散
	float diff = max(0.0, dot(L, vNrm));
	vec3 Ld = gl_FrontLightProduct[0].diffuse.xyz*diff;

	gl_FrontColor.rgb = Ad+As+Ld;

	//gl_FrontColor.rgb = 0.5 * gl_Normal.xyz + 0.5;
	gl_FrontColor.a = 1.0;
}
);

//! Kajiya-kaiモデル ピクセルシェーダ
const char kajiyakai_fs[] = RXSTR(
varying vec4 vPos;
varying vec3 vNrm;
void main(void)
{
	gl_FragColor = gl_Color;
}
);




//-----------------------------------------------------------------------------
// MARK:toonシェーダ
//-----------------------------------------------------------------------------

//! トゥーンレンダリング 頂点シェーダ
const char toon_vs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform vec3 lightPosition;
uniform vec3 eyePosition;
uniform float shininess;

// フラグメントシェーダに値を渡すための変数
varying vec4 vPos;
varying vec3 vNrm;
varying float fLightDiff;
varying float fLightSpec;
varying float fEdge;

void main(void)
{
	// 頂点位置と法線
	vPos = gl_ModelViewMatrix*gl_Vertex;
	vNrm = normalize(gl_NormalMatrix*gl_Normal);

	gl_Position = ftransform();
	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_ModelViewMatrix * gl_Vertex;

	// Calculate diffuse lighting
	vec3 N = normalize(vNrm);
	vec3 L = normalize(lightPosition-gl_Vertex.xyz);
	fLightDiff = max(0.0, dot(L, N));

	// Calculate specular lighting
	vec3 V = normalize(eyePosition-gl_Vertex.xyz);
	vec3 H = normalize(L+V);
	fLightSpec = pow(max(0.0, dot(H, N)), shininess);
	if(fLightSpec <= 0.0) fLightSpec = 0.0;

	// Perform edge detection
	fEdge = max(0.0, dot(V, N));	
}
);

//! トゥーンレンダリング ピクセルシェーダ
const char toon_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform vec4 Kd;
uniform vec4 Ks;
//uniform sampler1D texDiffRamp;
//uniform sampler1D texSpecRamp;
//uniform sampler1D texEdgeRamp;

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;
varying float fLightDiff;
varying float fLightSpec;
varying float fEdge;

void main(void)
{
//	fLightDiff = texture1D(texDiffRamp, fLightDiff).x;
//	fLightSpec = texture1D(texSpecRamp, fLightSpec).x;
//	fEdge      = texture1D(texEdgeRamp, fEdge).x;
	float ldiff = (fLightDiff > 0.75) ? 0.75 : ((fLightDiff > 0.5) ? 0.5 : 0.1);
	float lspec = (fLightSpec > 0.5) ? 1.0 : 0.0;
	float edge  = (fEdge > 0.5) ? 1.0 : 0.0;

	gl_FragColor = edge*(Kd*ldiff+Ks*lspec);
}
);



//-----------------------------------------------------------------------------
// MARK:Phongシェーダ
//-----------------------------------------------------------------------------

//! Phong 頂点シェーダ
const char phong_vs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

// フラグメントシェーダに値を渡すための変数
varying vec4 vPos;
varying vec3 vNrm;
varying vec3 vObjPos;

void main(void)
{
	// 頂点位置と法線
	vPos = gl_ModelViewMatrix*gl_Vertex;
	vNrm = normalize(gl_NormalMatrix*gl_Normal);

	gl_Position = ftransform();
	gl_TexCoord[0] = gl_TextureMatrix[0] * gl_ModelViewMatrix * gl_Vertex;
	
	vObjPos = gl_Vertex.xyz;
}
);

//! Phong ピクセルシェーダ
const char phong_fs[] = RXSTR(
#version 110 @
#extension GL_EXT_texture_array : enable @

uniform vec3 Ke; // 放射色
uniform vec3 Ka; // 環境光
uniform vec3 Kd; // 拡散反射
uniform vec3 Ks; // 鏡面反射
uniform float shine;
uniform vec3 La;	// ライト環境光
uniform vec3 Ld;	// ライト拡散反射光
uniform vec3 Ls;	// ライト鏡面反射光
uniform vec3 Lpos;	// ライト位置
uniform vec3 eyePosition;

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;
varying vec3 vObjPos;

void main(void)
{
	vec3 P = vObjPos.xyz;
	vec3 N = normalize(vNrm);
	
	// 放射色の計算
	vec3 emissive = Ke;

	// 環境光の計算
	vec3 ambient = Ka*La;

	// 拡散反射の計算
	vec3 L = normalize(Lpos-P);
	float diffuseLight = max(dot(L, N), 0.0);
	vec3 diffuse = Kd*Ld*diffuseLight;

	// 鏡面反射の計算
	vec3 V = normalize(eyePosition-P);
	vec3 H = normalize(L+V);
	float specularLight = pow(max(dot(H, N), 0.0), shine);
	if(diffuseLight <= 0.0) specularLight = 0.0;
	vec3 specular = Ks*Ls*specularLight;

	gl_FragColor.xyz = emissive+ambient+diffuse+specular;
	gl_FragColor.w = 1.0;
}
);

//-----------------------------------------------------------------------------
// MARK:Fresnelシェーダ
//-----------------------------------------------------------------------------

//! Fresnel 頂点シェーダ
const char fresnel_vs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform float fresnelBias;
uniform float fresnelScale; 
uniform float fresnelPower; 
uniform float etaRatio;
uniform vec3 eyePosition;

// フラグメントシェーダに値を渡すための変数
varying vec4 vPos;
varying vec3 vNrm;
varying float fNoise;
varying vec3 oR;
varying vec3 oT;
varying float oFresnel;

void main(void)
{
	// 頂点位置と法線
	fNoise = gl_Vertex.w;
	vec4 vert = gl_Vertex;
	vert.w = 1.0f;

	vPos = gl_ModelViewMatrix*vert;
	vNrm = normalize(gl_NormalMatrix*gl_Normal);

	gl_Position = gl_ModelViewProjectionMatrix*vert;//ftransform();
	gl_TexCoord[0] = gl_TextureMatrix[0]*gl_ModelViewMatrix*vert;

	// 入射，反射，屈折ベクトルの計算
	vec3 I = vPos.xyz-eyePosition;
	I = normalize(I);
	oR = reflect(I, vNrm);
	oT = refract(I, vNrm, etaRatio);

	// 反射因数の計算
	oFresnel = fresnelBias+fresnelScale*pow(min(0.0, 1.0-dot(I, vNrm)), fresnelPower);
}
);

//! Fresnel ピクセルシェーダ
const char fresnel_fs[] = RXSTR(
#version 120 @
#extension GL_EXT_texture_array : enable @

uniform samplerCube envmap;

uniform float maxNoise;
uniform float minNoise;
uniform vec3 FoamColor;

// バーテックスシェーダから受け取る変数
varying vec4 vPos;
varying vec3 vNrm;
varying float fNoise;
varying vec3 oR;
varying vec3 oT;
varying float oFresnel;

vec4 lerp(vec4 a, vec4 b, float s)
{
	return vec4(a + (b - a) * s);       
}

vec3 lerp(vec3 a, vec3 b, float s)
{
	return vec3(a + (b - a) * s);       
}

void main(void)
{
	// 反射環境色の取得
	vec4 reflecColor = textureCube(envmap, oR);
	reflecColor.a = 1.0;

	// 屈折環境色の計算
	vec4 refracColor;
	refracColor.rgb = textureCube(envmap, oT).rgb;
	refracColor.a = 1.0;

	vec4 cout = lerp(refracColor, reflecColor, oFresnel);

	if(fNoise > minNoise){
		float foam_rate = (fNoise-minNoise)/(maxNoise-minNoise);
		gl_FragColor.rgb = lerp(cout.rgb, FoamColor, foam_rate);
	}
	else{
		gl_FragColor.rgb = cout.rgb;
	}	

	gl_FragColor.a = oFresnel*0.5+0.5;
}
);



//-----------------------------------------------------------------------------
// MARK:pointspriteシェーダ
//-----------------------------------------------------------------------------

// vertex shader
const char ps_vs[] = RXSTR(
uniform float pointRadius;  // ポイントサイズ
uniform float pointScale;   // ピクセルスケールに変換するときの倍数
uniform float zCrossFront;
uniform float zCrossBack;
varying float vValid;
void main(void)
{
	if(gl_Vertex.z > zCrossFront || gl_Vertex.z < zCrossBack){
		vValid = 0.0;
	}
	else{
		vValid = 1.0;
	}

	// ウィンドウスペースでのポイントサイズを計算
	vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
	float dist = length(posEye);
	gl_PointSize = pointRadius*(pointScale/dist);

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	gl_FrontColor = gl_Color;
}
);

// pixel shader for rendering points as shaded spheres
const char ps_fs[] = RXSTR(
varying float vValid;
uniform vec3 lightDir;	// 光源方向
void main(void)
{
	if(vValid < 0.5) discard;

	//const vec3 lightDir = vec3(0.577, 0.577, 0.577);

	// テクスチャ座標から法線を計算(球として描画)
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0)+vec2(-1.0, 1.0);

	float mag = dot(N.xy, N.xy);	// 中心からの2乗距離
	if(mag > 1.0) discard;   // 円の外のピクセルは破棄
	N.z = sqrt(1.0-mag);

	// 光源方向と法線から表面色を計算
	float diffuse = max(0.0, dot(lightDir, N));

	gl_FragColor = gl_Color*diffuse;
}
);


//-----------------------------------------------------------------------------
// MARK:pointsprite2dシェーダ
//-----------------------------------------------------------------------------
// vertex shader
const char ps2d_vs[] = RXSTR(
uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
void main(void)
{
	// calculate window-space point size
	gl_PointSize = pointRadius*pointScale;

	gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

	gl_FrontColor = gl_Color;
}
);

// pixel shader for rendering points as shaded spheres
const char ps2d_fs[] = RXSTR(
void main(void)
{
	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);
	if (mag > 1.0) discard;   // kill pixels outside circle

	gl_FragColor = gl_Color;
}
);




//-----------------------------------------------------------------------------
// MARK:GLSLコンパイル
//-----------------------------------------------------------------------------
/*!
 * GLSLプログラムのコンパイル
 * @param[in] vsource vertexシェーダプログラム内容
 * @param[in] fsource pixel(fragment)シェーダプログラム内容
 * @return GLSLプログラム番号
 */
static GLuint CompileProgram(const char *vsource, const char *fsource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);
	
	glCompileShader(vertexShader);
	glCompileShader(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success) {
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}

/*!
 * GLSLシェーダコンパイル
 * @param[in] target ターゲット(GL_VERTEX_SHADER,GL_FRAGMENT_SHADER)
 * @param[in] shader シェーダコード
 * @return GLSLオブジェクト
 */
inline GLuint CompileGLSLShader(GLenum target, const char* shader)
{
	// GLSLオブジェクト作成
	GLuint object = glCreateShader(target);

	if(!object) return 0;

	glShaderSource(object, 1, &shader, NULL);
	glCompileShader(object);

	// コンパイル状態の確認
	GLint compiled = 0;
	glGetShaderiv(object, GL_COMPILE_STATUS, &compiled);

	if(!compiled){
		char temp[256] = "";
		glGetShaderInfoLog( object, 256, NULL, temp);
		fprintf(stderr, " Compile failed:\n%s\n", temp);

		glDeleteShader(object);
		return 0;
	}

	return object;
}

/*!
 * GLSLシェーダコンパイル
 * @param[in] target ターゲット(GL_VERTEX_SHADER,GL_FRAGMENT_SHADER)
 * @param[in] fn シェーダファイルパス
 * @return GLSLオブジェクト
 */
inline GLuint CompileGLSLShaderFromFile(GLenum target, const char* fn)
{
	FILE *fp;

	// バイナリとしてファイル読み込み
	fopen_s(&fp, fn, "rb");
	if(fp == NULL) return 0;

	// ファイルの末尾に移動し現在位置(ファイルサイズ)を取得
	fseek(fp, 0, SEEK_END);
	long size = ftell(fp);

	fseek(fp, 0, SEEK_SET);

	// シェーダの内容格納
	char *text = new char[size+1];
	fread(text, size, 1, fp);
	text[size] = '\0';

	//printf("%s\n", text);


	fclose(fp);

	// シェーダコンパイル
	printf("Compile %s\n", fn);
	GLuint object = CompileGLSLShader(target, text);

	delete [] text;

	return object;
}

/*!
 * バーテックスとフラグメントシェーダで構成されるGLSLプログラム作成
 * @param[in] vs バーテックスシェーダオブジェクト
 * @param[in] fs フラグメントシェーダオブジェクト
 * @return GLSLプログラムオブジェクト
 */
inline GLuint LinkGLSLProgram(GLuint vs, GLuint fs)
{
	// プログラムオブジェクト作成
	GLuint program = glCreateProgram();

	// シェーダオブジェクトを登録
	glAttachShader(program, vs);
	glAttachShader(program, fs);

	// プログラムのリンク
	glLinkProgram(program);

	// エラー出力
	GLint charsWritten, infoLogLength;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

	char * infoLog = new char[infoLogLength];
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf(infoLog);
	delete [] infoLog;

	// リンカテスト
	GLint linkSucceed = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &linkSucceed);
	if(linkSucceed == GL_FALSE){
		glDeleteProgram(program);
		return 0;
	}

	return program;
}


/*!
 * バーテックス/ジオメトリ/フラグメントシェーダで構成されるGLSLプログラム作成
 * @param[in] vs バーテックスシェーダオブジェクト
 * @param[in] gs ジオメトリシェーダオブジェクト
 * @param[in] inputType ジオメトリシェーダへの入力タイプ
 * @param[in] vertexOut バーテックスの出力
 * @param[in] outputType ジオメトリシェーダからの出力タイプ
 * @param[in] fs フラグメントシェーダオブジェクト
 * @return GLSLプログラムオブジェクト
 */
inline GLuint LinkGLSLProgram(GLuint vs, GLuint gs, GLint inputType, GLint vertexOut, GLint outputType, GLuint fs)
{
	// プログラムオブジェクト作成
	GLuint program = glCreateProgram();

	// シェーダオブジェクトを登録
	glAttachShader(program, vs);
	glAttachShader(program, gs);

	glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, inputType);
	glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, vertexOut);
	glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, outputType);
	glAttachShader(program, fs);

	// プログラムのリンク
	glLinkProgram(program);

	// エラー出力
	GLint charsWritten, infoLogLength;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLogLength);

	char * infoLog = new char[infoLogLength];
	glGetProgramInfoLog(program, infoLogLength, &charsWritten, infoLog);
	printf(infoLog);
	delete [] infoLog;

	// リンカテスト
	GLint linkSucceed = GL_FALSE;
	glGetProgramiv(program, GL_LINK_STATUS, &linkSucceed);
	if(linkSucceed == GL_FALSE){
		glDeleteProgram(program);
		return 0;
	}

	return program;
}


/*!
 * GLSLのコンパイル・リンク(ファイルより)
 * @param[in] vs 頂点シェーダファイルパス
 * @param[in] fs フラグメントシェーダファイルパス
 * @param[in] name プログラム名
 * @return GLSLオブジェクト
 */
inline rxGLSL CreateGLSLFromFile(const string &vs, const string &fs, const string &name)
{
	rxGLSL gs;
	gs.VertProg = vs;
	gs.FragProg = fs;
	gs.Name = name;

	GLuint v, f;
	printf("compile the vertex shader : %s\n", name.c_str());
	if(!(v = CompileGLSLShaderFromFile(GL_VERTEX_SHADER, vs.c_str()))){
		// skip the first three chars to deal with path differences
		v = CompileGLSLShaderFromFile(GL_VERTEX_SHADER, &vs.c_str()[3]);
	}

	printf("compile the fragment shader : %s\n", name.c_str());
	if(!(f = CompileGLSLShaderFromFile(GL_FRAGMENT_SHADER, fs.c_str()))){
		f = CompileGLSLShaderFromFile(GL_FRAGMENT_SHADER, &fs.c_str()[3]);
	}

	gs.Prog = LinkGLSLProgram(v, f);
	//gs.Prog = GLSL_CreateShaders(gs.VertProg.c_str(), gs.FragProg.c_str());

	return gs;
}

/*!
 * #versionなどのプリプロセッサを文字列として書かれたシェーダ中に含む場合，改行がうまくいかないので，
 *  #version 110 @ のように最後に@を付け，改行に変換する
 * @param[in] s  シェーダ文字列
 * @param[in] vs 変換後のシェーダ文字列
 */
inline void CreateGLSLShaderString(const char* s, vector<char> &vs)
{
	int idx = 0;
	char c = s[0];
	while(c != '\0'){
		if(c == '@') c = '\n'; // #versionなどを可能にするために@を改行に変換

		vs.push_back(c);
		idx++;
		c = s[idx];
	}
	vs.push_back('\0');
}

/*!
 * GLSLのコンパイル・リンク(文字列より)
 * @param[in] vs 頂点シェーダ内容
 * @param[in] fs フラグメントシェーダ内容
 * @param[in] name プログラム名
 * @return GLSLオブジェクト
 */
inline rxGLSL CreateGLSL(const char* vs, const char* fs, const string &name)
{
	rxGLSL gs;
	gs.VertProg = "from char";
	gs.FragProg = "from char";
	gs.Name = name;

	vector<char> vs1, fs1;
	CreateGLSLShaderString(vs, vs1);
	CreateGLSLShaderString(fs, fs1);
	
	//printf("vertex shader : %d\n%s\n", vs1.size(), &vs1[0]);
	//printf("pixel shader  : %d\n%s\n", fs1.size(), &fs1[0]);

	GLuint v, f;
	printf("compile the vertex shader : %s\n", name.c_str());
	v = CompileGLSLShader(GL_VERTEX_SHADER, &vs1[0]);
	printf("compile the fragment shader : %s\n", name.c_str());
	f = CompileGLSLShader(GL_FRAGMENT_SHADER, &fs1[0]);
	gs.Prog = LinkGLSLProgram(v, f);

	return gs;
}



#endif // #ifndef _RX_SHADERS_H_