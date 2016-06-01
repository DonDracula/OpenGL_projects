/*!
  @file shading.fp
	
  @brief GLSLフラグメントシェーダ
 
  @author Makoto Fujisawa
  @date 2011
*/
// FILE --shading.fp--
#version 120


//-----------------------------------------------------------------------------
// 変数
//-----------------------------------------------------------------------------
//
// バーテックスシェーダから受け取る変数
//
varying vec4 vPos;
varying vec3 vNrm;


//
// GLから設定される定数(uniform)
//
uniform float fresnelBias;
uniform float fresnelScale; 
uniform float fresnelPower; 
uniform float etaRatio;
uniform vec3 eyePosition;
uniform samplerCube envmap;


//-----------------------------------------------------------------------------
// 関数
//-----------------------------------------------------------------------------
vec4 lerp(vec4 a, vec4 b, float s)
{
	return vec4(a+(b-a)*s);       
}

vec3 lerp(vec3 a, vec3 b, float s)
{
	return vec3(a+(b-a)*s);       
}

//-----------------------------------------------------------------------------
// 反射モデル関数
//-----------------------------------------------------------------------------
/*!
 * Fresnel反射モデルによるシェーディング
 * @return 表面反射色
 */
vec4 FresnelShading(void)
{
	// 入射，反射，屈折ベクトルの計算
	vec3 N = normalize(vNrm);			// 法線ベクトル
	vec3 I = normalize(vPos.xyz-eyePosition);		// 入射ベクトル
	vec3 R = reflect(I, N);			// 反射ベクトル
	vec3 T = refract(I, N, etaRatio);	// 屈折ベクトル

	// 反射因数の計算
	float fresnel = fresnelBias+fresnelScale*pow(min(0.0, 1.0-dot(I, N)), fresnelPower);

	// 反射環境色の取得
	vec4 reflecColor = textureCube(envmap, R);
	reflecColor.a = 1.0;

	// 屈折環境色の計算
	vec4 refracColor;
	refracColor.rgb = textureCube(envmap, T).rgb;
	refracColor.a = 1.0;

	// 色を統合
	vec4 cout = lerp(refracColor, reflecColor, fresnel);
	cout.a = fresnel*0.5+0.5;

	return cout;
}



//-----------------------------------------------------------------------------
// エントリ関数
//-----------------------------------------------------------------------------
void main(void)
{	
	// 表面反射色
	gl_FragColor = FresnelShading();
}

