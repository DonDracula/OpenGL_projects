/*!
  @file rx_material.h
	
  @brief OpenGLライティング
 
*/


#ifndef _RX_MATERIAL_H_
#define _RX_MATERIAL_H_

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_utility.h"
#include "GL/glut.h"


using namespace std;

//-----------------------------------------------------------------------------
//! 材質データ
//-----------------------------------------------------------------------------
class rxMaterial
{
public:
	Vec4 m_vDiff, m_vSpec, m_vAmbi, m_vEmit;
	double m_fDiffS, m_fSpecS, m_fAmbiS, m_fEmitS;
	double m_fShin;

	Vec4 m_vColor;

	// 屈折，鏡面反射パラメータ
	double m_fEta;		//!< 屈折率
	double m_fBias;		//!< Fresnelバイアス
	double m_fPower;	//!< Fresnel指数
	double m_fScale;	//!< Fresnel倍率

	string m_strName;
	int m_iID;

	int m_iIllum;
	string m_strTexFile;
	unsigned int m_uTexName;

public:
	rxMaterial()
	{
		SetColor(Vec3(0.0, 0.0, 1.0), Vec3(1.0), Vec3(1.0), Vec3(0.0), 30.0);
		SetScale(1.0, 0.75, 0.2, 0.0);
		m_fEta = 0.9;
		m_fBias = 1.0;
		m_fPower = 0.98;
		m_fScale = 5.0;
	}

	void SetColor(const Vec4 &diff, const Vec4 &spec, const Vec4 &ambi, const Vec4 &emit, const double &shin)
	{
		m_vDiff = diff;
		m_vSpec = spec;
		m_vAmbi = ambi;
		m_vEmit = emit;
		m_fShin = shin;
	}

	void SetColor(const GLfloat diff[4], 
				  const GLfloat spec[4], 
				  const GLfloat ambi[4], 
				  const GLfloat emit[4], 
				  const GLfloat shine)
	{
		m_vDiff = Vec4(diff[0], diff[1], diff[2], diff[3]);
		m_vSpec = Vec4(spec[0], spec[1], spec[2], spec[3]);
		m_vAmbi = Vec4(ambi[0], ambi[1], ambi[2], ambi[3]);
		m_vEmit = Vec4(emit[0], emit[1], emit[2], emit[3]);
		m_fShin = shine;
	}

	void SetColor(const Vec3 &diff, const Vec3 &spec, const Vec3 &ambi, const Vec3 &emit, const double &shin)
	{
		m_vDiff = diff;	m_vDiff[3] = 1.0;
		m_vSpec = spec;	m_vSpec[3] = 1.0;
		m_vAmbi = ambi;	m_vAmbi[3] = 1.0;
		m_vEmit = emit;	m_vEmit[3] = 1.0;
		m_fShin = shin;
	}

	void SetDiff(const Vec4 &col){ m_vDiff = col; }
	void SetSpec(const Vec4 &col){ m_vSpec = col; }
	void SetAmbi(const Vec4 &col){ m_vAmbi = col; }
	void SetEmit(const Vec4 &col){ m_vEmit = col; }
	void SetDiff3(const Vec3 &col){ m_vDiff = col; m_vDiff[3] = 1.0; }
	void SetSpec3(const Vec3 &col){ m_vSpec = col; m_vSpec[3] = 1.0; }
	void SetAmbi3(const Vec3 &col){ m_vAmbi = col; m_vAmbi[3] = 1.0; }
	void SetEmit3(const Vec3 &col){ m_vEmit = col; m_vEmit[3] = 1.0; }

	void SetScale(const double &sdiff, const double &sspec, const double &sambi, const double &semit)
	{
		m_fDiffS = sdiff;
		m_fSpecS = sspec;
		m_fAmbiS = sambi;
		m_fEmitS = semit;
	}

	void SetRefrac(const double &eta, const double &bias, const double &power, const double &scale)
	{
		m_fEta = eta;
		m_fBias = bias;
		m_fPower = power;
		m_fScale = scale;
	}

	void SetGL(void)
	{
		GLfloat mat_diff[] = { (float)(m_fDiffS*m_vDiff[0]), (float)(m_fDiffS*m_vDiff[1]), (float)(m_fDiffS*m_vDiff[2]), (float)m_vDiff[3] };
		GLfloat mat_spec[] = { (float)(m_fSpecS*m_vSpec[0]), (float)(m_fSpecS*m_vSpec[1]), (float)(m_fSpecS*m_vSpec[2]), (float)m_vDiff[3] };
		GLfloat mat_ambi[] = { (float)(m_fAmbiS*m_vAmbi[0]), (float)(m_fAmbiS*m_vAmbi[1]), (float)(m_fAmbiS*m_vAmbi[2]), (float)m_vDiff[3] };
		GLfloat mat_shin[] = { (float)m_fShin };

		glMaterialfv(GL_FRONT, GL_DIFFUSE,  mat_diff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_spec);
		glMaterialfv(GL_FRONT, GL_AMBIENT,  mat_ambi); 
		glMaterialfv(GL_FRONT, GL_SHININESS,mat_shin);

		glColor3fv(mat_diff);
	}

	void SetGL(const Vec4 &diff, const Vec4 &spec, const Vec4 &ambi)
	{
		GLfloat mat_diff[] = { (float)(m_fDiffS*diff[0]), (float)(m_fDiffS*diff[1]), (float)(m_fDiffS*diff[2]), (float)diff[3] };
		GLfloat mat_spec[] = { (float)(m_fSpecS*spec[0]), (float)(m_fSpecS*spec[1]), (float)(m_fSpecS*spec[2]), (float)spec[3] };
		GLfloat mat_ambi[] = { (float)(m_fAmbiS*ambi[0]), (float)(m_fAmbiS*ambi[1]), (float)(m_fAmbiS*ambi[2]), (float)ambi[3] };
		GLfloat mat_shin[] = { (float)m_fShin };

		glMaterialfv(GL_FRONT, GL_DIFFUSE,  mat_diff);
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_spec);
		glMaterialfv(GL_FRONT, GL_AMBIENT,  mat_ambi); 
		glMaterialfv(GL_FRONT, GL_SHININESS,mat_shin);

		glColor4fv(mat_diff);
	}

	// アクセスメソッド
	Vec4 GetDiff(void) const { return m_vDiff*m_fDiffS; }
	Vec4 GetSpec(void) const { return m_vSpec*m_fSpecS; }
	Vec4 GetAmbi(void) const { return m_vAmbi*m_fAmbiS; }
	Vec4 GetEmit(void) const { return m_vEmit*m_fEmitS; }
	Vec3 GetDiff3(void) const { return Vec3(m_vDiff[0], m_vDiff[1], m_vDiff[2])*m_fDiffS; }
	Vec3 GetSpec3(void) const { return Vec3(m_vSpec[0], m_vSpec[1], m_vSpec[2])*m_fSpecS; }
	Vec3 GetAmbi3(void) const { return Vec3(m_vAmbi[0], m_vAmbi[1], m_vAmbi[2])*m_fAmbiS; }
	Vec3 GetEmit3(void) const { return Vec3(m_vEmit[0], m_vEmit[1], m_vEmit[2])*m_fEmitS; }
	double GetShin(void) const { return m_fShin; }

	Vec3 GetReflec(const Vec3 &irr, const Vec3 &nrm)
	{
		double ref_coef = m_fBias+m_fScale*pow((1.0+dot(irr, nrm)), m_fPower);
		RX_CLAMP(ref_coef, 0.0, 1.0);
		return ref_coef*GetSpec3();
	}

	Vec3 GetRefrac(const Vec3 &irr, const Vec3 &nrm)
	{
		double ref_coef = m_fBias+m_fScale*pow((1.0+dot(irr, nrm)), m_fPower);
		RX_CLAMP(ref_coef, 0.0, 1.0);
		return (1.0-ref_coef)*GetSpec3();
	}

	void Get(Vec3 &diff, Vec3 &spec, Vec3 &ambi, Vec3 &emit, double &shin)
	{
		diff = Vec3(m_vDiff[0], m_vDiff[1], m_vDiff[2]);
		spec = Vec3(m_vSpec[0], m_vSpec[1], m_vSpec[2]);
		ambi = Vec3(m_vAmbi[0], m_vAmbi[1], m_vAmbi[2]);
		emit = Vec3(m_vEmit[0], m_vEmit[1], m_vEmit[2]);
		shin = m_fShin;
	}
};

static rxMaterial g_matDefault;



#endif // #ifndef _RX_MATERIAL_H_
