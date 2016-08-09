/*! @file rx_glgui.h
	
	@brief 描画キャンパス上にGUI部品を描画
 
	@author Makoto Fujisawa
	@date   2009-02
*/
// FILE --rx_glgui.h--


#ifndef _RX_GL_GUI_H_
#define _RX_GL_GUI_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <cmath>

#include <iostream>
#include <sstream>

#include <vector>
#include <string>

#include <GL/glut.h>


//#define RX_USE_BOOST

#ifdef RX_USE_BOOST
#include <boost/function.hpp>
#include <boost/bind.hpp>
#endif


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
#define RX_SEL_BUFFER_SIZE 128

#ifdef RX_USE_BOOST
typedef boost::function<double (void)> FPGetFunc;
typedef boost::function<void (double)> FPSetFunc;
typedef boost::function<void (double)> FPCallback;
#else
typedef double (*FPGetFunc)(void);
typedef void (*FPSetFunc)(double);
typedef void (*FPCallback)(double);
#endif


//-----------------------------------------------------------------------------
// 共通関数
//-----------------------------------------------------------------------------
template<class T> 
inline string V2S(T val)
{
	ostringstream ss;
	ss << val;
	string str = ss.str();
	return str;
}

inline void DrawRect(double w, double h, double z = 0.0)
{
	glBegin(GL_QUADS);
	//glTexCoord2f(0.0, 0.0);	glVertex3d(-0.5*w, -0.5*h, z);
	//glTexCoord2f(1.0, 0.0);	glVertex3d( 0.5*w, -0.5*h, z);
	//glTexCoord2f(1.0, 1.0);	glVertex3d( 0.5*w,  0.5*h, z);
	//glTexCoord2f(0.0, 1.0);	glVertex3d(-0.5*w,  0.5*h, z);
	glTexCoord2f(0.0, 0.0);	glVertex2d(-0.5*w, -0.5*h);
	glTexCoord2f(1.0, 0.0);	glVertex2d( 0.5*w, -0.5*h);
	glTexCoord2f(1.0, 1.0);	glVertex2d( 0.5*w,  0.5*h);
	glTexCoord2f(0.0, 1.0);	glVertex2d(-0.5*w,  0.5*h);
	glEnd();
}

inline void DrawWireRect(double w, double h, double z = 0.0)
{
	glBegin(GL_LINE_LOOP);
	//glVertex3f(-0.5*w, -0.5*h, z);
	//glVertex3f( 0.5*w, -0.5*h, z);
	//glVertex3f( 0.5*w,  0.5*h, z);
	//glVertex3f(-0.5*w,  0.5*h, z);
	glVertex2d(-0.5*w, -0.5*h);
	glVertex2d( 0.5*w, -0.5*h);
	glVertex2d( 0.5*w,  0.5*h);
	glVertex2d(-0.5*w,  0.5*h);
	glEnd();
}


/*!
 * 円の描画
 * @param cen 円の中心
 * @param rad 円の半径
 */
inline void DrawCircle0(const Vec3 &cen, const double &rad)
{
	double t = 0.0;
	double dt = RX_PI/16.0;

	glPushMatrix();

	glTranslatef(cen[0], cen[1], cen[2]);
	glBegin(GL_POLYGON);
	do{
		glVertex2f(rad*cos(t), rad*sin(t));
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();

	glPopMatrix();
}

/*!
 * 円のワイヤーフレーム描画
 * @param cen 円の中心
 * @param rad 円の半径
 * @param n 分割数
 */
inline void DrawWireCircle0(const Vec3 &cen, const double &rad, const int &n)
{
	double t = 0.0;
	double dt = 2.0*RX_PI/(double)n;

	glPushMatrix();

	glTranslatef(cen[0], cen[1], cen[2]);
	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), rad*sin(t), 0.0);
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();

	glPopMatrix();
}

inline void DrawString0(string static_str, int x0, int y0)
{
	glRasterPos2f(x0, y0);

	int size = (int)static_str.size();
	for(int i = 0; i < size; ++i){
		char ic = static_str[i];
		//glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ic);
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ic);
	}
}


//-----------------------------------------------------------------------------
// Get,Set関数へのアクセスを支援するクラス
//-----------------------------------------------------------------------------
template<class Type> 
class rxAccessFuncT
{
protected:
#ifdef RX_USE_BOOST
	boost::function<Type (void)> m_fpGet;
	boost::function<void (Type)> m_fpSet;
#else
	Type (*m_fpGet)(void);
	void (*m_fpSet)(Type);
#endif

	Type *m_pValPtr;

	int m_bUseFP;

protected:
	// 仮想アクセスメソッド
	Type vGet(void){ return *m_pValPtr; }
	void vSet(Type x){ *m_pValPtr = x; }

public:
	rxAccessFuncT()
	{
		m_fpGet = NULL;
		m_fpSet = NULL;
		m_pValPtr = NULL;
		m_bUseFP = 0;
	}

	bool Get(Type &x)
	{
		if(m_fpGet != NULL){
#ifdef RX_USE_BOOST
			x = m_fpGet();
#else
			if(m_bUseFP){
				x = m_fpGet();
			}
			else{
				x = this->m_fpGet();
			}
#endif
			return true;
		}
		else{
			return false;
		}
	}
	bool Set(Type x)
	{
		if(m_fpSet != NULL){
#ifdef RX_USE_BOOST
			m_fpSet(x);
#else
			if(m_bUseFP){
				m_fpSet(x);
			}
			else{
				this->m_fpSet(x);
			}
#endif
			return true;
		}
		else{
			return false;
		}
	}

#ifdef RX_USE_BOOST
	void SetFunc(boost::function<Type (void)> get_func, boost::function<void (Type)> set_func)
#else
	void SetFunc(Type (*get_func)(void), void (*set_func)(Type))
#endif
	{
		m_fpGet = get_func;
		m_fpSet = set_func;
		m_bUseFP = true;
	}
	void SetPtr(Type *ptr)
	{
		m_pValPtr = ptr;
#ifdef RX_USE_BOOST
		m_fpGet = boost::bind(&rxAccessFuncT::vGet, this);
		m_fpSet = boost::bind(&rxAccessFuncT::vSet, this, _1);
#else
		m_fpGet = &rxAccessFuncT::vGet;
		m_fpSet = &rxAccessFuncT::vSet;
#endif
		m_bUseFP = false;
	}
};

typedef rxAccessFuncT<double> rxAccessFunc;



//-----------------------------------------------------------------------------
// MARK:rxWidgetGLGUIクラス
//  - ウィジット基底クラス
//-----------------------------------------------------------------------------
class rxWidgetGLGUI
{
public:
	string name;
	int posx, posy;

	bool enable;

	FPCallback m_fpCB;

public:
	rxWidgetGLGUI()
	{
		enable = true;
		m_fpCB = NULL;
	}
	virtual ~rxWidgetGLGUI(){}

	virtual void Draw(int idx, int name_len, int cx, int cy, bool sel) = 0;

	virtual void SetValue(double x) = 0;
	virtual void SetInitMouse(void) = 0;
	virtual void SetValueMouse(int dx, int res0) = 0;
	virtual void SetValueFunc(FPGetFunc get_func, FPSetFunc set_func) = 0;
	virtual void SetValuePtr(double *ptr) = 0;
	virtual void SetCallback(FPCallback fpCB)
	{
		m_fpCB = fpCB;
	}

	virtual void SetValue(int x){};

	virtual double GetValue(void) = 0;
};

//-----------------------------------------------------------------------------
// MARK:rxTextBoxGLGUIクラス
//  - テキストボックス
//-----------------------------------------------------------------------------
class rxTextBoxGLGUI : public rxWidgetGLGUI
{
public:
	string val_str;
	double val;

protected:
	rxAccessFunc fval;

public:
	rxTextBoxGLGUI() : rxWidgetGLGUI()
	{
		val = 0.0;
	}

	virtual ~rxTextBoxGLGUI(){}

	virtual void SetValue(double x)
	{
		val = x;
		val_str = V2S(x);

		fval.Set(x);
	}
	virtual void SetInitMouse(void)
	{
	}
	virtual void SetValueMouse(int dx, int res0 = -1)
	{
	}

	virtual double GetValue(void)
	{
		return val;
	}
	virtual void SetValueFunc(FPGetFunc get_func, FPSetFunc set_func)
	{
		fval.SetFunc(get_func, set_func);
	}
	virtual void SetValuePtr(double *ptr)
	{
		fval.SetPtr(ptr);
	}



	virtual void Draw(int idx, int name_len, int cx, int cy, bool sel)
	{
		if(!enable) return;

		// 値の更新(Get関数有りの場合)
		fval.Get(val);
		val_str = V2S(val);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

		int charw = glutBitmapWidth(GLUT_BITMAP_HELVETICA_12, 'W');
		double x;

		// 値を文字で描画
		glColor3f(0.0, 0.0, 0.0);
		DrawString0(name, 0, cy);

		x = name_len*charw;
		DrawString0(" : ", x, cy);

		x += 1*charw;
		DrawString0(val_str, x, cy);


		// テキストの枠を描画
		glPushMatrix();
		glTranslatef(x, cy, 0.0);

		double w = 1.1*val_str.size()*charw;
		double h = 15.0+2.0;

		glTranslatef(0.5*w, 0.0, 0.0);
		glLoadName(idx+1);

		// 枠線
		if(sel){
			glColor3f(1.0, 0.0, 0.0);
			glLineWidth(2.0);
		}
		else{
			glColor3f(0.0, 0.0, 0.0);
			glLineWidth(1.0);
		}
		DrawWireRect(w, h);

		// 枠内部
		glColor3f(1.0, 1.0, 1.0);

		DrawRect(w, h);
		glLoadName(-1);

		glPopMatrix();
	}

};


//-----------------------------------------------------------------------------
// MARK:rxSliderGLGUIクラス
//  - スライダー
//-----------------------------------------------------------------------------
class rxSliderGLGUI : public rxWidgetGLGUI
{
public:
	string val_str;		//!< 現在の値を文字列化したもの
	double val;			//!< 現在の値
	double val_init;	//!< マウスドラッグ時の初期値

	double range[2];	//!< 値の範囲
	int res;			//!< スライダー解像度

protected:
	rxAccessFunc fval;

public:
	rxSliderGLGUI() : rxWidgetGLGUI()
	{
		val = 0.0;
		res = 100;
	}

	virtual ~rxSliderGLGUI(){}

	virtual void SetValue(double x)
	{
		if(x < range[0]) x = range[0];
		if(x > range[1]) x = range[1];
		val = x;
		val_str = V2S(x);

		fval.Set(x);
		if(m_fpCB != NULL) m_fpCB(x);
	}

	virtual void SetInitMouse(void)
	{
		val_init = val;
	}
	virtual void SetValueMouse(int dx, int res0 = -1)
	{
		double r = 100.0;
		if(res0 != -1){
			r = res*res0;
		}

		double val1 = val_init+dx/r*(range[1]-range[0]);
		SetValue(val1);
	}

	virtual double GetValue(void)
	{
		return val;
	}
	virtual void SetValueFunc(FPGetFunc get_func, FPSetFunc set_func)
	{
		fval.SetFunc(get_func, set_func);
	}
	virtual void SetValuePtr(double *ptr)
	{
		fval.SetPtr(ptr);
	}


	virtual void Draw(int idx, int name_len, int cx, int cy, bool sel)
	{
		if(!enable) return;

		// 値の更新(Get関数有りの場合)
		fval.Get(val);
		val_str = V2S(val);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		double x;

		// 値を文字で描画
		//glColor4f(0.0, 0.0, 0.0, 1.0);
		DrawString0(name, 0, cy);

		x = name_len*9;
		DrawString0(" : ", x, cy);

		x += 3*9;
		DrawString0(val_str, x, cy);

		// スライダー描画
		glPushMatrix();
		glTranslatef(x, cy, 0.0);

		glPushMatrix();
		x = (val-range[0])/(range[1]-range[0]);
		glTranslatef(100*x, 0.0, 0.0);
		if(sel){
			glColor4d(1.0, 0.5, 0.5, 1.0);
		}
		else{
			glColor4d(0.7, 0.7, 0.7, 1.0);
		}

		// スライダーつまみ
		glLoadName(idx+1);
		DrawRect(15, 30, 0.12);
		glLoadName(-1);

		// スライダーつまみの影
		glTranslatef(2, -2, 0.0);
		glColor4d(0.3, 0.3, 0.3, 1.0);
		DrawRect(15, 30, 0.11);
		glPopMatrix();

		// ライン
		glTranslatef(cx, 0.0, 0.0);
		glColor4d(0.8, 0.8, 0.8, 1.0);
		DrawRect(100, 5, 0.1);

		glPopMatrix();
	}

};


//-----------------------------------------------------------------------------
// MARK:rxIntSliderGLGUIクラス
//  - 整数スライダー
//-----------------------------------------------------------------------------
class rxIntSliderGLGUI : public rxWidgetGLGUI
{
public:
	string val_str;		//!< 現在の値を文字列化したもの
	int val;			//!< 現在の値
	int val_init;	//!< マウスドラッグ時の初期値

	int range[2];	//!< 値の範囲
	int res;			//!< スライダー解像度

protected:
	rxAccessFunc fval;

public:
	rxIntSliderGLGUI() : rxWidgetGLGUI()
	{
		val = 0;
		res = 100;
	}

	virtual ~rxIntSliderGLGUI(){}

	virtual void SetValue(double x)
	{
		val = (int)x;

		if(val < range[0]) val = range[0];
		if(val > range[1]) val = range[1];
		val_str = V2S(val);

		fval.Set((double)val);
		if(m_fpCB != NULL) m_fpCB((double)val);
	}

	virtual void SetInitMouse(void)
	{
		val_init = val;
	}
	virtual void SetValueMouse(int dx, int res0 = -1)
	{
		double r = 100.0;
		if(res0 != -1){
			r = res*res0;
		}

		double val1 = val_init+dx/r*(range[1]-range[0]);
		SetValue(val1);
	}

	virtual double GetValue(void)
	{
		return (double)val;
	}
	virtual void SetValueFunc(FPGetFunc get_func, FPSetFunc set_func)
	{
		fval.SetFunc(get_func, set_func);
	}
	virtual void SetValuePtr(double *ptr)
	{
		fval.SetPtr(ptr);
	}


	virtual void Draw(int idx, int name_len, int cx, int cy, bool sel)
	{
		if(!enable) return;

		// 値の更新(Get関数有りの場合)
		double d;
		if(fval.Get(d)){
			val_str = V2S((int)d);
		}
		else{
			val_str = V2S(val);
		}

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		double x;

		// 値を文字で描画
		//glColor4f(0.0, 0.0, 0.0, 1.0);
		DrawString0(name, 0, cy);

		x = name_len*9;
		DrawString0(" : ", x, cy);

		x += 3*9;
		DrawString0(val_str, x, cy);

		// スライダー描画
		glPushMatrix();
		glTranslatef(x, cy, 0.0);

		glPushMatrix();
		x = (double)(val-range[0])/(double)(range[1]-range[0]);
		glTranslatef(100*x, 0.0, 0.0);
		if(sel){
			glColor4d(1.0, 0.5, 0.5, 1.0);
		}
		else{
			glColor4d(0.7, 0.7, 0.7, 1.0);
		}

		// スライダーつまみ
		glLoadName(idx+1);
		DrawRect(15, 30, 0.12);
		glLoadName(-1);

		// スライダーつまみの影
		glTranslatef(2, -2, 0.0);
		glColor4d(0.3, 0.3, 0.3, 1.0);
		DrawRect(15, 30, 0.11);
		glPopMatrix();

		// ライン
		glTranslatef(cx, 0.0, 0.0);
		glColor4d(0.8, 0.8, 0.8, 1.0);
		DrawRect(100, 5, 0.1);

		glPopMatrix();
	}

};


//-----------------------------------------------------------------------------
// MARK:GUI部品描画
//-----------------------------------------------------------------------------
class rxGLGUI
{
	vector<rxWidgetGLGUI*> m_vVals;
	int m_iW, m_iH;
	int m_fCurX, m_fCurY;
	int m_iSelItem;

	int m_iDrag;		//!< ドラッグ中か否か

	int m_iSx, m_iSy;	//!< ドラッグ開始位置
	int m_iPx, m_iPy;	//!< ドラッグ開始位置

	int m_iMaxLength;	//!< 変数名の最大文字数

public:
	//! デフォルトコンストラクタ
	rxGLGUI()
	{
		m_iW = 100;
		m_iH = 100;
		m_fCurX = 50;
		m_fCurY = 15;
		m_iSelItem = -1;

		m_iDrag = 0;

	}
	//! コンストラクタ
	rxGLGUI(int w, int h)
	{
		rxGLGUI();
		m_iW = w;
		m_iH = h;
	}

	//! デストラクタ
	~rxGLGUI()
	{
		for(int i = 0; i < (int)m_vVals.size(); ++i){
			delete m_vVals[i];
		}
	}
	
	void SetWindow(int w, int h)
	{
		m_iW = w;
		m_iH = h;
	}

	void CalMaxLength(void)
	{
		int max_l = 0;
		vector<rxWidgetGLGUI*>::iterator itr;
		for(itr = m_vVals.begin(); itr != m_vVals.end(); ++itr){
			int l = (int)((*itr)->name.size());
			if(l > max_l) max_l = l;
		}

		m_iMaxLength = max_l;
	}

	void SetValue(int i, double val)
	{
		m_vVals[i]->SetValue(val);
	}
	double GetValue(int i)
	{
		return m_vVals[i]->GetValue();
	}

	void SetValueFunc(int i, FPGetFunc fpGet, FPSetFunc fpSet)
	{
		m_vVals[i]->SetValueFunc(fpGet, fpSet);
	}
	void SetValuePtr(int i, double *ptr)
	{
		m_vVals[i]->SetValuePtr(ptr);
	}
	void SetCallback(int i, FPCallback fpCB)
	{
		m_vVals[i]->SetCallback(fpCB);
	}

	void Enable(int i, bool enable)
	{
		m_vVals[i]->enable = enable;
	}
	bool IsEnable(int i)
	{
		return m_vVals[i]->enable;
	}
	void FlipEnable(int i)
	{
		m_vVals[i]->enable = !m_vVals[i]->enable;
	}


	/*!
	 * テキストボックス追加
	 * @param[in] name ラベル
	 * @param[in] val 初期値
	 * @param[in] min_val,max_val 最小，最大値
	 * @return widget番号
	 */
	int SetTextBox(string name, double val)
	{
		// MARK:テキストボックス追加
		rxTextBoxGLGUI *wt = new rxTextBoxGLGUI;

		wt->posx = m_fCurX;
		wt->posy = m_fCurY;
		wt->name = name;
		wt->val = val;
		wt->val_str = V2S(val);
		wt->enable = true;

		m_vVals.push_back((rxWidgetGLGUI*)wt);

		CalMaxLength();

		m_fCurY += 30;

		return (int)m_vVals.size()-1;
	}


	/*!
	 * スライダー追加
	 * @param[in] name ラベル
	 * @param[in] val 初期値
	 * @param[in] min_val,max_val 最小，最大値
	 * @return widget番号
	 */
	int SetSlider(string name, double val, double min_val, double max_val, int res = 1000)
	{
		// MARK:スライダー追加
		rxSliderGLGUI *wt = new rxSliderGLGUI;

		wt->posx = m_fCurX;
		wt->posy = m_fCurY;
		wt->range[0] = min_val;
		wt->range[1] = max_val;
		wt->name = name;
		wt->val = val;
		wt->val_str = V2S(val);
		wt->res = res;
		wt->enable = true;

		m_vVals.push_back((rxWidgetGLGUI*)wt);

		CalMaxLength();

		m_fCurY += 30;

		return (int)m_vVals.size()-1;
	}
	
	/*!
	 * スライダー追加
	 * @param[in] name ラベル
	 * @param[in] val 初期値
	 * @param[in] min_val,max_val 最小，最大値
	 * @return widget番号
	 */
	int SetIntSlider(string name, int val, int min_val, int max_val, int spacing = 1)
	{
		// MARK:スライダー追加
		rxIntSliderGLGUI *wt = new rxIntSliderGLGUI;

		wt->posx = m_fCurX;
		wt->posy = m_fCurY;
		wt->range[0] = min_val;
		wt->range[1] = max_val;
		wt->name = name;
		wt->val = val;
		wt->val_str = V2S(val);
		wt->res = (max_val-min_val)/spacing;
		wt->enable = true;

		m_vVals.push_back((rxWidgetGLGUI*)wt);

		CalMaxLength();

		m_fCurY += 30;

		return (int)m_vVals.size()-1;
	}

	/*!
	 * 文字列描画
	 * @param[in] str 文字列
	 */
	void DrawGUI(bool oth = true)
	{
		// MARK:DrawGUI

		if(oth){
			glDisable(GL_LIGHTING);
			glDisable(GL_CULL_FACE);

			//glViewport(0, 0, m_iW, m_iH);

			glMatrixMode(GL_PROJECTION);
			glPushMatrix();
			glLoadIdentity();
			gluOrtho2D(0, m_iW, 0, m_iH);
			glMatrixMode(GL_MODELVIEW);
			glPushMatrix();
			glLoadIdentity();
		}

		glPushMatrix();
		glTranslatef(m_iW-(240+m_iMaxLength), 20, 0.0);

		int cx = 45;
		//int cx = m_iW-(120+m_iMaxLength);
		int cy = 15;
		int n = (int)m_vVals.size();
		//for(int i = 0; i < n; ++i){
		for(int i = n-1; i >= 0; --i){
			if(m_vVals[i]->enable){
				m_vVals[i]->Draw(i, m_iMaxLength, cx, cy, (i == m_iSelItem));
				cy += 34;
			}
			glLoadName(-1);
		}
		glPopMatrix();

		if(oth){
			glPopMatrix();
			glMatrixMode(GL_PROJECTION);
			glPopMatrix();
			glMatrixMode(GL_MODELVIEW);
		}
	}



	//
	// MARK:マウス
	//
	bool Mouse(int x, int y)
	{
		int idx;
		//cout << x << ", " << y << endl;
		if((idx = MousePickObject(x, y)) != -1){
			//cout << "pick item : " << idx << endl;
			m_iSelItem = idx;
			return true;
		}
		else{
			m_iSelItem = -1;
			return false;
		}
	}

	void SetValuePixelStart(int x, int y)
	{
		if(x < 0 || y < 0 || x > m_iW || y > m_iH) return;

		//cout << "pixel start : " << m_iSelItem << endl;
		// ドラッグ開始
		m_iDrag = 1;

		// ドラッグ開始点を記録
		m_iSx = x;
		m_iSy = y;

		m_iPx = x;
		m_iPy = y;

		if(m_iSelItem >= 0){
			m_vVals[m_iSelItem]->SetInitMouse();
		}
	}

	void SetValuePixelMotion(int x, int y, int res = -1)
	{
		if(x < 0 || y < 0 || x > m_iW || y > m_iH) return;

		if(m_iDrag){
			if(m_iSelItem < 0) return;
			//cout << "pixel motion : " << m_iSelItem << endl;

			// マウスポインタの位置の前回位置からの変位
			int dx = (x-m_iSx);
			//int dy = (y-m_iSy);

			//cout << "mouse (" << x << ", " << y << ")" << " - " << dx << endl;
			m_vVals[m_iSelItem]->SetValueMouse(dx, res);

			m_iPx = x;
			m_iPy = y;
		}

	}


	void SetValuePixelStop(int x, int y)
	{
		if(x < 0 || y < 0 || x > m_iW || y > m_iH || !m_iDrag) return;

		// ドラッグ終了点における変位を求める
		SetValuePixelMotion(x, y);

		// ドラッグ終了
		m_iDrag = 0;

		//cout << "pixel stop : " << m_iSelItem << endl;
	}


	//
	// MARK:キーボード
	//
	void Keyboard(unsigned char key, int x, int y, int mod = -1)
	{
		if(key == GLUT_KEY_LEFT){
			m_vVals[m_iSelItem]->SetValueMouse(1, -1);
		}
		else if(key == GLUT_KEY_RIGHT){
			m_vVals[m_iSelItem]->SetValueMouse(-1, -1);
		}
		//if((key >= "0" && key <= "9") || 
		//	key == "." || key == "-"){
		//	//m_vVals[m_iSelItem]->AddKey(key)
		//}
	}



	

	//
	// MARK:マウスピック
	//

	/*!
	 * マウス選択
	 * @param[in] x,y マウス座標
	 * @retval int ピックオブジェクト番号
	 */
	int MousePick(int x, int y)
	{
		// HCK:Pick
		GLuint selbuf[RX_SEL_BUFFER_SIZE];
		GLint hits;
		GLint viewport[4];

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_CULL_FACE);

		glViewport(0, 0, m_iW, m_iH);

		glGetIntegerv(GL_VIEWPORT, viewport);
		glSelectBuffer(RX_SEL_BUFFER_SIZE, selbuf);

		glRenderMode(GL_SELECT);

		glInitNames();
		glPushName(-1);

		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();

		gluPickMatrix(x, viewport[3]-y, 3.0, 3.0, viewport);
		gluOrtho2D(0, m_iW, 0, m_iH);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		DrawGUI(false);

		glLoadName(-1);

		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopName();

		hits = glRenderMode(GL_RENDER);
		
		int obj_number = MouseSelectHits(hits, selbuf);
		//cout << "hits : " << hits << endl;
		glMatrixMode(GL_MODELVIEW);

		return obj_number;
	}


	/*!
	 * OpenGLによるピックでヒットしたものから最小のデプス値を持つものを選択する
	 * @param hits ヒット数
	 * @param buf  選択バッファ
	 * @return ヒットしたオブジェクト番号
	 */
	int MouseSelectHits(GLint hits, GLuint buf[])
	{
		GLuint hit_name;

		float depth_min = 100.0f;
		float depth1 = 1.0f;
		float depth2 = 1.0f;

		GLuint depth_name;
		GLuint *ptr;
		
		// ヒットしたデータなし
		if(hits <= 0) return -1;
		
		// ポインタを作業用ptrへ渡す．
		ptr = (GLuint*)buf;
		for(int i = 0; i < hits; ++i){
			// 識別番号の階層の深さ
			depth_name = *ptr;
			ptr++;
			depth1 = (float) *ptr/0x7fffffff;
			ptr++;
			depth2 = (float) *ptr/0x7fffffff;
			ptr++;

			// 最小デプスの確認
			if(depth_min > depth1){
				depth_min = depth1;
				hit_name = *ptr;
			}
			ptr++;
		}
		return hit_name;
	}

	/*!
	 * オブジェクトのマウス選択
	 * @param[in] x,y マウス座標
	 * @retval true ピック成功
	 * @retval false ピック失敗
	 */
	int MousePickObject(int x, int y)
	{
		int obj_number;
		if((obj_number = MousePick(x, y)) != -1){
			return obj_number-1;
		}
		else{
			return -1;
		}

		//Zバッファ値をゲット
	//	object_depth = PickDepth(x,y);

		return -1;
	}

};


#endif // #ifndef _TEXTURE_H_