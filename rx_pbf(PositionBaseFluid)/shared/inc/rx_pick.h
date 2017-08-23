/*! @file rx_pick.h
	
	@brief OpenGLのセレクションモードを使った頂点ピック
		   矩形選択にも対応
 
	@author Makoto Fujisawa
	@date   2008-06, 2010-03, 2013-02
*/

#pragma warning (disable: 4244)

#ifndef _RX_PICK_H_
#define _RX_PICK_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>

#include <vector>
#include <algorithm>

#include <GL/glut.h>

#include "rx_utility.h"


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
using namespace std;

//! ピックされたオブジェクトに関する情報
struct rxPickInfo
{
	GLuint name;		//!< ヒットしたオブジェクトの名前
	float min_depth;	//!< プリミティブのデプス値の最小値
	float max_depth;	//!< プリミティブのデプス値の最大値
};

static inline bool CompFuncPickInfo(rxPickInfo a, rxPickInfo b)
{
	return a.min_depth < b.min_depth;
}

//-----------------------------------------------------------------------------
// OpenGLによるオブジェクトピック
//-----------------------------------------------------------------------------
class rxGLPick
{
	void (*m_fpDraw)(void*);	//!< ピック用描画関数
	void (*m_fpProj)(void*);	//!< 投影変換関数
	void *m_pDrawFuncPtr;		//!< 描画関数を呼び出すためのポインタ(メンバ関数ポインタ用)
	void *m_pProjFuncPtr;		//!< 投影変換関数を呼び出すためのポインタ(メンバ関数ポインタ用)
 
	int m_iSelBufferSize;		//!< セレクションバッファのサイズ
	GLuint* m_pSelBuffer;		//!< セレクションバッファ

	int m_iLastPick;			//!< 最後にピックされたオブジェクトの番号
 
public:
	rxGLPick()
	{
		m_iSelBufferSize = 4096;
		m_pSelBuffer = new GLuint[m_iSelBufferSize];
	}
	~rxGLPick()
	{
		if(m_pSelBuffer) delete [] m_pSelBuffer;
	}
 
	void Set(void (*draw)(void*), void* draw_ptr, void (*proj)(void*), void* proj_ptr)
	{
		m_fpDraw = draw;
		m_fpProj = proj;
		m_pDrawFuncPtr = draw_ptr;
		m_pProjFuncPtr = proj_ptr;
	}
 
protected:
	/*!
	 * OpenGLによるピックでヒットしたものから最小のデプス値を持つものを選択する
	 * @param hits ヒット数
	 * @param buf  選択バッファ
	 * @return ヒットしたオブジェクト番号
	 */
	vector<rxPickInfo> selectHits(GLint nhits, GLuint buf[])
	{
		vector<rxPickInfo> hits;
 
		float depth_min = 100.0f;
		float depth1 = 1.0f;
		float depth2 = 1.0f;
 
		GLuint depth_name;
		GLuint *ptr;
		
		// ヒットしたデータなし
		if(nhits <= 0) return hits;
		
		hits.resize(nhits);

		// ポインタを作業用ptrへ渡す．
		ptr = (GLuint*)buf;
		for(int i = 0; i < nhits; ++i){
			// ヒットしたオブジェクトの名前
			depth_name = *ptr;
			ptr++;

			// ヒットしたプリミティブのデプス値の最小値
			depth1 = (float) *ptr/0x7fffffff;
			ptr++;

			// ヒットしたプリミティブのデプス値の最大
			depth2 = (float) *ptr/0x7fffffff;
			ptr++;

			hits[i].name = (int)(*ptr);
			hits[i].min_depth = depth1;
			hits[i].max_depth = depth2;

			ptr++;
		}

		return hits;
	}
 
	/*!
	 * マウス選択
	 * @param[in] x,y 選択中心座標(マウス座標系)
	 * @param[in] w,h 選択範囲
	 * @retval int ピックオブジェクト番号
	 */
	vector<rxPickInfo> pick(int x, int y, int w, int h)
	{
		GLint viewport[4];
 
		glGetIntegerv(GL_VIEWPORT, viewport);
		glSelectBuffer(m_iSelBufferSize, m_pSelBuffer);
 
		glRenderMode(GL_SELECT);
 
		glInitNames();
		glPushName(0);
 
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
 
		gluPickMatrix(x, viewport[3]-y, w, h, viewport);
		m_fpProj(m_pProjFuncPtr);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
 
		m_fpDraw(m_pDrawFuncPtr);
		glLoadName(0);
 
		glMatrixMode(GL_PROJECTION);
		glPopMatrix();
		glPopName();
 
		GLint nhits;
		nhits = glRenderMode(GL_RENDER);
		
		vector<rxPickInfo> hits = selectHits(nhits, m_pSelBuffer);
		glMatrixMode(GL_MODELVIEW);

		if(nhits > 0) m_iLastPick = hits[0].name;
		return hits;
	}
 
 
public:
	/*!
	 * オブジェクトのマウス選択
	 * @param[in] x,y マウス座標
	 * @retval true ピック成功
	 * @retval false ピック失敗
	 */
	int Pick(int x, int y)
	{
		vector<rxPickInfo> hits = pick(x, y, 16, 16);
		if(hits.empty()){
			m_iLastPick = -1;
			return -1;
		}
		else{
			std::sort(hits.begin(), hits.end(), CompFuncPickInfo);
			m_iLastPick = hits[0].name;
			return hits[0].name;
		}
 
		return -1;
	}

	/*!
	 * オブジェクトのマウス選択(矩形範囲指定, 複数選択)
	 * @param[in] sx,sy 始点
	 * @param[in] ex,ey 終点
	 * @retval true ピック成功
	 * @retval false ピック失敗
	 */
	vector<rxPickInfo> Pick(int sx, int sy, int ex, int ey)
	{
		int x, y, w, h;

		if(ex < sx) RX_SWAP(ex, sx);
		if(ey < sy) RX_SWAP(ey, sy);

		x = (ex+sx)/2;
		y = (ey+sy)/2;
		w = ex-sx;
		h = ey-sy;

		return pick(x, y, w, h);
	}

	/*!
	 * 最後にピックされたオブジェクトの番号を返す
	 * @return ピックされたオブジェクトの番号
	 */
	int GetLastPick(void) const
	{
		return m_iLastPick;
	}
};


/*!
 * マウスドラッグ領域の描画(矩形と直線)
 * @param[in] type 1で直線,2で矩形
 * @param[in] p0,p1 マウスドラッグ開始点，終了点
 */
static inline void DrawRubber(int type, Vec2 p0, Vec2 p1, int w, int h)
{
	glDisable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, w, h, 0);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	if(type == 1){		// 矩形
		glColor3d(1.0, 1.0, 1.0);
		glPushMatrix();
		glBegin(GL_LINE_LOOP);
		glVertex2d(p0[0], p0[1]);
		glVertex2d(p1[0], p0[1]);
		glVertex2d(p1[0], p1[1]);
		glVertex2d(p0[0], p1[1]);
		glEnd();
		glPopMatrix();
	}
	else if(type == 2){	// 直線
		glDisable(GL_LIGHTING);
		glColor3d(0.0, 0.0, 0.0);
		glBegin(GL_LINES);
		glVertex2dv(p0.data);
		glVertex2dv(p1.data);
		glEnd();
	}
	
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

}



#endif // #ifndef _RX_PICK_H_