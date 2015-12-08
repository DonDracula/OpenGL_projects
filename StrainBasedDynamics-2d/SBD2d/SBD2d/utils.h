/*! @file utils.h
	
	@brief 様々な関数
 
	@author Makoto Fujisawa
	@date 2012
*/

#ifndef _RX_UTILS_H_
#define _RX_UTILS_H_

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// STL
#include <vector>
#include <string>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// Vec3や様々な関数
#include "rx_utility.h"	

using namespace std;

//-----------------------------------------------------------------------------
// 定数・定義
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// 数値計算
//-----------------------------------------------------------------------------
/*!
 * 2x2行列の行列式の計算
 *  | m[0] m[1] |
 *  | m[2] m[3] |
 * @param[in] m 元の行列
 * @return 行列式の値
 */
static inline double CalDetMat2x2(const double *m)
{
    return m[0]*m[3]-m[1]*m[2];
}
 
/*!
 * 2x2行列の逆行列の計算
 *  | m[0] m[1] |
 *  | m[2] m[3] |
 * @param[in] m 元の行列
 * @param[out] invm 逆行列
 * @return 逆行列の存在
 */
static inline bool CalInvMat2x2(const double *m, double *invm)
{
    double det = CalDetMat2x2(m);
    if(fabs(det) < RX_FEQ_EPS){
        return false;
    }
    else{
        double inv_det = 1.0/det;
        invm[0] =  inv_det*m[3];
        invm[1] = -inv_det*m[1];
        invm[2] = -inv_det*m[2];
        invm[3] =  inv_det*m[0];
        return true;
    }
}

/*!
 * 3x3行列の行列式の計算
 *  | m[0] m[1] m[2] |
 *  | m[3] m[4] m[5] |
 *  | m[6] m[7] m[8] |
 * @param[in] m 元の行列
 * @return 行列式の値
 */
static inline double CalDetMat3x3(const double *m)
{
    // a11a22a33+a21a32a13+a31a12a23-a11a32a23-a31a22a13-a21a12a33
    return m[0]*m[4]*m[8]+m[3]*m[7]*m[2]+m[6]*m[1]*m[5]
          -m[0]*m[7]*m[5]-m[6]*m[4]*m[2]-m[3]*m[1]*m[8];
}
 
/*!
 * 3x3行列の逆行列の計算
 *  | m[0] m[1] m[2] |
 *  | m[3] m[4] m[5] |
 *  | m[6] m[7] m[8] |
 * @param[in] m 元の行列
 * @param[out] invm 逆行列
 * @return 逆行列の存在
 */
static inline bool CalInvMat3x3(const double *m, double *invm)
{
    double det = CalDetMat3x3(m);
    if(fabs(det) < RX_FEQ_EPS){
        return false;
    }
    else{
        double inv_det = 1.0/det;
 
        invm[0] = inv_det*(m[4]*m[8]-m[5]*m[7]);
        invm[1] = inv_det*(m[2]*m[7]-m[1]*m[8]);
        invm[2] = inv_det*(m[1]*m[5]-m[2]*m[4]);
 
        invm[3] = inv_det*(m[5]*m[6]-m[3]*m[8]);
        invm[4] = inv_det*(m[0]*m[8]-m[2]*m[6]);
        invm[5] = inv_det*(m[2]*m[3]-m[0]*m[5]);
 
        invm[6] = inv_det*(m[3]*m[7]-m[4]*m[6]);
        invm[7] = inv_det*(m[1]*m[6]-m[0]*m[7]);
        invm[8] = inv_det*(m[0]*m[4]-m[1]*m[3]);
 
        return true;
    }
}


//-----------------------------------------------------------------------------
// OpenGL
//-----------------------------------------------------------------------------
/*!
 * 文字列描画
 * @param[in] static_str 文字列
 * @param[in] w,h ウィンドウサイズ
 */
static void DrawStrings(vector<string> &static_str, int w, int h)
{
	glDisable(GL_LIGHTING);
	// 平行投影にする
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
 
	float x0 = 5.0f;
	float y0 = h-20.0f;
 
	// 画面上部にテキスト描画
	for(int j = 0; j < (int)static_str.size(); ++j){
		glRasterPos2f(x0, y0);
 
		int size = (int)static_str[j].size();
		for(int i = 0; i < size; ++i){
			char ic = static_str[j][i];
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ic);
		}
 
		y0 -= 20;
	}
 
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
}



#endif // #ifndef _RX_UTILS_H_