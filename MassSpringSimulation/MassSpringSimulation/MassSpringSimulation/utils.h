/*! @file utils.h
	
	@brief 様々な関数
 
	@author Makoto Fujisawa
	@date 2012
*/

#ifndef _RX_UTILS_H_
#define _RX_UTILS_H_

#ifdef _DEBUG
#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "libpngd.lib")
#pragma comment(lib, "zlibd.lib")
#pragma comment(lib, "rx_modeld.lib")
#else
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng.lib")
#pragma comment(lib, "zlib.lib")
#pragma comment(lib, "rx_model.lib")
#endif

#pragma comment(lib, "glew32.lib")


#pragma warning (disable: 4101)

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <direct.h>


// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// 様々な関数
#include "rx_utility.h"

// マウスによる視点移動とマウスピック
#include "rx_trackball.h"
#include "rx_pick.h"

// テクスチャ＆画像入出力
#include "rx_gltexture.h"


using namespace std;

//-----------------------------------------------------------------------------
// 定数・定義
//-----------------------------------------------------------------------------
#define RXCOUT std::cout



//-----------------------------------------------------------------------------
// 関数
//-----------------------------------------------------------------------------
/*!
 * パスからファイル名を取り除いたパスを抽出
 * @param[in] path パス
 * @return フォルダパス
 */
inline string GetFolderPath(const string &path)
{
	size_t pos1;
 
	pos1 = path.rfind('\\');
	if(pos1 != string::npos){
		return path.substr(0, pos1+1);
		
	}
 
	pos1 = path.rfind('/');
	if(pos1 != string::npos){
		return path.substr(0, pos1+1);
	}
 
	return "";
}

/*!
 * ファイル名生成
 * @param head : 基本ファイル名
 * @param ext  : 拡張子
 * @param n    : 連番
 * @param d    : 連番桁数
 * @return 生成したファイル名
 */
static inline string CreateFileName(const string &head, const string &ext, int n, const int &d)
{
	string file_name = head;
	int dn = d-1;
	if(n > 0){
		dn = (int)(log10((double)n))+1;
	}
	else if(n == 0){
		dn = 1;
	}
	else{
		n = 0;
		dn = 1;
	}

	for(int i = 0; i < d-dn; ++i){
		file_name += "0";
	}

	file_name += RX_TO_STRING(n);
	if(!ext.empty() && ext[0] != '.') file_name += ".";
	file_name += ext;

	return file_name;
}

/*!
 * パスから拡張子を小文字にして取り出す
 * @param[in] path ファイルパス
 * @return (小文字化した)拡張子
 */
inline string GetExtension(const string &path)
{
	string ext;
	size_t pos1 = path.rfind('.');
	if(pos1 != string::npos){
		ext = path.substr(pos1+1, path.size()-pos1);
		string::iterator itr = ext.begin();
		while(itr != ext.end()){
			*itr = tolower(*itr);
			itr++;
		}
		itr = ext.end()-1;
		while(itr != ext.begin()){	// パスの最後に\0やスペースがあったときの対策
			if(*itr == 0 || *itr == 32){
				ext.erase(itr--);
			}
			else{
				itr--;
			}
		}
	}
 
	return ext;
}

/*!
 * ディレクトリ作成(多階層対応)
 * @param[in] dir 作成ディレクトリパス
 * @return 成功で1,失敗で0 (ディレクトリがすでにある場合も1を返す)
 */
static int MkDir(string dir)
{
	if(_mkdir(dir.c_str()) != 0){
		char cur_dir[512];
		_getcwd(cur_dir, 512);	// カレントフォルダを確保しておく
		if(_chdir(dir.c_str()) == 0){	// chdirでフォルダ存在チェック
			cout << "MkDir : " << dir << " is already exist." << endl;
			_chdir(cur_dir);	// カレントフォルダを元に戻す
			return 1;
		}
		else{
			size_t pos = dir.find_last_of("\\/");
			if(pos != string::npos){	// 多階層の可能性有り
				int parent = MkDir(dir.substr(0, pos+1));	// 親ディレクトリを再帰的に作成
				if(parent){
					if(_mkdir(dir.c_str()) == 0){
						return 1;
					}
					else{
						return 0;
					}
				}
			}
			else{
				return 0;
			}
		}
	}

	return 1;
}




//-----------------------------------------------------------------------------
// OpenGL
//-----------------------------------------------------------------------------
/*!
 * 現在の画面描画を画像ファイルとして保存
 * @param[in] fn ファイルパス
 */
static void SaveDisplay(const string &fn, int w_, int h_)
{
	int c_ = 4;
	unsigned char* data = new unsigned char[w_*h_*c_];

	glReadPixels(0, 0, w_, h_, GL_RGBA, GL_UNSIGNED_BYTE, data);

	// 上下反転
	int stride = w_*c_;
	for(int j = 0; j < h_/2; ++j){
		for(int i = 0; i < stride; ++i){
			unsigned char tmp = data[j*stride+i];
			data[j*stride+i] = data[(h_-j-1)*stride+i];
			data[(h_-j-1)*stride+i] = tmp;
		}
	}

	string ext = GetExtension(fn);
	if(ext == "bmp"){
		WriteBitmapFile(fn, data, w_, h_, c_, RX_BMP_WINDOWS_V3);
		cout << "saved the screen image to " << fn << endl;
	}
	else if(ext == "png"){
		WritePngFile(fn, data, w_, h_, c_);
		cout << "saved the screen image to " << fn << endl;
	}

	delete [] data;
}

/*!
 * xyz軸描画(x軸:赤,y軸:緑,z軸:青)
 * @param[in] len 軸の長さ
 */
inline int DrawAxis(double len, double line_width = 5.0)
{
	glLineWidth(line_width);

	// x軸
	glColor3f(1.0, 0.0, 0.0);
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(len, 0.0, 0.0);
	glEnd();

	// y軸
	glColor3f(0.0, 1.0, 0.0);
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, len, 0.0);
	glEnd();

	// z軸
	glColor3f(0.0, 0.0, 1.0);
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, 0.0, len);
	glEnd();

	return 1;
}

/*!
 * 円筒描画
 * @param[in] rad,len 半径と中心軸方向長さ
 * @param[in] slices  ポリゴン近似する際の分割数
 */
static void DrawCylinder(double rad, double len, int up, int slices)
{
	GLUquadricObj *qobj;
	qobj = gluNewQuadric();

	glPushMatrix();
	switch(up){
	case 0:
		glRotatef(-90.0, 0.0, 1.0, 0.0);
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	case 1:
		glRotatef(-90.0, 1.0, 0.0, 0.0);
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	case 2:
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	default:
		glTranslatef(0.0, 0.0, -0.5*len);
	}

	gluQuadricDrawStyle(qobj, GLU_FILL);
	gluQuadricNormals(qobj, GLU_SMOOTH);
	gluCylinder(qobj, rad, rad, len, slices, slices);

	glPushMatrix();
	glRotatef(180.0, 1.0, 0.0, 0.0);
	gluDisk(qobj, 0.0, rad, slices, slices);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(0.0, 0.0, len);
	gluDisk(qobj, 0.0, rad, slices, slices);
	glPopMatrix();

	glPopMatrix();
}

/*!
 * カプセル描画(円筒の両端に半球をつけた形)
 * @param[in] rad,len 半径と中心軸方向長さ
 * @param[in] slices  ポリゴン近似する際の分割数
 */
static void DrawCapsule(double rad, double len, int up, int slices)
{
	GLUquadricObj *qobj;
	qobj = gluNewQuadric();

	glPushMatrix();
	switch(up){
	case 0:
		glRotatef(-90.0, 0.0, 1.0, 0.0);
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	case 1:
		glRotatef(-90.0, 1.0, 0.0, 0.0);
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	case 2:
		glTranslatef(0.0, 0.0, -0.5*len);
		break;
	default:
		glTranslatef(0.0, 0.0, -0.5*len);
	}

	gluQuadricDrawStyle(qobj, GLU_FILL);
	gluQuadricNormals(qobj, GLU_SMOOTH);
	gluCylinder(qobj, rad, rad, len, slices, slices);

	glPushMatrix();
	glutSolidSphere(rad, slices, slices);
	glPopMatrix();

	glPushMatrix();
	glTranslatef(0.0, 0.0, len);
	glutSolidSphere(rad, slices, slices);
	glPopMatrix();

	glPopMatrix();

}







/*!
 * レイ/線分と三角形の交差
 * @param[in] P0,P1 レイ/線分の端点orレイ上の点
 * @param[in] V0,V1,V2 三角形の頂点座標
 * @param[out] I 交点座標
 * @retval 1 交点Iで交差 
 * @retval 0 交点なし
 * @retval 2 三角形の平面内
 * @retval -1 三角形が"degenerate"である(面積が0，つまり，線分か点になっている)
 */
static int IntersectSegmentTriangle(Vec3 P0, Vec3 P1,			// Segment
									Vec3 V0, Vec3 V1, Vec3 V2,	// Triangle
									Vec3 &I, Vec3 &n, float rp)			// Intersection point (return)
{
	// 三角形のエッジベクトルと法線
	Vec3 u = V1-V0;
	Vec3 v = V2-V0;
	n = Unit(cross(u, v));
	if(RXFunc::IsZeroVec(n)){
		return -1;	// 三角形が"degenerate"である(面積が0)
	}

	// 線分
	Vec3 dir = P1-P0;
	double a = dot(n, P0-V0);
	double b = dot(n, dir);
	if(fabs(b) < 1e-10){	// 線分と三角形平面が平行
		if(a == 0){
			return 2;	// 線分が平面上
		}
		else{
			return 0;	// 交点なし
		}
	}

	// 交点計算

	// 2端点がそれぞれ異なる面にあるかどうかを判定
	float r = -a/b;
	Vec3 offset = Vec3(0.0);
	float dn = 0;
	float sign_n = 1;
	if(a < 0){
		return 0;
	}

	if(r < 0.0){
		return 0;
	}
	else{
		if(fabs(a) > fabs(b)){
			return 0;
		}
		else{
			if(b > 0){
				return 0;
			}
		}
	}

	// 線分と平面の交点
	I = P0+r*dir;//+offset;

	// 交点が三角形内にあるかどうかの判定
	double uu, uv, vv, wu, wv, D;
	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	Vec3 w = I-V0;
	wu = dot(w, u);
	wv = dot(w, v);
	D = uv*uv-uu*vv;

	double s, t;
	s = (uv*wv-vv*wu)/D;
	if(s < 0.0 || s > 1.0){
		return 0;
	}
	
	t = (uv*wu-uu*wv)/D;
	if(t < 0.0 || (s+t) > 1.0){
		return 0;
	}

	return 1;
}



#endif // #ifndef _RX_UTILS_H_