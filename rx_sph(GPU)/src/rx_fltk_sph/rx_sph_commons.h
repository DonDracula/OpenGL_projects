/*! 
  @file rx_sph_commons.h

  @brief SPHプログラムの共通ヘッダ
 
  @author Makoto Fujisawa
  @date 2008
*/
// FILE --rx_sph_commons.h--

#ifndef _RX_SPH_COMMONS_H_
#define _RX_SPH_COMMONS_H_


//-----------------------------------------------------------------------------
// MARK:インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <ctime>
#include <cmath>
//#include <cctype>
#include <cstdio>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// STL
#include <vector>
#include <string>
#include <map>
#include <bitset>
#include <algorithm>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

// boost
#include <boost/config.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/fstream.hpp>

#include <boost/iostreams/stream.hpp>
#include <boost/algorithm/string.hpp>

// ユーティリティ
#include "rx_utility.h"

#include "rx_material.h"
#include "rx_matrix.h"

#include "rx_timer.h"

// CUDA
#include "rx_cu_common.cuh"

using namespace std;


//-----------------------------------------------------------------------------
// MARK:定義
//-----------------------------------------------------------------------------
typedef unsigned char byte;

#define STR(x) #x

#define RXFRMT(a) (boost::format("%.3f") % (a))
#define RXFRMTE(a) (boost::format("%.3e") % (a))


//-----------------------------------------------------------------------------
// 関数の定義
//-----------------------------------------------------------------------------
//! グラデーション色の生成
template<class T> 
inline void RX_COLOR_RAMP(T t, T *r)
{
	const int ncolors = 7;
	T c[ncolors][3] = {	{1.0, 0.0, 0.0},
						{1.0, 0.5, 0.0},
						{1.0, 1.0, 0.0},
						{0.0, 1.0, 0.0},
						{0.0, 1.0, 1.0},
						{0.0, 0.0, 1.0},
						{1.0, 0.0, 1.0} };
	t = t*(ncolors-1);
	int i = (int)t;
	T u = t-floor(t);
	r[0] = RX_LERP(c[i][0], c[i+1][0], u);
	r[1] = RX_LERP(c[i][1], c[i+1][1], u);
	r[2] = RX_LERP(c[i][2], c[i+1][2], u);
}



//
// CUDA用
//
#ifndef RXREAL
	#define RXREAL float
#endif
#ifndef RXREAL2
	#define RXREAL2 float2
#endif
#ifndef RXREAL3
	#define RXREAL3 float3
#endif

//! RXREAL2の出力オペレータ
inline string &operator<<(string &cb, const RXREAL2 &a)
{
	return cb << "(" << a.x << ", " << a.y << ")";
}
inline std::ostream &operator<<(std::ostream &out, const RXREAL2 &a)
{
	return out << "(" << a.x << ", " << a.y << ")" ;
}

//! RXREAL3の出力オペレータ
inline string &operator<<(string &cb, const RXREAL3 &a)
{
	return cb << "(" << a.x << ", " << a.y << ", " << a.z << ")" ;
}
inline std::ostream &operator<<(std::ostream &out, const RXREAL3 &a)
{
	return out << "(" << a.x << ", " << a.y << ", " << a.z << ")" ;
}

//! uint3の出力オペレータ
inline std::ostream &operator<<(std::ostream &out, const uint3 &a)
{
	return out << "(" << a.x << ", " << a.y << ", " << a.z << ")" ;
}

//! matrix3x3の出力オペレータ
inline std::ostream &operator<<(std::ostream &out, const matrix3x3 &a)
{
	return out << a.e[0] << "\n" << a.e[1] << "\n" << a.e[2];
}

//! matrix3x3の初期化(単位行列)
inline matrix3x3 Identity(void)
{
	matrix3x3 mat;
	mat.e[0] = make_float3(1.0f, 0.0f, 0.0f);
	mat.e[1] = make_float3(0.0f, 1.0f, 0.0f);
	mat.e[2] = make_float3(0.0f, 0.0f, 1.0f);
	return mat;
}

//! matrix3x3の逆行列
inline matrix3x3 Inverse(matrix3x3 m)
{
	float d = m.e[0].x*m.e[1].y*m.e[2].z- 
			  m.e[0].x*m.e[2].y*m.e[1].z+ 
			  m.e[1].x*m.e[2].y*m.e[0].z- 
			  m.e[1].x*m.e[0].y*m.e[2].z+ 
			  m.e[2].x*m.e[0].y*m.e[1].z- 
			  m.e[2].x*m.e[1].y*m.e[0].z;

	if(d == 0) d = 1;

	matrix3x3 inv_m;
	inv_m.e[0] = make_float3( (m.e[1].y*m.e[2].z-m.e[1].z*m.e[2].y)/d, -(m.e[0].y*m.e[2].z-m.e[0].z*m.e[2].y)/d, (m.e[0].y*m.e[1].z-m.e[0].z*m.e[1].y)/d);
	inv_m.e[1] = make_float3(-(m.e[1].x*m.e[2].z-m.e[1].z*m.e[2].x)/d, (m.e[0].x*m.e[2].z-m.e[0].z*m.e[2].x)/d, -(m.e[0].x*m.e[1].z-m.e[0].z*m.e[1].x)/d);
	inv_m.e[2] = make_float3( (m.e[1].x*m.e[2].y-m.e[1].y*m.e[2].x)/d, -(m.e[0].x*m.e[2].y-m.e[0].y*m.e[2].x)/d, (m.e[0].x*m.e[1].y-m.e[0].y*m.e[1].x)/d);
	
	return inv_m;
}


//! オイラー角から回転行列を生成
inline matrix3x3 EulerToMatrix(double pitch, double yaw, double roll)
{
	matrix3x3 mat;

	yaw   = RX_TO_RADIANS(yaw);
	pitch = RX_TO_RADIANS(pitch);
	roll  = RX_TO_RADIANS(roll);

	double cy = cos(yaw); 
	double sy = sin(yaw); 
	double cp = cos(pitch); 
	double sp = sin(pitch); 
	double cr = cos(roll);
	double sr = sin(roll);

	double cc = cy*cr; 
	double cs = cy*sr; 
	double sc = sy*cr; 
	double ss = sy*sr;

	mat.e[0] = make_float3(cc+sp*ss, cs-sp*sc, -sy*cp);
	mat.e[1] = make_float3(-cp*sr,   cp*cr,    -sp);
	mat.e[2] = make_float3(sc-sp*cs, ss+sp*cc, cy*cp);

	return mat;
}


inline void EulerToMatrix(double *m, double pitch, double yaw, double roll)
{
	yaw   = RX_TO_RADIANS(yaw);
	pitch = RX_TO_RADIANS(pitch);
	roll  = RX_TO_RADIANS(roll);
 
	double cy = cos(yaw); 
	double sy = sin(yaw); 
	double cp = cos(pitch); 
	double sp = sin(pitch); 
	double cr = cos(roll);
	double sr = sin(roll);
 
	double cc = cy*cr; 
	double cs = cy*sr; 
	double sc = sy*cr; 
	double ss = sy*sr;
 
	m[0]  = cc+sp*ss;
	m[1]  = cs-sp*sc;
	m[2]  = -sy*cp;
	m[3]  = 0.0;
	
	m[4]  = -cp*sr;
	m[5]  = cp*cr; 
	m[6]  = -sp;
	m[7]  = 0.0;
	
	m[8]  = sc-sp*cs;
	m[9]  = ss+sp*cc;
	m[10] = cy*cp;
	m[11] = 0.0;
 
	m[12] = 0.0;
	m[13] = 0.0;
	m[14] = 0.0;
	m[15] = 1.0;
}

inline void GetGLMatrix(matrix3x3 mat, float glmat[16])
{
	glmat[0]  = mat.e[0].x;
	glmat[1]  = mat.e[0].y;
	glmat[2]  = mat.e[0].z;
	glmat[3]  = 0.0;

	glmat[4]  = mat.e[1].x;
	glmat[5]  = mat.e[1].y;
	glmat[6]  = mat.e[1].z;
	glmat[7]  = 0.0;

	glmat[8]  = mat.e[2].x;
	glmat[9]  = mat.e[2].y;
	glmat[10] = mat.e[2].z;
	glmat[11] = 0.0;

	glmat[12] = 0.0;
	glmat[13] = 0.0;
	glmat[14] = 0.0;
	glmat[15] = 1.0;
}


/*!
 * 3x3行列からOpenGLの変換行列を取得
 * @param[in] 
 * @param[out] 
 * @return 
 */
inline void GetGLMatrix(RXREAL mat[9], GLfloat glmat[16])
{
	glmat[0]  = mat[0];
	glmat[1]  = mat[3]; 
	glmat[2]  = mat[6]; 
	glmat[3]  = 0.0f; 
	glmat[4]  = mat[1]; 
	glmat[5]  = mat[4]; 
	glmat[6]  = mat[7]; 
	glmat[7]  = 0.0f; 
	glmat[8]  = mat[2]; 
	glmat[9]  = mat[5]; 
	glmat[10] = mat[8]; 
	glmat[11] = 0.0f; 
	glmat[12] = 0.0f; 
	glmat[13] = 0.0f; 
	glmat[14] = 0.0f; 
	glmat[15] = 1.0f;
}


/*!
 * 青->緑->赤->白と変化するグラデーション色生成
 * @param[out] col 生成された色
 * @param[in] x 値
 * @param[in] xmin 最小値
 * @param[in] xmax 最大値
 */
inline void Gradation(double col[3], double x, const double xmin = 0.0, const double xmax = 1.0)
{
	double l = xmax-xmin;
	if(fabs(l) < 1e-10) return;
	
	const int ncolors = 7;
	double base[ncolors][3] = { {0.0, 0.0, 0.0},
								{0.0, 0.0, 1.0},
								{0.0, 1.0, 1.0},
								{0.0, 1.0, 0.0},
								{1.0, 1.0, 0.0},
								{1.0, 0.0, 0.0},
								{1.0, 1.0, 1.0} };
	x = RX_CLAMP(((x-xmin)/l), 0.0, 1.0)*(ncolors-1);
	int i = (int)x;
	double dx = x-floor(x);
	col[0] = RX_LERP(base[i][0], base[i+1][0], dx);
	col[1] = RX_LERP(base[i][1], base[i+1][1], dx);
	col[2] = RX_LERP(base[i][2], base[i+1][2], dx);
}


//-----------------------------------------------------------------------------
// MARK:ストリーム出力
//-----------------------------------------------------------------------------

//! テキストストリーム - コマンドプロンプトに表示すると共にログファイルにも保存
class rxCout : public boost::iostreams::sink
{
	string m_strLog;

public:
	rxCout(string fn)
	{
		m_strLog = fn;
	}

	std::streamsize write(const char* s, std::streamsize n)
	{
		string str;;
		str.resize(n);
		for(int i = 0; i < n; ++i){
			str[i] = s[i];
		}

		cout << str;

		boost::algorithm::replace_all(str, "\n", "");

		static std::ofstream fout(m_strLog, std::ios::out);
		fout << str << endl;
		return n;
	}
};

static boost::iostreams::stream<rxCout> RXCOUT("log/sph.log");
//#define RXCOUT std::cout


#define RXPRINT(x) RXCOUT << boost::format(x)





//-----------------------------------------------------------------------------
// コンテナ関連
//-----------------------------------------------------------------------------

/*!
 * vectorのidx番目の要素を削除(0スタート)
 * @param[in] src vectorコンテナ
 * @param[in] idx 削除要素インデックス
 * @return 削除の可否
 */
template<class T> 
inline bool EraseSTLVectori(vector<T> &src, int idx)
{
	vector<T>::iterator iter = src.begin();
	int i = 0;
	while(iter != src.end()){
		if(i == idx){
			src.erase(iter++);
			return true;
		}
		else{
			++iter;
		}

		i++;
	}

	return false;
}

/*!
 * vectorの特定の要素を削除
 * @param[in] src vectorコンテナ
 * @param[in] comp_func 削除条件関数
 * @return 削除の可否
 */
template<class T> 
inline bool EraseSTLVector(vector<T> &src, boost::function<bool (T)> comp_func)
{
	vector<T>::iterator iter = src.begin();
	while(iter != src.end()){
		if(comp_func(*iter)){
			src.erase(iter++);
			return true;
		}
		else{
			++iter;
		}
	}

	return false;
}

/*!
 * mapの特定の要素を削除
 * @param[in] src マップ
 * @param[in] comp_func 削除条件関数
 * @return 削除の可否
 */
template<class T1, class T2> 
inline bool EraseSTLMap(map<T1, T2> &src, boost::function<bool (pair<T1, T2>)> comp_func)
{
	map<T1, T2>::iterator iter = src.begin();
	while(iter != src.end()){
		if(comp_func(*iter)){
			src.erase(iter++);
		}
		else{
			++iter;
		}
	}

	return (iter != src.end());
}



/*!
 * "(x, y, z)"の形式の文字列からVec3型へ変換
 *  - (x)となっていたら(x, x, x)とする．
 * @param[in] s 文字列
 * @param[out] v 値
 * @return 要素記述数
 */
inline int StringToVec3(const string &s, Vec3 &v)
{
	int vcount = 0;
	size_t pos;
	v = Vec3(0.0);
	if((pos = s.find('(')) != string::npos){
		while(pos != string::npos && vcount < 3){
			size_t pos1 = pos;
			if((pos1 = s.find(',', pos+1)) != string::npos){
				v[vcount] = atof(s.substr(pos+1, (pos1-(pos+1))).c_str());
				vcount++;
				pos = pos1;
			}
			else if((pos1 = s.find(')', pos+1)) != string::npos){
				v[vcount] = atof(s.substr(pos+1, (pos1-(pos+1))).c_str());
				vcount++;
				break;
			}
			else{
				break;
			}
		}
	}
	if(vcount < 3){
		for(int i = vcount; i < 3; ++i){
			v[i] = v[vcount-1];
		}
	}

	return vcount;
}


//-----------------------------------------------------------------------------
// サーモグラフ
//-----------------------------------------------------------------------------
/*!
 * col0->col1と変化するサーモグラフ用の色生成
 * @param 
 * @return 
 */
template<class Type> 
inline void Gradation(Type x, Type minx, Type maxx, 
					  Type col0[3], Type col1[3], 
					  Type &r, Type &g, Type &b)
{
	Type l = maxx-minx;
	
	if(fabs(l) < 1e-8){
		r = col0[0]; g = col0[1]; b = col0[2];
		return;
	}

	if(x <= minx){
		r = col0[0]; g = col0[1]; b = col0[2];
	}
	else if(x >= maxx){
		r = col1[0]; g = col1[1]; b = col1[2];
	}
	else{
		Type col[3];
		for(int i = 0; i < 3; ++i){
			col[i] = col0[i]+(col1[i]-col0[i])*((x-minx)/l);
		}
		r = col[0]; g = col[1]; b = col[2];
	}
}

/*!
 * col0->col1と変化するサーモグラフ用の色生成
 * @param 
 * @return 
 */
template<class Type> 
inline Vec3 Gradation(Type x, Type minx, Type maxx, Type col0[3], Type col1[3])
{
	Type l = maxx-minx;
	Vec3 c;
	
	if(fabs(l) < 1e-8){
		c[0] = col0[0]; c[1] = col0[1]; c[2] = col0[2];
		return c;
	}

	if(x <= minx){
		c[0] = col0[0]; c[1] = col0[1]; c[2] = col0[2];
	}
	else if(x >= maxx){
		c[0] = col1[0]; c[1] = col1[1]; c[2] = col1[2];
	}
	else{
		Type col[3];
		for(int i = 0; i < 3; ++i){
			col[i] = col0[i]+(col1[i]-col0[i])*((x-minx)/l);
		}
		c[0] = col[0]; c[1] = col[1]; c[2] = col[2];
	}

	return c;
}


//-----------------------------------------------------------------------------
// MARK:データファイル出力
//-----------------------------------------------------------------------------
template<class Type> 
static void Output(int w, int h, Type *data, char *name)
{
	// HACK:output
	if(data == NULL) return;

	ofstream fo;
	fo.open(name);

	int size = w*h;
	Type min_ = data[0], max_ = data[0];
	for(int i = 0; i < size; ++i){
		if(data[i] < min_) min_ = data[i];
		if(data[i] > max_) max_ = data[i];
	}
	fo << "[" << min_ << ", " << max_ << "]" << endl;

	for(int j = 0; j < h; ++j){
		for(int i = 0; i < w; ++i){
			int idx = j*w+i;
			fo << data[idx] << " ";
		}
		fo << endl;
	}

	fo.close();
}



template<class Type> 
static void Output_v2(int w, int h, Type *x, Type *y, char *name)
{
	if(x == NULL || y == NULL) return;

	ofstream fo;
	fo.open(name);

	int size = w*h;
	Type minx = x[0], maxx = x[0];
	Type miny = y[0], maxy = y[0];
	for(int i = 0; i < size; ++i){
		if(x[i] < minx) minx = x[i];
		if(x[i] > maxx) maxx = x[i];
		if(y[i] < miny) miny = y[i];
		if(y[i] > maxy) maxy = y[i];
	}
	fo << "[ (" << minx << ", " << miny << "), (" << maxx << ", " << maxy << ") ]" << endl;

	for(int j = 0; j < h; ++j){
		for(int i = 0; i < w; ++i){
			int idx = j*w+i;
			fo << "(" << x[idx] << ", " << y[idx] << ") ";
		}
		fo << endl;
	}

	fo.close();
}



//-----------------------------------------------------------------------------
// MARK:文字列画面描画
//-----------------------------------------------------------------------------


/*!
 * 文字列描画
 * @param[in] cir_str 文字列循環バッファ
 * @param[in] static_str 静的な文字列バッファ
 * @param[in] w,h ウィンドウサイズ
 */
static void DrawStrings(vector<string> &static_str, int w, int h)
{
	// MRK:PrintD
	glDisable(GL_LIGHTING);
	//glColor3f(0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	float x0 = 5.0f;
	float y0 = h-20.0f;

	// 画面上部にスタティックなテキスト
	for(int j = 0; j < (int)static_str.size(); ++j){
		glRasterPos2f(x0, y0);

		int size = (int)static_str[j].size();
		for(int i = 0; i < size; ++i){
			char ic = static_str[j][i];
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ic);
		}

		y0 -= 20;
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}


//-----------------------------------------------------------------------------
// MARK:ランダムソート
//-----------------------------------------------------------------------------
/*!
 * [0, 1]の範囲の実数乱数の生成
 * @return 実数乱数
 */
inline double Random()
{
	return (double)rand()/(double)RAND_MAX;
}

/*!
 * 指定した範囲の実数乱数の生成
 * @param _max, _min 範囲
 * @return 実数乱数
 */
inline double Random(const double &_max, const double &_min)
{
	return (_max-_min)*Random()+_min;
}






//-----------------------------------------------------------------------------
// MARK:ファイル・フォルダ処理
//-----------------------------------------------------------------------------
/*!
 * ファイル，フォルダの存在確認
 * @param[in] path ファイル・フォルダパス
 */
inline bool ExistFile(const string &path_str)
{
	boost::filesystem::path fnph(path_str);
	return boost::filesystem::exists(fnph);
}


/*!
 * フォルダ区切りの検索
 * @param[in] str ファイル・フォルダパス
 * @param[out] pos 見つかった位置
 */
inline bool FindPathBound(const string &str, string::size_type &pos)
{
	if((pos = str.find_last_of("/")) == string::npos){
		if((pos = str.find_last_of("\\")) == string::npos){
			return false;
		}
	}
	
	return true;
}

/*!
 * フォルダの存在確認と作成
 * @param[in] path_str ファイル・フォルダパス
 */
inline bool CheckAndMakeDir(const string &path_str)
{
	if(!ExistFile(path_str)){
		vector<string> tmp_paths;
		tmp_paths.push_back(path_str);

		// パスを分解
		for(;;){
			string::size_type pos;
			if(!FindPathBound(tmp_paths.back(), pos)){
				break;
			}
			else{
				string str = tmp_paths.back().substr(0, pos);
				tmp_paths.push_back(str);
			}
		}

		// 分解したパスそれぞれ存在チェックし，なければ作成
		vector<string>::iterator itr = tmp_paths.end()-1;
		for( ; itr != tmp_paths.begin(); --itr){
			//cout << *itr << endl;
			if(!ExistFile(*itr)){
				boost::filesystem::create_directory(*itr);
			}
		}
	}

	return true;
}

/*!
 * ファイル名比較関数(拡張子)
 * @param[in] fn 比較したいファイル名
 * @param[in] ext 拡張子
 * @return fnの拡張子がextと同じならtrue
 */
inline bool SearchCompExt(const string &fn, const string &ext)
{
	return (fn.find(ext, 0) != string::npos);
}


/*!
 * ファイル名生成
 * @param head : 基本ファイル名
 * @param ext  : 拡張子
 * @param n    : 連番
 * @param d    : 連番桁数
 * @return 生成したファイル名
 */
inline string CreateFileName(const string &head, const string &ext, int n, const int &d)
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
 * 整数値の下一桁を返す
 * @param[in] x 整数値
 * @return xの下一桁
 */
inline int LowOneDigit(const int &x)
{
	int x1 = (x < 0) ? -x : x;
	double a = 10;

	//INT_MAX = 2147483647
	for(int i = 10; i >= 1; i--){
		a = pow(10.0, (double)i);
		while(x1 > a){
			x1 -= (int)a;
		}
	}

	return x1;
}

/*!
 * 0付きの数字を生成
 * @param[in] n 数字
 * @param[in] d 桁数
 * @return 0付きの数字(string)
 */
inline string GenZeroNo(int n, const int &d)
{
	string zero_no = "";
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
		zero_no += "0";
	}

	zero_no += RX_TO_STRING(n);

	return zero_no;
}


/*!
 * 秒数を hh:mm:ss の形式に変換
 * @param[in] sec 秒数
 * @param[in] use_msec ミリ秒まで含める(hh:mm:ss.msec)
 * @return hh:mm:ss形式の文字列
 */
inline string GenTimeString(double sec, bool use_msec = false)
{
	long value = (int)(1000*sec+0.5);	// ミリ秒

	unsigned int h = (unsigned int)(value/3600000);	// 時間
	value -= h*3600000;
	unsigned int m = (unsigned int)(value/60000);		// 分
	value -= m*60000;
	unsigned int s = (unsigned int)(value/1000);		// 秒
	value -= s*1000;
	unsigned int ms = (unsigned int)(value);			// ミリ秒

	stringstream ss;
	if(h > 0) ss << GenZeroNo(h, 2) << ":";
	ss << GenZeroNo(m, 2) << ":";
	ss << GenZeroNo(s, 2);
	if(use_msec) ss << "." << GenZeroNo(ms, 3);

	return ss.str();
}

/*!
 * 時刻を hh:mm:ss の形式に変換
 * @param[in] h,m,s 時,分,秒
 * @return hh:mm:ss形式の文字列
 */
inline string GenTimeString(int h, int m, int s)
{
	stringstream ss;
	if(h > 0) ss << GenZeroNo(h, 2) << ":";
	ss << GenZeroNo(m, 2) << ":";
	ss << GenZeroNo(s, 2);
	return ss.str();
}
/*!
 * パスからファイル名のみ取り出す
 * @param[in] path パス
 * @return ファイル名
 */
inline string GetFileName(const string &path)
{
	size_t pos1;
 
	pos1 = path.rfind('\\');
	if(pos1 != string::npos){
		return path.substr(pos1+1, path.size()-pos1-1);
	}
 
	pos1 = path.rfind('/');
	if(pos1 != string::npos){
		return path.substr(pos1+1, path.size()-pos1-1);
	}
 
	return path;
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









#endif // #ifndef _RX_SPH_COMMON_H_