/*!
  @file rx_utility.h

  @brief 様々な関数
 
　@author Makoto Fujisawa
　@date  
*/

#ifndef _RX_UTILITY_
#define _RX_UTILITY_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
//#include <cstdarg>
#include <cmath>
#include <ctime>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <vector>

#include "rx_vec.h"


//--------------------------------------------------------------------
// 定数
//--------------------------------------------------------------------
// 円周率
const double RX_M_PI = 3.14159265358979323846;
const double RX_PI = 3.14159265358979323846;

//! (4/3)*円周率
const double RX_V_PI = 4.18666666666666666667;

//! 平方根
const double RX_ROOT2 = 1.414213562;
const double RX_ROOT3 = 1.732050808;
const double RX_ROOT5 = 2.236067977;

//! 許容誤差
const double RX_FEQ_EPS = 1.0e-10;
const double RX_EPS = 1.0e-8;

//! ∞
const double RX_FEQ_INF = 1.0e10;
const double RX_FEQ_INFM = 0.999e10;

//! float型最大値
//#define RX_MAX_FLOAT 3.40282347e+38F
#define RX_MAX_FLOAT 3.4e+38F

//! 1/pi
const double RX_INV_PI = 0.318309886;

//! ルート3
const double RX_SQRT3 = 1.7320508;

//! degree -> radian の変換係数(pi/180.0)
const double RX_DEGREES_TO_RADIANS = 0.0174532925199432957692369076848;

//! radian -> degree の変換係数(180.0/pi)
const double RX_RADIANS_TO_DEGREES = 57.295779513082320876798154814114;



//--------------------------------------------------------------------
// マクロ(テンプレート関数)
//--------------------------------------------------------------------
//! ゼロ判定
template<class T> 
inline bool RX_IS_ZERO(const T &x){ return (fabs(x) < RX_FEQ_EPS); }

//! 許容誤差を含めた等値判定
template<class T> 
inline bool RX_FEQ(const T &a, const T &b){ return (fabs(a-b) < RX_FEQ_EPS); }

//! 許容誤差を含めた等値判定(Vec3用)
template<class T> 
inline bool RX_VEC3_FEQ(const T &a, const T &b, double eps = RX_FEQ_EPS)
{
	return ( (fabs(a[0]-b[0]) < eps) && (fabs(a[1]-b[1]) < eps) && (fabs(a[2]-b[2]) < eps) );
}

//! degree -> radian の変換
template<class T> 
inline T RX_TO_RADIANS(const T &x){ return static_cast<T>((x)*RX_DEGREES_TO_RADIANS); }

//! radian -> degree の変換
template<class T> 
inline T RX_TO_DEGREES(const T &x){ return static_cast<T>((x)*RX_RADIANS_TO_DEGREES); }

//! 最大値判定(2値)
template<class T> 
inline T RX_MAX(const T &a, const T &b){ return ((a > b) ? a : b); }

//! 最大値判定(3値)
template<class T> 
inline T RX_MAX3(const T &a, const T &b, const T &c){ return ( (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c)); }

//! 最小値判定(2値)
template<class T> 
inline T RX_MIN(const T &a, const T &b){ return ((a < b) ? a : b); }

//! 最小値判定(3値)
template<class T> 
inline T RX_MIN3(const T &a, const T &b, const T &c){ return ( (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c)); }

//! 値のクランプ(クランプした値を返す)
template<class T> 
inline T RX_CLAMP(const T &x, const T &a, const T &b){ return ((x < a) ? a : (x > b) ? b : x); }

////! 値のクランプ([a,b]でクランプ)
//template<class T> 
//inline void RX_CLAMP(T &x, const T &a, const T &b){ x = (x < a ? a : (x > b ? b : x)); }

//! フラグの切替(tに-1を指定したら反転，0か1を指定でそれを代入)
template<class T>
inline void RX_TOGGLE(T &flag, int t = -1){ flag = ((t == -1) ? !flag : ((t == 0) ? 0 : 1)); }

//! 1次元線型補間
template<class T>
inline T RX_LERP(const T &a, const T &b, const T &t){ return a + t*(b-a); }

//! スワップ
template<class T>
inline void RX_SWAP(T &a, T &b){ T c; c = a; a = b; b = c; }

//! 符号
template<class T>
inline T RX_SIGN(const T &x){ return (x >= 0 ? 1 : (x < 0 ? -1 : 0)); }

//! aの符号をbの符号にあわせる
template<class T>
inline T RX_SIGN2(const T &a, const T &b){ return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a); }

//! 絶対値
template<class T>
inline T RX_ABS(const T &x){ return (x < 0 ? -x : x); }

//! [0,1]の浮動小数点数乱数
inline double RX_FRAND(){ return rand()/(double)RAND_MAX; }


/*!
 * 様々な型のstringへの変換(stringstreamを使用)
 * @param[in] x 入力
 * @return string型に変換したもの
 */
template<typename T>
inline std::string RX_TO_STRING(const T &x)
{
	stringstream ss;
	ss << x;
	return ss.str();
}

//! string型に<<オペレータを設定
template<typename T>
inline std::string &operator<<(std::string &cb, const T &a)
{
	cb += RX_TO_STRING(a);
	return cb;
}


//! 3次元配列ファイル出力
template<class T>
inline void RX_OUTPUT(const std::string fn, const T *f, int nx, int ny, int nz)
{
	ofstream file;
	file.open(fn.c_str(), ios::out);
	if(!file || !file.is_open() || file.bad() || file.fail()){
		cout << "Invalid file specified" << endl;
		return;
	}

	for(int k = 0; k < nz; ++k){
		file << "[" << k << "]" << endl;
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				file << f[i+j*nx+k*nx*ny] << (i == nx-1 ? "" : " ");
			}
			file << endl;
		}
		file << endl;
	}

	file.close();
}

//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
// FOR
#ifndef RXFOR3
#define RXFOR3(i0, i1, j0, j1, k0, k1) for(int i = i0; i < i1; ++i) \
											for(int j = j0; j < j1; ++j) \
												for(int k = k0; k < k1; ++k)
#endif

#ifndef RXFOR3E
#define RXFOR3E(i0, i1, j0, j1, k0, k1) for(int i = i0; i <= i1; ++i) \
											for(int j = j0; j <= j1; ++j) \
												for(int k = k0; k <= k1; ++k)
#endif

#ifndef RXFOR2
#define RXFOR2(i0, i1, j0, j1) for(int i = i0; i < i1; ++i) \
								   for(int j = j0; j < j1; ++j)
#endif

#ifndef RXFOR2E
#define RXFOR2E(i0, i1, j0, j1) for(int i = i0; i <= i1; ++i) \
								    for(int j = j0; j <= j1; ++j)
#endif




//-----------------------------------------------------------------------------
// いろいろな関数
//-----------------------------------------------------------------------------
namespace RXFunc 
{
	/*!
	 * 指定した数の半角スペースを返す
	 * @param[in] n 
	 * @return 半角スペース
	 */
	inline std::string GenSpace(int n)
	{
		std::string spc;
		for(int i = 0; i < n; ++i){
			spc += " ";
		}
		return spc;
	}

	/*!
	 * Vec3の要素が全て正->true
	 * @param 
	 * @return 
	 */
	inline bool IsPositive(const Vec3 &x)
	{
		if(x[0] < -RX_FEQ_EPS)	return false;
		if(x[1] < -RX_FEQ_EPS)	return false;
		if(x[2] < -RX_FEQ_EPS)	return false;

		return true;
	}

	/*!
	 * 2次元での点xがmin_x,max_yで囲まれた矩形内にあるかどうかの判別
	 * @param 
	 * @return 
	 */
	inline bool InRange(const Vec2 &x, const Vec2 &min_x, const Vec2 &max_x)
	{
		if(x[0] < min_x[0]) return false;
		if(x[1] < min_x[1]) return false;
		if(x[0] > max_x[0]) return false;
		if(x[1] > max_x[1]) return false;

		return true;
	}


	/*!
	 * std::vectorコンテナ同士の内積計算
	 */
	template<class T>
	inline T DotProduct(const std::vector<T> &a, const std::vector<T> &b)
	{
		T d = static_cast<T>(0);
		for(int i = 0; i < (int)a.size(); ++i){
			d += a[i]*b[i];
		}
		return d;
	}


	/*!
	 * 2次元std::vectorコンテナ同士の内積計算
	 */
	template<class T>
	inline T DotProduct(const std::vector< std::vector<T> > &a, const std::vector< std::vector<T> > &b)
	{
		T d = static_cast<T>(0);
		int nx = (int)a.size();
		int ny = (int)a[0].size();
		for(int i = 0; i < nx; ++i){
			for(int j = 0; j < ny; ++j){
				d += a[i][j]*b[i][j];
			}
		}
		return d;
	}

	/*!
	 * 
	 * @param 
	 * @return 
	 */
	inline double Min3(const Vec3 &x)
	{
		return ( ( (x[0]<x[1]) && (x[0]<x[2]) ) ? x[0] : ( (x[1]<x[2]) ? x[1] : x[2] ) );
	}

	/*!
	 * 
	 * @param 
	 * @return 
	 */
	inline double Max3(const Vec3 &x)
	{
		return ( ( (x[0]>x[1]) && (x[0]>x[2]) ) ? x[0] : ( (x[1]>x[2]) ? x[1] : x[2] ) );
	}

	/*!
	 * 2値を比較して大きい方を返す関数
	 * @param x 比較する2値を格納したVec2
	 * @return 大きい方の値
	 */
	inline double Max2(const Vec2 &x)
	{
		return ( (x[0] > x[1]) ? x[0] : x[1] );
	}

	/*!
	 * 2値を比較して大きい方を返す関数
	 * @param x 比較する2値を格納したVec2
	 * @return 大きい方の値
	 */
	inline double Min2(const Vec2 &x)
	{
		return ( (x[0] < x[1]) ? x[0] : x[1] );
	}
		
	/*!
	 * [0,1]の乱数の生成
	 */
	inline double Frand(void)
	{
		return (double)(rand()/(1.0+RAND_MAX));
	}

		/*!
	 * [0,n]の乱数の生成
	 */
	inline int Nrand(int n)
	{
		return (int)(Frand()*n);
	}

	/*!
	 * [-1,1]の乱数の生成
	 */
	inline double SignedRand(void)
	{
		return 2*Frand()-1;
	}

	/*!
	 * [0,255]の整数乱数の生成
	 */
	inline char ByteRand(void)
	{
		return (char)(rand() & 0xff);
	}

	/*!
	 * 指定した範囲の実数乱数の生成
	 */
	inline double Rand(const double &_max, const double &_min)
	{
		return (_max-_min)*Frand()+_min;
	}

	/*!
	 * 指定した範囲の実数乱数の生成(Vec2)
	 */
	inline Vec2 Rand(const Vec2 &_max, const Vec2 &_min)
	{
		return Vec2((_max[0]-_min[0])*Frand()+_min[0], (_max[1]-_min[1])*Frand()+_min[1]);
	}

	/*!
	 * 指定した範囲の実数乱数の生成(Vec3)
	 * @param 
	 * @return 
	 */
	inline Vec3 Rand(const Vec3 &_max, const Vec3 &_min)
	{
		return Vec3((_max[0]-_min[0])*Frand()+_min[0], (_max[1]-_min[1])*Frand()+_min[1], (_max[2]-_min[2])*Frand()+_min[2]);
	}

	/*!
	 * 正規分布の実数乱数の生成
	 * @param[in] m 平均
	 * @param[in] s 標準偏差
	 */
	inline double NormalRand(double m, double s)
	{
		double x1, x2, w;
		do{
			x1 = 2.0*Frand()-1.0;
			x2 = 2.0*Frand()-1.0;
			w = x1*x1 + x2*x2;
		} while(w >= 1.0 || w < 1E-30);

		w = sqrt((-2.0*log(w))/w);

		x1 *= w;
		return x1*s+m;
	}
	
	/*!
	 * 余り計算(double用)
	 * @param 
	 * @return 
	 */
	inline double Mod(double a, double b)
	{
		int n = (int)(a/b);
		a -= n*b;
		if(a < 0)
			a += b;
		return a;
	}


	/*!
	 * 符号チェックを含めた平行根の計算
	 * @param[in] x 平方根を計算したい値
	 * @return 平方根の計算結果．xが負なら0.0
	 */
	inline double Sqrt(const double &x)
	{
		return x > 0.0 ? sqrt(x) : 0.0;
	}


	/*!
	 * 絶対値(Vec3)
	 * @param a
	 * @return 
	 */
	inline Vec3 Fabs(const Vec3& a)
	{
		return Vec3(fabs(a[0]),fabs(a[1]),fabs(a[2]));
	}

	/*!
	 * 絶対値(Vec2)
	 * @param a
	 * @return 
	 */
	inline Vec2 Fabs(const Vec2& a)
	{
		return Vec2(fabs(a[0]),fabs(a[1]));
	}

	/*!
	 * Vec3の要素の中で絶対値の最大値を返す
	 * @param x
	 * @return 
	 */
	inline double Max3Abs(const Vec3 &x)
	{
		return Max3(Fabs(x));
	}

	/*!
	 * Vec3配列の要素の中で絶対値の最大値を返す
	 * @param vx
	 * @return 
	 */
	inline double Max3Abs(const std::vector<Vec3> &vx)
	{
		double t_value, max_value = 0;
		for(std::vector<Vec3>::size_type i = 0; i < vx.size(); ++i){
			if((t_value = Max3Abs(vx[i])) > max_value) max_value = t_value;
		}
		return max_value;
	}

	/*!
	 * クランプ(与えられた値が指定した範囲外ならば，その境界に切り詰める)
	 * @param x 値
	 * @param a,b 境界値
	 * @return クランプされた値
	 */
	inline Vec3 Clamp(const Vec3& x, const Vec3& a, const Vec3& b)
	{
		return Vec3(RX_CLAMP(x[0],a[0],b[0]), RX_CLAMP(x[1],a[1],b[1]), RX_CLAMP(x[2],a[2],b[2]));
	}
	
	/*!
	 * returns true if the std::vector has a very small norm
	 * @param[in] A ベクトル値
	 * @return 
	 */
	inline bool IsZeroVec(const Vec3& A)
	{
		return (RX_IS_ZERO(A[0]) && RX_IS_ZERO(A[1]) && RX_IS_ZERO(A[2]));
	}


	/*!
	 * returns true if the std::vector is not very small
	 * @param[in] A ベクトル値
	 * @return 
	 */
	inline bool NonZeroVec(const Vec3& A)
	{
		return (!RX_IS_ZERO(A[0]) || !RX_IS_ZERO(A[1]) || !RX_IS_ZERO(A[2]));
	}


	/*!
	 * Vec3の各要素の平均値
	 * @param[in] A ベクトル値
	 * @return ベクトル値の平均
	 */
	inline double AverageVec(Vec3 A)
	{
		return (A[0]+A[1]+A[2])/3.0;
	}

 
	/*!
	 * 青->緑->赤->白と変化するサーモグラフ用の色生成
	 * @param[out] col 生成された色
	 * @param[in] x 値
	 * @param[in] xmin 最小値
	 * @param[in] xmax 最大値
	 */
	inline void Thermograph(double col[3], double x, const double xmin = 0.0, const double xmax = 1.0)
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
	inline Vec3 Thermograph(double x, const double xmin = 0.0, const double xmax = 1.0)
	{
		Vec3 color;
		Thermograph(color.data, x, xmin, xmax);
		return color;
	}
 
	/*!
	 * グラデーション色生成
	 * @param[out] col 生成された色
	 * @param[in] col1 x=xminのときの色
	 * @param[in] col2 x=xmaxのときの色
	 * @param[in] x 値
	 * @param[in] xmin 最小値
	 * @param[in] xmax 最大値
	 */
	inline void Gradation(double col[3], const double col1[3], const double col2[3], 
						  double x, const double xmin = 0.0, const double xmax = 1.0)
	{
		double l = xmax-xmin;
		if(fabs(l) < 1e-10) return;
    
		double dx = RX_CLAMP(((x-xmin)/l), 0.0, 1.0);
		col[0] = RX_LERP(col1[0], col2[0], dx);
		col[1] = RX_LERP(col1[1], col2[1], dx);
		col[2] = RX_LERP(col1[2], col2[2], dx);
	}
};



#endif // #ifndef _RX_UTILITY_
