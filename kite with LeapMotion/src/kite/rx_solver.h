/*! 
 @file solver.h

 @brief Fluid solver
 
 @author Taichi Okamoto
 @date 2008
*/

#include <cmath>

#ifndef _SOLVER
#define _SOLVER

//-----------------------------------------------------------------------------------
//変数
//-----------------------------------------------------------------------------------
extern double *g_u;//!< 速度場のx方向成分
extern double *g_v;//!< 速度場のy方向成分
extern double *g_w;//!< 速度場のz方向成分
extern double *g_u_prev;//!< 1手順前の速度場のx方向成分
extern double *g_v_prev;//!< 1手順前の速度場のy方向成分
extern double *g_w_prev;//!< 1手順前の速度場のz方向成分

extern double *g_dens;//!< 密度場(現在未実装)
extern double *g_dens_prev;//!< 1手順前の密度場(現在未実装)
			   
extern double *g_p_prev;//!< 1手順前の圧力場
			   
			   
//-----------------------------------------------------------------------------------
// マクロ
//-----------------------------------------------------------------------------------
#define IX(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*(k))//!< 3次元の格納
#define SWAP(x0,x) {double * tmp=x0;x0=x;x=tmp;}//!< 配列の入れ替え

#define GRID 16			//!< 1次元毎のグリッド分割数
#define F_FORCE -1.0	//!< 外力
#define STEP 0.01		//!< タイムステップ幅
#define VISC 0.0007		//!< 動粘性係数
#define DIFF 0.0		//!< 拡散係数
using namespace std;

//---------------------------------------------------------------------------------------------------------------------
// 流体シミュレータ
//---------------------------------------------------------------------------------------------------------------------
namespace fluid
{
	static int X_wind=0;//!< x方向の外力源(demo)
	static int Z_wind=1;//!< z方向の外力源(demo)

	static int V_field=0;//!< 速度場の視覚化ON/OFF(demo)
	static int D_tex=0;
	static int frc_view=0;

	void free_data ( void );

	//! メモリ確保
	void allocate_data ( void );
	//! 速度場の初期化
	void clear_data ( void );

	//! 速度場
	void vel_step ( int n, double * u, double * v, double * w, double * u0, double * v0, double * w0, double visc, double dt );

	//! 外力項
	void add_source ( int n, double * x, double * s, double dt );
	//! 拡散項
	void diffuse ( int n, int b, double * x, double * x0, double diff, double dt );
	//! 移流項
	void advect ( int n, int b, double * d, double * d0, double * u, double * v,double * w, double dt );
	//! Gauss-Seidal反復法
	void lin_solve ( int n, int b, double * x, double * x0, double a, double c );
	//! 境界条件
	void set_bnd ( int n, int b, double * x );
	//! 質量保存
	void project( int n, double * u, double * v, double * w, double * p, double * div );

	//! ユーザ入力
	void get_from_UI ( int N, double * d, double * u, double * v, double * w );

	//! OpenGL描画
	void draw_box(double mag);
	void draw_velocity(double mag, double scale);
}


#endif