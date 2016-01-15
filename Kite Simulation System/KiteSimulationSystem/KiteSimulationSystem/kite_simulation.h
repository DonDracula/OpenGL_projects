/*
	a class to make a simple kite simulation system
	date: 2015
*/

#ifndef KITESIMULATION_H
#define KITESIMULATION_H

#include <vector>

#include "rx_utility.h"			//Vector classes
#include "rx_matrix.h"			//Matrix classes
#include "rx_quaternion.h"		//Quaternion classes

#include "string_simulation.h"
#include "fluid_solver.h"
using namespace std;


#define DAMP_yaw 0.2
#define DAMP 0.0
//MACRO
#define COL_NUM 13							//凧の横方向分割点数->(4の倍数+1)
#define ROW_NUM 13							//凧の縦方向分割点数

#define RHO 1.21							//air density(1.2kg/m^3)
#define G_ACCE -9.8							//gravity

#define KITE_RHO 0.2						//凧の密度? 基本的には扱いやすいよう200g/1m^2
//テーブル関係
#define TABLE_S 0.68						//アスペクト比
#define TABLE_L 1.48						//アスペクト比

//deflection
#define BAR_SP (COL_NUM-1)					//骨の分割数
#define BAR_L 1.0							//骨の長さ
#define BAR_WIDTH 0.016						//骨断面の幅(正方形断面の場合,高さも兼ねる)
											//(およそ0.02mほどからはっきり変化がわかる)
#define INV_BAMBOO_E pow(10.0,-9.0)			//竹のヤング率の逆数
//quadrangle shape
class KiteSimulation
{
	struct quadrangle
	{
		int index[4];				//index of the 4 point of the quadrangle
		double S;					//surface area
		double mass;				//mass of the quadrangle
		Vec3 cg;					//center of gravity
		Vec3 normal;				//normal of the kite
		Vec3 local_inertia_mom;		//local moment of inertia of the quadrangle
		Vec3 local_inertia_pro;		//local moment of inertia of the quadrangle

		//kite design
		double b;					//width of the quadrangle
		double c;					//height of the quadrangle
	};
protected:

	double l_init;					//init length of string
	//Vec3 line_startPos;
	//Vec3 line_endPos;
	//kite model
	//kite design
	double k_b;						//width of the kite
	double k_c;						//height of the kite
	double k_AR;						//aspect ratio

	//double per_line_mass;
	double k_mass;					//mass of kite
	double k_S;						//surface area of kite
	Vec3 Wind;
	//Vec3 wind_vel;
	int k_p_num;						//凧を構成する点の数
	//ローカル座標(値は一定，座標の起点は糸目中心)
	vector<Vec3> k_local_cd;		//凧を構成する点の座標(ローカル系)
	//グローバル座標(レンダリング等に用いる)
	vector<Vec3> k_global_cd;		//凧を構成する点の座標(グローバル系)

	vector<Vec3> k_design_cd;			//poins' coordinates(design) 
	vector<Vec3> k_tex_cd;			//points' coordinates(texture)

	vector<quadrangle> element;		//quadrangle element
	int k_q_num;						//number of the quadrangle element

	Vec3 k_s;						//糸目中心(最終的にはローカル座標で格納)
	Vec3 k_cg;					//重心座標(ローカル座標ではない->ローカル座標では原点にあたる)

	rxMatrix3 inertia;			//inertia
	rxMatrix3 inertia_inv;		//inverse of the inertia

	//sim
	Vec3 k_pos;					//グローバル系における凧の重心位置
	Vec3 k_pos_pre;				//グローバル系における1step前の凧の重心位置

	Vec3 k_glb_s_pos;				//グローバル系における凧の糸目位置

	Vec3 k_omega;					//角速度

	rxQuaternion k_orientation;	//方向4元数

	Vec3 k_global_vel;			//グローバル系における移動速度
	Vec3 k_local_vel;				//ローカル系における凧が受ける速度

	Vec3 k_euler_angles;			//オイラー角(凧の姿勢)

	//loads-------------------------------------
	Vec3 k_frc;					//荷重
	Vec3 k_frc_pre;				//荷重
	Vec3 k_mom;					//モーメント
	Vec3 k_T_spring[2];			//ばねによる引張力

public:
	//init
	void initKite(void);		//init the para of kite
	void create_model_rec(void);		//create the model
	void create_model_yak(void);		
	void create_model_dia(void);		
	//void setPerVexMass(double m)	{this->per_line_mass = m;}
	//void setLineStartPoint(Vec3 startpoint) { this->line_startPos = startpoint; }

	//step update
	void step_simulation(double dt);

	//set the length of the string
	void set_length(double length){		l_init = length;	}
	double get_length()	{	return this->l_init;	}
	//set the wind
	void set_wind(double dt);
	//Vec3 get_wind() {return this->wind_vel;}

	//荷重とモーメントの計算
	void calc_loads(double dt);

	//calc the position
	void calc_line_pos(double dt);//糸(糸の自由端が凧の位置に対応)

	//draw
	void draw_kite(void);	//draw the kite

	void draw_options_01(void);	//プログラム確認用オプション
	void draw_options_02(void);	//プログラム確認用オプション
	void draw_options_03(void);	//プログラム確認用オプション

	//ファイル読み込み
	int  read_file(const string &file_name, vector<double> &datas);
	void read_file(const string &file_name);
	void read_table(void);					//.datからテーブルを読み込む

	//テーブルの参照
	double search_alpha_CD(double alpha,double AR);
	double search_alpha_CL(double alpha,double AR);
	double search_alpha_x(double alpha);		//α-xテーブル(迎え角から風心を求める)

	//レンダリング用座標取得
	void calc_render_pos(void);

	//calc the force from user
	Vec3 calc_UI_force(void);

	//deflection
	void initialize_deflection(void);	//init the deflection
	void calc_deflection(double P);		//calculate the deflection 
	void keep_long_deflection(void);		//長さを保つ

	//double calc_ex_Spring(double T,double dt);
};

/*!
 * Heronの公式
 * @note		3頂点の座標から三角形面積を求める
 * @param[in]	A 三角形頂点(Vec3型)
 * @param[in]	B 三角形頂点(Vec3型)
 * @param[in]	C 三角形頂点(Vec3型)
 * @return		三角形面積を返す(double型)
 */
inline double 
Heron(Vec3 A, Vec3 B, Vec3 C)
{
	Vec3 BA,CB,AC;	//辺
	double a,b,c;	//三辺の長さ
	double s,r;
	double TS;		//三角形の面積

	BA=B-A;
	CB=C-B;
	AC=A-C;
	//辺の長さ
	a=norm(BA);	//AとBの辺の長さ
	b=norm(CB);	//BとCの辺の長さ
	c=norm(AC);	//CとAの辺の長さ

	s=0.5*(a+b+c);
	r=s*(s-a)*(s-b)*(s-c);	//平方根の中身
	TS=sqrt(r);				//面積

	return TS;
}

/*!
 * vectorコンテナ同士の内積計算
 * @param 
 * @return 
 */
inline double 
DotProductV(double a[], double b[], int num)
{
	double d = 0.0;
	for(int i = 0; i < num; ++i){
		d += a[i]*b[i];
	}
	return d;
}
/*!
 * 共役勾配(CG)ソルバ
 * @param[in] func y = APを計算する関数オブジェクト
 * @param[in] b 右辺項
 * @param[out] x 解を格納する
 * @param[inout] max_iter 最大反復回数
 * @param[inout] tol 許容誤差
 */
inline int 
cgSolverMatrix(int num, double A[], 
					      double b[], double x[], int &max_iter, double &tol)
{
	int k, idx;
	int size = num;
	double tol2 = tol*tol;

#define LARGE_NUM 100 //十分な大きさを確保用

	double r[LARGE_NUM]={0},p[LARGE_NUM]={0},y[LARGE_NUM]={0};

	// 第0近似解に対する残差の計算
	for(idx = 0; idx < size ; ++idx){
		x[idx] = 0.0;
		r[idx] = b[idx];
		p[idx] = b[idx];
	}

	double resid;
	double norm2_b = DotProductV(b, b, num);
	if(fabs(norm2_b) < RX_FEQ_EPS) norm2_b = 1.0;
		
	double rr0 =DotProductV(r, r, num), rr1;
	if((resid = rr0/norm2_b) <= tol2){
		tol = sqrt(resid);
		max_iter = 0;

		return 0;
	}

	double alpha0, beta0;
	for(k = 0; k < max_iter; k++){
		// y = Ap の計算
		for(int i = 0; i < size; ++i){
			y[i] = 0.0;
			for(int j = 0; j < size; j++){
				y[i] += A[num*i+j]*p[j];
			}
		}

		// alpha = r*r/(P*AP)の計算
		alpha0 = rr0/DotProductV(p, y, num);

		// 解x、残差rの更新
		for(idx = 0; idx < size ; ++idx){
			x[idx] += alpha0*p[idx];
			r[idx] -= alpha0*y[idx];
		}

		// r_(k+1)*r_(k+1)
		rr1 = DotProductV(r, r, num);

		if((resid = rr1/norm2_b) <= tol2){
			tol = sqrt(resid);
			max_iter = k+1;

			return 0;     
		}

		beta0 = rr1/rr0;
		for(idx = 0; idx < size ; ++idx){
			p[idx] = r[idx]+beta0*p[idx];
		}

		rr0 = rr1;
	}

	tol = resid;

	return k;
}

#endif //KITESIMULATION_H