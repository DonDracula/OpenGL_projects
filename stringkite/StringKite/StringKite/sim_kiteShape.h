/*! 
 @file kite.h

 @brief Kite Simulation
*/

#ifndef _KITE
#define _KITE

#include <vector>
#include "rx_utility.h"		// Vector class
#include "rx_matrix.h"		// Matrix class
#include "rx_quaternion.h"	// Quaternion class
#include "sim_string.h"		//string simulation class

#include "sim_fluid.h"		//fluid simulation class

using namespace std;
//---------------------------------------------------------------------------------------------------------------------
//凧シミュレータ
//---------------------------------------------------------------------------------------------------------------------

class Kite3D{
public:
	//! structure "quadrangle"
	typedef  struct quadrangle
	{
		int index[4];			//四角形要素を構成する3点のインデックス
		double S;				//四角形要素の面積
		double mass;			//四角形要素の質量
		Vec3 cg;				//四角形要素の重心(ローカル系における座標)
		Vec3 normal;			//四角形要素の面法線
		Vec3 local_inertia_mom;	//四角形のローカル慣性モーメント
		Vec3 local_inertia_pro;	//四角形のローカル慣性モーメント

		//凧の設計
		double b;				//長方形の幅
		double c;				//長方形の高さ

	} quadrangle;

	//! structure "kite_tail"
	typedef struct kite_tail
	{
		int set_point;			//凧を構成する点群の中で尻尾と接続されている点番号
		double l;				//尻尾の長さ
		vector<Vec3> pos;		//質点位置
		vector<Vec3> pos_pre;	//1step前の質点位置
		vector<Vec3> vel;		//質点速度
		vector<Vec3> vel_pre;	//1step前の質点速度
		vector<Vec3> frc;		//質点に加わる合力

	} kite_tail;

	//! structure "kite_3d"
	//(すべての変数を利用しているとは限らない)
	typedef struct kite_3d
	{
		//model-------------------------------------
		//凧の設計
		double b;					//凧全体の最大幅
		double c;					//凧全体の最大高さ
		double AR;					//アスペクト比

		double mass;				//凧全体の質量
		double S;					//凧全体の面積

		int p_num;					//凧を構成する点の数
		//ローカル座標(値は一定，座標の起点は糸目中心)
		vector<Vec3> local_cd;		//凧を構成する点の座標(ローカル系)
		//グローバル座標(レンダリング等に用いる)
		vector<Vec3> global_cd;		//凧を構成する点の座標(グローバル系)

		vector<Vec3> design_cd;		//頂点座標(デザイン用座標)
		vector<Vec3> tex_cd;		//頂点座標(テクスチャ座標)

		vector<quadrangle> element;	//四角形要素
		int q_num;					//四角形要素数

		Vec3 s;						//糸目中心(最終的にはローカル座標で格納)
		Vec3 cg;					//重心座標(ローカル座標ではない->ローカル座標では原点にあたる)

		rxMatrix3 inertia;			//慣性テンソル
		rxMatrix3 inertia_inv;		//慣性テンソルの逆行列

		//sim---------------------------------------
		Vec3 pos;					//グローバル系における凧の重心位置
		Vec3 pos_pre;				//グローバル系における1step前の凧の重心位置

		Vec3 glb_s_pos;				//グローバル系における凧の糸目位置

		Vec3 omega;					//角速度

		rxQuaternion orientation;	//方向4元数

		Vec3 global_vel;			//グローバル系における移動速度
		Vec3 local_vel;				//ローカル系における凧が受ける速度

		Vec3 euler_angles;			//オイラー角(凧の姿勢)

		//loads-------------------------------------
		Vec3 frc;					//荷重
		Vec3 frc_pre;				//荷重
		Vec3 mom;					//モーメント
		Vec3 T_spring[2];			//ばねによる引張力

	} kite_3d;
	
private:
	//StringSimulation kiteString;

	kite_tail tail[2];//しっぽ情報格納

	//cd,cl data table 
	double CD_068_table[46];	//!< alpha_CD_068.dat格納用配列
	double CD_148_table[46];	//!< alpha_CD_148.dat格納用配列
	double CL_068_table[46];	//!< alpha_CL_068.dat格納用配列
	double CL_148_table[46];	//!< alpha_CL_148.dat格納用配列
	double x_table[71];		//alpha_x.dat格納用配列



	float tension_check;
	double ex_Sp_l;
	double ex_Sp_pos[2];
	double ex_Sp_vel[2];

//	Vec3 Wind;
	Vec3 kite_check;
	int v_point[4];

	double sp_l;		//点間隔
	double Wind_vel;	//気流速度
	double p_l[2];		//投影面積(長さ)


public:
		kite_3d kite;//凧情報格納
			//片方は力，他方は位置
	Vec3 Lift[2][2];
	Vec3 Drag[2][2];
	Vec3 T_tail[2][2];
	Vec3 T_string[2];
	Vec3 G[2];
		//static Vec3 ppppp;

		Kite3D();
		//初期化
		void setup(Vec3 pos);					//初始化风筝
		void initialize_sim(Vec3 pos);		//凧パラメータの初期化
		void initialize_options(void);	//オプションの初期化
		void create_model_rec(void);		//モデルの作成

		Vec3 getKiteMidPos() { return kite.glb_s_pos; }; 				//get bridle of kite
		void getLinked(double dt,Vec3 pos);

		void update1(double dt);
		void update2(double dt,Vec3 wind);
		//風
	//	void set_wind(double dt);

		//荷重とモーメントの計算
		void calc_loads(double dt);


		void calc_tail_pos(int n,double dt,Vec3 wind);//しっぽ

		//描画関係
		void draw(void);
		void draw_kite(void);	//凧本体
		//void draw_kite2(void);	//凧本体

		void draw_tail(void);	//しっぽ
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



		//たわみ関係
		void initialize_deflection(void);	//初期化
		void calc_deflection(double P);		//たわみの計算
		void keep_long_deflection(void);		//長さを保つ

		//tool
 
double	Heron(Vec3 A, Vec3 B, Vec3 C)
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
 
double	DotProductV(double a[], double b[], int num)
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
 
int	cgSolverMatrix(int num, double A[], 
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
};

#endif