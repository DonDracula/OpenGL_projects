/*! 
 @file kite.h

 @brief Kite Simulation
 
 @date 2015/7
*/

#ifndef _KITE
#define _KITE


#include <vector>

#include "rx_utility.h"		// Vector classes
#include "rx_matrix.h"		// Matrix classes
#include "rx_quaternion.h"	// Quaternion classes

#include "rx_solver.h"

using namespace std;

//macros
#define COL_NUM 13				//column number of the kite
#define ROW_NUM 13				//row number of the kite

#define RHO 1.21				//density of the air(1.2kg/m^3)
#define G_ACCE -9.8				//gravity

#define KITE_RHO 0.2			//density of the kite 200g/m^2

//table
#define TABLE_S 0.68			//aspect
#define TABLE_L 1.48			//aspect

//the string of the kite
#define LINE_SP 20				//the number of the particle in the line 
#define LINE_K 10.0*((double)LINE_SP)	//spring constant
#define LINE_D 1.0				//a constant to be used for the inner friction of the spring
#define LINE_RHO 0.2			//density of the mass
#define LINE_E 0.02				//抵抗係数

//しっぽ関連
#define TAIL_NUM 2							//しっぽの数

#define TAIL_SP 10							//しっぽの分割数
#define TAIL_K 10.0*((double)TAIL_SP)		//ばね定数
#define TAIL_D 1.0							//減衰係数
#define TAIL_RHO 0.2						//質量密度
#define TAIL_E 0.02							//抵抗係数

//たわみ関連
#define BAR_SP (COL_NUM-1)					//骨の分割数
#define BAR_L 1.0							//骨の長さ
#define BAR_WIDTH 0.016						//骨断面の幅(正方形断面の場合,高さも兼ねる)
											//(およそ0.02mほどからはっきり変化がわかる)
#define INV_BAMBOO_E pow(10.0,-9.0)			//竹のヤング率の逆数


//---------------------------------------------------------------------------------------------------------------------
//凧シミュレータ
//---------------------------------------------------------------------------------------------------------------------
namespace kite3d
{
	//! structure "quadrangle"
	struct quadrangle
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

	};

	//! structure "kite_tail"
	struct kite_tail
	{
		int set_point;			//凧を構成する点群の中で尻尾と接続されている点番号
		double l;				//尻尾の長さ
		vector<Vec3> pos;		//質点位置
		vector<Vec3> pos_pre;	//1step前の質点位置
		vector<Vec3> vel;		//質点速度
		vector<Vec3> vel_pre;	//1step前の質点速度
		vector<Vec3> frc;		//質点に加わる合力

	};

	//! structure "kite_3d"
	//(すべての変数を利用しているとは限らない)
	struct kite_3d
	{
		//initialize--------------------------------
		double l_init;				//糸の初期長さ
		double l_now;				//現在の糸の長さ
		double l_check;				//端点間の距離
		vector<Vec3> line_pos;		//糸を構成する質点の位置
		vector<Vec3> line_pos_pre;	//糸を構成する質点の1step前の位置
		vector<Vec3> line_vel;		//糸を構成する質点の速度
		vector<Vec3> line_vel_pre;	//糸を構成する質点の1step前の速度
		vector<Vec3> line_frc;		//糸を構成する質点に加わる合力

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

	};


	//初期化
	void initialize_sim(void);		//凧パラメータの初期化
	void initialize_options(void);	//オプションの初期化
	void create_model_rec(void);		//モデルの作成
	void create_model_yak(void);		//モデルの作成
	void create_model_dia(void);		//モデルの作成

	//ステップを進める
	void step_simulation(double dt);

	//風
	void set_wind(double dt);

	//荷重とモーメントの計算
	void calc_loads(double dt);

	//ポジションの計算
	void calc_line_pos(double dt);//糸(糸の自由端が凧の位置に対応)
	void calc_tail_pos(int n,double dt);//しっぽ

	//描画関係
	void draw_kite(void);	//凧本体
	void draw_line(void);	//凧糸
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

	//ユーザインタフェース(ハプティックデバイス)による力
	Vec3 calc_UI_force(void);

	//たわみ関係
	void initialize_deflection(void);	//初期化
	void calc_deflection(double P);		//たわみの計算
	void keep_long_deflection(void);		//長さを保つ

	double calc_ex_Spring(double T,double dt);
}


#endif //_KITE 