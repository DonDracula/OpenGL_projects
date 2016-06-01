/*! 
 @file kite.h

 @brief Kite Simulation(connect kite shape and kite string)
*/

#ifndef _STRINGKITE
#define _STRINGKITE

#include <vector>
#include "rx_utility.h"		// Vector class
#include "rx_matrix.h"		// Matrix class
#include "rx_quaternion.h"	// Quaternion class
#include "sim_string.h"		//string simulation class
#include "sim_kiteShape.h"	//kite shape simulation class

#include "sim_fluid.h"		//fluid simulation class

#define KITE_NUMBER 6                 //number of the kite shape

using namespace std;
//---------------------------------------------------------------------------------------------------------------------
//凧シミュレータ
//---------------------------------------------------------------------------------------------------------------------

class StringKite3D{
private:
	StringSimulation kite_String;
	Kite3D  kite_Shape;
	Kite3D  kite_Shape2;
	
	Kite3D kite3d_shape[KITE_NUMBER];	//凧糸の分割数 = 40, number of the kite3d, from index 10~40

	Vec3 Wind;

	double sp_l;		//点間隔
	double Wind_vel;	//気流速度

	int string_mid_point;

public:
		//static Vec3 ppppp;
		//extern Vec3 *spring_ce;
		Vec3 spring_ce;
		void readKite();
		StringKite3D();
		//初期化
		void setup();					//初始化风筝

		//ユーザインタフェース(ハプティックデバイス)による力
		Vec3 calc_UI_force(void);

		//对外端口 ， 连风筝用
	//	void setStartPos(Vec3 pos) {kite_String.firstParticle = pos;}
	//	Vec3 getStringLastPos() { return kite_String.lastParticle;}
	//	Vec3 getKiteMidPos() { return kite_Shape.kite.glb_s_pos; }; 				//get bridle of kite
		//ステップを進める
		void update(double dt);

		//風
		void set_wind(double dt);

		//ポジションの計算
		void update_line(double dt);//糸(糸の自由端が凧の位置に対応)

		//描画関係
		void draw(void);

};

#endif