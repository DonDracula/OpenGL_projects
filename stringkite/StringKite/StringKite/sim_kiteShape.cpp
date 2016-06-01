
// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>
//kite头文件
#include "sim_kiteShape.h"
#include "macros4cc.h"

//数値解析(行列,ベクトル,スカラ)
double	L_C_strains[LINE_SP];				//式(1)(辺のひずみに相当)ベクトル(式(5)のBベクトル)(m*1)
double	L_C_grad[LINE_SP][3*(LINE_SP+1)];	//式(2)(辺の傾きに相当)行列(m*3n)
double	L_IM_Cg[3*(LINE_SP+1)][LINE_SP];	//Inv_M×C_gradの転置行列(3n*m)
double	L_Cg_IM_Cg[LINE_SP*LINE_SP];		//式(5)(Ax=B)のA行列(m*m)
double	L_Delta_lambda[LINE_SP];			//δλベクトル(m*1)
double	L_Delta_x[3*(LINE_SP+1)];			//答えを受け取るδxベクトル(3n*1)

double	T_C_strains[TAIL_SP];				//式(1)(辺のひずみに相当)ベクトル(式(5)のBベクトル)(m*1)
double	T_C_grad[TAIL_SP][3*(TAIL_SP+1)];	//式(2)(辺の傾きに相当)行列(m*3n)
double	T_IM_Cg[3*(TAIL_SP+1)][TAIL_SP];	//Inv_M×C_gradの転置行列(3n*m)
double	T_Cg_IM_Cg[TAIL_SP*TAIL_SP];		//式(5)(Ax=B)のA行列(m*m)
double	T_Delta_lambda[TAIL_SP];			//δλベクトル(m*1)
double	T_Delta_x[3*(TAIL_SP+1)];			//答えを受け取るδxベクトル(3n*1)

//deflection
Vec2 point[(BAR_SP+1)];

//---------------------------------------------------------------------------------
//凧シミュレータ
//---------------------------------------------------------------------------------

Kite3D::Kite3D(){
}

//初始化风筝
void Kite3D::setup(Vec3 pos)
{
	initialize_options();		//オプションの初期化
	initialize_sim(pos);			//凧パラメータの初期化
	create_model_rec();			//長方形モデルの作成
	//たわみの初期化
	initialize_deflection();
}


/*!
 * @note 凧パラメータの初期化
 */
 
void Kite3D::initialize_sim(Vec3 pos)
{
//	originPos = Vec3();
	kite_check=Vec3();
	tension_check=0.0f;

	ex_Sp_l=1.0;
	ex_Sp_pos[0]=0.0;
	ex_Sp_pos[1]=ex_Sp_l;
	ex_Sp_vel[0]=0.0;
	ex_Sp_vel[1]=0.0;

//凧糸，凧のしっぽに関しては
	
	////风筝线初始化
	//kiteString.setup();
	//しっぽの長さ
	tail[0].l=0.8;	
	tail[1].l=0.8;
	
	//	kiteString.setBeginPos(Vec3(-10,0,0));
	//凧本体のパラメータの初期化
	kite.pos=pos;	//初期位置等于风筝线末端
	kite.pos_pre=kite.pos;				//1step前の位置

	kite.global_vel=Vec3();				//グローバル系における速度
	kite.local_vel=Vec3();				//ローカル系における速度

	kite.frc=Vec3();			//荷重
	kite.frc_pre=kite.frc;		//荷重
	kite.T_spring[0]=Vec3();	//ばねによる引張力
	kite.T_spring[1]=Vec3();	//ばねによる引張力

	//凧の回転に関するパラメータの初期化
	kite.omega=Vec3();		//角速度

	//方向4元数の初期化
	//初期姿勢(角度)
	double roll=0.0;
	double pitch=60.0;
	double yow=0.0;
	double rdrd=RX_DEGREES_TO_RADIANS * RX_DEGREES_TO_RADIANS;//degreeからradに変換するため
	Vec3 dummy=Vec3(roll*rdrd,pitch*rdrd,yow*rdrd);//(roll,pitch,yow)

	kite.orientation.SetEulerAngles(dummy);//初期姿勢の格納

	for(int i=0;i<2;i++)
	{
		Lift[0][i]=Vec3();
		Lift[1][i]=Vec3();
		Drag[0][i]=Vec3();
		Drag[1][i]=Vec3();
		T_tail[0][i]=Vec3();
		T_tail[1][i]=Vec3();
		T_string[i]=Vec3();
		G[i]=Vec3();
	}
}

/*!c
 * @note オプション(凧糸・尻尾)の分割数分の要素確保
 */
 
void Kite3D::initialize_options(void)
{
	int i=0,j=0;//counter

	//尻尾の分割数分ループ
	for(i=0;i<=TAIL_SP;i++)
	{
		//尻尾の本数分ループ
		for(j=0;j<TAIL_NUM;j++)
		{
			tail[j].pos.push_back(Vec3());
			tail[j].pos_pre.push_back(Vec3());
			tail[j].vel.push_back(Vec3());
			tail[j].vel_pre.push_back(Vec3());
			tail[j].frc.push_back(Vec3());
		}
	}
}

/*!
 * @note たわみの初期化
 */
 
void Kite3D::initialize_deflection(void)
{
	int i=0;

	for(i=0;i<=BAR_SP;i++)
	{
		point[i]=Vec2();//0で初期化
		point[i].data[0]=((double)i)*kite.b/((double)BAR_SP);
	}

	sp_l=kite.b/((double)BAR_SP);//点間隔
	Wind_vel=0.0;//気流速度の初期化

	//投影面積(長さ)
	p_l[0]=point[BAR_SP].data[0]-point[0].data[0];
	//最大たわみ
	p_l[1]=0.0;
}

/*!
 * @note たわみの計算
 * @param[in] P 凧に加わる荷重
*/
 
void Kite3D::calc_deflection(double P)
{
	p_l[1]=0.0;//最大たわみの初期化

	int i=0,j=0;

	//たわみ計算
	//位置が絡まない部分の計算
	double coef[2]={0.0,0.0};
	coef[0]=0.25*P;
	coef[1]=1.0/pow(BAR_WIDTH,4.0);
	coef[1]*=INV_BAMBOO_E;

	//位置が絡む部分の計算
	for(i=0;i<=BAR_SP;i++)
	{
		j=i;
		if((BAR_SP/2)<i)
		{
			j=BAR_SP-i;
		}

		double var=0.0;
		var=3.0*kite.b*kite.b-4.0*point[j].data[0]*point[j].data[0];
		var*=point[j].data[0];

		point[i].data[1]=var*coef[0]*coef[1];

		//最大たわみのチェック
		if(point[i].data[1]>p_l[1])
		{
			p_l[1]=point[i].data[1];
		}
	}

	p_l[1]=fabs(p_l[1]);

	keep_long_deflection();
}

/*!
 * @note たわみが生じた際に棒の長さを保つ
 */ 
void Kite3D::keep_long_deflection(void)
{
	int i=0;

	double l_2=sp_l*sp_l;//点間隔の2乗
	double y=0.0,y_2=0.0;
	double x=0.0,x_2=0.0;

	for(i=0;i<BAR_SP;i++)
	{
		//距離のy方向成分を求める
		y=point[i].data[1]-point[i+1].data[1];
		y=fabs(y);
		//距離のy方向成分の2乗
		y_2=y*y;

		//可変部分(x方向成分)を求める
		x_2=l_2-y_2;	//(斜辺^2-y方向成分^2)
		x=sqrt(x_2);
		x=fabs(x);		//念のため絶対値計算により正にする
		
		point[i+1].data[0]=point[i].data[0]+x;
	}

	p_l[0]=point[BAR_SP].data[0]-point[0].data[0];//投影面積(長さ)
}

/*!
 * @note 凧のデザインからモデル作成
 */
 
void Kite3D::create_model_rec(void)
{
	//最終的には入力されたデザインから
	//点群と三角形ポリゴンを作成

	int i=0,j=0;//カウンタ

//入力---------------------------------
	//double b=0.85;//最大横幅
	//double c=0.8;//最大縦幅
	double b=0.8;//最大横幅
	double c=1.2;//最大縦幅

	vector<Vec3> point;	//点群
	int p_num=0;		//点群数

	vector<Vec3> tex_p;	//テクスチャ座標

	for(i=0;i<ROW_NUM;i++)//縦方向(x方向)
	{
		for(j=0;j<COL_NUM;j++)//横方向(y方向)
		{
			point.push_back(Vec3(((double)i)*c/((double)(ROW_NUM-1)),((double)j)*b/((double)(COL_NUM-1)),0.0 ));
			p_num++;//登録点数加算
			tex_p.push_back(Vec3( -((double)i)/((double)(ROW_NUM-1)),-((double)j)/((double)(COL_NUM-1)),0.0));
		}
	}
	v_point[0]=0;
	v_point[1]=COL_NUM-1;
	v_point[2]=COL_NUM*ROW_NUM-1;
	v_point[3]=COL_NUM*(ROW_NUM-1);

	vector<int> q_index[4];	//四角形を構成する4点のインデックス
	int q_num=0;			//四角形の数

	for(i=0;i<(ROW_NUM-1);i++)
	{
		for(j=0;j<(COL_NUM-1);j++)
		{
			//四角形を格納していく
			q_index[0].push_back(i*COL_NUM+j);
			q_index[1].push_back(i*COL_NUM+(j+1));
			q_index[2].push_back((i+1)*COL_NUM+(j+1));
			q_index[3].push_back((i+1)*COL_NUM+j);
			q_num++;
		}
	}
	//------------------------------------------------------------------------------------------------------------------------------------------------------15gai
	Vec3 itome=Vec3(0.3*c,0.5*b,0.0);//糸目中心
	//Vec3 itome = Vec3(0.5*b,0.3*c,0.0);

	//入力から計算用モデル作成
	kite.b=b;
	kite.c=c;
	kite.AR=b/c;

	kite.p_num=p_num;	//頂点数格納
	kite.q_num=q_num;	//四角形要素数格納

	kite.s=itome;		//糸目中心取り込み

	for(i=0;i<p_num;i++)
	{
		kite.tex_cd.push_back(tex_p[i]);

		kite.local_cd.push_back(point[i]);	//ローカル座標は後で重心基準になる

		kite.design_cd.push_back(point[i]-kite.s);//デザイン座標は糸目中心を基準にしている
		kite.global_cd.push_back(QVRotate(kite.orientation,kite.design_cd[i]));	//グローバル座標
		kite.global_cd[i]+=kite.pos;
	}
	
	kite.S=0.0; kite.mass=0.0;	//初期化
	kite.element.resize(q_num);	//四角形要素数分element確保
	for(i=0;i<q_num;i++)
	{
		kite.element[i].b=kite.b;
		kite.element[i].c=kite.c;

		//四角形を構成する4点のインデックス
		kite.element[i].index[0]=(int)q_index[0][i];
		kite.element[i].index[1]=(int)q_index[1][i];
		kite.element[i].index[2]=(int)q_index[2][i];
		kite.element[i].index[3]=(int)q_index[3][i];

		//ヘロンの公式より四角形の面積を計算
		kite.element[i].S=Heron(kite.local_cd[kite.element[i].index[0]],
							kite.local_cd[kite.element[i].index[1]],
							kite.local_cd[kite.element[i].index[2]])
						+ Heron(kite.local_cd[kite.element[i].index[0]],
							kite.local_cd[kite.element[i].index[2]],
							kite.local_cd[kite.element[i].index[3]]);
		//wxLogMessage(_T("S=%lf"),kite.element[i].S);

		//四角形の質量(面積*密度)
		kite.element[i].mass=kite.element[i].S*KITE_RHO;

		Vec3 v1,v2;
		double dummy;
		//四角形の面法線
		v1=kite.local_cd[kite.element[i].index[1]]-kite.local_cd[kite.element[i].index[0]];
		v2=kite.local_cd[kite.element[i].index[2]]-kite.local_cd[kite.element[i].index[1]];
		kite.element[i].normal=cross(v1,v2);	//外積
		dummy=unitize(kite.element[i].normal);	//正規化

		//四角形の重心
		kite.element[i].cg=((kite.local_cd[kite.element[i].index[0]]+
							kite.local_cd[kite.element[i].index[1]]+
							kite.local_cd[kite.element[i].index[2]]+
							kite.local_cd[kite.element[i].index[3]]))*0.25;
		
		//四角形のローカル慣性テンソル
		double Ixx=0.0,Iyy=0.0,Izz=0.0;//慣性モーメント項
		double Ixy=0.0,Ixz=0.0,Iyz=0.0;//慣性乗積

		Vec3 dif;
		//四角形要素の底辺長さbを求める
		dif=kite.local_cd[kite.element[i].index[0]]-kite.local_cd[kite.element[i].index[1]];
		b=norm(dif);
		//四角形要素の高さcを求める
		dif=kite.local_cd[kite.element[i].index[0]]-kite.local_cd[kite.element[i].index[3]];
		c=norm(dif);

		//慣性モーメント
		//(1/12)*rho*b*c*c*c etc...
		Ixx=0.0833333*KITE_RHO*b*c*c*c;
		Iyy=0.0833333*KITE_RHO*b*b*b*c;
		Izz=0.0833333*KITE_RHO*b*c*(b*b+c*c);

		kite.element[i].local_inertia_mom.data[0]=Ixx;
		kite.element[i].local_inertia_mom.data[1]=Iyy;
		kite.element[i].local_inertia_mom.data[2]=Izz;

		//慣性乗積
		//(1/16)*rho*b*b*c*c etc...
		Ixy=-0.0625*KITE_RHO*b*b*c*c;
		Ixz=0.0;
		Iyz=0.0;
		
		kite.element[i].local_inertia_pro.data[0]=Ixy;
		kite.element[i].local_inertia_pro.data[1]=Ixz;
		kite.element[i].local_inertia_pro.data[2]=Iyz;

		//凧全体の面積･質量を計算
		kite.S+=kite.element[i].S;
		kite.mass+=kite.element[i].mass;
	}

	//凧全体の重心を計算
	Vec3 mom=Vec3();
	for(i=0;i<q_num;i++)
	{
		mom+=kite.element[i].mass*kite.element[i].cg;
	}
	double kite_mass_inv=1.0/kite.mass;
	kite.cg=mom*kite_mass_inv;//重心

	//重心基準の座標へと変換
	for(i=0;i<q_num;i++)
	{
		kite.element[i].cg-=kite.cg;
	}
	kite.s-=kite.cg;//糸目中心も
	//ローカル座標も
	for(i=0;i<p_num;i++)
	{
		kite.local_cd[i]-=kite.cg;
	}

	//慣性テンソル作成
	double Ixx,Iyy,Izz,Ixy,Ixz,Iyz;
	Ixx=0.0;Iyy=0.0;Izz=0.0;//慣性モーメント項
	Ixy=0.0;Ixz=0.0;Iyz=0.0;//慣性乗積
	for(i=0;i<q_num;i++)
	{
		Ixx+=kite.element[i].local_inertia_mom.data[0]+
			kite.element[i].mass*
			(kite.element[i].cg.data[1]*kite.element[i].cg.data[1]+
			kite.element[i].cg.data[2]*kite.element[i].cg.data[2]);
		Iyy+=kite.element[i].local_inertia_mom.data[1]+
			kite.element[i].mass*
			(kite.element[i].cg.data[0]*kite.element[i].cg.data[0]+
			kite.element[i].cg.data[2]*kite.element[i].cg.data[2]);
		Izz+=kite.element[i].local_inertia_mom.data[2]+
			kite.element[i].mass*
			(kite.element[i].cg.data[1]*kite.element[i].cg.data[1]+
			kite.element[i].cg.data[0]*kite.element[i].cg.data[0]);

		Ixy+=kite.element[i].local_inertia_pro.data[0]+
			kite.element[i].mass*
			(kite.element[i].cg.data[0]*kite.element[i].cg.data[1]);
		Ixz+=kite.element[i].local_inertia_pro.data[1]+
			kite.element[i].mass*
			(kite.element[i].cg.data[0]*kite.element[i].cg.data[2]);
		Iyz+=kite.element[i].local_inertia_pro.data[2]+
			kite.element[i].mass*
			(kite.element[i].cg.data[2]*kite.element[i].cg.data[1]);
	}

	kite.inertia.SetValue(Ixx,-Ixy,-Ixz,-Ixy,Iyy,-Iyz,-Ixz,-Iyz,Izz);//慣性テンソルに格納
	kite.inertia_inv=kite.inertia.Inverse();//逆行列

	kite.glb_s_pos=QVRotate(kite.orientation,kite.s)+kite.pos;


//しっぽのセット--------------------------------------------------

	//しっぽの接続端点位置セット
	tail[0].set_point=(COL_NUM*(ROW_NUM-1)+1)+1;
	tail[0].pos[0]=kite.global_cd[tail[0].set_point];
	tail[1].set_point=COL_NUM*ROW_NUM-3;
	tail[1].pos[0]=kite.global_cd[tail[1].set_point];

	for(i=0;i<=TAIL_SP;i++)//要素数分ループ
	{
		for(j=0;j<TAIL_NUM;j++)//しっぽ数分ループ
		{
			tail[j].pos[i]=tail[j].pos[0]-Vec3(0.0,0.0,tail[j].l*((double)i)/((double)TAIL_SP));
			tail[j].pos_pre[i]=tail[j].pos[i];
		}
	}
//----------------------------------------------------------------
}
//从数据库查看角度对应的cd系数
/*!
 * @note α-CDテーブル(迎え角から抗力係数を求める)
 * @param[in] alpha
 * @return CD
 */

 
double Kite3D::search_alpha_CD(double alpha,double AR)
{
	double CD[3];
	double terminal;
	int index=(int)alpha;

	if(0>index)//負でも一応計算できるように
	{
		CD[0]=CD_068_table[0];
		CD[1]=CD_148_table[0];
	}
	else if(45<=index)//限界越え
	{
		CD[0]=CD_068_table[45]*(1.0-cos(2.0*alpha*RX_DEGREES_TO_RADIANS));
		CD[1]=CD_148_table[45]*(1.0-cos(2.0*alpha*RX_DEGREES_TO_RADIANS));
	}
	else
	{
		terminal=alpha-(double)index;//補間用
		CD[0]=terminal*CD_068_table[index+1]+(1.0-terminal)*CD_068_table[index];//線形補間
		CD[1]=terminal*CD_148_table[index+1]+(1.0-terminal)*CD_148_table[index];//線形補間
	}

	terminal=(AR-TABLE_S)/(TABLE_L-TABLE_S);
	CD[2]=terminal*CD[0]+(1.0-terminal)*CD[1];//線形補間

	return CD[2];
}
//从数据库查看角度对应的扬力系数cl
/*!
 * @note α-CLテーブル(迎え角から揚力係数を求める)
 * @param[in] alpha
 * @return CL
 */
 
double Kite3D::search_alpha_CL(double alpha,double AR)
{
	double CL[3];
	double terminal;
	int index=(int)alpha;

	if(0>index)//負でも一応計算できるように
	{
		CL[0]=CL_068_table[0];
		CL[1]=CL_148_table[0];
	}
	else if(90<=index)//限界越え
	{
		CL[0]=CL_068_table[0];
		CL[1]=CL_148_table[0];
	}
	else if(45<=index&&90>index)//限界越え
	{
		CL[0]=CL_068_table[45]*sin(2.0*alpha*RX_DEGREES_TO_RADIANS);
		CL[1]=CL_148_table[45]*sin(2.0*alpha*RX_DEGREES_TO_RADIANS);
	}
	else
	{
		terminal=alpha-(double)index;//補間用
		CL[0]=terminal*CL_068_table[index+1]+(1.0-terminal)*CL_068_table[index];//線形補間
		CL[1]=terminal*CL_148_table[index+1]+(1.0-terminal)*CL_148_table[index];//線形補間
	}

	terminal=(AR-TABLE_S)/(TABLE_L-TABLE_S);
	CL[2]=terminal*CL[0]+(1.0-terminal)*CL[1];//線形補間

	return CL[2];
}
//从数据库查看角度对应风心距离x
/*!
 * @note α-xテーブル(迎え角から風心を求める)
 * @param[in] alpha
 * @return x
 */
 
double Kite3D::search_alpha_x(double alpha)
{
	double x_coefficient;
	double terminal;
	int index=((int)alpha)-20;

	if(0>index)//α-x曲線の直線部分
	{
		x_coefficient=x_table[0];
	}
	else if(70<=index)//限界超え
	{
		x_coefficient=x_table[70];
	}
	else
	{
		terminal=alpha-(double)(index+20);//補間用
		if(0.01>terminal)
		{
			terminal=0.0;
		}
		else if(1.0<(terminal+0.01))
		{
			terminal=1.0;
		}
		x_coefficient=terminal*x_table[index+1]+(1.0-terminal)*x_table[index];//線形補間
	}

	return x_coefficient;
}


/*!
 * @note レンダリング用座標取得
 */
 
void Kite3D::calc_render_pos(void)
{
	int i=0,j=0;

	//たわみ
	for(i=0;i<ROW_NUM;i++)
	{
		for(j=0;j<COL_NUM;j++)
		{
			kite.design_cd[(i*COL_NUM)+j].data[2]=p_l[1]-point[j].data[1];
		}
	}

	for(i=0;i<kite.p_num;i++)
	{
		kite.global_cd[i]=QVRotate(kite.orientation,kite.design_cd[i]);	//回転
		kite.global_cd[i]+=kite.glb_s_pos;								//平行移動
	}

}

void Kite3D::getLinked(double dt,Vec3 pos)
{
		double Inv_dt=1.0/dt;
	//糸目位置とのリンク
	kite.glb_s_pos = pos;
	kite.pos=kite.glb_s_pos-QVRotate(kite.orientation,kite.s);
	kite.global_vel=(kite.pos-kite.pos_pre)*Inv_dt;
	kite.pos_pre=kite.pos;
}

void Kite3D::update1(double dt)
{
		int i=0;
//------------------------
	for(i=0;i<2;i++)
	{
		Lift[0][i]=Vec3();
		Lift[1][i]=Vec3();
		Drag[0][i]=Vec3();
		Drag[1][i]=Vec3();
		T_tail[0][i]=Vec3();
		T_tail[1][i]=Vec3();
	}

		//荷重とモーメントの計算，计算扭力
	calc_loads(dt);
	//並進処理
	kite.pos+=kite.global_vel*dt+0.5*kite.frc*dt*dt/kite.mass;	//位置の更新
	kite.global_vel+=kite.frc*dt/kite.mass;						//速度の更新
//-----------------------------------------------
#define DAMP_yaw 0.2
#define DAMP 0.0

	//回転の処理
	//ローカル系における角速度を計算
	Vec3 I_omega=kite.inertia*kite.omega;
	kite.omega+=kite.inertia_inv*(kite.mom-cross(kite.omega,I_omega)-Vec3(DAMP*kite.omega.data[0],DAMP*kite.omega.data[1],DAMP_yaw*kite.omega.data[2]))*dt;
	//新しい4元数を計算
	kite.orientation+=(kite.orientation*kite.omega)*(0.5*dt);
	//方向4元数の正規化
	kite.orientation.normalize();

//-----------------------------------------------*/

	kite.glb_s_pos=QVRotate(kite.orientation,kite.s)+kite.pos;	//グローバル系の糸目位置更新
}

void Kite3D::update2(double dt,Vec3 wind)
{
	int i =0;
	kite.glb_s_pos=QVRotate(kite.orientation,kite.s)+kite.pos;	//グローバル系の糸目位置更新

		//update_line(dt);
	//しっぽの挙動
	for(i=0;i<TAIL_NUM;i++)
	{
		calc_tail_pos(i,dt,wind);
	}

	//ローカル系における速度を計算
	//まず凧の移動速度と風速から相対速度を求めておく
	kite.local_vel=kite.global_vel-wind;								//相対速度(凧速度ベクトル-風ベクトル)
	kite.local_vel=QVRotate(kite.orientation.inverse(),kite.local_vel);	//ローカル系へコンバート

	//オイラー角取得(一応)
	Vec3 dummy=kite.orientation.GetEulerAngles();
	kite.euler_angles=dummy;

	//レンダリング用座標取得
	calc_render_pos();
	
	//グローバル
	G[1]=kite.pos;
	T_string[1]=kite.glb_s_pos;
	for( i=0;i<2;i++)
	{
		//位置
		T_tail[i][1]=kite.global_cd[tail[i].set_point];
		Lift[i][1]=QVRotate(kite.orientation,Lift[i][1])+kite.pos;
		Drag[i][1]=QVRotate(kite.orientation,Drag[i][1])+kite.pos;
		//力
		T_tail[i][0]=QVRotate(kite.orientation,T_tail[i][0]);
		Lift[i][0]=QVRotate(kite.orientation,Lift[i][0]);
		Drag[i][0]=QVRotate(kite.orientation,Drag[i][0]);
	}
}
 
void Kite3D::calc_tail_pos(int n,double dt,Vec3 wind)
{
	int i=0,j=0,k=0;

	//Vec3 wind_vel=Wind-kite.global_vel;
	Vec3 wind_vel=Vec3();
	double wind_norm=0.0;

	double L=tail[n].l/((double)TAIL_SP);
	double Inv_dt=1.0/dt;

	Vec3 g_vec=Vec3(0.0,0.0,1.0);
	g_vec*=(L*TAIL_RHO)*G_ACCE;

	//接続端点速度の算出
	tail[n].vel[0]=kite.global_vel;	//速度

	tail[n].pos[0]=kite.global_cd[tail[n].set_point];	//位置

	//初期化
	for(i=0;i<=TAIL_SP;i++)
	{
		tail[n].frc[i]=Vec3();
	}

	//抗力,重力を加える
	for(i=0;i<=TAIL_SP;i++)
	{
		wind_vel=wind-tail[n].vel[i];
		wind_norm=unitize(wind_vel);

		tail[n].frc[i]+=wind_vel*(wind_norm*wind_norm)*TAIL_E;
		tail[n].frc[i]+=g_vec;
	}

	//ばねの力
	for(i=0;i<TAIL_SP;i++)
	{
		Vec3 d=tail[n].pos[i]-tail[n].pos[i+1];
		Vec3 vel=tail[n].vel[i]-tail[n].vel[i+1];

		Vec3 f1=-(TAIL_K * (norm(d) - L) + TAIL_D * ( dot(vel,d) / norm(d) )) * ( d / norm(d) );
		Vec3 f2 = -f1;

		tail[n].frc[i]+=f1;
		tail[n].frc[i+1]+=f2;
	}

	//kite.T_spring[1]=tail[n].frc[0];

	//速度・位置の更新
	for(i=1;i<=TAIL_SP;i++)
	{
		Vec3 Ae=Vec3();

		Ae=tail[n].frc[i]/(L*LINE_RHO);

		tail[n].pos[i]+=tail[n].vel[i]*dt+0.5*Ae*dt*dt;
		tail[n].vel[i]+=Ae*dt;
	}

	//各質点の質量の逆数
	double InvMass[TAIL_SP+1];
	for(i=0;i<=TAIL_SP;i++)
	{
		if(0==i)
		{
			InvMass[i]=1.0/(L*TAIL_RHO+kite.mass);
		}
		else
		{
			InvMass[i]=1.0/(L*TAIL_RHO);
		}
	}

	//*fast projection
	int flag=0;
	double ee=0.01;//閾値(*100%)
	for(;;)
	{
		flag=0;
		//TAIL_SPがばねの本数に対応
		for(i=0;i<TAIL_SP;i++)
		{
			//辺ごとの式(1)を格納する
			Vec3 l=tail[n].pos[i+1]-tail[n].pos[i];
			double Inv_E_L=1.0/L;//ばねの初期長の逆数
			//式(1)
			T_C_strains[i]=norm2(l)*Inv_E_L-L;

			//ここで，式(1)が誤差範囲内に収まっているかを判定する
			double eps=((1.0+ee)*(1.0+ee)-1.0)*L;
			if(fabs(T_C_strains[i])>eps)//誤差範囲内に収まっていない場合はflagに+する
			{
				flag++;
			}

			//式(2)
			l=2.0*Inv_E_L*(tail[n].pos[i+1]-tail[n].pos[i]);

			int in[6];//index
			//式(2)の格納
			//i番目の質点に関する値
			T_C_grad[i][3*i]=l.data[0];		in[0]=3*i;
			T_C_grad[i][3*i+1]=l.data[1];	in[1]=3*i+1;
			T_C_grad[i][3*i+2]=l.data[2];	in[2]=3*i+2;
			//(i+1)番目の質点に関する値
			T_C_grad[i][3*(i+1)]=-l.data[0];	in[3]=3*(i+1);
			T_C_grad[i][3*(i+1)+1]=-l.data[1];	in[4]=3*(i+1)+1;
			T_C_grad[i][3*(i+1)+2]=-l.data[2];	in[5]=3*(i+1)+2;

			//Mの逆行列×C_gradの転置の格納
			//i番目の質点に関する値
			T_IM_Cg[3*i][i]=l.data[0]*InvMass[i]*dt*dt;
			T_IM_Cg[3*i+1][i]=l.data[1]*InvMass[i]*dt*dt;
			T_IM_Cg[3*i+2][i]=l.data[2]*InvMass[i]*dt*dt;
			//(i+1)番目の質点に関する値
			T_IM_Cg[3*(i+1)][i]=-l.data[0]*InvMass[i+1]*dt*dt;
			T_IM_Cg[3*(i+1)+1][i]=-l.data[1]*InvMass[i+1]*dt*dt;
			T_IM_Cg[3*(i+1)+2][i]=-l.data[2]*InvMass[i+1]*dt*dt;

			for(k=0;k<TAIL_SP;k++)
			{//初期化
				T_Cg_IM_Cg[TAIL_SP*i+k]=0.0;
			}
			//式(5)のA行列をつくる
			for(j=0;j<TAIL_SP;j++)
			{
				for(k=0;k<6;k++)
				{
					T_Cg_IM_Cg[TAIL_SP*i+j]+=T_C_grad[i][in[k]]*T_IM_Cg[in[k]][j];
				}
			}
		}
		if(0==flag) break;//もしすべての辺が誤差範囲内に収まっているのならば，反復を打ち切る

		//共役勾配法でδλベクトルを求める
		int bl;
		int m_it=3;
		double cg_eps=1e-5;
		bl=cgSolverMatrix(TAIL_SP,T_Cg_IM_Cg,T_C_strains,T_Delta_lambda,m_it,cg_eps);
		//答えを受け取るδxベクトルをつくる
		for(i=0;i<3*(TAIL_SP+1);i++)
		{
			T_Delta_x[i]=0.0;//初期化
			
			for(j=0;j<TAIL_SP;j++)
			{
				T_Delta_x[i]+=T_IM_Cg[i][j]*T_Delta_lambda[j];
			}
		}
		//真の位置と真の速度
		i=0;
		for(j=0;j<=TAIL_SP;j++)
		{

			if(0==j)//固定点の処理
			{
				i=i+3;

			}
			else if(!(0==j))//固定されてない点の処理
			{
				tail[n].pos[j].data[0]+=T_Delta_x[i];i++;
				tail[n].pos[j].data[1]+=T_Delta_x[i];i++;
				tail[n].pos[j].data[2]+=T_Delta_x[i];i++;
				tail[n].vel[j]=(tail[n].pos[j]-tail[n].pos_pre[j])*Inv_dt;
			}
		}
	}//*/

	for(i=0;i<=TAIL_SP;i++)
	{
		tail[n].pos_pre[i]=tail[n].pos[i];//コピー
		tail[n].vel_pre[i]=tail[n].vel[i];//コピー
	}
}

/*!
 * @note 風のセット
 */

//计算扭力
 
void Kite3D::calc_loads(double dt)
{
	//荷重とモーメントの初期化
	kite.frc=Vec3();
	kite.mom=Vec3();

	int i=0;	//カウンタ

	Vec3 Fb,Mb;
	Fb=Vec3();	//合力を格納する
	Mb=Vec3();	//モーメントの和を格納する

	double Fb_nrm=0.0;//合力の法線方向成分

	//色々な値を格納する
	double tmp=0.0;
	Vec3 tmp_vec=Vec3();

	Vec3 local_vel=Vec3();	//ローカル系における速度
	Vec3 xz_vel=Vec3();		//x-z平面における速度
	double xz_speed=0.0;	//速度の大きさ
	Vec3 yz_vel=Vec3();		//y-z平面における速度
	double yz_speed=0.0;	//速度の大きさ

	//Vec3 force_vec=Vec3();	//風力の加わる方向
	Vec3 Lift_vec=Vec3();	//揚力の加わる方向
	Vec3 Drag_vec=Vec3();	//抗力の加わる方向
	Vec3 L=Vec3();			//揚力
	Vec3 D=Vec3();			//抗力

	double alpha=0.0;	//迎え角
	double psi=0.0;		//迎え角
	double CD=0.0;		//抗力係数
	double CL=0.0;		//揚力係数
	double cw=0.0;		//風圧中心係数
	Vec3 x_wind=Vec3();	//風圧中心

	for(i=0;i<kite.q_num;i++)
	{
		//四角形要素の速度を計算
		tmp_vec=cross(kite.omega,kite.element[i].cg);	//角速度成分
		local_vel=kite.local_vel+tmp_vec;				//ローカル系における速度

		//x-z平面とy-z平面に速度を投影する
		xz_vel=local_vel;	//速度のコピー
		xz_vel.data[1]=0.0;	//y成分の除去
		yz_vel=local_vel;	//速度のコピー
		yz_vel.data[0]=0.0;	//x成分の除去

//xz-------------------------------------------

		//抗力の加わる方向の取得
		xz_speed=unitize(xz_vel);//速度の大きさの取得と速度ベクトルの正規化
		Drag_vec=-xz_vel;

		//揚力の作用する方向の取得
		Lift_vec=cross(Drag_vec,-kite.element[i].normal);
		Lift_vec=cross(Lift_vec,Drag_vec);
		tmp=unitize(Lift_vec);//正規化

		//迎え角を調べる
		tmp=dot(Drag_vec,kite.element[i].normal);//cos(alpha)を求める
		if(1.0<tmp)			//本来ありえないが
		{					//cos(alpha)が1.0を上回った場合の修正
			tmp=1.0;
		}
		else if(-1.0>tmp)	//本来ありえないが
		{					//cos(alpha)が-1.0を下回った場合の修正
			tmp=-1.0;
		}
		alpha=RX_TO_DEGREES(asin(tmp));	//alphaを求める
		if(0.0>alpha)	//法線方向の逆に移動している場合
		{				//(風が裏から表に向かって流れている場合)
			alpha=-alpha;
		}
//*
		//迎え角から形状抵抗係数と風圧中心座標を求める
		cw=search_alpha_x(alpha);//係数
		CD=search_alpha_CD(alpha,kite.AR);
		CL=search_alpha_CL(alpha,kite.AR);

		//揚力，抗力を計算
		L=0.5*CL*RHO*(kite.element[i].S*p_l[0])*(xz_speed*xz_speed)*Lift_vec;
		D=0.5*CD*RHO*(kite.element[i].S*p_l[0])*(xz_speed*xz_speed)*Drag_vec;

		//凧全体での合力を計算
		Fb+=L+D;

		Lift[0][0]+=L;
		Drag[0][0]+=D;

		//凧全体でのモーメント総計
		if(0.0<xz_vel.data[0])
		{
			cw=1.0-cw;
		}
		x_wind=Vec3((cw-0.5)*kite.element[i].c+kite.element[i].cg.data[0], kite.element[i].cg.data[1], 0.0);

		tmp_vec=cross(x_wind,L);
		Mb+=tmp_vec;
		tmp_vec=cross(x_wind,D);
		Mb+=tmp_vec;

		Lift[0][1]=Vec3((cw)*kite.element[i].c-kite.cg.data[0],0.0,0.0);
		Drag[0][1]=Vec3((cw)*kite.element[i].c-kite.cg.data[0],0.0,0.0);

		//合力の法線方向成分
		Fb_nrm+=dot((L+D),-kite.element[i].normal);

//yz-------------------------------------------*/

//*
		//抗力の加わる方向の取得
		yz_speed=unitize(yz_vel);//速度の大きさの取得と速度ベクトルの正規化
		Drag_vec=-yz_vel;

		//揚力の作用する方向の取得
		Lift_vec=cross(Drag_vec,-kite.element[i].normal);
		Lift_vec=cross(Lift_vec,Drag_vec);
		tmp=unitize(Lift_vec);//正規化

		//迎え角を調べる
		tmp=dot(Drag_vec,kite.element[i].normal);//cos(psi)を求める
		if(1.0<tmp)			//本来ありえないが
		{					//cos(psi)が1.0を上回った場合の修正
			tmp=1.0;
		}
		else if(-1.0>tmp)	//本来ありえないが
		{					//cos(psi)が-1.0を下回った場合の修正
			tmp=-1.0;
		}
		psi=RX_TO_DEGREES(asin(tmp));	//psiを求める
		if(0.0>psi)
		{
			psi=-psi;
		}

		//迎え角から形状抵抗係数と風心座標を求める
		//迎え角から形状抵抗係数と風圧中心座標を求める
		cw=search_alpha_x(psi);//係数
		CD=search_alpha_CD(psi,kite.AR);
		CL=search_alpha_CL(psi,kite.AR);

//#define PSI_SP 0.02734
//
//		cw+=PSI_SP;//alpha用の関数をpsiに利用するための修正

		//揚力，抗力を計算
		L=0.5*CL*RHO*(kite.element[i].S*p_l[0])*(yz_speed*yz_speed)*Lift_vec;
		D=0.5*CD*RHO*(kite.element[i].S*p_l[0])*(yz_speed*yz_speed)*Drag_vec;

		//凧全体での合力を計算
		Fb+=L+D;
		Lift[1][0]+=L;
		Drag[1][0]+=D;

		//凧全体でのモーメント総計
		if(0.0<yz_vel.data[1])
		{
			cw=1.0-cw;
		}
		x_wind=Vec3(kite.element[i].cg.data[0], (cw-0.5)*kite.element[i].b+kite.element[i].cg.data[1], 0.0);

		kite_check=Vec3(0.0,(cw)*kite.b-kite.cg.data[1],0.0);
		tmp_vec=cross(x_wind,L);
		Mb+=tmp_vec;
		tmp_vec=cross(x_wind,D);
		Mb+=tmp_vec;

		Lift[1][1]=Vec3(0.0,(cw)*kite.b-kite.cg.data[1],0.0);
		Drag[1][1]=Vec3(0.0,(cw)*kite.b-kite.cg.data[1],0.0);

		//合力の法線方向成分
		Fb_nrm+=dot((L+D),-kite.element[i].normal);

//---------------------------------------------*/
	}
	kite_check=QVRotate(kite.orientation,kite_check);
	
	//たわみ計算より投影面積(p_l)を求める
	calc_deflection(Fb_nrm);

	//しっぽからの引張力
	double ce=0.0;
	ce=1.2;
	Vec3 tail_frc=Vec3();
	for(i=0;i<TAIL_NUM;i++)
	{
		tail_frc=QVRotate(kite.orientation.inverse(),tail[i].frc[0]);//コンバート
		Fb+=ce*tail_frc;
		Mb+=cross(kite.local_cd[tail[i].set_point],ce*tail_frc);

		T_tail[i][0]+=ce*tail_frc;
	}

	//荷重をローカル系からグローバル系へとコンバート
	kite.frc=QVRotate(kite.orientation,Fb);

	//モーメント
	kite.mom+=Mb;

	//重力(グローバル系で加える)
	Vec3 g_vec=Vec3(0.0,0.0,1.0);
	g_vec*=kite.mass*G_ACCE;
	kite.frc+=g_vec;

	G[0]=g_vec;//グローバル

	//1step前の張力を合力に加算
	kite.frc+=kite.T_spring[0];
	T_string[0]=kite.T_spring[0];

	//張力によるモーメント
	kite.T_spring[0]=QVRotate(kite.orientation.inverse(),kite.T_spring[0]);//グローバル->ローカル
	Mb=cross(kite.s,kite.T_spring[0]);//張力によるモーメント
	kite.mom+=Mb;
}


//文件读取函数
 
int Kite3D::read_file(const string &file_name, vector<double> &datas)
{
	// 拡張子チェック
	string file_ext = file_name.substr(file_name.size()-3, file_name.size()-1);

	int n=-1;
	int m=0;

	// ファイルからの読み込み
	if(file_ext == "dat"){	// 点データの読み込み
		double x;
	
		//open
		FILE *fp;
		if((fp = fopen(file_name.c_str(), "r")) == NULL) return -1;

		//データ数
		fscanf(fp, "%d", &n);
		if(0>n||4<n) return -1;

		if(0==n)
		{
			m=91;
		}
		else
		{
			m=46;
		}
	
		int count = 0;
		for(int i = 0; i < m; ++i){
			fscanf(fp, "%lf", &x);
			datas.push_back(x);
			count++;
		}
		fclose(fp);

		return n;
	}
	else{
		return -1;
	}

}

 
void Kite3D::read_file(const string &file_name)
{
	vector<double> datas;
	int i;

	int number=read_file(file_name, datas);

	if(0==number)
	{
		for(i=0;i<=70;i++)
		{
			x_table[i]=datas[i];
			x_table[i]+=0.25+0.02734;
		}
	}
	else if(1==number)
	{
		for(i=0;i<=45;i++)
		{
			CD_068_table[i]=datas[i];
		}
	}
	else if(2==number)
	{
		for(i=0;i<=45;i++)
		{
			CD_148_table[i]=datas[i];
		}
	}
	else if(3==number)
	{
		for(i=0;i<=45;i++)
		{
			CL_068_table[i]=datas[i];
		}
	}
	else if(4==number)
	{
		for(i=0;i<=45;i++)
		{
			CL_148_table[i]=datas[i];
		}
	}
}

	//--------------------------------------------------------------------------------------------------------------15gai
//画图函数
void Kite3D::draw(void)
{
	
	draw_kite();

	draw_tail();
	//kiteString.draw();
	draw_options_01();
//	draw_options_02();
//	draw_options_03();
}
void 
	Kite3D::draw_kite(void)
{
	int i=0,j=0,k=0;

	//凧の描画
	
	for(i=0;i<kite.q_num;i++)
	{
		glBegin(GL_QUADS);
			//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
			glTexCoord2d(kite.tex_cd[kite.element[i].index[0]][1], kite.tex_cd[kite.element[i].index[0]][0]);
			glVertex3d(kite.global_cd[kite.element[i].index[0]][0], kite.global_cd[kite.element[i].index[0]][2], -kite.global_cd[kite.element[i].index[0]][1]);

			//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
			glTexCoord2d(kite.tex_cd[kite.element[i].index[1]][1], kite.tex_cd[kite.element[i].index[1]][0]);
			glVertex3d(kite.global_cd[kite.element[i].index[1]][0], kite.global_cd[kite.element[i].index[1]][2], -kite.global_cd[kite.element[i].index[1]][1]);

			//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
			glTexCoord2d(kite.tex_cd[kite.element[i].index[2]][1], kite.tex_cd[kite.element[i].index[2]][0]);
			glVertex3d(kite.global_cd[kite.element[i].index[2]][0], kite.global_cd[kite.element[i].index[2]][2], -kite.global_cd[kite.element[i].index[2]][1]);

			//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
			glTexCoord2d(kite.tex_cd[kite.element[i].index[3]][1], kite.tex_cd[kite.element[i].index[3]][0]);
			glVertex3d(kite.global_cd[kite.element[i].index[3]][0], kite.global_cd[kite.element[i].index[3]][2], -kite.global_cd[kite.element[i].index[3]][1]);

		glEnd();
	}
	
}

 
void Kite3D::draw_tail(void)
{
	glPushMatrix();
		glDisable(GL_LIGHTING);
		glColor3f ( 0.5f, 0.5f, 0.5f );//gray
		//glColor3f ( 1.0f, 1.0f, 1.0f );//white
		glLineWidth ( 5.0f );

		glBegin ( GL_LINE_STRIP );
		for(int i=0;i<=TAIL_SP;i++)
		{
			glVertex3d ( tail[0].pos[i].data[0],tail[0].pos[i].data[2],-tail[0].pos[i].data[1]);
		}
		glEnd ();

		glBegin ( GL_LINE_STRIP );
		for(int i=0;i<=TAIL_SP;i++)
		{
			glVertex3d ( tail[1].pos[i].data[0],tail[1].pos[i].data[2],-tail[1].pos[i].data[1]);
		}
		glEnd ();

		glEnable(GL_LIGHTING);
	glPopMatrix();
}

//各力の作用点表示
 
void Kite3D::draw_options_01(void)
{
	glDisable(GL_LIGHTING);

	glColor3f ( 1.0f, 0.0f, 0.0f );//red
	glPushMatrix();
	//重心
		glTranslated(G[1].data[0],G[1].data[2],-G[1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();

	glColor3f ( 1.0f, 0.0f, 1.0f );//purple
	glPushMatrix();
	//作用点1
		glTranslated(Lift[0][1].data[0],Lift[0][1].data[2],-Lift[0][1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();
	glColor3f ( 0.0f, 1.0f, 0.0f );//green
	glPushMatrix();
	//作用点2
		glTranslated(Lift[1][1].data[0],Lift[1][1].data[2],-Lift[1][1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();

	glColor3f ( 1.0f, 1.0f, 0.0f );//yellow
	glPushMatrix();
	//しっぽ1
		glTranslated(T_tail[0][1].data[0],T_tail[0][1].data[2],-T_tail[0][1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();
	glPushMatrix();
	//しっぽ2
		glTranslated(T_tail[1][1].data[0],T_tail[1][1].data[2],-T_tail[1][1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();

	glColor3f ( 0.0f, 0.0f, 1.0f );//blue
	glPushMatrix();
	//糸目
		glTranslated(T_string[1].data[0],T_string[1].data[2],-T_string[1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();

	glEnable(GL_LIGHTING);
}

//各力の大きさ表示
 
void Kite3D::draw_options_02(void)
{
	double scale=0.1;
	glDisable(GL_LIGHTING);

	glLineWidth ( 2.0f );
	
	//重心
	glColor3f ( 1.0f, 0.0f, 0.0f );//red
	glBegin ( GL_LINES );
		glVertex3d ( G[1].data[0],G[1].data[2],-G[1].data[1] );
		glVertex3d ( G[1].data[0]+scale*G[0].data[0],G[1].data[2]+scale*G[0].data[2],-(G[1].data[1]+scale*G[0].data[1]) );
	glEnd ();

	//揚力
	glColor3f ( 1.0f, 0.0f, 1.0f );//purple
	glBegin ( GL_LINES );
		glVertex3d ( Lift[0][1].data[0],Lift[0][1].data[2],-Lift[0][1].data[1]);
		glVertex3d ( Lift[0][1].data[0]+scale*Lift[0][0].data[0],Lift[0][1].data[2]+scale*Lift[0][0].data[2],-(Lift[0][1].data[1]+scale*Lift[0][0].data[1]));
	glEnd ();
	glBegin ( GL_LINES );
		glVertex3d ( Lift[1][1].data[0],Lift[1][1].data[2],-Lift[1][1].data[1]);
		glVertex3d ( Lift[1][1].data[0]+scale*Lift[1][0].data[0],Lift[1][1].data[2]+scale*Lift[1][0].data[2],-(Lift[1][1].data[1]+scale*Lift[1][0].data[1]));
	glEnd ();
	//抗力
	glColor3f ( 0.0f, 1.0f, 0.0f );//green
	glBegin ( GL_LINES );
		glVertex3d ( Drag[0][1].data[0],Drag[0][1].data[2],-Drag[0][1].data[1]);
		glVertex3d ( Drag[0][1].data[0]+scale*Drag[0][0].data[0],Drag[0][1].data[2]+scale*Drag[0][0].data[2],-(Drag[0][1].data[1]+scale*Drag[0][0].data[1]));
	glEnd ();
	glBegin ( GL_LINES );
		glVertex3d ( Drag[1][1].data[0],Drag[1][1].data[2],-Drag[1][1].data[1]);
		glVertex3d ( Drag[1][1].data[0]+scale*Drag[1][0].data[0],Drag[1][1].data[2]+scale*Drag[1][0].data[2],-(Drag[1][1].data[1]+scale*Drag[1][0].data[1]));
	glEnd ();

	//しっぽ1
	glColor3f ( 1.0f, 1.0f, 0.0f );//yellow
	glBegin ( GL_LINES );
		glVertex3d ( T_tail[0][1].data[0],T_tail[0][1].data[2],-T_tail[0][1].data[1]);
		glVertex3d ( T_tail[0][1].data[0]+scale*T_tail[0][0].data[0],T_tail[0][1].data[2]+scale*T_tail[0][0].data[2],-(T_tail[0][1].data[1]+scale*T_tail[0][0].data[1]));
	glEnd ();
	//しっぽ2
	glBegin ( GL_LINES );
		glVertex3d ( T_tail[1][1].data[0],T_tail[1][1].data[2],-T_tail[1][1].data[1]);
		glVertex3d ( T_tail[1][1].data[0]+scale*T_tail[1][0].data[0],T_tail[1][1].data[2]+scale*T_tail[1][0].data[2],-(T_tail[1][1].data[1]+scale*T_tail[1][0].data[1]));
	glEnd ();

	//糸目
	glColor3f ( 1.0f, 0.5f, 0.0f );//orange
	glBegin ( GL_LINES );
		glVertex3d ( T_string[1].data[0],T_string[1].data[2],-T_string[1].data[1]);
		glVertex3d ( T_string[1].data[0]+scale*T_string[0].data[0],T_string[1].data[2]+scale*T_string[0].data[2],-(T_string[1].data[1]+scale*T_string[0].data[1]));
	glEnd ();

	glEnable(GL_LIGHTING);
}

//合力表示
 
void Kite3D::draw_options_03(void)
{
	double scale=0.1;
	glDisable(GL_LIGHTING);

	glColor3f ( 1.0f, 0.0f, 0.0f );//red
	glBegin ( GL_LINES );
		glVertex3d ( G[1].data[0],G[1].data[2],-G[1].data[1] );
		glVertex3d ( G[1].data[0]+scale*kite.frc.data[0],G[1].data[2]+scale*kite.frc.data[2],-(G[1].data[1]+scale*kite.frc.data[1]) );
	glEnd ();

	glEnable(GL_LIGHTING);
}
