/*
	a calss to create the kite
	date 2015
*/		

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "kite_simulation.h"

LineSimulation line;

double CD_068_table[46];	//!< alpha_CD_068.dat格納用配列
double CD_148_table[46];	//!< alpha_CD_148.dat格納用配列
double CL_068_table[46];	//!< alpha_CL_068.dat格納用配列
double CL_148_table[46];	//!< alpha_CL_148.dat格納用配列
double x_table[71];		//alpha_x.dat格納用配列

int nsteps = 0;

//KiteShape kite;
//force, pos
Vec3 Lift[2][2];
Vec3 Drag[2][2];
Vec3 T_string[2];
Vec3 G[2];				//globle position use to draw the force point of the kite

float tension_check;
double ex_Sp_l;
double ex_Sp_pos[2];
double ex_Sp_vel[2];


Vec3 kite_check;
int v_point[4];

Vec3 spring_ce;			//a para to calcualte the user control speed, which can influence the first point of the spring

//deflection
Vec2 point[(BAR_SP+1)];
double sp_l;		//点間隔
double Wind_vel;	//気流速度
double p_l[2];		//投影面積(長さ)

////数値計算用
//double Heron(Vec3 A, Vec3 B, Vec3 C);	//ヘロンの公式
//double DotProductV(double a[], double b[], int num);
//int cgSolverMatrix(int num, double A[], double b[], double x[], int &max_iter, double &tol);

//init kite func
void KiteSimulation::initKite(void)		//init the para of kite
{
	set_length(8.0);
	//setLineStartPoint(line.getStartPoint());
	//setLineEndPoint(line.getEndPoint());

	kite_check = Vec3();
	tension_check = 0.0;

	ex_Sp_l=1.0;
	ex_Sp_pos[0]=0.0;
	ex_Sp_pos[1]=ex_Sp_l;
	ex_Sp_vel[0]=0.0;
	ex_Sp_vel[1]=0.0;

	//init the line
	line.InitLine();

	//init the kite
	k_pos = line.getEndPoint();			//kite's init pos equals to the end point position of string
	//k_pos = Vec3(-1,0,2);
	k_pos_pre = k_pos;					//pos 1 step ago
	k_global_vel = Vec3();			//global vel
	k_local_vel = Vec3();				//local vel

	k_frc = Vec3();					//荷重
	k_frc_pre = k_frc;					//previous 荷重
	//k_T_spring[0]	= Vec3();			//ばねによる引張力
	k_T_spring[0]	= line.calTension();	

	k_T_spring[1]	= Vec3();			//ばねによる引張力
	Wind = Vec3();

	//init the rotation para of the kite
	k_omega = Vec3();					//angular velocity 

	//init the direction
	//Initial position (angle)
	double roll  =0.0;
	double pitch = 60.0;
	double yow = 0.0;
	double rdrd = RX_DEGREES_TO_RADIANS* RX_DEGREES_TO_RADIANS;//degreeからradに変換するため
	Vec3 dummy=Vec3(roll*rdrd,pitch*rdrd,yow*rdrd);//(roll,pitch,yow)

	k_orientation.SetEulerAngles(dummy);				//初期姿勢の格納

	for(int i= 0; i< 2; i++)
	{
		Lift[0][i] = Vec3();
		Lift[1][i] = Vec3();

		Drag[0][i] = Vec3();
		Drag[1][i] = Vec3();

		T_string[i] = Vec3();
		G[i] = Vec3();
	}
}

//init the deflection
void KiteSimulation::initialize_deflection(void)	
{
	int i=0;

	for(i= 0;i<=BAR_SP;i++)
	{
		point[i]=Vec2();		//0で初期化
		point[i].data[0] = ((double)i)*k_b/((double)BAR_SP);
	}

	sp_l = k_b/((double)BAR_SP);		//点間隔
	Wind_vel=0.0;//init the wind

	// Projection area 
	p_l[0] = point[BAR_SP].data[0]-point[0].data[0];
	//max deflection
	p_l[1] = 0.0;
}



void KiteSimulation::calc_deflection(double P)		//calculate the deflection 
{
	p_l[1]=0.0;//init the max deflection

	int i=0,j=0;

	//calc the deflection
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
		var=3.0*k_b*k_b-4.0*point[j].data[0]*point[j].data[0];
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
void KiteSimulation::keep_long_deflection(void)		//長さを保つ
{
	int i=0;

	double l_2=sp_l*sp_l;//点間隔の2乗
	double y=0.0,y_2=0.0;
	double x=0.0,x_2=0.0;

	for(i=0;i<BAR_SP;i++)
	{
		//calc the y direction of the distance
		y=point[i].data[1]-point[i+1].data[1];
		y=fabs(y);
		
		y_2=y*y;

		//calc the x direction of the distance
		x_2=l_2-y_2;	//(斜辺^2-y方向成分^2)
		x=sqrt(x_2);
		x=fabs(x);		//念のため絶対値計算により正にする
		
		point[i+1].data[0]=point[i].data[0]+x;
	}

	p_l[0]=point[BAR_SP].data[0]-point[0].data[0];//projected area
}

void KiteSimulation::create_model_rec(void)		//create the model
{
	//最終的には入力されたデザインから
	//点群と三角形ポリゴンを作成
	int i=0,j=0;//counter

	//input---------------------------------
	double b_max=0.8;//max kite width
	double c_max=1.2;//max kite height

	vector<Vec3> pointGroups;	//Point groups
	int pointgroup_num=0;		//number of the point groups

	vector<Vec3> tex_p;	//texture pos

	for(i=0;i<ROW_NUM;i++)//x direction
	{
		for(j=0;j<COL_NUM;j++)//y direction
		{
			pointGroups.push_back(Vec3(((double)i)*c_max/((double)(ROW_NUM-1)),((double)j)*b_max/((double)(COL_NUM-1)),0.0 ));
			pointgroup_num++;	
			tex_p.push_back(Vec3( -((double)i)/((double)(ROW_NUM-1)),-((double)j)/((double)(COL_NUM-1)),0.0));
		}
	}
	v_point[0]=0;
	v_point[1]=COL_NUM-1;
	v_point[2]=COL_NUM*ROW_NUM-1;
	v_point[3]=COL_NUM*(ROW_NUM-1);

	vector<int> q_index[4];	//index of the 4 points of the quadrangle
	int quadrangel_num=0;			//number of the quadrangle

	for(i=0;i<(ROW_NUM-1);i++)
	{
		for(j=0;j<(COL_NUM-1);j++)
		{
			//store the quadrangle
			q_index[0].push_back(i*COL_NUM+j);
			q_index[1].push_back(i*COL_NUM+(j+1));
			q_index[2].push_back((i+1)*COL_NUM+(j+1));
			q_index[3].push_back((i+1)*COL_NUM+j);
			quadrangel_num++;
		}
	}

	Vec3 itome=Vec3(0.3*c_max,0.5*b_max,0.0);//point which connects with the string

	//create the model from input
	k_b=b_max;
	k_c=c_max;
	k_AR=b_max/c_max;

	k_p_num=pointgroup_num;	//頂点数格納
	k_q_num=quadrangel_num;	//四角形要素数格納

	k_s=itome;		//糸目中心取り込み

	for(i=0;i<pointgroup_num;i++)
	{
		k_tex_cd.push_back(tex_p[i]);

		k_local_cd.push_back(pointGroups[i]);	//ローカル座標は後で重心基準になる

		k_design_cd.push_back(pointGroups[i]-k_s);//デザイン座標は糸目中心を基準にしている
		k_global_cd.push_back(QVRotate(k_orientation,k_design_cd[i]));	//グローバル座標
		k_global_cd[i]+=k_pos;
	}
	
	k_S=0.0; k_mass=0.0;	//init
	element.resize(quadrangel_num);	//四角形要素数分element確保
	for(i=0;i<quadrangel_num;i++)
	{
		element[i].b=k_b;
		element[i].c=k_c;

		//inxdex of the 4 points of the quadrangle
		element[i].index[0]=(int)q_index[0][i];
		element[i].index[1]=(int)q_index[1][i];
		element[i].index[2]=(int)q_index[2][i];
		element[i].index[3]=(int)q_index[3][i];

		//ヘロンの公式より四角形の面積を計算
		element[i].S=Heron(k_local_cd[element[i].index[0]],
							k_local_cd[element[i].index[1]],
							k_local_cd[element[i].index[2]])
						+ Heron(k_local_cd[element[i].index[0]],
							k_local_cd[element[i].index[2]],
							k_local_cd[element[i].index[3]]);

		//mass of the quadrangle
		element[i].mass=element[i].S*KITE_RHO;

		Vec3 v1,v2;
		double dummy;
		//normal of the quadrangle
		v1=k_local_cd[element[i].index[1]]-k_local_cd[element[i].index[0]];
		v2=k_local_cd[element[i].index[2]]-k_local_cd[element[i].index[1]];
		element[i].normal=cross(v1,v2);	//外積
		dummy=unitize(element[i].normal);	//normalization

		//center gravity of the quadrangle
		element[i].cg=((k_local_cd[element[i].index[0]]+
							k_local_cd[element[i].index[1]]+
							k_local_cd[element[i].index[2]]+
							k_local_cd[element[i].index[3]]))*0.25;
		
		//四角形のローカル慣性テンソル
		double Ixx=0.0,Iyy=0.0,Izz=0.0;//慣性モーメント項
		double Ixy=0.0,Ixz=0.0,Iyz=0.0;//慣性乗積

		Vec3 dif;
		//四角形要素の底辺長さbを求める
		dif=k_local_cd[element[i].index[0]]-k_local_cd[element[i].index[1]];
		b_max=norm(dif);
		//四角形要素の高さcを求める
		dif=k_local_cd[element[i].index[0]]-k_local_cd[element[i].index[3]];
		c_max=norm(dif);

		//慣性モーメント
		Ixx=0.0833333*KITE_RHO*b_max*c_max*c_max*c_max;
		Iyy=0.0833333*KITE_RHO*b_max*b_max*b_max*c_max;
		Izz=0.0833333*KITE_RHO*b_max*c_max*(b_max*b_max+c_max*c_max);

		element[i].local_inertia_mom.data[0]=Ixx;
		element[i].local_inertia_mom.data[1]=Iyy;
		element[i].local_inertia_mom.data[2]=Izz;

		//慣性乗積
		Ixy=-0.0625*KITE_RHO*b_max*b_max*c_max*c_max;
		Ixz=0.0;
		Iyz=0.0;
		
		element[i].local_inertia_pro.data[0]=Ixy;
		element[i].local_inertia_pro.data[1]=Ixz;
		element[i].local_inertia_pro.data[2]=Iyz;

		//calculate the area and mass of the kite
		k_S+=element[i].S;
		k_mass+=element[i].mass;
	}
	cout<<k_mass<<endl;
	//calc the center gravity of the kite
	Vec3 mom=Vec3();
	for(i=0;i<quadrangel_num;i++)
	{
		mom+=element[i].mass*element[i].cg;
	}
	double kite_mass_inv=1.0/k_mass;
	k_cg=mom*kite_mass_inv;//重心

	//重心基準の座標へと変換
	for(i=0;i<quadrangel_num;i++)
	{
		element[i].cg-=k_cg;
	}
	k_s-=k_cg;//糸目中心も
	//ローカル座標も
	for(i=0;i<pointgroup_num;i++)
	{
		k_local_cd[i]-=k_cg;
	}

	//慣性テンソル作成
	double Ixx,Iyy,Izz,Ixy,Ixz,Iyz;
	Ixx=0.0;Iyy=0.0;Izz=0.0;//慣性モーメント項
	Ixy=0.0;Ixz=0.0;Iyz=0.0;//慣性乗積
	for(i=0;i<quadrangel_num;i++)
	{
		Ixx+=element[i].local_inertia_mom.data[0]+
			element[i].mass*
			(element[i].cg.data[1]*element[i].cg.data[1]+
			element[i].cg.data[2]*element[i].cg.data[2]);
		Iyy+=element[i].local_inertia_mom.data[1]+
			element[i].mass*
			(element[i].cg.data[0]*element[i].cg.data[0]+
			element[i].cg.data[2]*element[i].cg.data[2]);
		Izz+=element[i].local_inertia_mom.data[2]+
			element[i].mass*
			(element[i].cg.data[1]*element[i].cg.data[1]+
			element[i].cg.data[0]*element[i].cg.data[0]);

		Ixy+=element[i].local_inertia_pro.data[0]+
			element[i].mass*
			(element[i].cg.data[0]*element[i].cg.data[1]);
		Ixz+=element[i].local_inertia_pro.data[1]+
			element[i].mass*
			(element[i].cg.data[0]*element[i].cg.data[2]);
		Iyz+=element[i].local_inertia_pro.data[2]+
			element[i].mass*
			(element[i].cg.data[2]*element[i].cg.data[1]);
	}

	inertia.SetValue(Ixx,-Ixy,-Ixz,-Ixy,Iyy,-Iyz,-Ixz,-Iyz,Izz);//慣性テンソルに格納
	inertia_inv=inertia.Inverse();//逆行列

	k_glb_s_pos=QVRotate(k_orientation,k_s)+k_pos;

}
void KiteSimulation::create_model_yak(void)
{
}
void KiteSimulation::create_model_dia(void)
{
}

//テーブルの参照
//calculate the resistance cofficience from the attack angle
double KiteSimulation::search_alpha_CD(double alpha,double AR)
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
//calculate the lift coefficient from the attack angle
double KiteSimulation::search_alpha_CL(double alpha,double AR)
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

//calc the center of the kite from the attack angle
double KiteSimulation::search_alpha_x(double alpha)		//α-xテーブル(迎え角から風心を求める)
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

//step update
void KiteSimulation::step_simulation(double dt)
{
	int i= 0;
	for(i=0;i<2;i++)
	{
		Lift[0][i] = Vec3();
		Lift[1][i] = Vec3();
		Drag[0][i]=Vec3();
		Drag[1][i]=Vec3();
	}

	//set the wind
	set_wind(dt);
	//Vec3 mm = wind_vel;

	//calc 荷重とmoment
	calc_loads(dt);
	//calc the translation
	k_pos += k_global_vel*dt + 0.5*k_frc*dt*dt/k_mass;	//update the pos
	k_global_vel +=k_frc*dt/k_mass;						//update the velcolity

	//calculate the rotation
	//calc the angel speed from the local coordinate 
	Vec3 I_omega=inertia*k_omega;
	k_omega+=inertia_inv*(k_mom-cross(k_omega,I_omega)-Vec3(DAMP*k_omega.data[0],DAMP*k_omega.data[1],DAMP_yaw*k_omega.data[2]))*dt;
	//新しい4元数を計算
	k_orientation+=(k_orientation*k_omega)*(0.5*dt);
	//方向4元数の正規化
	k_orientation.normalize();

	k_glb_s_pos=QVRotate(k_orientation,k_s)+k_pos;	//グローバル系の糸目位置更新

	//line pos
	calc_line_pos(dt);

	k_glb_s_pos=QVRotate(k_orientation,k_s)+k_pos;	//グローバル系の糸目位置更新


	//calc the velocity form local coordinate
	//まず凧の移動速度と風速から相対速度を求めておく
	k_local_vel=k_global_vel-Wind;								//相対速度(凧速度ベクトル-風ベクトル)
	k_local_vel=QVRotate(k_orientation.inverse(),k_local_vel);	//ローカル系へコンバート

	//get the euler angles
	Vec3 dummy=k_orientation.GetEulerAngles();
	k_euler_angles=dummy;

	//calc the pos to render
	calc_render_pos();

	//global
	G[1]=k_pos;
	T_string[1]=k_glb_s_pos;
	for(i=0;i<2;i++)
	{
		//位置
		Lift[i][1]=QVRotate(k_orientation,Lift[i][1])+k_pos;
		Drag[i][1]=QVRotate(k_orientation,Drag[i][1])+k_pos;
		//力
		Lift[i][0]=QVRotate(k_orientation,Lift[i][0]);
		Drag[i][0]=QVRotate(k_orientation,Drag[i][0]);
	}
}

//レンダリング用座標取得
void KiteSimulation::calc_render_pos(void)
{
	int i=0,j=0;

	//たわみ
	for(i=0;i<ROW_NUM;i++)
	{
		for(j=0;j<COL_NUM;j++)
		{
			k_design_cd[(i*COL_NUM)+j].data[2]=p_l[1]-point[j].data[1];
		}
	}

	for(i=0;i<k_p_num;i++)
	{
		k_global_cd[i]=QVRotate(k_orientation,k_design_cd[i]);	//rotation
		k_global_cd[i]+=k_glb_s_pos;								//translation
	}
	
}

//set the wind
void KiteSimulation::set_wind(double dt)
{
	#define WIND_STR 6.0//*fabs(sin(dt*StepNo))

	int i,j,k;	//counter
	int N=GRID;	//number of the grid
	double h;	//グリッド幅

	h = (l_init+1.0)*2.0/(double)N;

	double x,y,z;
	double ef=1.0;

	Vec3 kite_pos=Vec3(-k_pos.data[1],k_pos.data[2],k_pos.data[0]);

	for(i=1;i<=N;i++){//x座標
		if( -l_init+(i-1)*h<kite_pos.data[0] && kite_pos.data[0]<=-l_init+i*h ){
			for(j=1;j<=N;j++){//y座標
				if( -l_init+(j-1)*h<kite_pos.data[1] && kite_pos.data[1]<=-l_init+j*h ){
					for(k=1;k<=N;k++){//z座標
						if( -l_init+(k-1)*h<kite_pos.data[2] && kite_pos.data[2]<=-l_init+k*h ){
							
							x= g_w[IX(i,j,k)];
							y=-g_u[IX(i,j,k)];
							z= g_v[IX(i,j,k)];

							Wind=ef*l_init*1.0*Vec3(x,y,z);//気流ベクトルセット
						}
					}
				}
			}
		}
	}

}

//荷重とモーメントの計算
void KiteSimulation::calc_loads(double dt)
{
	//init the frc and moment
	k_frc=Vec3();
	k_mom=Vec3();

	int i=0;	//counter

	Vec3 Fb,Mb;
	Fb=Vec3();	//store the resultant force
	Mb=Vec3();	//モーメントの和を格納する

	double Fb_nrm=0.0;//normal of the resultant force

	//色々な値を格納する
	double tmp=0.0;
	Vec3 tmp_vec=Vec3();

	Vec3 local_vel=Vec3();	//local speed
	Vec3 xz_vel=Vec3();		//x-z speed
	double xz_speed=0.0;	//speed
	Vec3 yz_vel=Vec3();		//y-z平面における速度
	double yz_speed=0.0;	//速度の大きさ

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

	for(i=0;i<k_q_num;i++)
	{
		//四角形要素の速度を計算
		tmp_vec=cross(k_omega,element[i].cg);	//角速度成分
		local_vel=k_local_vel+tmp_vec;				//ローカル系における速度

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
		Lift_vec=cross(Drag_vec,-element[i].normal);
		Lift_vec=cross(Lift_vec,Drag_vec);
		tmp=unitize(Lift_vec);//正規化

		//迎え角を調べる
		tmp=dot(Drag_vec,element[i].normal);//cos(alpha)を求める
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
		CD=search_alpha_CD(alpha,k_AR);
		CL=search_alpha_CL(alpha,k_AR);

		//揚力，抗力を計算
		L=0.5*CL*RHO*(element[i].S*p_l[0])*(xz_speed*xz_speed)*Lift_vec;
		D=0.5*CD*RHO*(element[i].S*p_l[0])*(xz_speed*xz_speed)*Drag_vec;

		//凧全体での合力を計算
		Fb+=L+D;

		Lift[0][0]+=L;
		Drag[0][0]+=D;

		//凧全体でのモーメント総計
		if(0.0<xz_vel.data[0])
		{
			cw=1.0-cw;
		}
		x_wind=Vec3((cw-0.5)*element[i].c+element[i].cg.data[0], element[i].cg.data[1], 0.0);

		tmp_vec=cross(x_wind,L);
		Mb+=tmp_vec;
		tmp_vec=cross(x_wind,D);
		Mb+=tmp_vec;

		Lift[0][1]=Vec3((cw)*element[i].c-k_cg.data[0],0.0,0.0);
		Drag[0][1]=Vec3((cw)*element[i].c-k_cg.data[0],0.0,0.0);

		//合力の法線方向成分
		Fb_nrm+=dot((L+D),-element[i].normal);

//yz-------------------------------------------*/
		//抗力の加わる方向の取得
		yz_speed=unitize(yz_vel);//速度の大きさの取得と速度ベクトルの正規化
		Drag_vec=-yz_vel;

		//揚力の作用する方向の取得
		Lift_vec=cross(Drag_vec,-element[i].normal);
		Lift_vec=cross(Lift_vec,Drag_vec);
		tmp=unitize(Lift_vec);//正規化

		//迎え角を調べる
		tmp=dot(Drag_vec,element[i].normal);//cos(psi)を求める
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
		CD=search_alpha_CD(psi,k_AR);
		CL=search_alpha_CL(psi,k_AR);

//#define PSI_SP 0.02734
//
//		cw+=PSI_SP;//alpha用の関数をpsiに利用するための修正

		//揚力，抗力を計算
		L=0.5*CL*RHO*(element[i].S*p_l[0])*(yz_speed*yz_speed)*Lift_vec;
		D=0.5*CD*RHO*(element[i].S*p_l[0])*(yz_speed*yz_speed)*Drag_vec;

		//凧全体での合力を計算
		Fb+=L+D;
		Lift[1][0]+=L;
		Drag[1][0]+=D;

		//凧全体でのモーメント総計
		if(0.0<yz_vel.data[1])
		{
			cw=1.0-cw;
		}
		x_wind=Vec3(element[i].cg.data[0], (cw-0.5)*element[i].b+element[i].cg.data[1], 0.0);

		kite_check=Vec3(0.0,(cw)*k_b-k_cg.data[1],0.0);
		tmp_vec=cross(x_wind,L);
		Mb+=tmp_vec;
		tmp_vec=cross(x_wind,D);
		Mb+=tmp_vec;

		Lift[1][1]=Vec3(0.0,(cw)*k_b-k_cg.data[1],0.0);
		Drag[1][1]=Vec3(0.0,(cw)*k_b-k_cg.data[1],0.0);

		//合力の法線方向成分
		Fb_nrm+=dot((L+D),-element[i].normal);

//---------------------------------------------*/
	}
	kite_check=QVRotate(k_orientation,kite_check);
	
	//たわみ計算より投影面積(p_l)を求める
	calc_deflection(Fb_nrm);

	//荷重をローカル系からグローバル系へとコンバート
	k_frc=QVRotate(k_orientation,Fb);

	//モーメント
	k_mom+=Mb;

	//重力(グローバル系で加える)
	Vec3 g_vec=Vec3(0.0,0.0,1.0);
	g_vec*=k_mass*G_ACCE;
	k_frc+=g_vec;

	G[0]=g_vec;//グローバル

	//1step前の張力を合力に加算
	k_frc+=k_T_spring[0];
	T_string[0]=k_T_spring[0];

	//張力によるモーメント
	k_T_spring[0]=QVRotate(k_orientation.inverse(),k_T_spring[0]);//グローバル->ローカル
	Mb=cross(k_s,k_T_spring[0]);//張力によるモーメント
	k_mom+=Mb;
}
//useless now
//calc the force from user
Vec3  KiteSimulation::calc_UI_force(void)
{
	Vec3 UI_vec=Vec3();
	double ce=0.03;

	UI_vec.data[0]=spring_ce[0]*ce;
	UI_vec.data[1]=-spring_ce[2]*ce;
	UI_vec.data[2]=spring_ce[1]*ce;

	return UI_vec;
}

//calc the position
void  KiteSimulation::calc_line_pos(double dt)//糸(糸の自由端が凧の位置に対応)
{
	//calculate wind speed
	int line_e =  0.2;
	double wind_norm=0.0;
	Vec3 wind_vel=Wind;
	
	wind_norm=unitize(wind_vel);
	Vec3 w_s= wind_vel*(wind_norm*wind_norm)*0.02;

	double Inv_dt=1.0/dt;
	int i;
	//糸目位置とのリンク

	line.setEndPoint(k_glb_s_pos);

	//kite.line_pos[0]+=kite3d::calc_UI_force()*dt;
	//k_T_spring[0]	= line.calTension();			//ばねによる引張力
	//k_T_spring[0]	= Vec3(-2.7,0,1);			//ばねによる引張力
	//init the windspeed to the line
	line.setWindSpeed(w_s);

	// update the line
	line.line_update();

	//糸目位置とのリンク
	k_glb_s_pos = line.getEndPoint();
	k_pos=k_glb_s_pos-QVRotate(k_orientation,k_s);
	k_global_vel=(k_pos-k_pos_pre)*5;
	k_pos_pre=k_pos;

//	k_l_check=norm((kite.line_pos[LINE_SP]-kite.line_pos[0]));

}

//ファイル読み込み
int   KiteSimulation::read_file(const string &file_name, vector<double> &datas)
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
void  KiteSimulation::read_file(const string &file_name)
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


//draw
void  KiteSimulation::draw_kite(void)	//draw the kite
{
	// draw line
	glPushMatrix();
		line.draw_Line();
	glPopMatrix();
		//draw teapot
	//glPushMatrix();
	//	static const GLfloat difg[] = { 0.4f, 0.6f, 0.4f, 1.0f };	// 拡散色 : 緑
	//		glMaterialfv(GL_FRONT, GL_DIFFUSE, difg);
	//		glTranslatef(line.getEndPoint().data[0],line.getEndPoint().data[1],line.getEndPoint().data[2]);
	//		glutSolidTeapot(0.1);									//draw sphere
	//glPopMatrix();

	//draw kite shape
	int i=0,j=0,k=0;

	for(i=0;i<k_q_num;i++)
	{
		//glBegin(GL_QUADS);
		glBegin(GL_QUADS);

		//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
		glTexCoord2d(k_tex_cd[element[i].index[0]][1], k_tex_cd[element[i].index[0]][0]);
		glVertex3d(k_global_cd[element[i].index[0]][0], k_global_cd[element[i].index[0]][2], -k_global_cd[element[i].index[0]][1]);

		//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
		glTexCoord2d(k_tex_cd[element[i].index[1]][1], k_tex_cd[element[i].index[1]][0]);
		glVertex3d(k_global_cd[element[i].index[1]][0], k_global_cd[element[i].index[1]][2], -k_global_cd[element[i].index[1]][1]);

		//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
		glTexCoord2d(k_tex_cd[element[i].index[2]][1], k_tex_cd[element[i].index[2]][0]);
		glVertex3d(k_global_cd[element[i].index[2]][0], k_global_cd[element[i].index[2]][2], -k_global_cd[element[i].index[2]][1]);

		//glNormal3d(-kite.element[i].normal[1],kite.element[i].normal[2],kite.element[i].normal[0]);
		glTexCoord2d(k_tex_cd[element[i].index[3]][1], k_tex_cd[element[i].index[3]][0]);
		glVertex3d(k_global_cd[element[i].index[3]][0], k_global_cd[element[i].index[3]][2], -k_global_cd[element[i].index[3]][1]);

		glEnd();
	}
}

void  KiteSimulation::read_table(void)					//.datからテーブルを読み込む
{
}



void  KiteSimulation::draw_options_01(void)	//プログラム確認用オプション
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

	glColor3f ( 0.0f, 0.0f, 1.0f );//blue
	glPushMatrix();
	//糸目
		glTranslated(T_string[1].data[0],T_string[1].data[2],-T_string[1].data[1]);
		glutSolidSphere(0.03,15,15);
	glPopMatrix();

	glEnable(GL_LIGHTING);
}
void  KiteSimulation::draw_options_02(void)	//プログラム確認用オプション
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

	//糸目
	glColor3f ( 1.0f, 0.5f, 0.0f );//orange
	glBegin ( GL_LINES );
		glVertex3d ( T_string[1].data[0],T_string[1].data[2],-T_string[1].data[1]);
		glVertex3d ( T_string[1].data[0]+scale*T_string[0].data[0],T_string[1].data[2]+scale*T_string[0].data[2],-(T_string[1].data[1]+scale*T_string[0].data[1]));
	glEnd ();

	glEnable(GL_LIGHTING);
}
void  KiteSimulation::draw_options_03(void)	//プログラム確認用オプション
{
		double scale=0.1;
	glDisable(GL_LIGHTING);

	glColor3f ( 1.0f, 0.0f, 0.0f );//red
	glBegin ( GL_LINES );
		glVertex3d ( G[1].data[0],G[1].data[2],-G[1].data[1] );
		glVertex3d ( G[1].data[0]+scale*k_frc.data[0],G[1].data[2]+scale*k_frc.data[2],-(G[1].data[1]+scale*k_frc.data[1]) );
	glEnd ();

	glEnable(GL_LIGHTING);
}

