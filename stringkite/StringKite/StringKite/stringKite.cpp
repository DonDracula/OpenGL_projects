
// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>
//kite头文件
#include "stringKite.h"
#include "macros4cc.h"

//---------------------------------------------------------------------------------
//凧シミュレータ
//---------------------------------------------------------------------------------
StringKite3D::StringKite3D()
{
}

void StringKite3D::readKite()
{

	kite_Shape.read_file("alpha_CD_068.dat");
	kite_Shape.read_file("alpha_CD_148.dat");
	kite_Shape.read_file("alpha_CL_068.dat");
	kite_Shape.read_file("alpha_CL_148.dat");
	kite_Shape.read_file("alpha_x.dat");
	//kite 2
	kite_Shape2.read_file("alpha_CD_068.dat");
	kite_Shape2.read_file("alpha_CD_148.dat");
	kite_Shape2.read_file("alpha_CL_068.dat");
	kite_Shape2.read_file("alpha_CL_148.dat");
	kite_Shape2.read_file("alpha_x.dat");

	for(int i =0;i<KITE_NUMBER;i++)
	{
		kite3d_shape[i].read_file("alpha_CD_068.dat");
		kite3d_shape[i].read_file("alpha_CD_148.dat");
		kite3d_shape[i].read_file("alpha_CL_068.dat");
		kite3d_shape[i].read_file("alpha_CL_148.dat");
		kite3d_shape[i].read_file("alpha_x.dat");
	}

}
//初始化风筝
void StringKite3D::setup()
{
	kite_String.setup();
	kite_Shape.setup(kite_String.lastParticle);
	
	kite_Shape2.setup(kite_String.midParticle);
///kite_Shape2.getKiteMidPos();

	for(int i = 0; i< KITE_NUMBER;i++)
	{
	
		kite3d_shape[i].setup(kite_String.getParticlePos(i));
	}
}
 
Vec3 StringKite3D::calc_UI_force(void)
{
	Vec3 UI_vec=Vec3();
	double ce=0.03;

	UI_vec.data[0]=spring_ce[0]*ce;
	UI_vec.data[1]=spring_ce[1]*ce;
	UI_vec.data[2]=spring_ce[2]*ce;

	return UI_vec;
}

/*!
 * @note シミュレーションを進める
 */
void StringKite3D::update(double dt)
{
	int i=0;
//------------------------
	//準備
	set_wind(dt);//風のセット

	//kite_Shape2.update1(dt);
	//kite_Shape.update1(dt);

	for(int i=0; i<KITE_NUMBER;i++)
	{
		kite3d_shape[i].update1(dt);
	}

	//凧糸
	//更新风筝线位置
	update_line(dt);
//	kite_Shape2.update2(dt,Wind);
//	kite_Shape.update2(dt,Wind);

	for(int i=0; i<KITE_NUMBER;i++)
	{
		kite3d_shape[i].update2(dt,Wind);
	}

}


void StringKite3D::update_line(double dt)
{
	//string_mid_point = (int)(kite_String.GetNumOfVertices()-1)/2;
	//set windspeed of the string
	Vec3 wind_vel=Wind;
	double wind_norm=0.0;
	wind_norm=unitize(wind_vel);
	kite_String.setWindSpeed(wind_vel*(wind_norm*wind_norm)*LINE_E);

	kite_String.FixVertex(0,kite_String.firstParticle);
	//糸目位置とのリンク, 风筝中心的速度位置赋予风筝线
//kite 2(...............。。。。。。。。。。。。。。。。。。.....................................。待解决)
	//kite_String.setMidVel(kite_Shape2.kite.global_vel);
	//kite_String.FixVertex(string_mid_point,kite_Shape2.kite.glb_s_pos);

	//kite 1

	kite_String.FixVertex(kite_String.GetNumOfVertices()-1,kite3d_shape[5].kite.glb_s_pos);
	//kite_String.setLastVel(kite3d_shape[5].kite.global_vel);

	for(int i= 0;i<KITE_NUMBER;i++)
	{
		kite_String.setIndex_Vel(kite_String.getIndex(i),kite3d_shape[i].kite.global_vel);
	}


	// for Haptic device interface
	kite_String.firstParticle +=calc_UI_force()*dt;

	//风筝线更新
	kite_String.update();

	//kite1糸目位置とのリンク
	//kite_Shape.getLinked(dt,kite_String.lastParticle);
	//kite_Shape2.getLinked(dt,kite_String.midParticle);

	for(int i= 0;i<KITE_NUMBER;i++)
	{
		kite3d_shape[i].getLinked(dt,kite_String.getParticlePos(i));

		kite3d_shape[i].T_string[0] = kite_String.calTension(kite_String.getIndex(i));
	}
	//
	//kite2(...............。。。。。。。。。。。。。。。)
//	kite_Shape2.kite.T_spring[0]=kite_String.calTension(string_mid_point);
	//kite_Shape.kite.T_spring[0]=kite_String.calTension(kite_String.GetNumOfVertices()-1);


//-----------------------------------------------------------------*/

}


/*!
 * @note 風のセット
 */
 
void StringKite3D::set_wind(double dt)
{
#define WIND_STR 6.0//*fabs(sin(dt*StepNo))

	/*/
	//if(StepNo<400)
	if(Z_wind==1)//Z_wind==1
	{
		Wind=Vec3(WIND_STR,0.0,0.0);
	}
	//else if(StepNo>=400)
	else if(X_wind==1)
	{
		Wind=Vec3(0.0,WIND_STR,0.0);
	}
	//*/

//*----------------------------------------------------
	int i,j,k;	//カウンタ
	int N=GRID;	//グリッド数
	double h;	//グリッド幅

	h = (kite_String.Length+1.0)*2.0/(double)N;

	double x,y,z;
	double ef=1.0;

	Vec3 kite_pos=Vec3(-kite_Shape.kite.pos.data[1],kite_Shape.kite.pos.data[2],kite_Shape.kite.pos.data[0]);

	for(i=1;i<=N;i++){//x座標
		if( -kite_String.Length+(i-1)*h<kite_pos.data[0] && kite_pos.data[0]<=-kite_String.Length+i*h ){
			for(j=1;j<=N;j++){//y座標
				if( -kite_String.Length+(j-1)*h<kite_pos.data[1] && kite_pos.data[1]<=-kite_String.Length+j*h ){
					for(k=1;k<=N;k++){//z座標
						if( -kite_String.Length+(k-1)*h<kite_pos.data[2] && kite_pos.data[2]<=-kite_String.Length+k*h ){

							x= g_w[IX(i,j,k)];
							y=-g_u[IX(i,j,k)];
							z= g_v[IX(i,j,k)];
							
							Wind=ef*kite_String.Length*2.0*Vec3(x,y,z);//気流ベクトルセット
							
						}
					}
				}
			}
		}
	}

//----------------------------------------------------*/
}


//画图函数
void StringKite3D::draw(void)
{
	kite_String.draw();
	//kite_Shape.draw();
	//kite_Shape2.draw();

	for(int i= 0;i<KITE_NUMBER;i++)
	{
		kite3d_shape[i].draw();
	}
}
