
/*
@fiel glmain.cpp

@brief basic structure of GLUT 

@date 2015-05
*/


#pragma comment (lib, "glew32.lib")

#ifdef _DEBUG
#pragma comment(lib, "libjpegd.lib")
#pragma comment(lib, "libpngd.lib")
#pragma comment(lib, "zlibd.lib")
#else
#pragma comment(lib, "libjpeg.lib")
#pragma comment(lib, "libpng.lib")
#pragma comment(lib, "zlib.lib")
#endif

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// STL
#include <vector>
#include <string>

// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"
#include "rx_trackball.h"

#include "stringKite.h"
#include "rx_shadowmap.h"
#include "macros4cc.h"

// テクスチャ
#include "rx_gltexture.h"
using namespace std;

//define program name
const std::string RX_PROGRAM_NAME = "Fly Kite";

// キューブマップ
rxCubeMapData g_CubeMap;	//!< キューブマップ
bool g_bUseCubeMap;			//!< キューブマップ使用フラグ

// ウィンドウ情報
int g_iWinW = 720;		//!< 描画ウィンドウの幅
int g_iWinH = 720;		//!< 描画ウィンドウの高さ
int g_iWinX = 100;		//!< 描画ウィンドウ位置x
int g_iWinY = 100;		//!< 描画ウィンドウ位置y

// GLSLシェーダ
rxGLSL g_glslShading;

bool g_bIdle = false;	//!< アイドル状態

//! 背景色
double g_fBGColor[4] = {0.8, 0.9, 1.0, 1.0};

//! トラックボールによる視点移動
rxTrackball g_tbView;

int	nsteps = 0;
//extern Vec3 spring_ce;
//extern kite3d::kite_3d kite;

StringKite3D stringKite3d;

Vec3 kiteMidPos;

//function declaration & definition 
void OnExitApp(int id =-1);
void Idle(void);
void CleanGL(void);

//! 反射パラメータ
struct rxReflecParameter
{
	float shininess;	//!< 鏡面反射の広がり(Phong反射モデル)

	// for Fresnel反射モデル
	float eta;
	float bias;
	float power;
	float scale;

	rxReflecParameter()
	{
		shininess = 30.0f;
		eta = 0.97f;
		bias = 0.1f;
		power = 0.98f;
		scale = 1.0f;
	}
};
// 光源
Vec3 g_v3LightPos(1, 2, 0);			//!< 光源位置
rxReflecParameter g_Ref;
//control procedure
/*
idle cantrol,in which, ON is true,false is OFF
*/
void SwitchIdle(int on)
{
	g_bIdle = (on==-1)?!g_bIdle:(on?true:false);
	glutIdleFunc((g_bIdle?Idle:0));
	cout<<"idle"<<(g_bIdle?"on":"off")<<endl;
}

//display on fullscreen / window
void SwitchFullScreen(void)
{
	static int fullscreen =0;
	static int pos0[2] = {0,0};
	static int win0[2] = {500,500};
	if(fullscreen){
		glutPositionWindow(pos0[0],pos0[1]);
		glutReshapeWindow(win0[0],win0[1]);
	}
	else{
		pos0[0] = glutGet(GLUT_WINDOW_X);
		pos0[1] = glutGet(GLUT_WINDOW_Y);
		win0[0] = glutGet(GLUT_WINDOW_WIDTH);
		win0[1] = glutGet(GLUT_WINDOW_HEIGHT);

		glutFullScreen();
	}
	fullscreen ^=1;
}

//Exit the application
void OnExitApp(int id)
{
	CleanGL();
	exit(1);
}

//display function

void Projection(void)
{
	gluPerspective(45.0f,(float)g_iWinW/(float)g_iWinH,0.2f,1000.0f);
}
inline void glLightdv(GLenum light, GLenum pname, const double *light_pos)
{
	GLfloat light_pos_f[4];
	for(int i = 0; i < 2; ++i) light_pos_f[i] = (GLfloat)light_pos[i];
	light_pos_f[3] = 1.0f;
	glLightfv(GL_LIGHT0, GL_POSITION, light_pos_f);
}

void RenderScene(void)
{
	glPushMatrix();

	// 光源位置設定 - 描画物体に対して固定する場合はここで設定
	glLightdv(GL_LIGHT0, GL_POSITION, g_v3LightPos.data);

	// 視点位置
	Vec3 eye_pos(0.0);
	g_tbView.CalLocalPos(eye_pos.data, Vec3(0.0).data);

	// GLSLシェーダをセット
	glUseProgram(g_glslShading.Prog);

	glUniform1f(glGetUniformLocation(g_glslShading.Prog, "etaRatio"),     g_Ref.eta);
	glUniform1f(glGetUniformLocation(g_glslShading.Prog, "fresnelBias"),  g_Ref.bias);
	glUniform1f(glGetUniformLocation(g_glslShading.Prog, "fresnelPower"), g_Ref.power);
	glUniform1f(glGetUniformLocation(g_glslShading.Prog, "fresnelScale"), g_Ref.scale);
	glUniform3f(glGetUniformLocation(g_glslShading.Prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);
	glUniform1i(glGetUniformLocation(g_glslShading.Prog, "envmap"), 0);

	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	//画风筝1
	stringKite3d.draw();
	//画风筝2

	glPushMatrix();
	double mag = 5.0+1.0;
	glTranslated(-mag, -mag, -mag);
	if(fluid::V_field) fluid::draw_velocity(2*mag, 0.1);	// 速度の視覚化
	//画盒子（环境）
	fluid::draw_box(2*mag);
	glPopMatrix();
	glUseProgram(0);
	glPopMatrix();
}

//envent handle
void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPushMatrix();

	g_tbView.Apply();	// マウスによる回転・平行移動の適用

		if(g_bUseCubeMap){
		// キューブマップの描画
		glDisable(GL_LIGHTING);
		glDisable(GL_CULL_FACE);
		glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
		glColor3d(0, 0, 0);
		DrawCubeMap(g_CubeMap, 100.0);
	}

	RenderScene();

	glPopMatrix();

	glutSwapBuffers();
}

//idle handle
void Idle(void)
{
	int n = GRID;
	double visc = VISC;
	double dt = STEP;
	// 最初から風を定常状態にしたい場合
	//if(nsteps == 0){
	//	for(int i = 0; i < 30; ++i){
	//		fluid_get_from_UI(n, dens_prev, u_prev, v_prev, w_prev);
	//		fluid_vel_step(n, u, v, w, u_prev, v_prev, w_prev, visc, Dt);
	//	}
	//}
	
	fluid::get_from_UI(n, g_dens_prev, g_u_prev, g_v_prev, g_w_prev);
	fluid::vel_step(n, g_u, g_v, g_w, g_u_prev, g_v_prev, g_w_prev, visc, dt);
	//风筝1更新
	stringKite3d.update(dt);

	//stringKite.spring_ce = Vec3(0.0, 0.0, 0.0);
	nsteps++;

	glutPostRedisplay();
}

//init the GL
void InitGL(void)
{
	cout <<"OpenGL ver." << glGetString(GL_VERSION) << endl;

	GLenum err = glewInit();
	if(err == GLEW_OK){
		cout<<"GLEW OK : GLew ver. "<< glewGetString(GLEW_VERSION)<<endl;
	}
	else{
		cout << "GLEW Error :"<<glewGetErrorString(err) << endl;
	}

	//multisample
	GLint buf,sbuf;
	glGetIntegerv(GL_SAMPLER_BUFFER,&buf);
	cout<<"number of sample buffers is "<<buf<<endl;
	glGetIntegerv(GL_SAMPLES,&sbuf);
	cout<<"number of samples is " <<sbuf << endl;

	glClearColor((GLfloat)g_fBGColor[0],(GLfloat)g_fBGColor[1],(GLfloat)g_fBGColor[2],1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glEnable(GL_MULTISAMPLE);

	//nomalize the polygon
	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	//set up the light
	static const GLfloat lpos[] = {0.0f,0.0f,1.0f,1.0f};//position of the light
	static const GLfloat difw[] = {0.7f,0.7f,0.7f,1.0f};//diffuse
	static const GLfloat spec[] = {0.2f,0.2f,0.2f,1.0f};//specular
	static const GLfloat ambi[] = {0.1f,0.1f,0.1f,1.0f};//ambient
	glLightfv(GL_LIGHT0,GL_POSITION,lpos);
	glLightfv(GL_LIGHT0,GL_DIFFUSE,difw);
	glLightfv(GL_LIGHT0,GL_SPECULAR,spec);
	glLightfv(GL_LIGHT0,GL_AMBIENT,ambi);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	glShadeModel(GL_SMOOTH);
	
	stringKite3d.readKite();

	//风筝1初始化
//	kite_3d.setStartPos(Vec3(-1,0,0));	
	stringKite3d.setup();
	//kiteMidPos = kite_3d.getKiteMidPos();
	
	//
	////风筝2初始化
	////import data
	//kite_3d2.read_file("alpha_CD_068.dat");
	//kite_3d2.read_file("alpha_CD_148.dat");
	//kite_3d2.read_file("alpha_CL_068.dat");
	//kite_3d2.read_file("alpha_CL_148.dat");
	//kite_3d2.read_file("alpha_x.dat");

	//kite_3d2.setStartPos(kite_3d.getKiteMidPos());
	////kite_3d2.setStartPos(Vec3(1,0,0));
	//kite_3d2.setup();

	//init the trackball
	g_tbView.SetScaling(-15.0f);

		// 流体シミュレータの初期化
	//initialize_fluid_sim
	fluid::allocate_data();
	fluid::clear_data();


	//kite_3d.spring_ce=Vec3();
}

//resize the window
void Resize(int w,int h)
{
	g_iWinW =w;
	g_iWinH = h;
	glViewport(0,0,w,h);
	g_tbView.SetRegion(w,h);

	//set the Parameters of the projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	Projection();

	//set up hte modelview
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/*
set the mouse control button
@param[in] button: mouse button(GLUT_LEFT_BUTTON,GLUT_MIDDLE_BUTTON.GLUT_RIGHT_BUTTON)
@param[in] state: the state of the mouse button(GLUT_UP.GLUT_DOWN)
@param[in] x,y the positon of the mouse button

*/
void Mouse(int button,int state,int x,int y)
{
	if(x<0||y<0) return;

	int mod = glutGetModifiers();		//get the state mode of SHIFT,CTRL,ALT

	if(button == GLUT_LEFT_BUTTON){
		if(state == GLUT_DOWN)
		{
			g_tbView.Start(x,y,mod+1);
		}
		else if(state == GLUT_UP)
		{
			g_tbView.Stop(x,y);
		}
	}
	else if (button == GLUT_MIDDLE_BUTTON){
	}
	else if(button == GLUT_RIGHT_BUTTON)
	{
	}
	glutPostRedisplay();
}

//motion function(put down the button)
//@param[in] x,y :the positon of the mouse
void Motion(int x,int y)
{
	if(x<0 || y<0) return;
	int mod = glutGetModifiers();		//get the state mode of Shift,Ctrl,Alt
	g_tbView.Motion(x,y);
	glutPostRedisplay();
}

//motion function()
void PassiveMotion(int x,int y)
{
	if(x<0 ||y<0) return;
}

/*keybord function
@param[in] key, the type of the key
@param[in] x,y : the current position of the mouse
*/
void Keyboard(unsigned char key, int x, int y)
{
		int mod = glutGetModifiers();	// SHIFT,CTRL,ALTの状態取得

	switch(key){
	case '\033':  // '\033' は ESC の ASCII コード
		OnExitApp();
		break;

	case ' ':	// アニメーション1ステップだけ実行
		Idle();
		break;

	case 's':	// アニメーションON/OFF
		SwitchIdle(-1);
		break;

	case 'F':	// フルスクリーン
		SwitchFullScreen();
		break;

	case 'i': //初期化
		//kite_3d.initialize_sim();
		break;

	case 'x': //x方向の風
		fluid::Z_wind=0;
		if(fluid::X_wind==0){		
			fluid::X_wind=1;
		}
		break;

	case 'z': //z方向の風
		fluid::X_wind=0;
		if(fluid::Z_wind == 0){
			fluid::Z_wind = 1;
		}
		break;

	case 'u': //x方向一様流
		fluid::X_wind = 0;
		fluid::Z_wind = 0;
		break;

	case 'v': //速度の視覚化ON/OFF
		fluid::V_field ^= 1;
		break;

	case 'd':
		fluid::D_tex ^= 1;
		break;

	case 'f':
		fluid::frc_view ^= 1;
		break;

	case 'h':
		cout << "'s' key : simulation start/stop" << endl;
		cout << "'F' key : fullscreen" << endl;
		cout << "'i' key : initialize the scene" << endl;
		cout << "'x','y','z' key : change the wind direction" << endl;
		cout << "'v' key : visualize the wind field" << endl;
		break;

	default:
		break;
	}

	glutPostRedisplay();
}

//special key
void SpecialKey(int key,int x,int y)
{
	double d = 10.0;
	switch(key){
	case GLUT_KEY_LEFT:
		stringKite3d.spring_ce = Vec3(-d, 0.0, 0.0);
		break;

	case GLUT_KEY_RIGHT:
		stringKite3d.spring_ce = Vec3(d, 0.0,  0.0);
		break;

	case GLUT_KEY_UP:
		stringKite3d.spring_ce = Vec3( 0, d, 0.0);
		break;

	case GLUT_KEY_DOWN:
		stringKite3d.spring_ce = Vec3(0.0, -d, 0.0);
		break;

	default:
		break;
	}

	glutPostRedisplay();
}

//clean the GL
void CleanGL(void)
{
	fluid::free_data();
}

//main menu
//@param[in] id :main ID
void OnMainMenu(int id)
{
	Keyboard((unsigned char)id,0,0);
}

//create the right click menu
void InitMenu(void)
{
	//main menu
	glutCreateMenu(OnMainMenu);
	glutAddMenuEntry("------------------------",         -1);
	glutAddMenuEntry("Toggle fullscreen [f]",         'f');
	glutAddMenuEntry("Toggle animation [s]" ,           's');
	glutAddMenuEntry("Step animation [ ]",              ' ');
	glutAddMenuEntry("------------------------",         -1);
	glutAddMenuEntry("Quit [ESC]",                       '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}


int main(int argc, char *argv[])
{
	glutInitWindowPosition(g_iWinX,g_iWinY);
	glutInitWindowSize(g_iWinW,g_iWinH);
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_ACCUM);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(Display);
	glutReshapeFunc(Resize);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutPassiveMotionFunc(PassiveMotion);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialKey);

	SwitchIdle(0);

	InitGL();
	InitMenu();

	Keyboard('h', 0, 0);

	glutMainLoop();

	CleanGL();

	return 0;

}