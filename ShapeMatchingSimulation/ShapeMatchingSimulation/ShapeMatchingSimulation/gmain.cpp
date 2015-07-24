/*
file glmain.cpp
brief the simlulation of rope using shape-matching method
date 2015-06
*/

#pragma comment (lib,"glew32.lib")

//include files
//--------------------------------------------------------------------------------
#include <vector>
#include "rx_utility.h"
#include "deformable.h"
#include "rx_trackball.h"
//define program name
const string RX_PROGRAM_NAME = "ShapeMatchingSimulation Demo";

//windows size
int g_iWinW = 720;		//Width of the window
int g_iWinH = 720;		//Height of the window
int g_iWinX = 100;		//x position of the window
int g_iWinY = 100;		//y position of the window

bool g_bIdle = false;		//state of the idle
//int g_iDraw = RXD_FACE;	//!< drawing flag

double g_fAng = 0.0;	//!< 描画回転角

//background color
double g_fBGColor[4] = {0.8,0.9,1.0,1.0};

//rxtrackball,to control the viewpoint
rxTrackball g_tbView;

double g_fDt = 0.03;				//update time

//overlappiing regions
struct rxObject2D
{
	Deformable *deform;
	int vstart, vend;			//!< 全体の頂点数における位置
};
vector<rxObject2D> g_Obj;

//function declaration & definition 
void OnExitApp(int id =-1);
void Idle(void);
void Timer(int value);
void CleanGL(void);
m2Vector pos1(-1.0f,0.5f),pos2(1.0f,0.5f),pos3(1.0f,1.5f),pos4(-1.0f,1.5f);

//Deformable *mDeformable;

//init the object
void InitObject(void)
{
	rxObject2D obj;
	obj.deform = new Deformable();
	obj.vstart = 0;
	obj.vend = 3;

//	obj.deform->addVertex(pos1,0.01);
//	obj.deform->addVertex(pos2,0.01);
//	obj.deform->addVertex(pos3,0.01);
//	obj.deform->addVertex(pos4,0.01);

	obj.deform->params.setDefaults();
	char filename[256];
	strcpy(filename, "scene1.txt");
	obj.deform->loadFromFile(filename);				//load the points from file

	//set the bound one region
	obj.deform->params.bounds.min.x = -2;
	obj.deform->params.bounds.min.y = -2;

	obj.deform->params.bounds.max.x = 2;
	obj.deform->params.bounds.max.y = 2;
	obj.deform->params.timeStep = 0.02f;

	g_Obj.push_back(obj);
	cout<<"size"<<g_Obj.size();
}

//control procedure
//idle control,in which,ON is true,false is OFF
void SwitchIdle(int on)
{
	g_bIdle = (on == -1) ? !g_bIdle : (on ? true : false);
	glutIdleFunc((g_bIdle ? Idle : 0));
	cout << "idle " << (g_bIdle ? "on" : "off") << endl;
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
	gluPerspective(30.0f,(float)g_iWinW/(float)g_iWinH,0.2f,1000.0f);
}

void RenderScene(void)
{
	glPushMatrix();
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

	glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	static const GLfloat difg[] = { 0.4f, 0.6f, 0.4f, 1.0f };	// 拡散色 : 緑
	//draw object------------------------------------------------------------

	glColor3f(0.0, 1.0, 1.0);

	// ground
	glPushMatrix();
	glMaterialfv(GL_FRONT, GL_DIFFUSE, difg);
	glTranslatef(0, -2.5, 0);
	glScaled(10.0, 0.2, 10.0);
	glutSolidCube(1.0);
	glPopMatrix();

	//draw shadow ends here

	glLineWidth(4);

	//start drawing the rope
	glColor3ub(255,255,0);									//set color to yellow

	glLineWidth(4);
	glPointSize(4);
	
	glBegin(GL_POINTS);

	for(int index=0;index<g_Obj[0].deform->getNumVertices();++index)
	{	
		m2Vector m =  g_Obj[0].deform->getVertexPos(index);
		glVertex2f(g_Obj[0].deform->getVertexPos(index).x,g_Obj[0].deform->getVertexPos(index).y);
	}
	/*
	for(int index=0;index<g_Obj[0].deform->getNumVertices();index++)
	{
		glPushMatrix();
		glTranslatef(g_Obj[0].deform->getVertexPos(index).x,g_Obj[0].deform->getVertexPos(index).y,0);
		glutSolidSphere(0.1,32,32);									//draw sphere
		glPopMatrix();
	}
	*/
	glEnd();

	//drawing the rope ends here

	glPopMatrix();
}

//event handle
void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	g_tbView.Apply();		//mouse control, which allows to rotate and move

	glPushMatrix();
	//glRotated(g_fAng,0,1,0);
	RenderScene();
	glPopMatrix();

	glPopMatrix();

	//word 
	vector<string> strs;
	strs.push_back("\"s\" key : idle on/off");
	strs.push_back("SHIFT+\"f\" key : fullscreen on/off" );
	strs.push_back("\"v\",\"e\",\"f\" key : switch vertex,edge,face drawing");
	//glColor3d(1.0,1.0,1.0);
	//drawString();                 //draw the string view of the teapot

	glutSwapBuffers();
}

//idle handle
void Idle(void)
{
	float dt = 0.002;

	g_Obj[0].deform->timeStep();
	glutPostRedisplay();
}

//timer handle
void Timer(int value)
{
	float dt = g_fDt;
	float max_dt = 0.002;										//maximun possible dt is 0.002 seconds

	int numOfIterations = (int)(dt/max_dt)+1;					//calculate number if iteratins to be made at this update depending on max_dt an dt
	if(numOfIterations !=0)										//avoid division by zero
		dt = dt/numOfIterations;								//dt should be updated according to numOfTerations

	for(int i = 0;i<numOfIterations;++i)						//iterate simulations "numOfIterations" times
		g_Obj[0].deform->timeStep();

	glutPostRedisplay();
	glutTimerFunc(g_fDt*1000,Timer,0);							//call the glutTimerfunc every g_fDt*1000(30)millisecond
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

	glClearColor((GLfloat)g_fBGColor[0],(GLfloat)g_fBGColor[1],(GLfloat)g_fBGColor[2],1.0f);		//background
	glClearDepth(1.0f);										//depth buffer setup

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

	glShadeModel(GL_SMOOTH);							// Select Smooth Shading
	glHint (GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);		// Set Perspective Calculations To Most Accurate

	//init the trackball
	g_tbView.SetScaling(-10.0f);
	InitObject();

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
	//	if(x<0 || y<0) return;
	//	int mod = glutGetModifiers();		//get the state mode of Shift,Ctrl,Alt
	//	g_tbView.Motion(x,y);

	/*			Vec3 ray_from, ray_to;
	Vec3 init_pos = Vec3(0.0);
	g_tbView.CalLocalPos(ray_from, init_pos);
	g_tbView.GetRayTo(x, y, 45, ray_to);

	Vec3 dir = Unit(ray_to-ray_from);	// 視点からマウス位置へのベクトル
	m2Vector dddd=  m2Vector(dir.data[0],dir.data[1]);

	int v = mDeformable->getNumVertices();

	m2Vector cur_pos =mDeformable->getVertexPos(1);
	cout<<"numofVertices[1] position:" << cur_pos.x<<"and"<<cur_pos.y<<endl;
	m2Vector new_pos = m2Vector(ray_from.data[0],ray_from[1])+dddd*10;
	cout<<"new  position~~~~~~~~~~~~~~~~~~:" << new_pos.x<<"and"<<new_pos.y<<endl;
	//	g_vObj[g_iPickedObj].deform->FixVertex(1, new_pos);
	mDeformable->fixVertex(1,new_pos);
	*/
	//	glutPostRedisplay();
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
	int mod = glutGetModifiers();		//the state(Shift,Ctrl,Alt)


	switch(key){
	case '\033': // the ASCII code of ESC
		OnExitApp();
		break;

	case ' ':	
		break;

	case 'g':	//animation ON/OFF
		SwitchIdle(-1);
		break;

	case 'F':   //Fullscreen -> Shift+f
		SwitchFullScreen();
		break;

	case 'd':
		//Idle();
		break;
	case 'a':

		break;
	case 'w':

		break;
	case 's':

		break;
	case 'z':

		break;
	case 'x':

		break;

	default:	
		{
			int idx = 0;
			//		while(RX_DRAW_STR[2*idx]!="-1"&&(int)RX_DRAW_STR[2*idx+1][0]!=key) idx++;
			//	if(RX_DRAW_STR[2*idx]!="-1"){
			//			g_iDraw=(0x01<<idx);
			//		}
		}
		break;
	}

	glutPostRedisplay();
}

//special key
void SpecialKey(int key,int x,int y)
{

	switch(key){
	case GLUT_KEY_LEFT:

		break;

	case GLUT_KEY_RIGHT:

		break;

	case GLUT_KEY_UP:

		break;

	case GLUT_KEY_DOWN:

		break;

	default:
		break;
	}

	glutPostRedisplay();
}

//clean the GL
void CleanGL(void)
{

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

	SwitchIdle(1);

	InitGL();
	InitMenu();

	glutMainLoop();

	CleanGL();
	return 0;

}