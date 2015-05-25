/*
@fiel glmain.cpp

@brief the simulation of rope using mass-spring method

@date 2015-05
*/

#pragma comment (lib,"glew32.lib")

//include files
//--------------------------------------------------------------------------------

#include "spring.h"
#include "rx_trackball.h"
//define program name
const string RX_PROGRAM_NAME = "Feel My Love";

// drawing flag
enum
{
	RXD_VERTEX		= 0x0001,	//!< ’¸“_•`‰æ
	RXD_EDGE		= 0x0002,	//!< ƒGƒbƒW•`‰æ
	RXD_FACE		= 0x0004,	//!< –Ê•`‰æ
	RXD_NORMAL		= 0x0008,	//!< –@ü•`‰æ
};
const string RX_DRAW_STR[] = {
	"Vertex",	"v", 
	"Edge",		"e", 
	"Face",		"f", 
	"Normal",	"n", 
	"-1"
};

//windows size
int g_iWinW = 720;		//Width of the window
int g_iWinH = 720;		//Height of the window
int g_iWinX = 100;		//x position of the window
int g_iWinY = 100;		//y position of the window

bool g_bIdle = false;		//state of the idle
int g_iDraw = RXD_FACE;	//!< drawing flag

double g_fAng = 0.0;	//!< •`‰æ‰ñ“]Šp

//background color
double g_fBGColor[4] = {0.8,0.9,1.0,1.0};

//rxtrackball,to control the viewpoint
rxTrackball g_tbView;

//function declaration & definition 
void OnExitApp(int id =-1);
void Idle(void);
void CleanGL(void);
void drawHeart(float size );


RopeSimulation* ropeSimulation = new RopeSimulation(
													40,						//40 particles(Masses)
													0.05f,					//each mass has a weight of 30g
													10000.0f,				//springConstant in the rope
													0.1f,					//normal lenght of springs in the rope
													0.2f,					//inner firction constant of spring
													Vec3(0,-9.81f,0),		//gravitational acceleration
													0.02f,					//air friction constant
													100.0f,					//ground repulsion constant
													0.2f,					//ground friction constant
													2.0f,					//ground absorption constant
													-1.5f);					//ground height
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

void RenderScene(void)
{
	glPushMatrix();

	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

	glColorMaterial(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	//draw object------------------------------------------------------------

	//glutSolidTeapot(1.0f);		//we draw a teapot here

	/* // Draw A Plane To Represent The Ground (Different Colors To Create A Fade)
	glBegin(GL_QUADS);
		glColor3ub(0,0,255);								// Set Color To Light Blue
		glVertex3f(20,ropeSimulation->groundHeight,20);
		glVertex3f(-20,ropeSimulation->groundHeight,20);
		glColor3ub(0,0,0);									// Set Color To Black
		glVertex3f(-20,ropeSimulation->groundHeight,-20);
		glVertex3f(20,ropeSimulation->groundHeight,-20);
	glEnd();
	*/
	/*
		// Start Drawing Shadow Of The Rope
	glColor3ub(0,0,0);
	for(int a=0;a<ropeSimulation->numOfMasses-1;++a)
	{
		Mass* mass1 = ropeSimulation->getMass(a);
		Vec3* pos1 = &mass1->pos;

		Mass* mass2 = ropeSimulation->getMass(a+1);
		Vec3* pos2=&mass2->pos;

		glLineWidth(2);
		glBegin(GL_LINES);
			glVertex3f(pos1->data[0],ropeSimulation->groundHeight,pos1->data[2]);		// Draw Shadow At groundHeight
			glVertex3f(pos2->data[0],ropeSimulation->groundHeight,pos2->data[2]);
		glEnd();	
	}
	// Drawing Shadow Ends Here.
	*/
	// Start Drawing The Rope.
	glColor3ub(255,255,0);								//set color to yellow
	for(int a=0;a<ropeSimulation->numOfMasses-1;++a)
	{
		Mass* mass1=ropeSimulation->getMass(a);
		Vec3* pos1 = &mass1->pos;

		Mass* mass2=ropeSimulation->getMass(a+1);
		Vec3* pos2 = &mass2->pos;

		glLineWidth(4);
		glBegin(GL_LINES);
			glVertex3f(pos1->data[0],pos1->data[1],pos1->data[1]);
			glVertex3f(pos2->data[0],pos2->data[2],pos2->data[2]);
		glEnd();
	}
	//drawing the rope ends here

	glPopMatrix();
}

//envent handle
void Display(void)
{

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	g_tbView.Apply();		//mouse control, which allows to rotate and move

	glPushMatrix();
	glRotated(g_fAng,0,1,0);
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
	g_fAng +=0.5f;
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

	ropeSimulation->getMass(ropeSimulation->numOfMasses -1)->vel.data[2] = 10.0f;

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
	g_tbView.SetScaling(-5.0f);
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
	int mod = glutGetModifiers();		//the state(Shift,Ctrl,Alt)
	Vec3 ropeConnectionVel;

	switch(key){
	case '\033': // the ASCII code of ESC
		OnExitApp();
		break;

	case ' ':	//idle
		Idle();
		break;

	case 'g':	//animation ON/OFF
		SwitchIdle(-1);
		break;

	case 'F':   //Fullscreen
		SwitchFullScreen();
		break;

	case 'D':
		ropeConnectionVel.data[0] +=3.0f;
		break;
	case 'A':
		ropeConnectionVel.data[0] -=3.0f;
		break;
	case 'W':
		ropeConnectionVel.data[2] -=3.0f;
		break;
	case 'S':
		ropeConnectionVel.data[2] +=3.0f;
		break;
	case 'Z':
		ropeConnectionVel.data[1] +=3.0f;
		break;
	case 'X':
		ropeConnectionVel.data[1] -=3.0f;
		break;

	default:	
		{
			int idx = 0;
			while(RX_DRAW_STR[2*idx]!="-1"&&(int)RX_DRAW_STR[2*idx+1][0]!=key) idx++;
			if(RX_DRAW_STR[2*idx]!="-1"){
				g_iDraw=(0x01<<idx);
			}
		}
		break;
	}
	
	ropeSimulation->setRopeConnectionVel(ropeConnectionVel);			// Set The Obtained ropeConnectionVel In The Simulation
	float dt=0.01f;
//	float dt = milliseconds / 1000.0f;										// Let's Convert Milliseconds To Seconds

//	float maxPossible_dt = 0.002f;											// Maximum Possible dt Is 0.002 Seconds
																			// This Is Needed To Prevent Pass Over Of A Non-Precise dt Value

  //	int numOfIterations = (int)(dt / maxPossible_dt) + 1;					// Calculate Number Of Iterations To Be Made At This Update Depending On maxPossible_dt And dt
	//if (numOfIterations != 0)												// Avoid Division By Zero
//		dt = dt / numOfIterations;											// dt Should Be Updated According To numOfIterations

	//for (int a = 0; a < numOfIterations; ++a)								// We Need To Iterate Simulations "numOfIterations" Times
		ropeSimulation->operate(dt);

	glutPostRedisplay();
}

//special key
void SpecialKey(int key,int x,int y)
{
	Vec3 ropeConnectionVel;
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
	ropeSimulation->release();					// Release The ropeSimulation
	delete(ropeSimulation);						// Delete The ropeSimulation
	ropeSimulation = NULL;
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

	glutMainLoop();

	CleanGL();
	return 0;
	
}