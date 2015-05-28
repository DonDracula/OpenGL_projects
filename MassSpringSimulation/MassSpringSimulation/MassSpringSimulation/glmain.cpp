/*
	file glmain.cpp
	brief the simlulation of rope using mass-spring method
	date 2015-05
*/

#pragma comment (lib,"glew32.lib")

//include files
//--------------------------------------------------------------------------------

#include "ropesimulator.h"
#include "rx_trackball.h"
//define program name
const string RX_PROGRAM_NAME = "MassSpringSimulation Demo";

//windows size
int g_iWinW = 720;		//Width of the window
int g_iWinH = 720;		//Height of the window
int g_iWinX = 100;		//x position of the window
int g_iWinY = 100;		//y position of the window

bool g_bIdle = false;		//state of the idle
//int g_iDraw = RXD_FACE;	//!< drawing flag

double g_fAng = 0.0;	//!< •`‰æ‰ñ“]Šp

//background color
double g_fBGColor[4] = {0.8,0.9,1.0,1.0};

//rxtrackball,to control the viewpoint
rxTrackball g_tbView;

double g_fDt = 0.03;				//update time

//function declaration & definition 
void OnExitApp(int id =-1);
void Idle(void);
void Timer(int value);
void CleanGL(void);

RopeSimulator* ropeSimulator = new RopeSimulator(
													80,						//40 particles(Masses)
													0.05f,					//each mass has a weight of 30g
													1000.0f,					//springConstant in the rope
													0.05f,					//normal lenght of springs in the rope
													0.2f,					//inner firction constant of spring
													Vec3(0,-9.81f,0),		//gravitational acceleration
													0.02f,					//air friction constant
													100.0f,					//ground repulsion constant
													0.2f,					//ground friction constant
													2.0f,					//ground absorption constant
													-2.5f);					//ground height

//control procedure
//idle control,in which,ON is true,false is OFF
void SwitchIdle(int on)
{
	g_bIdle = (on==-1)?!g_bIdle:(on?true:false);
	if(g_bIdle) glutTimerFunc(g_fDt*1000,Timer,0);
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

	//Draw a plane to represent the ground
	glBegin(GL_QUADS);
		glColor3ub(0,0,255);						//set color to light blue
		glVertex3f(20,ropeSimulator->getGroundHeight(),20);
		glVertex3f(-20,ropeSimulator->getGroundHeight(),20);
		glColor3ub(0,0,0);							//set color to black
		glVertex3f(-20,ropeSimulator->getGroundHeight(),-20);
		glVertex3f(20,ropeSimulator->getGroundHeight(),-20);
	glEnd();

	//start drawing shadow of the rope
	glColor3ub(0,0,0);
	for(int index = 0;index<ropeSimulator->getNumOfMasses()-1;++index)
	{
		Mass* mass1 = ropeSimulator->getMass(index);
		Vec3* pos1 = &mass1->getPos();

		Mass* mass2 = ropeSimulator->getMass(index);
		Vec3* pos2 = &mass2->getPos();

		glLineWidth(2);
		glBegin(GL_LINES);
			glVertex3f(pos1->data[0],ropeSimulator->getGroundHeight(),pos1->data[2]);			//draw shadow at groundheight
			glVertex3f(pos2->data[0],ropeSimulator->getGroundHeight(),pos2->data[2]);
		glEnd();
	}
	//draw shadow ends here

	//start drawing the rope
	glColor3ub(255,255,0);									//set color to yellow
	for(int index=0;index<ropeSimulator->getNumOfMasses()-1;++index)
	{
		Mass* mass1=ropeSimulator->getMass(index);
		Vec3* pos1= &mass1->getPos();

		Mass* mass2 = ropeSimulator->getMass(index+1);
		Vec3* pos2= &mass2->getPos();

		glLineWidth(4);
		glBegin(GL_LINES);
			glVertex3dv(pos1->data);
			glVertex3dv(pos2->data);
		glEnd();

	}

	for(int count=0; count<ropeSimulator->getNumOfMasses();++count)
	{
		Mass* mass1=ropeSimulator->getMass(count);
		Vec3 &pos1 = mass1->getPos();
		glPushMatrix();
		glTranslatef(pos1[0],pos1[1],pos1[2]);
		glutSolidSphere(0.02,32,32);									//draw sphere
		glPopMatrix();
	}
	//drawing teh rope ends here

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

	ropeSimulator->operate(dt);
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
		ropeSimulator->operate(dt);

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

	ropeSimulator->getMass(ropeSimulator->getNumOfMasses() -1)->getVel().data[2]= 10.0f;

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
	Vec3 ropeConnectionVel(0.0);

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

	case 'F':   //Fullscreen -> Shift+f
		SwitchFullScreen();
		break;

	case 'd':
		ropeConnectionVel.data[0] +=3.0f;
		break;
	case 'a':
		ropeConnectionVel.data[0] -=3.0f;
		break;
	case 'w':
		ropeConnectionVel.data[2] -=3.0f;
		break;
	case 's':
		ropeConnectionVel.data[2] +=3.0f;
		break;
	case 'z':
		ropeConnectionVel.data[1] +=3.0f;
		break;
	case 'x':
		ropeConnectionVel.data[1] -=3.0f;
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
	
	ropeSimulator->setRopeConnectionVel(ropeConnectionVel);			// Set The Obtained ropeConnectionVel In The Simulation
	
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
	ropeSimulator->release();					// Release The ropeSimulation
	delete(ropeSimulator);						// Delete The ropeSimulation
	ropeSimulator = NULL;
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