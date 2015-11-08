/*! 
  @file glmain.cpp
	
  @brief GLUT基本構成
 
  @author Makoto Fujisawa
  @date 2010-02
*/

#pragma comment (lib, "glew32.lib")


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "utils.h"
#include "rx_trackball.h"
#include "Leap.h"

using namespace Leap;
//-----------------------------------------------------------------------------
// 定義/定数
//-----------------------------------------------------------------------------
const string RX_PROGRAM_NAME = "gl_simple";

// 描画フラグ
enum
{
	RXD_VERTEX		= 0x0001,	//!< 頂点描画
	RXD_EDGE		= 0x0002,	//!< エッジ描画
	RXD_FACE		= 0x0004,	//!< 面描画
	RXD_NORMAL		= 0x0008,	//!< 法線描画
};
const string RX_DRAW_STR[] = {
	"Vertex",	"v", 
	"Edge",		"e", 
	"Face",		"f", 
	"Normal",	"n", 
	"-1"
};


//-----------------------------------------------------------------------------
// グローバル変数
//-----------------------------------------------------------------------------
// ウィンドウ情報
int g_iWinW = 720;		//!< 描画ウィンドウの幅
int g_iWinH = 720;		//!< 描画ウィンドウの高さ
int g_iWinX = 100;		//!< 描画ウィンドウ位置x
int g_iWinY = 100;		//!< 描画ウィンドウ位置y

bool g_bIdle = false;	//!< アイドル状態
int g_iDraw = RXD_FACE;	//!< 描画フラグ

double g_fAng = 0.0;	//!< 描画回転角

//! 背景色
double g_fBGColor[4] = {0.0, 0.0, 0.0, 1.0};

//! トラックボールによる視点移動
rxTrackball g_tbView;


//-----------------------------------------------------------------------------
// 関数プロトタイプ宣言
//-----------------------------------------------------------------------------
void OnExitApp(int id = -1);
void Idle(void);
void CleanGL(void);



//-----------------------------------------------------------------------------
// アプリケーション制御
//-----------------------------------------------------------------------------

/*!
 * アイドル関数のON/OFF
 * @param[in] on trueでON, falseでOFF
 */
void SwitchIdle(int on)
{
	g_bIdle = (on == -1) ? !g_bIdle : (on ? true : false);
	glutIdleFunc((g_bIdle ? Idle : 0));
	cout << "idle " << (g_bIdle ? "on" : "off") << endl;
}

/*!
 * フルスクリーン/ウィンドウ表示の切り替え
 */
void SwitchFullScreen(void)
{
	static int fullscreen = 0;		// フルスクリーン状態
	static int pos0[2] = { 0, 0 };
	static int win0[2] = { 500, 500 };
	if(fullscreen){
		glutPositionWindow(pos0[0], pos0[1]);
		glutReshapeWindow(win0[0], win0[1]);
	}
	else{
		pos0[0] = glutGet(GLUT_WINDOW_X);
		pos0[1] = glutGet(GLUT_WINDOW_Y);
		win0[0] = glutGet(GLUT_WINDOW_WIDTH);
		win0[1] = glutGet(GLUT_WINDOW_HEIGHT);

		glutFullScreen();
	}
	fullscreen ^= 1;
}

/*!
 * アプリケーション終了 兼 Quitボタンのイベントハンドラ
 * @param[in] id ボタンのイベントID
 */
void OnExitApp(int id)
{
	CleanGL();

	exit(1);
}


//-----------------------------------------------------------------------------
// 描画関数など
//-----------------------------------------------------------------------------
/*!
 * 透視投影変換
 */
void Projection(void)
{
	gluPerspective(45.0f, (float)g_iWinW/(float)g_iWinH, 0.2f, 1000.0f);
	//glOrtho(-1, 1, -1, 1, -1, 1);
	//gluOrtho2D(0, g_iWinW, 0, g_iWinH);
}

void RenderScene(void)
{
	glPushMatrix();

	glEnable(GL_LIGHTING);

	switch(g_iDraw){
	case RXD_FACE: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); break;
	case RXD_EDGE: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); break;
	case RXD_VERTEX: glPolygonMode(GL_FRONT_AND_BACK, GL_POINT); break;
	}

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// オブジェクト描画
	glColor3d(1.0, 1.0, 1.0);
	glutSolidTeapot(1.0);

	glPopMatrix();
}


//------------------------------------------------------------------
// OpenGLキャンバス用のイベントハンドラ
//------------------------------------------------------------------
/*!
 * 再描画イベント処理関数
 */
void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	g_tbView.Apply();	// マウスによる回転・平行移動の適用

	glPushMatrix();
	glRotated(g_fAng, 0, 1, 0);
	RenderScene();
	glPopMatrix();

	glPopMatrix();

	// 文字列描画
	vector<string> strs;
	strs.push_back("\"s\" key : idle on/off");
	strs.push_back("SHIFT+\"f\" key : fullscreen on/off");
	strs.push_back("\"v\",\"e\",\"f\" key : switch vertex,edge,face drawing");
	glColor3d(1.0, 1.0, 1.0);
	DrawStrings(strs, g_iWinW, g_iWinH);

	glutSwapBuffers();
}


/*! 
 * リサイズイベント処理関数
 * @param[in] w キャンバス幅(ピクセル数)
 * @param[in] h キャンバス高さ(ピクセル数)
 */
void Resize(int w, int h)
{
	g_iWinW = w;
	g_iWinH = h;

	glViewport(0, 0, w, h);
	g_tbView.SetRegion(w, h);

	// 透視変換行列の設定
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	Projection();

	// モデルビュー変換行列の設定
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

/*!
 * マウスイベント処理関数
 * @param[in] button マウスボタン(GLUT_LEFT_BUTTON,GLUT_MIDDLE_BUTTON,GLUT_RIGHT_BUTTON)
 * @param[in] state マウスボタンの状態(GLUT_UP, GLUT_DOWN)
 * @param[in] x,y マウス座標(スクリーン座標系)
 */
void Mouse(int button, int state, int x, int y)
{
	if(x < 0 || y < 0) return;

	int mod = glutGetModifiers();	// SHIFT,CTRL,ALTの状態取得

	if(button == GLUT_LEFT_BUTTON){
		if(state == GLUT_DOWN){
			g_tbView.Start(x, y, mod+1);
		}
		else if(state == GLUT_UP){
			g_tbView.Stop(x, y);
		}
	}
	else if(button == GLUT_MIDDLE_BUTTON){
	}
	else if(button == GLUT_RIGHT_BUTTON){
	}

	glutPostRedisplay();
}

/*!
 * モーションイベント処理関数(マウスボタンを押したままドラッグ)
 * @param[in] x,y マウス座標(スクリーン座標系)
 */
void Motion(int x, int y)
{
	if(x < 0 || y < 0) return;
	int mod = glutGetModifiers();	// SHIFT,CTRL,ALTの状態取得
	g_tbView.Motion(x, y);
	glutPostRedisplay();
}

/*!
 * モーションイベント処理関数(マウスボタンを押さない移動)
 * @param[in] x,y マウス座標(スクリーン座標系)
 */
void PassiveMotion(int x, int y)
{
	if(x < 0 || y < 0) return;
}


/*! 
 * アイドルイベント処理関数
 */
void Idle(void)
{
	g_fAng += 0.5;
	glutPostRedisplay();
}

/*!
 * キーボードイベント処理関数
 * @param[in] key キーの種類
 * @param[in] x,y キーが押されたときのマウス座標(スクリーン座標系)
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

	default:	// 描画フラグの処理
		{
			int idx = 0;
			while(RX_DRAW_STR[2*idx] != "-1" && (int)RX_DRAW_STR[2*idx+1][0] != key) idx++;
			if(RX_DRAW_STR[2*idx] != "-1"){
				//g_iDraw ^= (0x01 << idx);	// エッジや面を同時に描画する際はこちらを使用
				g_iDraw = (0x01 << idx);
			}
		}
		break;
	}

	glutPostRedisplay();
}

class LeapEventListener : public Listener {
  public:
    virtual void onConnect(const Controller&);
    virtual void onDisconnect(const Controller&);
    virtual void onFrame(const Controller&);
};


void LeapEventListener::onConnect(const Controller& controller) {
    std::cout << "Connected" << std::endl;
    // Enable gestures, set Config values:
    controller.enableGesture(Gesture::TYPE_SWIPE);
    controller.config().setFloat("Gesture.Swipe.MinLength", 200.0);
    controller.config().save();
}

//Not dispatched when running in a debugger
void LeapEventListener::onDisconnect(const Controller& controller) {
    std::cout << "Disconnected" << std::endl;
}

void LeapEventListener::onFrame(const Controller& controller) {
    std::cout << "New frame available" << std::endl;
    Frame frame = controller.frame();
    // Process this frame's data...
}
/*!
 * 特殊キーボードイベント処理関数
 * @param[in] key キーの種類
 * @param[in] x,y キーが押されたときのマウス座標(スクリーン座標系)
 */
void SpecialKey(int key, int x, int y)
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



/*! 
 * GLの初期化関数
 */
void InitGL(void)
{
	cout << "OpenGL Ver. " << glGetString(GL_VERSION) << endl;

	GLenum err = glewInit();
	if(err == GLEW_OK){
		cout << "GLEW OK : Glew Ver. " << glewGetString(GLEW_VERSION) << endl;
	}
	else{
		cout << "GLEW Error : " << glewGetErrorString(err) << endl;
	}

	// マルチサンプリングの対応状況確認
	GLint buf, sbuf;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &buf);
	cout << "number of sample buffers is " << buf << endl;
	glGetIntegerv(GL_SAMPLES, &sbuf);
	cout << "number of samples is " << sbuf << endl;

	glClearColor((GLfloat)g_fBGColor[0], (GLfloat)g_fBGColor[1], (GLfloat)g_fBGColor[2], 1.0f);
	glClearDepth(1.0f);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glEnable(GL_MULTISAMPLE);

	// ポリゴン法線設定
	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	// 光源設定
	static const GLfloat lpos[] = { 0.0f, 0.0f, 1.0f, 1.0f };	// 光源位置
	static const GLfloat difw[] = { 0.7f, 0.7f, 0.7f, 1.0f };	// 拡散色 : 白
	static const GLfloat spec[] = { 0.2f, 0.2f, 0.2f, 1.0f };	// 鏡面反射色
	static const GLfloat ambi[] = { 0.1f, 0.1f, 0.1f, 1.0f };	// 環境光
	glLightfv(GL_LIGHT0, GL_POSITION, lpos);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  difw);
	glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambi);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);

	glShadeModel(GL_SMOOTH);

	// トラックボール初期姿勢
	g_tbView.SetScaling(-5.0);
	//g_tbView.SetTranslation(0.0, -2.0);
}

/*! 
 * GLの終了関数
 */
void CleanGL(void)
{
}

/*!
 * メインメニュー
 * @param[in] id メニューID
 */
void OnMainMenu(int id)
{
	Keyboard((unsigned char)id, 0, 0);
}

/*!
 * GLUTの右クリックメニュー作成
 */
void InitMenu(void)
{
	// メインメニュー
	glutCreateMenu(OnMainMenu);

	// 描画メニュー
	glutAddMenuEntry(" -------- draw --------- ", -1);
	int count = 0;
	while(RX_DRAW_STR[2*count] != "-1"){
		string label = RX_DRAW_STR[2*count];
		glutAddMenuEntry(label.c_str(),	RX_DRAW_STR[2*count+1][0]);
		count++;
	}
	glutAddMenuEntry(" ------------------------ ",	-1);
	glutAddMenuEntry("Toggle fullscreen [F]",		'F');
	glutAddMenuEntry("Toggle animation [s]",		's');
	glutAddMenuEntry("Step animation [ ]",			' ');
	glutAddMenuEntry(" ------------------------ ",	-1);
	glutAddMenuEntry("Quit [ESC]",					'\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
}



/*!
 * メインルーチン
 * @param[in] argc コマンドライン引数の数
 * @param[in] argv コマンドライン引数
 */
int main(int argc, char *argv[])
{
Controller controller;
LeapEventListener listener;
controller.addListener(listener);




	glutInitWindowPosition(g_iWinX, g_iWinY);
	glutInitWindowSize(g_iWinW, g_iWinH);
	glutInit(&argc, argv);
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
	    // Remove the sample listener when done
    controller.removeListener(listener);
	return 0;
}


