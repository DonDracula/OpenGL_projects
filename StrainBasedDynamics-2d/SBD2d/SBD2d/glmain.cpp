

/*! 
	 glmain.cpp
	 date: 2015
*/


#pragma comment (lib, "glew32.lib")

#ifdef _DEBUG
#pragma comment(lib, "rx_modeld.lib")
#else
#pragma comment(lib, "rx_model.lib")
#endif


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "utils.h"
#include "rx_atom_ini.h"
#include "rx_model.h"

#include "rx_sbd_2d.h"

using namespace std;


//-----------------------------------------------------------------------------
// 定義/定数
//-----------------------------------------------------------------------------
const string RX_PROGRAM_NAME = "sbd2d";

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

//! 背景色
double g_fBGColor[4] = {0.0, 0.0, 0.0, 1.0};

int g_iDraw = RXD_FACE;	//!< 描画フラグ

// 設定ファイル
rxINI *g_pINI = 0;

// 変形メッシュ
rxSBD2D *g_pS = 0;

// 描画領域の大きさ
Vec2 g_v2EnvMin = Vec2(-1.4);
Vec2 g_v2EnvMax = Vec2(1.4);

// 頂点マウスピック
int g_iPickedVertex = -1;


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

/*!
 * 設定ファイル読み込み
 * @param[in] fn 設定ファイル名(拡張子無し)
 */
void ReadConfig(const string &fn)
{
	int n;

	// ユーザ管理設定ファイル(読み込みのみ)
	rxINI *cfg = new rxINI();
	cfg->Set<int>("sbd", "n", &n, 64);

	if(!(cfg->Load(fn+".cfg"))){
		cout << "Failed to open the " << fn << ".cfg file!" << endl;
		if(cfg->Save(fn+".cfg")){
			cout << "save : " << fn << ".cfg" << endl;
		}
	}
	g_pS = new rxSBD2D(n);

	// アプリケーション管理設定ファイル
	g_pINI = new rxINI();

	g_pINI->Set("window", "width",  &g_iWinW, 720);
	g_pINI->Set("window", "height", &g_iWinH, 720);
	g_pINI->Set("window", "pos_x",  &g_iWinX, 100);
	g_pINI->Set("window", "pos_y",  &g_iWinY, 100);

	if(!(g_pINI->Load(fn+".ini"))){
		cout << "Failed opening the " << fn << " file!" << endl;
	}
}

/*!
 * 設定ファイル書き込み
 * @param[in] fn 設定ファイル名(拡張子無し)
 */
void WriteConfig(const string &fn)
{
	g_iWinX = glutGet(GLUT_WINDOW_X);
	g_iWinY = glutGet(GLUT_WINDOW_Y);

	if(g_pINI != NULL && g_pINI->Save(fn+".ini")){
		cout << "save : " << fn << endl;
	}
}

//-----------------------------------------------------------------------------
// 描画関数など
//-----------------------------------------------------------------------------
/*!
 * 透視投影変換
 */
void Projection(void)
{
	//gluPerspective(45.0f, (float)g_iWinW/(float)g_iWinH, 0.2f, 1000.0f);
	gluOrtho2D(g_v2EnvMin[0], g_v2EnvMax[0], g_v2EnvMin[1], g_v2EnvMax[1]);
}

void RenderScene(void)
{
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// オブジェクト描画
	glColor3d(1.0, 0.9, 0.5);
	if(g_pS) g_pS->Draw(g_iDraw);
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

	RenderScene();

	glPopMatrix();

	// 文字列描画
	vector<string> strs;
	strs.push_back("'s' : idle on/off");
	strs.push_back("SHIFT+'f' : fullscreen on/off");
	strs.push_back("'v','e','f' : switch vertex,edge,face drawing");
	strs.push_back("'r' : reset");
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
			Vec2 mpos(x/(double)g_iWinW, (g_iWinH-y-1)/(double)g_iWinH);
			mpos *= g_v2EnvMax-g_v2EnvMin;
			mpos += g_v2EnvMin;

			int picked = -1;
			picked = g_pS->Search(mpos, 0.05);
			if(picked != -1){
				g_iPickedVertex = picked;
				g_pS->SetFix(g_iPickedVertex, mpos);
			}
			else if(g_iPickedVertex){
				g_iPickedVertex = -1;
			}

		}
		else if(state == GLUT_UP){
			if(g_iPickedVertex != -1){
				g_pS->UnsetFix(g_iPickedVertex);
				g_iPickedVertex = -1;
			}
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
	if(g_iPickedVertex != -1){
		Vec2 mpos(x/(double)g_iWinW, (g_iWinH-y-1)/(double)g_iWinH);
		mpos *= g_v2EnvMax-g_v2EnvMin;
		mpos += g_v2EnvMin;
		g_pS->SetFix(g_iPickedVertex, mpos);
	}

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
	if(g_pS){
		g_pS->Update(0.01);
	}
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

	case 'r':	// 再生成
		if(g_pS) g_pS->Init();
		break;

	default:	// 描画フラグの処理
		{
			int idx = 0;
			while(RX_DRAW_STR[2*idx] != "-1" && (int)RX_DRAW_STR[2*idx+1][0] != key) idx++;
			if(RX_DRAW_STR[2*idx] != "-1"){
				g_iDraw ^= (0x01 << idx);	// エッジや面を同時に描画する際はこちらを使用
				//g_iDraw = (0x01 << idx);
			}
		}
		break;
	}

	glutPostRedisplay();
}

/*!
 * 特殊キーボードイベント処理関数
 * @param[in] key キーの種類
 * @param[in] x,y キーが押されたときのマウス座標(スクリーン座標系)
 */
void SpecialKey(int key, int x, int y)
{
	switch(key){
	case GLUT_KEY_RIGHT:break;
	case GLUT_KEY_LEFT:	break;
	case GLUT_KEY_UP:	break;
	case GLUT_KEY_DOWN:	break;
	default:	break;
	}

	glutPostRedisplay();
}



/*! 
 * GLの初期化関数
 */
void InitGL(void)
{
	//cout << "OpenGL Ver. " << glGetString(GL_VERSION) << endl;

	GLenum err = glewInit();
	if(err == GLEW_OK){
		//cout << "GLEW OK : Glew Ver. " << glewGetString(GLEW_VERSION) << endl;
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
}

/*! 
 * GLの終了関数
 */
void CleanGL(void)
{
	WriteConfig(RX_PROGRAM_NAME);
	if(g_pINI) delete g_pINI;

	if(g_pS) delete g_pS;
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
	glutAddMenuEntry("Toggle fullscreen [f]",		'f');
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
	ReadConfig(RX_PROGRAM_NAME);

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

	return 0;
}


