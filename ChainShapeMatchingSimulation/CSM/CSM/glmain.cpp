/*! @file glmain.cpp

@brief GLUT基本構成

@date   2015-06
*/

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "utils.h"

//shape matching method
#include "shape_matching.h"

// 3D model
#include "rx_model.h"

// shadow map
#include "rx_shadow.h"

using namespace std;


//-----------------------------------------------------------------------------
// definition
//-----------------------------------------------------------------------------
const string RX_PROGRAM_NAME = "Chain Shape Matching";

const GLfloat RX_FOV = 45.0f;

//const GLfloat RX_NEAR_CLIP = 0.1f;
//const GLfloat RX_FAR_CLIP  = 100.0f;
//
//const GLfloat RX_LIGHT0_POS[4] = {  0.5f, 3.0f, 0.5f, 0.0f };

const double RX_GOUND_HEIGHT = -1.0;

//// vertex color
//const Vec3 VERTEX_COLOR[] = { Vec3(1.0, 0.0, 0.0), 
//	Vec3(0.0, 1.0, 0.0), 
//	Vec3(0.0, 0.0, 1.0), 
//	Vec3(0.0, 1.0, 1.0), 
//	Vec3(1.0, 0.0, 1.0), 
//	Vec3(1.0, 1.0, 0.0) };
//


enum{
	RXD_VERTEX		= (1 << 0),  
	RXD_EDGE		= (1 << 1), 
	RXD_FACE		= (1 << 2),
	RXD_NORMAL		= (1 << 3),
	RXD_CELL		= (1 << 4), 
};

bool g_bIdle = false;

//windows size
int g_iWinW = 1280;		//Width of the window
int g_iWinH = 1280;		//Height of the window
int g_iWinX = 100;		//x position of the window
int g_iWinY = 50;		//y position of the window

bool g_bFullScreen = false;		

// mouse state
int g_iMouseButton = -1;

//! 描画フラグ
int g_iDraw = RXD_FACE;

// キーの状態
int g_iKeyMod = 0;

// background color
double g_fBGColor[3] = {0.8, 0.8, 0.9};

// trackball
rxTrackball g_tbView;

//mouse pick 
rxGLPick g_Pick;
vector<int> g_vSelectedVertices;
int g_iPickedObj = -1;
double g_fPickDist = 1.0;	//!< ピックされた点までの距離

//// polygon object
//struct rxObject
//{
//	string filename;			//!< ファイル名
//	rxPolygons poly;			//!< ポリゴン
//	ShapeMatching *deform;	//!< シェイプマッチング法による変形
//	int vstart, vend;			//!< 全体の頂点数における位置
//};
//vector<rxObject> g_vObj;

struct rxRopeObject
{
	ShapeMatching *ropeDeform;
	int vstart,vend;
};
vector<rxRopeObject> g_ropeObj;

// shadow map
rxShadowMap g_ShadowMap;
int g_iShadowMapSize = 4096;

// save the image
int g_iSaveImageSpacing = -1;	//!< 画像保存間隔(=-1なら保存しない)

int g_iStep = 0;

double beta = 0.8;
int regionSize = 3;
//-----------------------------------------------------------------------------
//function declaration & definition 
//-----------------------------------------------------------------------------
void OnExitApp(int id = -1);

void Idle(void);
void CleanGL(void);

/*
//-----------------------------------------------------------------------------
// メッシュ変形
//-----------------------------------------------------------------------------

void ReadFile(const string &fn, rxPolygons &poly, Vec3 cen, Vec3 ext)
{
RxModel::Read(fn, poly, cen, ext, Vec3(0.0));

// テクスチャ読み込み
if(!poly.materials.empty()){
string parent_path = GetFolderPath(fn);
rxMTL::iterator iter = poly.materials.begin();
for(; iter != poly.materials.end(); ++iter){
if(iter->second.tex_file.empty()) continue;

//iter->second.tex_file = parent_path+iter->second.tex_file;
std::cout << iter->first << " : " << iter->second.tex_file << endl;
LoadGLTexture(iter->second.tex_file, iter->second.tex_name, true, false);
}
}
}
*/
//Vec3 g_v3Cen = Vec3(0.0, -1.0, 0.0);
//double g_fRad = 0.5;

Vec3 g_v3EnvMin(-3.0, RX_GOUND_HEIGHT, -3.0);
Vec3 g_v3EnvMax(3.0, RX_GOUND_HEIGHT+3.0, 3.0);
double g_fDt = 0.01;

//rxNNGrid *m_pNNGrid;			//!< 分割グリッドによる近傍探索
//vector<double> g_vVrts;
//vector<int> g_vTris;
vector<Vec3> masses;
//vector<Particle> particles; 

//init the object
void InitObject(float springLength,int totalmass)
{
	rxRopeObject ropeObj;
	ropeObj.ropeDeform = new ShapeMatching((int)g_ropeObj.size());

	int vn= regionSize;

	ropeObj.vstart = (g_ropeObj.empty() ? 0 : g_ropeObj.back().vend+1);
	ropeObj.vend = ropeObj.vstart+(vn-1);
	vector<vector<int>> clusters;

	for (int index=0;index<totalmass;index++)
	{

		masses.push_back(Vec3(index*springLength,0,0));
	}
	
	//vector<int> c;
	//for(int i = 3;i<masses.size()-4;i++)
	//{
	//	int v = i;
	//	c.push_back(v);
	//}clusters.push_back(c);

	//vector<int> c2;
	//for(int i = 0;i<masses.size()-6;i++)
	//{
	//	int v = i;
	//	c2.push_back(v);
	//}clusters.push_back(c2);

	//	vector<int> c3;
	//for(int i = 5;i<masses.size();i++)
	//{
	//	int v = i;
	//	c3.push_back(v);
	//}clusters.push_back(c3);

	for(int i = 0;i<masses.size()-regionSize+1;i++)
	{
		int v = i;
		vector<int> c;
		for(int startPos= i; startPos < i+regionSize; startPos++)
		{
			c.push_back(startPos);
		}
		clusters.push_back(c);
		
	}

	cout<<"masses size "<<masses.size()<<endl;
	for(int i = 0;i<masses.size();i++)
	{
		ropeObj.ropeDeform->AddVertex(masses[i],0.1);
	}

	//cluster
	for(int i = 0; i< (int)clusters.size(); ++i)
	{
		ropeObj.ropeDeform->AddCluster(clusters[i]);
	}

	cout <<"cluster size::::::::::::::::::::::::::::::::"<<clusters.size()<<endl;
	// Shape Matchingの設定

	ropeObj.ropeDeform->SetSimulationSpace(g_v3EnvMin, g_v3EnvMax);
	ropeObj.ropeDeform->SetTimeStep(g_fDt);
	ropeObj.ropeDeform->SetCollisionFunc(0);
	ropeObj.ropeDeform->SetStiffness(0.9, beta);

	g_ropeObj.push_back(ropeObj);

	g_iStep = 0;
	g_vSelectedVertices.clear();
	g_iPickedObj = -1;
}


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
	static int pos0[2] = { 0, 0 };
	static int win0[2] = { 500, 500 };
	if(g_bFullScreen){
		glutPositionWindow(pos0[0], pos0[1]);
		glutReshapeWindow(win0[0], win0[1]);
		g_bFullScreen = false;
	}
	else{
		pos0[0] = glutGet(GLUT_WINDOW_X);
		pos0[1] = glutGet(GLUT_WINDOW_Y);
		win0[0] = glutGet(GLUT_WINDOW_WIDTH);
		win0[1] = glutGet(GLUT_WINDOW_HEIGHT);

		glutFullScreen();
		g_bFullScreen = true;
	}

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
	gluPerspective(RX_FOV, (double)g_iWinW/(double)g_iWinH, 0.01f, 20.0f);
}
static void Projection_s(void* x){ Projection(); }

/*!
* シーン描画
*/
void RenderScene(int draw)
{
	// 材質
	static const GLfloat difw[] = { 0.8f, 0.8f, 0.8f, 1.0f };	// 拡散色 : 白
	static const GLfloat difr[] = { 1.0f, 0.4f, 0.4f, 1.0f };	// 拡散色 : 赤
	static const GLfloat difg[] = { 0.4f, 0.6f, 0.4f, 1.0f };	// 拡散色 : 緑
	static const GLfloat difb[] = { 0.4f, 0.4f, 1.0f, 1.0f };	// 拡散色 : 青
	static const GLfloat spec[] = { 0.3f, 0.3f, 0.3f, 1.0f };	// 鏡面反射色
	static const GLfloat ambi[] = { 0.1f, 0.1f, 0.1f, 1.0f };	// 環境光


	glPushMatrix();

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// 材質を設定
	glMaterialfv(GL_FRONT, GL_SPECULAR, spec);
	glMaterialfv(GL_FRONT, GL_AMBIENT,  ambi);
	glMaterialf(GL_FRONT, GL_SHININESS, 50.0);

	// draw objects
	glColor3f(0.0, 0.0, 1.0);


	// 床
	glPushMatrix();
	glMaterialfv(GL_FRONT, GL_DIFFUSE, difg);
	glTranslatef(0, RX_GOUND_HEIGHT-0.105, 0);
	glScaled(10.0, 0.2, 10.0);
	glutSolidCube(1.0);
	glPopMatrix();

	// オブジェクト
	glPushMatrix();
	glColor3ub(255,255,0);									//set color to yellow

	glBegin(GL_POINTS);
	glPointSize(10);
	//	cout << "rope number ============"<<g_ropeObj[0].ropeDeform->GetNumOfVertices()<<endl;
	//	vector<rxRopeObject>::iterator itr = g_ropeObj.begin();
	for(int i =0; i<g_ropeObj.size();i++)
	{
		for(int index=0;index<g_ropeObj[i].ropeDeform->GetNumOfVertices();index++)
		{
			glVertex3dv(g_ropeObj[i].ropeDeform->GetVertexPos(index).data);

		}
		glEnd();
		//drawing the rope ends here

		for(int count=0; count<g_ropeObj[i].ropeDeform->GetNumOfVertices();++count)
		{
			glPushMatrix();
			glMaterialfv(GL_FRONT, GL_DIFFUSE, difr);
			glTranslatef(g_ropeObj[i].ropeDeform->GetVertexPos(count).data[0],g_ropeObj[i].ropeDeform->GetVertexPos(count).data[1],g_ropeObj[i].ropeDeform->GetVertexPos(count).data[2]);
			glutSolidSphere(0.02,32,32);									//draw sphere
			glPopMatrix();
		}		
	}
	glPopMatrix();

	// 球
	//glPushMatrix();
	//glDisable(GL_LIGHTING);
	//glTranslatef(g_v3Cen[0], g_v3Cen[1], g_v3Cen[2]);
	//glColor3d(1.0, 1.0, 0.0);
	//glutWireSphere(g_fRad, 32, 32);
	//glPopMatrix();


	glPopMatrix();
}
static void RenderScene_s(void* x){ RenderScene((g_iDraw & 15)); }

void RenderSceneForShadow(void)
{
	RenderScene((g_iDraw & 15));
}

//------------------------------------------------------------------
// OpenGLキャンバス用のイベントハンドラ
//------------------------------------------------------------------
/*!
* 再描画イベント処理関数
*/
void Display(void)
{
	glClearColor(0.8, 0.8, 0.9, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	GLfloat sky_color[4] = {0.9f, 0.9f, 1.0f, 1.0f};

	// FIX
	//glEnable(GL_MULTISAMPLE);
	Vec4 light_dir = Vec4(0.2, 1.0, 0.5, 0.0);
	normalize(light_dir);
	GLfloat light_dirf[4];
	for(int i = 0; i < 4; ++i) light_dirf[i] = (GLfloat)light_dir[i];

	Vec3 eye_pos(0.0), eye_dir(0.0);
	g_tbView.CalLocalPos(eye_pos.data, Vec3(0.0).data);
	g_tbView.CalLocalRot(eye_dir.data, Vec3(0.0, 0.0, -1.0).data);
	vector<Vec3> bcns, bsls;
	bcns.push_back(Vec3(0.0));
	bsls.push_back(Vec3(2.0));
	g_ShadowMap.MakeShadowMap(light_dir, eye_pos, eye_dir, RenderSceneForShadow, bcns, bsls);

	// フレームバッファとデプスバッファをクリアする
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glPushMatrix();
	g_tbView.Apply();	// マウスによる回転・平行移動の適用

	// 光源設定
	glLightfv(GL_LIGHT0, GL_POSITION, light_dirf);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, sky_color);

	g_ShadowMap.RenderSceneWithShadow(RenderSceneForShadow, g_iWinW, g_iWinH);

	//g_ShadowMap.DrawDepthTex();
	//g_ShadowMap.OverviewCam(RenderScene2, g_iWinW, g_iWinH);

	glutSwapBuffers();
}


void DisplayForPick(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();

	// マウスによる回転・平行移動の適用
	g_tbView.Apply();

	// 頂点描画
	int c = 1;
	//	vector<rxObject>::iterator itr = g_vObj.begin();
	//	while(itr != g_vObj.end()){
	vector<rxRopeObject>::iterator itr = g_ropeObj.begin();
	while(itr != g_ropeObj.end()){
		int vn = (int)itr->ropeDeform->GetNumOfVertices();
		glDisable(GL_LIGHTING);
		glPointSize(16.0);
		for(int i = 0; i < vn; ++i){
			glLoadName(c++);
			glBegin(GL_POINTS);
			glVertex3dv((itr->ropeDeform->GetVertexPos(i)).data);
			glEnd();
		}

		itr++;
	}
	glPopMatrix();

	glutSwapBuffers();
}
static void DisplayForPick_s(void* x){ DisplayForPick(); }


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

	g_ShadowMap.CalAllFrustums(w, h, RX_FOV);
}

void ClearPick(void)
{
	g_vSelectedVertices.clear();
	if(g_iPickedObj != -1){
		//for(int i = 0; i < g_vObj[g_iPickedObj].deform->GetNumOfVertices(); ++i) g_vObj[g_iPickedObj].deform->UnFixVertex(i);
	//	for(int i = 0; i < g_ropeObj[g_iPickedObj].ropeDeform->GetNumOfVertices(); ++i) g_ropeObj[g_iPickedObj].ropeDeform->UnFixVertex(i);
		g_iPickedObj = -1;
	}
}

/*!
* マウスイベント処理関数
* @param[in] button マウスボタン(GLUT_LEFT_BUTTON,GLUT_MIDDLE_BUTTON,GLUT_RIGHT_BUTTON)
* @param[in] state マウスボタンの状態(GLUT_UP, GLUT_DOWN)
* @param[in] x,y マウス座標(スクリーン座標系)
*/
void Mouse(int button, int state, int x, int y)
{
	g_iKeyMod = glutGetModifiers();	// SHIFT,CTRL,ALTの状態取得

	if(button == GLUT_LEFT_BUTTON){
		if(state == GLUT_DOWN){
			int hit = g_Pick.Pick(x, y);
			if(hit >= 1){
				int v = hit-1;

				// どのオブジェクトの頂点なのかを判別
				g_iPickedObj = -1;
				//int size = (int)g_vObj.size();
				cout <<"g_ropePOBjJJJJJJJJJJJJJ"<<g_ropeObj.size()<<endl;
				int size = (int)g_ropeObj.size();
			//	cout <<"size sieze"<<size<<endl;
				for(int i = 0; i < size; ++i){
					//if(g_vObj[i].vstart <= v && v <= g_vObj[i].vend){
					if(g_ropeObj[i].vstart<=v&&v<=g_ropeObj[i].vend){
						g_iPickedObj = i;
						break;
					}
					//}
				}

				if(g_iPickedObj != -1){
					vector<int> vrts;
					vrts.push_back(v);
					cout << "vertex " << v << " is selected - " << g_iPickedObj << endl;
					g_vSelectedVertices.resize(1);
					//	g_vSelectedVertices[0] = v-g_vObj[g_iPickedObj].vstart;
					g_vSelectedVertices[0] = v-g_ropeObj[g_iPickedObj].vstart;

					// 視点からピック点までの距離計算
					Vec3 ray_from;
					Vec3 init_pos = Vec3(0.0);
					g_tbView.CalLocalPos(ray_from, init_pos);
					//Vec3 pos = g_vObj[g_iPickedObj].deform->GetVertexPos(v);
					Vec3 pos = g_ropeObj[g_iPickedObj].ropeDeform->GetVertexPos(v);

					g_fPickDist = length(pos-ray_from);
				}
			}
			else{
				g_tbView.Start(x, y, g_iKeyMod+1);
				ClearPick();
			}
		}
		else if(state == GLUT_UP){
			g_tbView.Stop(x, y);
			ClearPick();
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
	if(g_vSelectedVertices.empty()){
		g_tbView.Motion(x, y);
	}
	else{
		Vec3 ray_from, ray_to;
		Vec3 init_pos = Vec3(0.0);
		g_tbView.CalLocalPos(ray_from, init_pos);
		g_tbView.GetRayTo(x, y, RX_FOV, ray_to);

		Vec3 dir = Unit(ray_to-ray_from);	// 視点からマウス位置へのベクトル

		int v = g_vSelectedVertices[0];
		//Vec3 cur_pos = g_vObj[g_iPickedObj].deform->GetVertexPos(v);
		Vec3 cur_pos = g_ropeObj[0].ropeDeform->GetVertexPos(0);
		Vec3 new_pos = ray_from+dir*g_fPickDist;
		//g_vObj[g_iPickedObj].deform->FixVertex(v, new_pos);
		g_ropeObj[0].ropeDeform->FixVertex(0,new_pos);
	}



	glutPostRedisplay();
}

/*!
* モーションイベント処理関数(マウスボタンを押さない移動)
* @param[in] x,y マウス座標(スクリーン座標系)
*/
void PassiveMotion(int x, int y)
{
}


/*! 
* アイドルイベント処理関数
*/
void Idle(void)
{
	int size = (int)g_ropeObj.size();
	for(int i = 0; i < size; ++i){
		g_ropeObj[i].ropeDeform->Update();

	}
///	g_ropeObj[0].ropeDeform->m_vGoalPos()

	//if(g_iSaveImageSpacing > 0 && (g_iStep%g_iSaveImageSpacing == 0)){
	//	SaveDisplay(CreateFileName("img/display_", ".png", g_iStep, 5), g_iWinW, g_iWinH);
	//}

	g_iStep++;
	glutPostRedisplay();
}

/*!
* キーボードイベント処理関数
* @param[in] key キーの種類
* @param[in] x,y キーが押されたときのマウス座標(スクリーン座標系)
*/
void Keyboard(unsigned char key, int x, int y)
{
	// MARK:Keyboard
	g_iKeyMod = glutGetModifiers();	// SHIFT,CTRL,ALTの状態取得

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

	case 'v':
		g_iDraw ^= RXD_VERTEX;
		break;

	case 'e':
		g_iDraw ^= RXD_EDGE;
		break;

	case 'f':
		g_iDraw ^= RXD_FACE;
		break;

	case 'n':
		g_iDraw ^= RXD_NORMAL;
		break;

	case 'd':
		g_iDraw ^= RXD_CELL;
		break;

	case 'S':
		//g_iSaveImageSpacing = (g_iSaveImageSpacing < 0) ? 1 : -1;
		break;

	case 'c':	// 選択の解除
		ClearPick();
		break;

	default:
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
	case GLUT_KEY_LEFT:
		break;
	case GLUT_KEY_RIGHT:
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
	RXCOUT << "OpenGL Ver. " << glGetString(GL_VERSION) << endl;

	glewInit();

	// マルチサンプリングの対応状況確認
	GLint buf, sbuf;
	glGetIntegerv(GL_SAMPLE_BUFFERS, &buf);
	cout << "number of sample buffers is " << buf << endl;
	glGetIntegerv(GL_SAMPLES, &sbuf);
	cout << "number of samples is " << sbuf << endl;

	// マルチサンプリングによるアンチエイリアス
	//glEnable(GL_MULTISAMPLE);

	// 背景色
	glClearColor((GLfloat)g_fBGColor[0], (GLfloat)g_fBGColor[1], (GLfloat)g_fBGColor[2], 1.0f);
	glClearDepth(1.0f);


	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	// ポリゴン設定
	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	// 光源の初期設定
	static const GLfloat difw[] = { 1.0f, 1.0f, 1.0f, 1.0f };	// 拡散色 : 白
	static const GLfloat spec[] = { 0.6f, 0.6f, 0.6f, 1.0f };	// 鏡面反射色
	static const GLfloat ambi[] = { 0.3f, 0.3f, 0.3f, 1.0f };	// 環境光
	//glLightfv(GL_LIGHT0, GL_POSITION, RX_LIGHT0_POS);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  difw);
	glLightfv(GL_LIGHT0, GL_SPECULAR, spec);
	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambi);
	glEnable(GL_LIGHTING);

	glShadeModel(GL_SMOOTH);

	glEnable(GL_AUTO_NORMAL);
	glEnable(GL_NORMALIZE);

	// 視点の初期化
	g_tbView.SetScaling(-3.0);
	g_tbView.SetTranslation(0.0, 0.5);

	// シャドウマップ初期化
	//g_ShadowMap.InitShadow(g_iShadowMapSize, g_iShadowMapSize);
	g_ShadowMap.InitShadowMap(g_iWinW, g_iWinH, RX_FOV, 4, 4096);
	g_ShadowMap.SetShadowType(3);

	// Shape Matching
	InitObject(0.05f,40);
	g_ropeObj[0].ropeDeform->FixVertex(0,Vec3(0,0,0));
}

/*! 
* GLの終了関数
*/
void CleanGL(void)
{
	//vector<rxObject>::iterator itr = g_vObj.begin();
	//while(itr != g_vObj.end()){
	vector<rxRopeObject>::iterator itr = g_ropeObj.begin();
	while(itr != g_ropeObj.end()){	
		if(itr->ropeDeform) delete (itr->ropeDeform);
		itr->ropeDeform = 0;
		itr++;
	}

//	if(m_pNNGrid) delete m_pNNGrid;
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
	//MkDir("img");

	glutInitWindowPosition(g_iWinX, g_iWinY);
	glutInitWindowSize(g_iWinW, g_iWinH);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutCreateWindow(argv[0]);

	glutDisplayFunc(Display);
	glutReshapeFunc(Resize);
	glutMouseFunc(Mouse);
	glutMotionFunc(Motion);
	glutPassiveMotionFunc(PassiveMotion);
	glutKeyboardFunc(Keyboard);
	glutSpecialFunc(SpecialKey);
	//glutCloseFunc(CleanGL);

	SwitchIdle(1);

	InitGL();
	InitMenu();

	g_Pick.Set(DisplayForPick_s, 0, Projection_s, 0);

	glutMainLoop();

	return 0;
}


