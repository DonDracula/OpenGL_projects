/*!
  @file rx_fltk_glcanvas.h
	
  @brief FLTKによるOpenGLウィンドウクラス
 
*/

#ifndef _RX_FLTK_GLCANVAS_H_
#define _RX_FLTK_GLCANVAS_H_



//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <iostream>

// STL
#include <vector>
#include <string>

// FLTK
#include <FL/Fl.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Spinner.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Round_Button.H>
#include <FL/Fl_Check_Button.H>

#include "rx_sph_commons.h"
#include "rx_sph_config.h"
#include "rx_fltk_widgets.h"

#include "rx_trackball.h"
#include "rx_model.h"
#include "rx_pov.h"

#include "rx_texture.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>


using namespace std;

//-----------------------------------------------------------------------------
// 定義/定数
//-----------------------------------------------------------------------------
class rxFlWindow;
class rxParticleSystemBase;
struct rxCell;

class rxMCMeshCPU;
class rxMCMeshGPU;

#define RX_USE_GPU

#ifdef RX_USE_GPU
	#define RXSPH rxSPH_GPU
#else
	#define RXSPH rxSPH
#endif


// 描画フラグ
enum
{
	RXD_PARTICLE		= 0x0001,	//!< パーティクル
	RXD_VELOCITY		= 0x0002,	//!< 速度場
	RXD_NORMAL			= 0x0004,	//!< 法線
	RXD_BBOX			= 0x0008,	//!< AABB(シミュレーション空間)
	RXD_CELLS			= 0x0010,	//!< 近傍探索用分割セル
	RXD_MESH			= 0x0020,	//!< メッシュ
	RXD_SOLID			= 0x0040,	//!< 固体
	RXD_REFRAC			= 0x0080,	//!< 屈折描画

	RXD_PARAMS			= 0x0100,	//!< パラメータ画面描画
	RXD_ANISOTROPICS	= 0x0200,	//!< 異方性カーネル
};
const string RX_DRAW_STR[] = {
	"Particle",					"p", 
	"Velocity",					"v", 
	"Normal",					"n",
	"AABB (Simulation Space)", 	"b",
	"Cells", 					"d",
	"Mesh", 					"m",
	"Solid", 					"o",
	"Refrac", 					"r",
	"Params", 					"P",
	"Anisotoropics", 			"A",
	"-1"
};

// パーティクル描画方法
enum
{
	RXP_POINTSPRITE = 0, 
	RXP_POINT, 
	RXP_POINT_NONE, 

	RXP_END, 
};
const string RX_PARTICLE_DRAW[] = {
	"Point Sprite", "^1", 
	"GL_POINT", 	"^2", 
	"None", 		"^3", 
	"-1"
};


// 三角形メッシュ生成法
enum
{
	RXM_MC_CPU = 0, 
	RXM_MC_GPU, 
};
const string RX_TRIANGULATION_METHOD[] = {
	"Marching Cubes (CPU)",	   "", 
	"Marching Cubes (GPU)",    "", 
	"-1"
};

// 固体描画
enum
{
	RXS_VERTEX		= 0x0001, 
	RXS_EDGE		= 0x0002, 
	RXS_FACE		= 0x0004, 
	RXS_NORMAL		= 0x0008, 
	RXS_MOVE		= 0x0010,

	RXS_END
};
const string RXS_STR[] = {
	"Vertex", "", 
	"Edge",   "", 
	"Face",   "", 
	"Normal", "", 
	"Move",   "", 
	"-1"
};


//! 描画領域サイズ候補
const string RX_CANVAS_SIZE_STR[] = {
	"1920x1080",	"",
	"1280x720",		"", 
	"1024x768",		"", 
	"800x800",		"", 
	"800x600",		"", 
	"640x480",		"", 
	"-1", 
};


//! SPH設定
enum SettingMode
{
	ID_SPH_MESH = 0,	// メッシュ生成
	ID_SPH_INPUT,		// パーティクルデータ出力
	ID_SPH_OUTPUT,		// パーティクルデータ入力
	ID_SPH_MESH_OUTPUT, // メッシュ出力
	ID_SPH_INLET,		// 流入境界
	ID_SPH_VC, 
	ID_SPH_ANISOTROPIC, // 異方性カーネル

	ID_SPH_END, 
};


//-----------------------------------------------------------------------------
//! rxFlGLWindowクラス - fltkによるOpenGLウィンドウ
//-----------------------------------------------------------------------------
class rxFlGLWindow : public Fl_Gl_Window
{
protected:
	int m_iWinW;					//!< 描画ウィンドウの幅
	int m_iWinH;					//!< 描画ウィンドウの高さ
	int m_iMouseButton;				//!< マウスボタンの状態
	int m_iKeyMod;					//!< 修飾キーの状態
	rxTrackball m_tbView;			//!< トラックボール

	double m_fBGColor[3];			//!< 背景色
	bool m_bAnimation;				//!< アニメーションON/OFF
	bool m_bFullscreen;				//!< フルスクリーンON/OFF

	rxFlWindow *m_pParent;			//!< 親クラス

	vector<rxPolygons> m_vPolys;	//!< ポリゴンオブジェクト

	// FTGL
	unsigned long m_ulFontSize;		//!< フォントサイズ

	//
	// 粒子法関連変数
	//
	rxParticleSystemBase *m_pPS;	//!< SPH
	double m_fDt;					//!< タイムステップ幅
	double m_fGravity;				//!< 重力加速度

	int m_iCurrentStep;				//!< 現在のステップ数
	bool m_bPause;					//!< シミュレーションのポーズフラグ

	// シーン
	//string m_strCurrentScene;		//!< 現在のシーンの名前
	//vector<string> m_vSceneFiles;	//!< シーンファイルリスト
	//int m_iSceneFileNum;			//!< シーンファイルの数
	rxSPHConfig m_Scene;

	int m_iSimuSetting;				//!< ミュレーション設定保存用

	// 固体移動フラグ
	bool m_bSolidMove;

	// パーティクル情報出力
	string m_strSphOutputName0 ;
	string m_strSphOutputHeader;

	// パーティクル情報入力
	string m_strSphInputName0 ;
	string m_strSphInputHeader;

	//
	// メッシュ
	//
	uint m_iNumVrts, m_iNumTris;	//!< 生成されたメッシュの頂点数とメッシュ数
	int m_iVertexStore;				//!< サンプリングボリューム数に対する予想される頂点数(nx*ny*store)

	int m_iMeshMaxN;				//!< メッシュ化グリッド数(境界がもっとも長い軸方向の分割数)
	int m_iMeshN[3];				//!< メッシュ化グリッド数

	Vec3 m_vMeshBoundaryExt;		//!< メッシュ境界ボックスの各辺の長さの1/2
	Vec3 m_vMeshBoundaryCen;		//!< メッシュ境界ボックスの中心座標

	rxPolygons m_Poly;				//!< メッシュ
	//vector<rxPolygons*> m_vSolidPoly;//!< 固体メッシュ
	rxMaterialOBJ m_matPoly;

	GLuint m_uVrtVBO;				//!< メッシュ頂点(VBO)
	GLuint m_uTriVBO;				//!< メッシュポリゴン(VBO)
	GLuint m_uNrmVBO;				//!< メッシュ頂点法線(VBO)
	int m_iDimVBO;

	// メッシュ出力
	int m_iSaveMeshSpacing;

	// 背景画像
	bool m_bUseCubeMap;				//!< キューブマップ使用フラグ
	rxCubeMapData m_CubeMap;		//!< キューブマップ

	// メッシュ生成
	rxMCMeshCPU *m_pMCMeshCPU;
	rxMCMeshGPU *m_pMCMeshGPU;


public:
	// 描画フラグ
	int m_iDraw;					//!< 描画フラグ
	int m_iDrawPS;					//!< パーティクル描画方法
	int m_iColorType;				//!< パーティクル描画時の色
	int m_iTriangulationMethod;		//!< 三角形メッシュ生成法
	int m_iSolidDraw;				//!< 固体描画

	// シミュレーション設定
	bitset<32> m_bsSimuSetting;		//!< ミュレーション設定フラグ
	double m_fVScale;				//!< ベクトル場描画時のスケール
	double m_fMeshThr;				//!< 陰関数メッシュ化時の閾値

	// シーンリスト
	vector<string> m_vSceneTitles;	//!< シーンファイルリスト
	int m_iCurrentSceneIdx;			//!< 現在のシーンファイル

	// 画像出力
	int m_iSaveImageSpacing;		//!< 画像保存間隔(=-1なら保存しない)

public:
	//! コンストラクタ
	rxFlGLWindow(int x, int y, int w, int h, const char* l, void *parent);

	//! デストラクタ
	~rxFlGLWindow();

public:
	// OpenGL初期化
	void InitGL(void);

	// OpenGL描画
	void Projection(void);
	vector<string> SetDrawString(void);
	void ReDisplay(void);

	// GUIコールバック
	void Display(void);
	void Resize(int w, int h);
	void Mouse(int button, int state, int x, int y);
	void Motion(int x, int y);
	void PassiveMotion(int x, int y);
	void Idle(void);
	void Timer(void);
	void Keyboard(int key, int x, int y);
	void SpecialKey(int key, int x, int y);

	// 視点
	void InitView(void);

	// アニメーション
	static void OnTimer_s(void* x);
	static void OnIdle_s(void* x);
	
	// アニメーション切り替え
	bool SwitchIdle(int on);

	// フルスクリーン切り替え
	void SwitchFullScreen(int win = 1);
	int  IsFullScreen(void);


	// ファイル入出力
	void OpenFile(const string &fn);
	void SaveFile(const string &fn);
	void SaveDisplay(const string &fn);
	void SaveDisplay(const int &stp);
	
	void SaveMesh(const string fn, rxPolygons &polys);
	void SaveMesh(const int &stp, rxPolygons &polys);

	// FTGLフォント設定
	int SetupFonts(const char* file);


public:
	// メッシュ生成
	bool CalMeshSPH(int nmax, double thr = 1000.0);
	bool ResetMesh(void);
	void SetMeshCreation(void);
	RXREAL GetImplicitSPH(double x, double y, double z);

protected:
	bool calMeshSPH_CPU(int nmax, double thr = 1000.0);
	bool calMeshSPH_GPU(int nmax, double thr = 1000.0);

	// SPH
	void InitSPH(rxSPHConfig &sph_scene);
	void InitSPH(void){ InitSPH(m_Scene); }
	void AddSphere(void);

	void StepPS(double dt);
	void ComputeFPS(void);

	void DivideString(const string &org, vector<string> &div);

	void DrawParticleVector(RXREAL *prts, RXREAL *vels, int n, int d, double *c0, double *c1, double len = 0.1);
	void DrawParticlePoints(unsigned int vbo, int n, unsigned int color_vbo = 0, RXREAL *data = 0);
	void DrawSubParticles(void);

	void CreateVBO(GLuint* vbo, unsigned int size);
	void DeleteVBO(GLuint* vbo);

	void DrawLiquidSurface(void);
	void SetParticleColorType(int type, int change = 0);

	void RenderSphScene(void);
	

private:
	//! 描画コールバック関数のオーバーライド
	void draw(void)
	{
		if(!context_valid()) InitGL();
		if(!valid()) Resize(w(), h());
		Display();    // OpenGL描画
	}

	//! リサイズコールバック関数のオーバーライド
	void resize(int x_, int y_, int w_, int h_)
	{
		Fl_Gl_Window::resize(x_, y_, w_, h_);
		//Resize(w_, h_);
	}


public:
	// イベントハンドラ
	int handle(int e);	// handle関数のオーバーライド

	// メニューイベントハンドラ
	static void OnMenuDraw_s(Fl_Widget *widget, void* x);
	inline void OnMenuDraw(double val, string label);

	static void OnMenuSimulation_s(Fl_Widget *widget, void* x);
	inline void OnMenuSimulation(double val, string label);

	static void OnMenuParticle_s(Fl_Widget *widget, void* x);
	inline void OnMenuParticle(double val, string label);
	inline void OnMenuParticleColor(double val, string label);
	inline void OnMenuParticleDraw(double val, string label);

	static void OnMenuSolid_s(Fl_Widget *widget, void* x);
	inline void OnMenuSolid(double val, string label);

	static void OnMenuTriangulation_s(Fl_Widget *widget, void* x);
	inline void OnMenuTriangulation(double val, string label);

	static void OnMenuScene_s(Fl_Widget *widget, void* x);
	inline void OnMenuScene(double val, string label);
};




#endif // #ifdef _RX_FLTK_GLCANVAS_H_
