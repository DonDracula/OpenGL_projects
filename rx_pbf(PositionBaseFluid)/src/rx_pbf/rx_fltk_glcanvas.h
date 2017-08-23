/*!
  @file rx_fltk_glcanvas.h
	
  @brief FLTKによるOpenGLウィンドウクラス
 
  @author Makoto Fujisawa 
  @date   2011-09
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

#include "rx_texture.h"


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


#if defined(RX_USE_GPU)
	#define RXSPH rxPBDSPH_GPU
#else
	#define RXSPH rxPBDSPH
#endif



// 描画フラグ
enum
{
	RXD_PARTICLE		= 0x0001,	//!< パーティクル
	RXD_VELOCITY		= 0x0002,	//!< 速度場
	RXD_BBOX			= 0x0004,	//!< AABB(シミュレーション空間)
	RXD_AXIS			= 0x0008,   //!< 軸

	RXD_CELLS			= 0x0010,	//!< 近傍探索用分割セル
	RXD_MESH			= 0x0020,	//!< メッシュ
	RXD_SOLID			= 0x0040,	//!< 固体
	RXD_REFRAC			= 0x0080,	//!< 屈折描画

	RXD_BPARTICLE		= 0x0100,	//!< 境界パーティクル
	RXD_BPARTICLE_ENV	= 0x0200,	//!< 境界パーティクル描画
	RXD_ZPLANE_CLIP		= 0x0400,	//!< 断面
	RXD_PARAMS			= 0x0800,	//!< パラメータ画面描画
};
const string RX_DRAW_STR[] = {
	"Particle",					"p", 
	"Velocity",					"v", 
	"AABB (Simulation Space)", 	"b",
	"Axis",				 		"",

	"Cells", 					"d",
	"Mesh", 					"m",
	"Solid", 					"o",
	"Refrac", 					"r",
	
	"Boundary Particle",		"B",
	"Env. Boundary Prt.",		"", 
	"Cross Section",			"",
	"Params", 					"P",
	"-1"
};

// パーティクル描画方法
enum
{
	RXP_POINTSPRITE = 0, 
	RXP_POINT, 
	RXP_SPHERE, 
	RXP_POINT_NONE, 

	RXP_END, 
};
const string RX_PARTICLE_DRAW[] = {
	"Point Sprite",		"^1", 
	"GL_POINT", 		"^2", 
	"glutSolidSphere", 	"^3", 
	"None", 			"^4", 
	"-1"
};

// パーティクル描画色方法
enum
{
	RXP_COLOR_CONSTANT = 0, 
	RXP_COLOR_DENSITY, 
	RXP_COLOR_RAMP, 
	RXP_COLOR_END, 
};
const string RX_PARTICLE_COLOR[] = {
	"Constant", 	"", 
	"Density", 		"", 
	"Ramp",			"", 
	"-1"
};


// 固体描画
enum
{
	RXS_VERTEX		= 0x0001, 
	RXS_EDGE		= 0x0002, 
	RXS_FACE		= 0x0004, 
	RXS_NORMAL		= 0x0008, 
	RXS_END
};
const string RXS_STR[] = {
	"Vertex", "", 
	"Edge",   "", 
	"Face",   "", 
	"Normal", "", 
	"-1"
};

// 液体表面描画
enum
{
	RXL_VERTEX		= 0x0001, 
	RXL_EDGE		= 0x0002, 
	RXL_FACE		= 0x0004, 
	RXL_NORMAL		= 0x0008, 
	RXL_END
};
const string RXL_STR[] = {
	"Vertex", "", 
	"Edge",   "", 
	"Face",   "", 
	"Normal", "", 
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
	RX_SPH_MESH			= 0x0001,	// メッシュ生成
	RX_SPH_INPUT		= 0x0002,		// パーティクルデータ出力
	RX_SPH_OUTPUT		= 0x0004,		// パーティクルデータ入力
	RX_SPH_MESH_OUTPUT	= 0x0008, // メッシュ出力
	RX_SPH_INLET		= 0x0010,		// 流入境界

	RX_SPH_END			= 0x0040, 
};

//! ピックされたオブジェクトに関する情報
struct rxPickInfo
{
	GLuint name;		//!< ヒットしたオブジェクトの名前
	float min_depth;	//!< プリミティブのデプス値の最小値
	float max_depth;	//!< プリミティブのデプス値の最大値
};

static inline bool CompFuncPickInfo(rxPickInfo a, rxPickInfo b)
{
	return a.min_depth < b.min_depth;
}

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
	rxSceneConfig m_Scene;


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


	int m_iPickedParticle;			//!< マウスピックされたパーティクル
	int m_iSelBufferSize;			//!< セレクションバッファのサイズ
	GLuint* m_pSelBuffer;			//!< セレクションバッファ


public:
	// 描画フラグ
	int m_iDraw;					//!< 描画フラグ
	int m_iDrawPS;					//!< パーティクル描画方法
	int m_iColorType;				//!< パーティクル描画時の色
	int m_iSolidDraw;				//!< 固体描画
	int m_iLiquidDraw;				//!< 液体表面描画

	// シミュレーション設定
	int m_iSimuSetting;		//!< ミュレーション設定フラグ
	double m_fVScale;				//!< ベクトル場描画時のスケール
	double m_fMeshThr;				//!< 陰関数メッシュ化時の閾値

	double m_fClipPlane[2];			//!< クリップ平面(z軸に垂直な平面)

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

	// マウスピック用
	static void Projection_s(void* x);
	static void DisplayForPick_s(void* x);
	void DisplayForPick(void);
	bool PickParticle(int x, int y);

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


public:
	// メッシュ生成
	bool CalMeshSPH(int nmax, double thr = 1000.0);
	bool ResetMesh(void);
	void SetMeshCreation(void);
	RXREAL GetImplicitSPH(double x, double y, double z);

protected:
	bool calMeshSPH_CPU(int nmax, double thr = 1000.0);
	bool calMeshSPH_GPU(int nmax, double thr = 1000.0);
	
	// マウスピック
	vector<rxPickInfo> selectHits(GLint nhits, GLuint buf[]);
	vector<rxPickInfo> pick(int x, int y, int w, int h);

public:
	// SPH
	void InitSPH(rxSceneConfig &sph_scene);
	void InitSPH(void){ InitSPH(m_Scene); }
	void AddSphere(void);

	void StepPS(double dt);
	void ComputeFPS(void);

	void DivideString(const string &org, vector<string> &div);

	void DrawParticleVector(RXREAL *prts, RXREAL *vels, int n, int d, double *c0, double *c1, double len = 0.1);
	void DrawParticlePoints(unsigned int vbo, int n, unsigned int color_vbo = 0, RXREAL *data = 0, int offset = 0);

	void CreateVBO(GLuint* vbo, unsigned int size);
	void DeleteVBO(GLuint* vbo);

	void DrawLiquidSurface(void);
	void SetParticleColorType(int type);

	void RenderSphScene(void);

	bool IsArtificialPressureOn(void);

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

	static void OnMenuMesh_s(Fl_Widget *widget, void* x);
	inline void OnMenuMesh(double val, string label);
	inline void OnMenuMeshSolid(double val, string label);
	inline void OnMenuMeshLiquid(double val, string label);

	static void OnMenuScene_s(Fl_Widget *widget, void* x);
	inline void OnMenuScene(double val, string label);
};




#endif // #ifdef _RX_FLTK_GLCANVAS_H_
