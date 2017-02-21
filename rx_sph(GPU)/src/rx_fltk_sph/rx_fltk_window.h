/*!
  @file rx_fltk_window.h
	
  @brief FLTKによるウィンドウクラス
 
*/

#ifndef _RX_FLTK_WINDOW_H_
#define _RX_FLTK_WINDOW_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Box.H>
#include <FL/filename.H>		// ファイル名
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_File_Icon.H>


#include "rx_fltk_glcanvas.h"

using namespace std;

//-----------------------------------------------------------------------------
// MARK:定義/定数
//-----------------------------------------------------------------------------
const string RX_PROGRAM_NAME = "rx_fltk_sph";
extern vector<string> g_vDefaultFiles;



//-----------------------------------------------------------------------------
//! rxFlWindowクラス - fltkによるウィンドウ
//-----------------------------------------------------------------------------
class rxFlWindow : public Fl_Double_Window
{
protected:
	// メンバ変数
	int m_iWinW;		//!< 描画ウィンドウの幅
	int m_iWinH;		//!< 描画ウィンドウの高さ
	int m_iWinX;		//!< 描画ウィンドウ位置x
	int m_iWinY;		//!< 描画ウィンドウ位置y

	// ウィジット
	Fl_Menu_Bar *m_pMenuBar;	//!< メニューバー

	Fl_Value_Slider *m_pSliderMeshThr;
	Fl_Value_Slider *m_pSliderVScale;
	Fl_Check_Button *m_pCheckRefraction;
	Fl_Check_Button *m_pCheckMesh;

	rxFlDndBox *m_pDndBox;		//!< D&D領域
	Fl_Box *m_pBoxStatus;		//!< ステータスバー

	rxFlGLWindow *m_pGLCanvas;	//!< OpenGL描画キャンパス

	// ファイル情報
	string m_strFileName;		//!< 読み込んだファイル名
	string m_strFullPath;		//!< 読み込んだファイル名(フルパス)

	char *m_pStatusLabel;		//!< ステータスバー文字列

	bool m_bFullScreen;			//!< フルスクリーン表示
	int m_iIdle;				//!< アイドル,タイマーの状態

public:
	//! コンストラクタ
	rxFlWindow(int w, int h, const char* title);

	//! デストラクタ
	~rxFlWindow();

public:
	// コールバック関数
	static void OnMenuFile_s(Fl_Widget *widget, void* x);
	inline void OnMenuFileOpen(void);
	inline void OnMenuFileSave(void);
	inline void OnMenuFileSaveFrame(void);
	inline void OnMenuFileQuit(void);
	static void OnMenuStep_s(Fl_Widget *widget, void* x);
	inline void OnMenuStep(string label);
	static void OnMenuWindow_s(Fl_Widget *widget, void* x);
	inline void OnMenuWindow(string label);
	static void OnMenuHelpVersion_s(Fl_Widget *widget, void* x);

	static void OnDnd_s(Fl_Widget *widget, void* x);
	inline void OnDnd(void);

	static void OnButtonStart_s(Fl_Widget *widget, void* x);
	inline void OnButtonStart(void);
	static void OnButtonApply_s(Fl_Widget *widget, void* x);
	inline void OnButtonApply(Fl_Widget *widget);

	static void OnSliderDraw_s(Fl_Widget *widget, void* x);
	inline void OnSliderDraw(Fl_Widget *widget);
	static void OnCheckDraw_s(Fl_Widget *widget, void* x);
	inline void OnCheckDraw(Fl_Widget *widget);


	// メニューの更新
	void UpdateMenuState(void);

	// ステータスバー文字列の変更
	void SetStatusLabel(const string &label);

	// フルスクリーン切替
	void SwitchFullScreen(void);
	bool IsFullScreen(void) const;

	// ファイル入出力
	void Open(const string &fn);
	void Save(const string &fn);
	void OpenImage(const string &fn);

	// 設定ファイル
	void ReadConfig(const string &fn);
	void WriteConfig(const string &fn);

protected:
	int handle(int ev);
};





#endif // #ifdef _RX_FLTK_WINDOW_H_
