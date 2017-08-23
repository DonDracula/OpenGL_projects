/*!
  @file main.cpp
	
  @brief FLTKとOpenGL
 
  @author Makoto Fujisawa
  @date   2011-08
*/

#ifdef _DEBUG
#pragma comment(lib, "fltkd.lib")
#pragma comment(lib, "fltkgld.lib")
#pragma comment(lib, "fltkimagesd.lib")
#pragma comment(lib, "fltkjpegd.lib")
#pragma comment(lib, "fltkpngd.lib")
#pragma comment(lib, "fltkzlibd.lib")
#else
#pragma comment(lib, "fltk.lib")
#pragma comment(lib, "fltkgl.lib")
#pragma comment(lib, "fltkimages.lib")
#pragma comment(lib, "fltkjpeg.lib")
#pragma comment(lib, "fltkpng.lib")
#pragma comment(lib, "fltkzlib.lib")
#endif


#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "cudart.lib")

#ifdef _DEBUG
#pragma comment(lib, "rx_modeld.lib")
#else
#pragma comment(lib, "rx_model.lib")
#endif




// コマンドプロンプトを出したくない場合はここをコメントアウト
//#pragma comment(linker, "/subsystem:\"windows\" /entry:\"mainCRTStartup\"")


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_fltk_window.h"

// CUDA
#include "rx_cu_funcs.cuh"

//-----------------------------------------------------------------------------
// メイン関数
//-----------------------------------------------------------------------------
/*!
 * メインルーチン
 * @param[in] argc コマンドライン引数の数
 * @param[in] argv コマンドライン引数
 */
int main(int argc, char *argv[])
{
	// コマンドライン引数
	if(argc >= 2){
		for(int i = 1; i < argc; ++i){
			string fn = argv[i];
			g_vDefaultFiles.push_back(fn);
		}
	}

	glutInit(&argc, argv);
	CuInit(argc, argv);

	Fl::visual(FL_DOUBLE | FL_INDEX);
	Fl::get_system_colors();
	fl_register_images();
	Fl::scheme("gtk+");

	rxFlWindow win(480, 480, "opengl application");
	return Fl::run();
}


