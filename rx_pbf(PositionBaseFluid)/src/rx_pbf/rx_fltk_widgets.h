/*!
  @file rx_fltk_widgets.h
	
  @brief FLTKによるカスタムウィジット
 
  @author Makoto Fujisawa 
  @date   2011-08
*/

#ifndef _RX_FLTK_WIDGETS_H_
#define _RX_FLTK_WIDGETS_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <sstream>
#include <string>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Output.H>
#include <FL/Fl_Slider.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Shared_Image.H> // 画像

#include <GL/glew.h>
#include <GL/glut.h>

#ifdef WIN32
#include <windows.h>
#include <commdlg.h>
#endif



using namespace std;




//-----------------------------------------------------------------------------
// 文字列処理関数
//-----------------------------------------------------------------------------



/*!
 * stringからchar[]に変換(\0終わり)
 * @param[in] s string文字列
 * @return char文字列
 */
inline char* RX_TO_CHAR(const string &s)
{
	if(s.empty()) return 0;

	int n = (int)s.size();
	char* c = new char[n+1];
	for(int i = 0; i < n; ++i) c[i] = s[i];
	c[n] = '\0';
	return c;
}

/*!
 * stringからchar[]に変換(\0終わり)
 * @param[in] s string文字列
 * @param[out] c char文字列
 */
inline void RX_TO_CHAR(const string &s, char c[])
{
	if(s.empty()) return;

	int n = (int)s.size();
	for(int i = 0; i < n; ++i) c[i] = s[i];
	c[n] = '\0';
}



/*!
 * パスからファイル名を取り除いたパスを抽出
 * @param[in] path パス
 * @return フォルダパス
 */
inline string GetFolderPath(const string &path)
{
	size_t pos1;
 
	pos1 = path.rfind('\\');
	if(pos1 != string::npos){
		return path.substr(0, pos1+1);
		
	}
 
	pos1 = path.rfind('/');
	if(pos1 != string::npos){
		return path.substr(0, pos1+1);
	}
 
	return "";
}




//-----------------------------------------------------------------------------
// FLTK関連関数
//-----------------------------------------------------------------------------
/*!
 * メニューアイテム(FL_MENU_TOGGLE)の状態を変更
 * @param[in] menubar メニューバーオブジェクト
 * @param[in] name メニュー名("File/Open"など)
 * @param[in] state トグルON/OFF
 * @param[in] enable 有効/無効
 * @return メニューが存在しなければ-1を返す
 */
inline int SetMenuItemState(Fl_Menu_Bar *menubar, string name, int state, int enable = true)
{
	Fl_Menu_Item *m = (Fl_Menu_Item*)menubar->find_item(RX_TO_CHAR(name));
	if(!m) return -1;

	if(enable){
		m->activate();
	}
	else{
		m->deactivate();
	}

	if(state){
		m->set();
	}
	else{
		m->clear();
	}
	return(0);
}



//-----------------------------------------------------------------------------
//! rxFlDndBoxクラス - ドラッグアンドドロップ用ボックス
//-----------------------------------------------------------------------------
class rxFlDndBox : public Fl_Box
{
	// MARK:rxFlDndBox
protected:
	// メンバ変数
	int m_iEvent;	//!< イベントID

	string m_strEventText;
	int m_iEventTextLen;

public:
	//! コンストラクタ
	rxFlDndBox(int x, int y, int w, int h, const char *l = 0)
		 : Fl_Box(x, y, w, h, l), m_iEvent(FL_NO_EVENT), m_iEventTextLen(0)
	{
		labeltype(FL_NO_LABEL);
		box(FL_NO_BOX);
		clear_visible_focus();
	}

	//! デストラクタ
	virtual ~rxFlDndBox()
	{
	}

public:
	// メンバ関数
	static void CallbackS(void *v)
	{
		rxFlDndBox *w = (rxFlDndBox*)v;
		w->do_callback();
	}

	int Event()
	{
		return m_iEvent;
	}

	string EventText()
	{
		return m_strEventText;
	}

	int EventTextLength()
	{
		return m_iEventTextLen;
	}

	int handle(int e)
	{
		switch(e){
			case FL_DND_ENTER:
				cout << "rxFlDndBox::FL_DND_ENTER" << endl;
				m_iEvent = e;
				return 1;
			case FL_DND_RELEASE:
				cout << "rxFlDndBox::FL_DND_RELEASE" << endl;
				m_iEvent = e;
				return 1;
			case FL_DND_LEAVE:
				cout << "rxFlDndBox::FL_DND_LEAVE" << endl;
				m_iEvent = e;
				return 1;
			case FL_DND_DRAG:
				cout << "rxFlDndBox::FL_DND_DRAG" << endl;
				m_iEvent = e;
				return 1;


			case FL_PASTE:
				cout << "rxFlDndBox::FL_PASTE" << endl;
				m_iEvent = e;

				m_iEventTextLen = Fl::event_length();
				m_strEventText = Fl::event_text();

				if(callback() && ((when() & FL_WHEN_RELEASE) || (when() & FL_WHEN_CHANGED)))
					Fl::add_timeout(0.0, rxFlDndBox::CallbackS, (void*)this);
				return 1;
		}

		return Fl_Box::handle(e);
	}
};



//-----------------------------------------------------------------------------
// FLTKでの画像読み込み
//-----------------------------------------------------------------------------
inline unsigned char* ReadImageByFLTK(const string &fn, int &w, int &h, int &c)
{
	char* cfn = RX_TO_CHAR(fn);
	Fl_Shared_Image *simg = Fl_Shared_Image::get(cfn);

	if(simg->count() != 1){
		return 0;
	}

	w = simg->w();
	h = simg->h();
	c = simg->d();
	unsigned char *dat = new unsigned char[w*h*c];

	// ファイル出力
	const char *buf = simg->data()[0];
	unsigned char r, g, b, a;
	for(int j = 0; j < h; ++j){
		for(int i = 0; i < w; ++i){
			long idx = j*w*c+i*c;
			r = g = b = a = 0;
			switch(c){
			case 1:
				r = g = b = *(buf+idx);
				break;

			case 3:
				r = *(buf+idx+0);
				g = *(buf+idx+1);
				b = *(buf+idx+2);
				break;

			case 4:
				r = *(buf+idx+0);
				g = *(buf+idx+1);
				b = *(buf+idx+2);
				a = *(buf+idx+3);
				break;
							
			default:
				break;
			}

			dat[idx+0] = r;
			dat[idx+1] = g;
			dat[idx+2] = b;
		}
	}

	return dat;
}

/*!
 * OpenGLテクスチャ登録
 * @param[in] fn ファイル名
 * @param[inout] tex_name テクスチャ名(0なら新たに生成)
 * @param[in] mipmap ミップマップ使用フラグ
 * @param[in] compress テクスチャ圧縮使用フラグ
 */
static int LoadGLTexture(const string &fn, GLuint &tex_name, bool mipmap, bool compress)
{
	// 画像読み込み
	unsigned char *buf = 0;
	int w, h, c;
	buf = ReadImageByFLTK(fn, w, h, c);
	if(buf == 0){
		return -1;
	}

	GLuint iformat, format;

	// 画像フォーマット
	format = GL_RGBA;
	if(c == 1){
		format = GL_LUMINANCE;
	}
	else if(c == 3){
		format = GL_RGB;
	}
 
	// OpenGL内部の格納フォーマット
	if(compress){
		iformat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
		if(c == 1){
			iformat = GL_COMPRESSED_LUMINANCE_ARB;
		}
		else if(c == 3){
			iformat = GL_COMPRESSED_RGB_S3TC_DXT1_EXT ;
		}
	}
	else{
		iformat = GL_RGBA;
		if(c == 1){
			iformat = GL_LUMINANCE;
		}
		else if(c == 3){
			iformat = GL_RGB;
		}
	}
 
	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
 
	// テクスチャ作成
	if(tex_name == 0){
		glGenTextures(1, &tex_name);
 
		glBindTexture(GL_TEXTURE_2D, tex_name);
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR));
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
 
		if(mipmap){
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 6);
		}
 
		glTexImage2D(GL_TEXTURE_2D, 0, iformat, w, h, 0, format, GL_UNSIGNED_BYTE, buf);
 
		if(mipmap){
			glGenerateMipmapEXT(GL_TEXTURE_2D);
		}
	}
	else{
		glBindTexture(GL_TEXTURE_2D, tex_name);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, format, GL_UNSIGNED_BYTE, pimg);
		glTexImage2D(GL_TEXTURE_2D, 0, iformat, w, h, 0, format, GL_UNSIGNED_BYTE, buf);

		if(mipmap){
			glGenerateMipmapEXT(GL_TEXTURE_2D);
		}
	}

	delete [] buf;
 
	glBindTexture(GL_TEXTURE_2D, 0);

	return 1;
}




//-----------------------------------------------------------------------------
// MARK:ファイルダイアログ 
//-----------------------------------------------------------------------------
/*!
 * "|"で区切られたファイルフィルタを説明部と拡張子指定部に分解
 *  [例] "Text Files|*.txt;*.log;*.dat|All Files|*.*"
 *       -> descs : "Text Files", "All Files"
 *       -> exts  : "*.txt;*.log;*.dat", "*.*"
 * @param[in] filter "|"で区切られたファイルフィルタ文字列
 * @param[out] descs 説明部
 * @param[out] exts  拡張子指定部
 * @return フィルタ数( = descs.size() = exts.size() )
 */
inline int ParseFilterString(const string &filter, vector<string> &descs, vector<string> &exts)
{
	int nfilter = 0;
	size_t pos0 = 0, pos1 = 0;
	do{
		// フィルタ説明の抽出
		pos1 = filter.find('|', pos0);
		if(pos0 < pos1){
			descs.push_back(filter.substr(pos0, pos1-pos0));
			pos0 = pos1+1;

			// フィルタ拡張子の抽出
			pos1 = filter.find('|', pos0);
			if(pos0 < pos1){
				exts.push_back(filter.substr(pos0, pos1-pos0));
				nfilter++;
			}
		}

		pos0 = pos1+1;
	}while(pos1 != string::npos);

	return nfilter;
}


/*!
 * fltkのFile_Chooser用のフィルタ生成
 *  [例] "Text Files (*.{txt,log,dat})\tAll Files (*)"
 * @param[in] descs 説明部
 * @param[in] exts  拡張子指定部
 * @return フィルタ文字列
 */
inline string GenFlFileFilter(const vector<string> &descs, const vector<string> &exts)
{
	int nfilter = (int)descs.size();
	string fl_filter;
	for(int i = 0; i < nfilter; ++i){
		// 説明
		fl_filter += descs[i];

		// 拡張子
		fl_filter += " (";

		// *.txt;*.log;*.dat のような文字列から，txt,log,datを抽出
		vector<string> es;
		size_t epos0 = 0, epos1 = 0;
		do{
			epos1 = exts[i].find(';', epos0);
			if(epos0 < epos1){
				// "*."と";"の間の文字列を抽出
				es.push_back(exts[i].substr(epos0+2, epos1-epos0-2));
			}
			epos0 = epos1+1;
		}while(epos1 != string::npos);

		// fltkのフィルタ形式に変換 *.{txt,log,dat}
		if((int)es.size() > 2){
			// 複数拡張子指定時
			fl_filter += "*.{";
			for(int j = 0; j < (int)es.size(); ++j){
				fl_filter += es[j];
				if(j != (int)es.size()-1) fl_filter += ",";
			}
			fl_filter += "}";
		}
		else if(!es.empty() && es[0] != "*"){
			// 単一拡張子
			fl_filter += "*."+es[0];
		}
		else{
			// 任意拡張子(*)
			fl_filter += "*";
		}

		fl_filter += ")";
		if(i < nfilter-1) fl_filter += "\t";
	}

	return fl_filter;
}

/*!
 * Win32のGetOpenFileName用のフィルタ生成(ヌル文字区切り)
 *  [例] "Text Files\0*.txt;*.log;*.dat\0All Files\0*.*\0\0"
 * @param[in] descs 説明部
 * @param[in] exts  拡張子指定部
 * @param[out] cfilter フィルタ文字列
 * @param[out] n フィルタ文字列最大サイズ
 */
inline int GenWin32FileFilter(const vector<string> &descs, const vector<string> &exts, char cfilter[], int n)
{
	int nfilter = (int)descs.size();
	int c = 0, j = 0, k = 0;
	while(c < n){
		if(k%2 == 0){
			if(k != 0 && j == 0){	// 区切りのヌル文字
				cfilter[c++] = NULL;
			}

			// 説明
			cfilter[c++] = descs[k/2][j++];
			if(j >= (int)descs[k/2].size()){
				j = 0;
				k++;
			}
		}
		else{
			if(k != 0 && j == 0){	// 区切りのヌル文字
				cfilter[c++] = NULL;
			}

			// 拡張子
			cfilter[c++] = exts[k/2][j++];
			if(j >= (int)exts[k/2].size()){
				j = 0;
				k++;
			}
		}

		if(k >= nfilter*2){	// 最後のヌル文字x2
			cfilter[c++] = NULL;
			cfilter[c++] = NULL;
			break;
		}
	}
	return 1;
}

/*!
 * ファイル選択ダイアログの表示
 *  Windowsではエクスプローラ形式，その他ではfltkのFile_Chooser
 * @param[out] fns 選択されたファイル
 * @param[in] title ダイアログタイトル
 * @param[in] filter ファイルフィルタ("|"区切り)
 * @param[in] multi 複数選択の可否
 */
inline int ShowFileDialog(vector<string> &fns, 
						  const string &title, const string &filter, bool multi = false)
{
	vector<string> descs, exts;
	int nfilter = ParseFilterString(filter, descs, exts);

#ifdef WIN32

	OPENFILENAME ofn;
	memset((void*)&ofn, 0, sizeof(OPENFILENAME));

	// ファイル選択ダイアログの設定
	ofn.lStructSize  = sizeof(OPENFILENAME);
	ofn.Flags       |= OFN_NOVALIDATE;          // 無効な文字が入ったファイル名を有効にする(/からはじまるファイルパスを有効に)
	ofn.Flags       |= OFN_HIDEREADONLY;        // 読み取り専用チェックボックスを隠す
	ofn.Flags       |= OFN_EXPLORER;            // 新しいエクスプローラウィンドウを使用
	ofn.Flags       |= OFN_ENABLESIZING;        // ダイアログリサイズ可
	ofn.Flags       |= OFN_NOCHANGEDIR;         // 過去に開いたフォルダをデフォルトにする
	if(multi) ofn.Flags |= OFN_ALLOWMULTISELECT;	// 複数選択
	ofn.nMaxFile     = 4096-1;
	ofn.lpstrFile    = new char[4096];
	ofn.lpstrFile[0] = 0;
	ofn.hwndOwner    = GetForegroundWindow();
	ofn.lpstrTitle   = title.c_str();

	// ファイルフィルタ
	char cfilter[1024];
	GenWin32FileFilter(descs, exts, cfilter, 1024);
	ofn.lpstrFilter  = cfilter;

	// ファイル選択ダイアログを開く
	int err = GetOpenFileName(&ofn);
	if(err == 0){
		err = CommDlgExtendedError();
		if(err == 0) return 0;
		fprintf(stderr, "CommDlgExtendedError() code = %d", err);
		return 0;
	}

	// 複数ファイル選択時はofn.lpstrFileにディレクトリ名とファイル名(複数)がヌル文字で区切られて格納されている．
	// 最後のファイル名の後には2つの連続したヌル文字が格納されている．
	string tmp;
	vector<string> fns0;
	int null_num = 0;
	for(int i = 0; i < (int)ofn.nMaxFile; ++i){
		if(ofn.lpstrFile[i] == NULL){	// ヌル文字による区切り
			if(!null_num){
				fns0.push_back(tmp);
				tmp.clear();
			}
			null_num++;
		}
		else{
			tmp.push_back(ofn.lpstrFile[i]);
			null_num = 0;
		}

		// 2つの連続したヌル文字が見つかればループを抜ける
		if(null_num >= 2) break;
	}

	// 複数ファイル選択時はfnsのサイズが3以上
	int n = (int)fns0.size();
	//vector<string> fns;
	if(n >= 3){
		fns.resize(n-1);
		for(int i = 0; i < n-1; ++i){
			// ディレクトリ名とファイル名を合成
			fns[i] = fns0[0]+"/"+fns0[i+1];
		}
		n -= 1;
	}
	else{
		// 単一ファイル
		fns.resize(1);
		fns[0] = fns0[0];
		n = 1;
	}

#else
	string fl_filter = GenFlFileFilter(descs, exts);
	//cout << fl_filter << endl;
	//fl_filter = "Movie Files (*.{avi,mp4,flv,mov,mkv})\tAll Files (*)";

	int t = (multi ? Fl_File_Chooser::MULTI : Fl_File_Chooser::SINGLE);
	Fl_File_Chooser *fc = new Fl_File_Chooser(".", fl_filter.c_str(), t, title.c_str());
	fc->show();
	while(fc->visible()){
		Fl::wait();
	}

	int n = fc->count();
	for(int i = 0; i < n; ++i){
		if(!fc->value(i+1)) continue;
		fns.push_back(fc->value(i+1));
	}
	n = (int)fns.size();

	delete fc;

#endif

	return n;
}


#endif // #ifndef _RX_FLTK_WIDGETS_H_