/*! @file rx_texture.h
	
	@brief GLテクスチャ作成
	@note rx_gltexture.hと同時に読み込まないこと
 
	@author Makoto Fujisawa
	@date   2008
*/


#ifndef _RX_TEXTURE_H_
#define _RX_TEXTURE_H_

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdlib>

#include <GL/glew.h>
#include <GL/glut.h>

#include <vector>
#include <string>

#include <iostream>
#include <fstream>

using namespace std;

//! テクスチャ用画像構造体
struct rxTexImage
{
	int w, h, c;
	vector<unsigned char> buf;
};



//-----------------------------------------------------------------------------
// テクスチャ
//-----------------------------------------------------------------------------
/*!
 * テクスチャの作成
 * @param[out] tex_name テクスチャ名
 * @param[in] size_x,size_y テクスチャの大きさ
 */
static void CreateTexture(GLuint &tex_name, unsigned int size_x, unsigned int size_y)
{
	// GLテクスチャ生成
	glGenTextures(1, &tex_name);
	glBindTexture(GL_TEXTURE_2D, tex_name);

	// テクスチャパラメータ
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	// テクスチャ領域確保
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	//gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, size_x, size_y, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}

/*!
 * テクスチャ削除
 * @param[out] tex_name テクスチャ名
 */
static void DeleteTexture(GLuint &tex_name)
{
	glDeleteTextures(1, &tex_name);

	tex_name = 0;
}


/*!
 * 立方体描画
 * @param[in] size 一辺の長さ 
 */
static void DrawCube(double size = 1.0)
{
	glBegin( GL_QUADS );
	glNormal3d( 1.0, 0.0, 0.0 );
	glTexCoord2d(0.0, 0.0);	glVertex3d( size/2.0,  size/2.0, -size/2.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d( size/2.0,  size/2.0,  size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d( size/2.0, -size/2.0,  size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d( size/2.0, -size/2.0, -size/2.0);

	glNormal3d(-1.0, 0.0, 0.0);
	glTexCoord2d(0.0, 0.0);	glVertex3d(-size/2.0,  size/2.0,  size/2.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d(-size/2.0,  size/2.0, -size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d(-size/2.0, -size/2.0, -size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d(-size/2.0, -size/2.0,  size/2.0);

	glNormal3d(0.0, 0.0, 1.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d(-size/2.0,  size/2.0,  size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d(-size/2.0, -size/2.0,  size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d( size/2.0, -size/2.0,  size/2.0);
	glTexCoord2d(0.0, 0.0);	glVertex3d( size/2.0,  size/2.0,  size/2.0);

	glNormal3d(0.0, 0.0, -1.0);
	glTexCoord2d(0.0, 0.0);	glVertex3d(-size/2.0,  size/2.0, -size/2.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d( size/2.0,  size/2.0, -size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d( size/2.0, -size/2.0, -size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d(-size/2.0, -size/2.0, -size/2.0);

	glNormal3d(0.0, -1.0, 0.0);
	glTexCoord2d(0.0, 0.0);	glVertex3d(-size/2.0, -size/2.0,  size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d(-size/2.0, -size/2.0, -size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d (size/2.0, -size/2.0, -size/2.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d (size/2.0, -size/2.0,  size/2.0);

	glNormal3d(0.0, 1.0, 0.0);
	glTexCoord2d(0.0, 0.0);	glVertex3d(-size/2.0,  size/2.0,  size/2.0);
	glTexCoord2d(1.0, 0.0);	glVertex3d( size/2.0,  size/2.0,  size/2.0);
	glTexCoord2d(1.0, 1.0);	glVertex3d( size/2.0,  size/2.0, -size/2.0);
	glTexCoord2d(0.0, 1.0);	glVertex3d(-size/2.0,  size/2.0, -size/2.0);
	glEnd();
	
}


static bool LoadRawImage(const string &fn, GLuint &tex_name, const int &w0, const int &h0)
{
	const int w = 64;
	const int h = 64;

	GLubyte image[h][w][4];

	// テクスチャ画像の読み込み
	std::ifstream file(fn.c_str(), std::ios::in | std::ios::binary);
	if(file){
		file.read((char *)image, sizeof image);
		file.close();
	}
	else {
		std::cerr << fn << " が開けません" << std::endl;
		return false;
	}

	// テクスチャ作成
	if(tex_name == 0){
		glGenTextures(1, &tex_name);
	}
	
	glBindTexture(GL_TEXTURE_2D, tex_name);

	// テクスチャ画像はバイト単位に詰め込まれている
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	// テクスチャパラメータ
	glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

	// テクスチャの割り当て
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);

	return true;
}



//-----------------------------------------------------------------------------
// MARK:画面保存
//-----------------------------------------------------------------------------

/*! 
 * フレームバッファのRGB情報を一時的なバッファに確保
 * @param[in] w,h 切り取り領域の大きさ
 * @return バッファ(w×h×3)
 */
static unsigned char* SaveFrameBufferToBuf(int w, int h)
{
	float *image = new float[w*h*3];

	glRasterPos2i(0, h);
	glReadPixels(0, 0, w, h, GL_RGB, GL_FLOAT, image);

	unsigned char *byte_image = new unsigned char[w*h*3];

	unsigned char pval0;
	float pval1;
	for(int i = 0; i < w; ++i){
		for(int j = 0; j < h; ++j){
			for(int k = 0; k < 3; ++k){
				pval1 = image[3*(w*(h-j-1)+i)+k]*255.0f;

				if(pval1 < 0){
					pval0 = 0;
				}
				else if(pval1 > 255.0f){
					pval0 = 255;
				}
				else{
					pval0 = (unsigned char) pval1;
				}

				byte_image[3*(w*j+i)+k] = pval0;
			}
		}
	}
	delete [] image;

	return byte_image;
}


/*! 
 * フレームバッファのRGB情報を一時的なバッファに確保
 * @param[in] w,h 切り取り領域の大きさ
 * @param[out] tex_img バッファ(w×h×3)
 */
static bool SaveFrameBufferToBuf(int w, int h, rxTexImage &tex_img)
{
	tex_img.buf.resize(w*h*3);

	glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, &tex_img.buf[0]);

	// 上下反転
	int stride = w*3;
	for(int j = 0; j < h/2; ++j){
		for(int i = 0; i < stride; ++i){
			unsigned char tmp = tex_img.buf[j*stride+i];
			tex_img.buf[j*stride+i] = tex_img.buf[(h-j-1)*stride+i];
			tex_img.buf[(h-j-1)*stride+i] = tmp;
		}
	}

	tex_img.w = w;
	tex_img.h = h;
	tex_img.c = 3;
	
	return true;
}






//-----------------------------------------------------------------------------
// MARK:テクスチャデータの基底クラス
//-----------------------------------------------------------------------------
class rxTexObj
{
protected:
	GLenum m_iTarget;
	bool m_bValid;
	bool m_bManageObjects;
	GLuint m_iTexName;


public:
	// managed = true : デストラクタでオブジェクトを破棄したいとき
	rxTexObj(GLenum tgt, bool managed) : m_iTarget(tgt), m_bValid(false), m_bManageObjects(managed) {}

	~rxTexObj()
	{
		if(m_bManageObjects) Delete();
	}

	void Bind(void)
	{
		if(!m_bValid){
			Generate();
		}
		glBindTexture(m_iTarget, m_iTexName);
	}

	void UnBind(void)
	{
		glBindTexture(m_iTarget, 0);
	}

	void SetParameter(GLenum pname, GLint i){ glTexParameteri(m_iTarget, pname, i); }
	void SetParameter(GLenum pname, GLfloat f){ glTexParameterf(m_iTarget, pname, f);	}
	void SetParameter(GLenum pname, GLint *ip){ glTexParameteriv(m_iTarget, pname, ip); }
	void SetParameter(GLenum pname, GLfloat *fp){ glTexParameterfv(m_iTarget, pname, fp); }

	void Enable(void){ glEnable(m_iTarget); }
	void Disable(void){ glDisable(m_iTarget); }

	GLuint GetTexName(void) const { return m_iTexName; }
	GLenum GetTexTarget(void) const { return m_iTarget; }

	void Delete()
	{
		if(m_bValid){
			glDeleteTextures(1, &m_iTexName);
			m_bValid = false; 
		}
	}
			
	void Generate()
	{ 
		glGenTextures(1, &m_iTexName);
		m_bValid = true;
	}

public:
	bool IsValid() const { return m_bValid; }
};

//-----------------------------------------------------------------------------
// MARK:2Dテクスチャデータ
//-----------------------------------------------------------------------------
template <class T>
class rxTexObj2D_T : public rxTexObj
{
public:
	T *m_pImage;
	int m_iW, m_iH, m_iC;
	GLenum m_iFormat;

public:
	//! コンストラクタ
	rxTexObj2D_T(bool managed = false) : rxTexObj(GL_TEXTURE_2D, managed)
	{
		m_pImage = NULL;
		m_iW = 0;
		m_iH = 0;
		m_iC = 3;
		m_iFormat = GL_RGB;
	}

	//! デストラクタ
	~rxTexObj2D_T()
	{
		if(m_pImage != NULL) delete [] m_pImage;
	}


	/*!
	 * テクスチャサイズの設定
	 * @param[in] iW,iH テクスチャの縦横ピクセル数
	 * @param[in] iC テクスチャの1ピクセルごとのチャンネル数
	 */
	void SetSize(int iW, int iH, int iC = 3)
	{
		if(m_iW != iW || m_iH != iH || m_iC != iC){
			m_iW = iW;
			m_iH = iH;
			m_iC = iC;
			m_iFormat = (m_iC == 4) ? GL_RGBA : GL_RGB;

			if(m_pImage != NULL) delete [] m_pImage;
			m_pImage = new T[m_iW*m_iH*m_iC];
		}
	}

	/*!
	 * テクスチャサイズの取得
	 * @param[out] iW,iH テクスチャの縦横ピクセル数
	 * @param[out] iC テクスチャの1ピクセルごとのチャンネル数
	 */
	void GetSize(int &iW, int &iH, int &iC)
	{
		iW = m_iW;
		iH = m_iH;
		iC = m_iC;
	}

	/*!
	 * ピクセルの色情報の取得
	 * @param[in]  ic,jc ピクセル座標
	 * @param[out] r,g,b ピクセルの色
	 */
	void GetColor(int ic, int jc, T &r, T &g, T &b)
	{
		r = m_pImage[m_iC*(m_iW*jc+ic)+0];
		g = m_pImage[m_iC*(m_iW*jc+ic)+1];
		b = m_pImage[m_iC*(m_iW*jc+ic)+2];
	}

	/*!
	 * ピクセルの色情報の取得(4チャンネルRGBA)
	 * @param[in]  ic,jc ピクセル座標
	 * @param[out] r,g,b,a ピクセルの色
	 */
	void GetColor4(int ic, int jc, T &r, T &g, T &b, T &a)
	{
		r = m_pImage[m_iC*(m_iW*jc+ic)+0];
		g = m_pImage[m_iC*(m_iW*jc+ic)+1];
		b = m_pImage[m_iC*(m_iW*jc+ic)+2];
		if(m_iC == 4) a = m_pImage[m_iC*(m_iW*jc+ic)+3];
	}

	/*!
	 * ピクセルの色情報の設定
	 * @param[in] ic,jc ピクセル座標
	 * @param[in] r,g,b ピクセルの色
	 */
	void SetColor(int ic, int jc, const T &r, const T &g, const T &b)
	{
		m_pImage[m_iC*(m_iW*jc+ic)+0] = r;
		m_pImage[m_iC*(m_iW*jc+ic)+1] = g;
		m_pImage[m_iC*(m_iW*jc+ic)+2] = b;
		if(m_iC == 4) m_pImage[m_iC*(m_iW*jc+ic)+3] = 0;
	}

	/*!
	 * ピクセルの色情報の設定(4チャンネルRGBA)
	 * @param[in] ic,jc ピクセル座標
	 * @param[in] r,g,b,a ピクセルの色
	 */
	void SetColor(int ic, int jc, const T &r, const T &g, const T &b, const T &a)
	{
		m_pImage[m_iC*(m_iW*jc+ic)+0] = r;
		m_pImage[m_iC*(m_iW*jc+ic)+1] = g;
		m_pImage[m_iC*(m_iW*jc+ic)+2] = b;
		if(m_iC == 4) m_pImage[m_iC*(m_iW*jc+ic)+3] = a;
	}

	/*!
	 * テクスチャメモリへ画像データを転送
	 */
	void SetTexture(void){
		GLenum type = GL_UNSIGNED_BYTE;

		if(sizeof(T) == 1){
			type = GL_UNSIGNED_BYTE;
		}
		else if(sizeof(T) == 4){
			type = GL_FLOAT;
		}
		
		glTexImage2D(m_iTarget, 0, GL_RGBA, m_iW, m_iH, 0, m_iFormat, type, m_pImage);
	}

};

typedef rxTexObj2D_T<unsigned char> rxTexObj2D;
typedef rxTexObj2D_T<float> rxTexObj2Df;



//-----------------------------------------------------------------------------
// MARK:キューブマップテクスチャデータ
//-----------------------------------------------------------------------------
class rxCubeMapData
{
public:
	rxTexObj2D tex[6];
	GLenum id;
};



//-----------------------------------------------------------------------------
// MARK:テクスチャ関数
//-----------------------------------------------------------------------------
/*!
 * テクスチャをOpenGLに登録
 * @param tex_width  テクスチャ画像のピクセル数(横)
 * @param tex_height テクスチャ画像のピクセル数(縦)
 * @param image テクスチャ画像を格納した配列
 * @param tex_obj 2Dテクスチャオブジェクト
 */
static void BindTexture(const int &tex_width, const int &tex_height, GLubyte *image, rxTexObj2D &tex_obj)
{
	tex_obj.SetSize(tex_width, tex_height, 4);

	int ic, jc, idx;
	for(jc = 0; jc < tex_height; ++jc){
		for(ic = 0; ic < tex_width; ++ic){
			idx = 3*(jc*tex_width+ic);
			tex_obj.SetColor(ic, jc, image[idx], image[idx+1], image[idx+2]);
		}
	}

	// テクスチャ登録
	tex_obj.Bind();

	// テクスチャパラメータの設定
	tex_obj.SetParameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	tex_obj.SetParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	tex_obj.SetTexture();
	tex_obj.UnBind();
}

/*!
 * テクスチャをOpenGLに登録
 * @param tex_width  テクスチャ画像のピクセル数(横)
 * @param tex_height テクスチャ画像のピクセル数(縦)
 * @param image テクスチャ画像を格納した配列
 * @param tex_obj 2Dテクスチャオブジェクト
 */
static void BindTexture(rxTexObj2D &tex_obj)
{
	// テクスチャ登録
	tex_obj.Bind();

	// テクスチャパラメータの設定
	tex_obj.SetParameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	tex_obj.SetParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	tex_obj.SetTexture();
	tex_obj.UnBind();
}



/*! テクスチャをビットマップで保存
 * @param[in] file_name 保存ファイル名
 * @param[in] tex_obj 保存したいテクスチャオブジェクト
 * @param[out] tex_img バッファ(w×h×3)
 * @retval true 保存成功
 * @retval false 保存失敗
 */
static bool SaveTextureToBuf(const string &fn, rxTexObj2D &tex_obj, rxTexImage &tex_img)
{
	int w = tex_obj.m_iW;
	int h = tex_obj.m_iH;
	int c = 3;
	tex_img.buf.resize(w*h*c);

	int ic, jc, idx;
	for(jc = 0; jc < h; ++jc){
		for(ic = 0; ic < w; ++ic){
			idx = 3*(jc*w+ic);
			tex_img.buf[idx+0] = tex_obj.m_pImage[idx];
			tex_img.buf[idx+1] = tex_obj.m_pImage[idx+1];
			tex_img.buf[idx+2] = tex_obj.m_pImage[idx+2];
		}
	}

	tex_img.w = w;
	tex_img.h = h;
	tex_img.c = 3;

	return true;
}

/*! 
 * ReadTexture テクスチャの読み込み
 * @param[in] path テクスチャ画像のパス
 * @param[out] tex_obj テクスチャオブジェクト
 * @param[in] tex_img バッファ
 * @return テクスチャが読み込めたかどうか
 */
static bool ReadTextureFromBuf(const string &fn, rxTexObj2D &tex_obj, const rxTexImage &tex_img)
{
	if(tex_img.buf.empty()){
		return false;
	}

	int w = tex_img.w;
	int h = tex_img.h;
	int c = tex_img.c;

	tex_obj.SetSize(w, h, c);

	int ic, jc;
	for(jc = 0; jc < tex_obj.m_iH; ++jc){
		for(ic = 0; ic < tex_obj.m_iW; ++ic){
			int idx = 3*(jc*w+ic);
			tex_obj.SetColor(ic, h-jc-1, tex_img.buf[idx+0], tex_img.buf[idx+1], tex_img.buf[idx+2]);
		}
	}

	// テクスチャ登録
	tex_obj.Bind();

	// テクスチャパラメータの設定
	tex_obj.SetParameter(GL_TEXTURE_WRAP_S, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_WRAP_T, GL_REPEAT);
	tex_obj.SetParameter(GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	tex_obj.SetParameter(GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	tex_obj.SetTexture();

	return true;
}


/*!
 * OpenGLテクスチャ登録
 * @param[in] fn ファイル名
 * @param[inout] tex_name テクスチャ名(0なら新たに生成)
 * @param[in] mipmap ミップマップ使用フラグ
 * @param[in] compress テクスチャ圧縮使用フラグ
 */
static int LoadGLTextureFromBuf(const string &fn, GLuint &tex_name, const rxTexImage &tex_img, 
								bool mipmap, bool compress)
{
	// 画像読み込み
	if(tex_img.buf.empty()){
		return false;
	}

	int w = tex_img.w;
	int h = tex_img.h;
	int c = tex_img.c;
	
	//cout << "image : " << w << " x " << h << " x " << c << endl;
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
 
		glTexImage2D(GL_TEXTURE_2D, 0, iformat, w, h, 0, format, GL_UNSIGNED_BYTE, &tex_img.buf[0]);
 
		if(mipmap){
			glGenerateMipmapEXT(GL_TEXTURE_2D);
		}
	}
	else{
		glBindTexture(GL_TEXTURE_2D, tex_name);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, format, GL_UNSIGNED_BYTE, pimg);
		glTexImage2D(GL_TEXTURE_2D, 0, iformat, w, h, 0, format, GL_UNSIGNED_BYTE, &tex_img.buf[0]);

		if(mipmap){
			glGenerateMipmapEXT(GL_TEXTURE_2D);
		}
	}
 
	glBindTexture(GL_TEXTURE_2D, 0);

	return 1;
}


#if 0
/*! 
 * 環境マップ用のキューブマップテクスチャの読み込み
 * @param[in] fn[6] テクスチャ画像(6枚)のパス(x+,x-,y+,y-,z+,z-)(右,左,上,下,後,前)
 * @param[out] cube_map rxCubeMapData型
 * @retval true  キューブマップ用画像の読み込み成功
 * @retval false キューブマップ用画像の読み込み失敗
 */
static bool LoadCubeMapTexture(const string fn[6], rxCubeMapData &cube_map)
{
	GLuint tex_name;
	glGenTextures(1, &tex_name);
	glBindTexture(GL_TEXTURE_CUBE_MAP, tex_name);

	// キューブマップテクスチャパラメータの設定
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);		// 画像境界の扱いの指定
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	// 画像フィルタの指定
	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 6);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	
	GLenum target[6] = { GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
						 GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
						 GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };

	for(int i = 0; i < 6; ++i){
		int w, h, c;
		unsigned char* pimg;
		pimg = ReadImageFile(fn[i], w, h, c);
		if(!pimg){
			return false;
		}

		GLuint format;
		format = GL_RGBA;
		if(c == 1){
			format = GL_LUMINANCE;
		}
		else if(c == 3){
			format = GL_RGB;
		}

		GLuint iformat;
		iformat = GL_RGBA;
		if(c == 1){
			iformat = GL_LUMINANCE;
		}
		else if(c == 3){
			iformat = GL_RGB;
		}

		gluBuild2DMipmaps(target[i], format, w, h, iformat, GL_UNSIGNED_BYTE, pimg); 


		free(pimg);	
	}

	glBindTexture(GL_TEXTURE_2D, 0);	

	cube_map.id = tex_name;

	return true;
}

/*! 
 * 環境マップ用のキューブマップテクスチャの読み込み
 * @param cube_map キューブマップデータ
 * @param base キューブマップ用画像のファイル名のベース部分
 * @param ext キューブマップ用画像のファイルの拡張子
 * @retval true  キューブマップ用画像の読み込み成功
 * @retval false キューブマップ用画像の読み込み失敗
 */
static bool LoadCubeMap(rxCubeMapData &cube_map, string base, string ext)
{
	// キューブマップ用画像の読み込み(x+,x-,y+,y-,z+,z-)(右,左,上,下,後,前)
	string fn[6];
	fn[0] = base+"posx"+ext;
	fn[1] = base+"negx"+ext;
	fn[2] = base+"posy"+ext;
	fn[3] = base+"negy"+ext;
	fn[4] = base+"posz"+ext;
	fn[5] = base+"negz"+ext;

	if(!LoadCubeMapTexture(fn, cube_map)){
		return false;
	}

	return true;
}
#endif


/*! 
 * キューブマップテクスチャを内部に貼り付けた立方体の描画
 * @param[in] cube_map キューブマップデータ
 * @param[in] side 立方体の一辺の長さ
 */
static void DrawCubeMap(const rxCubeMapData &cube_map, double side)
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glDisable(GL_CULL_FACE);

	// bind textures
//	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_CUBE_MAP);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map.id);

	// initialize object linear texgen
	glPushMatrix();
	GLfloat s_plane[] = { 1.0, 0.0, 0.0, 0.0 };
	GLfloat t_plane[] = { 0.0, 1.0, 0.0, 0.0 };
	GLfloat r_plane[] = { 0.0, 0.0, 1.0, 0.0 };
	glTexGenfv(GL_S, GL_OBJECT_PLANE, s_plane);
	glTexGenfv(GL_T, GL_OBJECT_PLANE, t_plane);
	glTexGenfv(GL_R, GL_OBJECT_PLANE, r_plane);
	glPopMatrix();

	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);

	glEnable(GL_TEXTURE_GEN_S);
	glEnable(GL_TEXTURE_GEN_T);
	glEnable(GL_TEXTURE_GEN_R);

	glPushMatrix();
	glutSolidCube(side);
	glPopMatrix();

	glDisable(GL_TEXTURE_GEN_S);
	glDisable(GL_TEXTURE_GEN_T);
	glDisable(GL_TEXTURE_GEN_R);

	glDisable(GL_TEXTURE_CUBE_MAP);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
}


#endif // #ifndef _TEXTURE_H_