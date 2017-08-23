/*! 
  @file rx_bitmap.h
	
  @brief ビットマップファイル読み込み，書き出し
	- http://ja.wikipedia.org/wiki/Windows_bitmap

	- OS/2(V1), Windows(V3,V4,V5)の非圧縮ビットマップに対応
	- ビットフィールド付きビットマップ，カラーパレット付きビットマップに対応

	- RLE, JPEG, PNG圧縮は未対応
	- V4,V5のカラーマネジメント，プロファイルには未対応
 
  @author Makoto Fujisawa
  @date 2011-06
*/
// FILE --rx_bitmap.h--

#ifndef _RX_BITMAP_H_
#define _RX_BITMAP_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>

#include <vector>
#include <string>


//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
enum rxBitmapType
{
	RX_BMP_OS2_V1, 
	RX_BMP_OS2_V2, 
	RX_BMP_WINDOWS_V3, 
	RX_BMP_WINDOWS_V4, 
	RX_BMP_WINDOWS_V5, 
};

//-----------------------------------------------------------------------------
// 構造体
//-----------------------------------------------------------------------------
//! ICM(Image Color Management)定義 (36ビット,CIEXYZTRIPLE型と同じ)
//  - http://www.adobe.com/jp/support/techguides/color/colormodels/ciexyz.html
struct rxICMColor
{
	long red[3];	//!< R用x,y,z
	long green[3];	//!< G用x,y,z
	long blue[3];	//!< B用x,y,z
};

//! カラーパレット用RGB構造体
struct rxBmpRGB
{
	unsigned char r, g, b;	//!< 赤，緑，青成分
};

//! カラーパレット用RGB構造体(予約領域含む)
struct rxBmpRGB4
{
	unsigned char r, g, b;	//!< 赤，緑，青成分
	unsigned char res;		//!< 予約領域
};

//! ファイルヘッダ (14バイト-2バイト)
// - http://msdn.microsoft.com/en-us/library/dd183374(VS.85).aspx
struct rxBmpFileHeader
{
	//char type[2];			//!< ファイルタイプ(2バイト) - BMPなら'B''M'
	unsigned int size;		//!< ファイルサイズ(4バイト)
	char reserved[4];		//!< 予約領域1,2(2バイト x 2)
	unsigned int offset;	//!< イメージデータまでのオフセット(4バイト) - INFOなら54
};

//! 情報ヘッダ(COREタイプ(OS/2),12バイト)
// - http://msdn.microsoft.com/en-us/library/dd183372(VS.85).aspx
struct rxBmpInfoHeaderOS2
{
	unsigned int size;			//!< ヘッダサイズ(4バイト) - 12バイト
	unsigned short width;		//!< 画像の幅(2バイト) - ピクセル数,1以上
	unsigned short height;		//!< 画像の高さ(2バイト) - ピクセル数,1以上
	unsigned short planes;		//!< 画像のチャンネル数(2バイト) - 1
	unsigned short bitcount;	//!< ビット数/ピクセル(2バイト) - 1,4,8,24
};

//! 情報ヘッダ(INFO,V4,V5タイプ(Windows),40,108,124バイト)
// - INFO : http://msdn.microsoft.com/en-us/library/dd183376(v=VS.85).aspx
// - V4   : http://msdn.microsoft.com/en-us/library/dd183380(VS.85).aspx
// - V5   : http://msdn.microsoft.com/en-us/library/dd183381(VS.85).aspx
struct rxBmpInfoHeaderWin
{
	unsigned int size;			//!< ヘッダサイズ(4バイト) - 40,108,124バイト
	int width;					//!< 画像の幅(4バイト) - ピクセル数,1以上
	int height;					//!< 画像の高さ(4バイト) - ピクセル数,0以外,マイナスの場合はトップダウン形式
	unsigned short planes;		//!< 画像のチャンネル数(2バイト) - 1
	unsigned short bitcount;	//!< ビット数/ピクセル(2バイト) - 0,1,4,8,16,24,32
	unsigned int compression;	//!< 圧縮形式(4バイト) - 0(非圧縮), 1(8ビットRLE), 2(4ビットRLE), 3(ビットフィールド付き非圧縮), 4(JPEG), 5(PNG)
	unsigned int image_size;	//!< 画像データのサイズ(4バイト) - バイト数
	unsigned int xpixel;		//!< 水平解像度(4バイト) - ピクセル数/メートル
	unsigned int ypixel;		//!< 垂直解像度(4バイト) - ピクセル数/メートル
	unsigned int num_color_idx;		//!< 使用する色数(4バイト) - カラーパレットに格納される色数
	unsigned int num_important_idx;	//!< 重要な色数(4バイト) - カラーパレットの重要色の数(表示に必要な数)

	// V4,V5タイプ用
	unsigned int red_mask;		//!< 赤成分のカラーマスク(4バイト)
	unsigned int green_mask;	//!< 緑成分のカラーマスク(4バイト)
	unsigned int blue_mask;		//!< 青成分のカラーマスク(4バイト)
	unsigned int alpha_mask;	//!< アルファ成分のカラーマスク(4バイト)
	unsigned int color_space;	//!< 色空間(4バイト) - 0
	rxICMColor icm;				//!< CIE XYZ(36バイト)
	//CIEXYZTRIPLE icm;
	unsigned int red_gamma;		//!< 赤成分のガンマ値(4バイト)
	unsigned int green_gamma;	//!< 緑成分のガンマ値(4バイト)
	unsigned int blue_gamma;	//!< 青成分のガンマ値(4バイト)

	// V5タイプ用
	unsigned int intent;		//!< sRGB色空間タイプ(4バイト) - ICC32準拠 : 1(Saturation),2(Relative Colorimetric),4(Perceptual),8(Absolute Colorimetic)
	unsigned int profile_data;	//!< プロファイルデータのオフセット(4バイト) - 情報ヘッダの先頭からのオフセットバイト数
	unsigned int profile_size;	//!< プロファイルデータのサイズ(4バイト) - バイト数
	unsigned int reserved;		//!< 予約領域(4バイト) - 0
};

//! ビットフィールド(12バイト)
// - INFOタイプでbitcountが16か32，compressionが3の場合に情報ヘッダの直後に存在
struct rxBmpBitField
{
	unsigned int red_mask;		//!< 赤成分のカラーマスク(4バイト)
	unsigned int green_mask;	//!< 緑成分のカラーマスク(4バイト)
	unsigned int blue_mask;		//!< 青成分のカラーマスク(4バイト)
};


//-----------------------------------------------------------------------------
// 固定小数点数の変換関数(ガンマ値用)
//-----------------------------------------------------------------------------
/*!
 * 浮動小数点数(float)から固定小数点数への変換
 *  - ガンマ値格納用
 *  - 8.8固定小数点数
 *  - 9～16ビットが小数部
 * @param[in] x 浮動小数点数
 * @param[in] shift 固定小数点数の小数部桁数
 * @return 固定小数点数
 */
inline int FloatToFixForGamma(float x)
{
	int power = 1 << 16;
	return (int)(power*x) & 0x00ffff00;
}

/*!
 * 固定小数点数から浮動小数点数(float)への変換
 * @param[in] x 固定小数点数
 * @param[in] shift 固定小数点数の小数部桁数
 * @return 浮動小数点数
 */
inline float FixToFloatForGamma(int x)
{
	int power = 1 << 16;
	return (float)x/(float)power;
}


//-----------------------------------------------------------------------------
// BMPファイルの読み込みと書き込み
//-----------------------------------------------------------------------------
/*!
 * BMPファイルの読み込み
 * @param[in] fn ファイル名
 * @param[out] w,h 画像サイズ
 * @param[out] c 画像の色深度
 * @return 画像データ
 */
static unsigned char* ReadBitmapFile(const std::string &fn, int &w, int &h, int &c)
{
	// ファイルをバイナリモードで開く
	FILE *fp;
	if((fp = fopen(fn.c_str(), "rb")) == NULL){
		fprintf(stderr, "bitmap error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	//fseek(fp, 0L, SEEK_SET);

	// ファイルヘッダの読み込み
	char type[2];
	rxBmpFileHeader file_header;
	fread(type, sizeof(char), 2, fp);						// 2バイト
	fread(&file_header, sizeof(file_header), 1, fp);		// 12バイト

	// 情報ヘッダサイズの読み込み
	unsigned int info_header_size = 0;
	fread(&info_header_size, sizeof(unsigned int), 1, fp);	// 4バイト
	fseek(fp, (long)(sizeof(char)*14), SEEK_SET);

	int bitcount = 0;	// ピクセルビット数
	
	std::vector<rxBmpRGB4> cpalette;	// カラーパレット
	bool use_cmask = false;		// カラーマスク使用フラグ
	unsigned int cmask[4] = {0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000};	// カラーマスク(RGBA)
	int cmask_shift = 0;		// カラーマスクのビット数

	// 情報ヘッダの読み込み
	if(info_header_size == 12 || info_header_size == 64){	// OS/2
		//printf("OS/2\n");
		rxBmpInfoHeaderOS2 info_header;
		fread(&info_header, sizeof(char), info_header_size, fp);

		w = (int)info_header.width;
		h = (int)info_header.height;
		c = (int)ceil((double)info_header.bitcount/8.0);
		bitcount = (int)info_header.bitcount;

		// カラーパレット
		if(bitcount <= 8){
			std::vector<rxBmpRGB> cpalette3;
			int n = 1 << bitcount;
			cpalette3.resize(n);
			fread(&cpalette3[0], sizeof(rxBmpRGB), n, fp);

			cpalette.resize(n);
			for(int i = 0; i < n; ++i){
				cpalette[i].r = cpalette3[i].r;
				cpalette[i].g = cpalette3[i].g;
				cpalette[i].b = cpalette3[i].b;
				cpalette[i].res = 0;
			}
		}
	}
	else if(info_header_size >= 40){	// Windows
		//printf("Windows\n");
		rxBmpInfoHeaderWin info_header;
		fread(&info_header, sizeof(char), info_header_size, fp);	

		w = (int)info_header.width;
		h = (int)info_header.height;
		c = (int)ceil((double)info_header.bitcount/8.0);
		bitcount = (int)info_header.bitcount;

		// ビットフィールド
		if(info_header.compression == 3 && (bitcount == 16 || bitcount == 32)){
			// マスク値を求める
			use_cmask = true;
			if(info_header_size == 40){
				fread(cmask, sizeof(unsigned int), 3, fp);
			}
			else{
				cmask[0] = info_header.red_mask;
				cmask[1] = info_header.green_mask;
				cmask[2] = info_header.blue_mask;
				cmask[3] = info_header.alpha_mask;
			}
			// マスクのビット数(青成分のマスクで1のビットの数を数える)
			cmask_shift = (int)(log((double)(cmask[2]+1))/log(2.0));
		}

		// カラーパレット
		if(bitcount >= 1 && bitcount <= 8 && info_header.num_color_idx){
			std::vector<rxBmpRGB4> cpalette;
			int n = info_header.num_color_idx;
			cpalette.resize(n);
			fread(&cpalette[0], sizeof(rxBmpRGB4), n, fp);
		}
	}
	else{
		fclose(fp);
		return 0;
	}

	bool flip = true;
	if(h < 0){
		// トップダウン形式(画像を上から下へ記録)の場合
		h = -h;
		flip = false;
	}
	
	if(!cpalette.empty() || use_cmask){
		c = 3;
	}

	// 出力データ
	unsigned char* img = new unsigned char[w*h*c];

	if(!cpalette.empty()){
		// データを取得して，カラーパレットを適応し，画像データを作成
		int bitmask = (1 << bitcount)-1;
		int div = 8/bitcount;

		int m = (w*h)/div;	// 各ピクセルのビット数が8ビット(1バイト)としたときの総ピクセル数
		unsigned char* bit_buf = new unsigned char[m];
		fread(bit_buf, sizeof(unsigned char), (size_t)(long)(m), fp);

		unsigned char* buf = bit_buf;
		for(int i = 0; i < m; ++i){
			unsigned char b = *buf;
			for(int j = 0; j < div; ++j){
				int k = (b >> bitcount*(div-j-1)) & bitmask;
				rxBmpRGB4 c = cpalette[k];
				int idx = 3*(div*i+j);
				img[idx+0] = c.r;
				img[idx+1] = c.g;
				img[idx+2] = c.b;
			}
			buf++;
		}

		free(bit_buf);
	}
	else if(use_cmask){
		// データを取得して，ビットフィールドを適用し，画像データを作成
		// bitcountは16か32
		int bitmask = (bitcount == 16 ? 0x0000ffff : 0xffffffff);
		int div = 32/bitcount;
		int m = (w*h)/div;	// 各ピクセルのビット数が32ビット(4バイト)としたときの総ピクセル数
		unsigned int* bit_buf = new unsigned int[m];
		fread(bit_buf, sizeof(unsigned int), (size_t)(long)(m), fp);

		unsigned int* buf = bit_buf;
		for(int i = 0; i < m; ++i){
			unsigned int b = *buf;
			for(int j = 0; j < div; ++j){
				unsigned int c = (b >> bitcount*(div-j-1)) & bitmask;
				int idx = 3*(div*i+j);
				img[idx+0] = (cmask[0] & c) >> cmask_shift*2;
				img[idx+1] = (cmask[1] & c) >> cmask_shift;
				img[idx+2] = (cmask[2] & c);

			}

			buf++;
		}

		free(bit_buf);
	}
	else{
		// 画像データの取得
		fread(img, sizeof(unsigned char), (size_t)(long)(w*h*c), fp);
	}


	// BGR -> RGB
	for(int j = 0; j < h; ++j){
		for(int i = 0; i < w; ++i){
			int idx = 3*(i+j*w);
			unsigned char tmp = img[idx+0];
			img[idx+0] = img[idx+2];
			img[idx+2] = tmp;
		}
	}

	// 上下反転
	if(flip){
		int stride = w*3;
		for(int j = 0; j < h/2; ++j){
			for(int i = 0; i < stride; ++i){
				unsigned char tmp = img[j*stride+i];
				img[j*stride+i] = img[(h-j-1)*stride+i];
				img[(h-j-1)*stride+i] = tmp;
			}
		}
	}

	fclose(fp);
	return img;
}

/*!
 * BMPファイルの書き込み(INFOタイプ)
 * @param[in] fn ファイル名
 * @param[in] img 画像データ
 * @param[in] w,h 画像サイズ
 * @param[in] c 画像の色深度
 * @param[in] type ビットマップ形式
 */
static int WriteBitmapFile(const std::string &fn, unsigned char *img, int w, int h, int c, int type = RX_BMP_WINDOWS_V3)
{
	// ファイルをバイナリモードで開く
	FILE *fp;
	if((fp = fopen(fn.c_str(), "wb")) == NULL){
		fprintf(stderr, "bitmap error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	// ファイルヘッダ
	char file_type[2] = {'B', 'M'};
	fwrite(file_type, sizeof(char), 2, fp);

	rxBmpFileHeader file_header;
	file_header.size = w*h*c+54;
	file_header.reserved[0] = 0;
	file_header.reserved[1] = 0;
	file_header.reserved[2] = 0;
	file_header.reserved[3] = 0;
	file_header.offset = 54;

	fwrite(&file_header, sizeof(file_header), 1, fp);

	if(type == RX_BMP_OS2_V1 || type == RX_BMP_OS2_V2){
		// 情報ヘッダ(OS/2)
		rxBmpInfoHeaderOS2 info_header;
		info_header.size = 12;
		info_header.width = (unsigned short)w;
		info_header.height = (unsigned short)h;
		info_header.planes = 1;
		info_header.bitcount = 8*c;

		fwrite(&info_header, sizeof(info_header), 1, fp);	// 12バイト
	}
	else if(type >= RX_BMP_WINDOWS_V3){
		// 情報ヘッダ(Windows)
		rxBmpInfoHeaderWin info_header;
		info_header.width = w;
		info_header.height = h;
		info_header.planes = 1;
		info_header.bitcount = 8*c;
		info_header.compression = 0;	// 非圧縮
		info_header.image_size = w*h*c;
		info_header.xpixel = 0;
		info_header.ypixel = 0;
		info_header.num_color_idx = 0;
		info_header.num_important_idx = 0;

		if(type == RX_BMP_WINDOWS_V3){
			info_header.size = 40;
		}
		else if(type >= RX_BMP_WINDOWS_V4){
			info_header.size = 108;

			// カラーマスク
			info_header.red_mask   = 0xff0000;
			info_header.green_mask = 0x00ff00;	
			info_header.blue_mask  = 0x0000ff;	
			info_header.alpha_mask = 0x000000;	

			// 色空間
			info_header.color_space = 0;

			// ICM
			rxICMColor icm;
			icm.red[0] = icm.red[1] = icm.red[2] = 0;
			icm.green[0] = icm.green[1] = icm.green[2] = 0;
			icm.blue[0] = icm.blue[1] = icm.blue[2] = 0;
			info_header.icm = icm;

			// ガンマ値
			float gamma = 1.1f;
			info_header.red_gamma = FloatToFixForGamma(gamma);
			info_header.green_gamma = FloatToFixForGamma(gamma);
			info_header.blue_gamma = FloatToFixForGamma(gamma);

			if(type == RX_BMP_WINDOWS_V5){
				info_header.size = 124;

				// V5タイプ用
				info_header.intent = 1;
				info_header.profile_data = 0;
				info_header.profile_size = 0;
				info_header.reserved = 0;
			}
		}
		
		fwrite(&info_header, sizeof(char), info_header.size, fp);
	}
	else{
		return 0;
	}


	// 画像データ
	unsigned char *img_buf = new unsigned char[w*h*c];

	// RGB -> BGR と 上下反転
	for(int j = 0; j < h; ++j){
		for(int i = 0; i < w; ++i){
			int idx0 = c*(i+j*w);
			int idx1 = c*(i+(h-j-1)*w);
			img_buf[idx0+0] = img[idx1+2];
			img_buf[idx0+1] = img[idx1+1];
			img_buf[idx0+2] = img[idx1+0];
		}
	}

	fwrite(img_buf, sizeof(unsigned char), (size_t)(long)(w*h*c), fp);

	delete [] img_buf;

	fclose(fp);

	return 1;
}


#endif // #ifndef _RX_BITMAP_H_
