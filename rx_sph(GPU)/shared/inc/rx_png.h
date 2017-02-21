/*! 
  @file rx_png.h
	
  @brief PNGファイル読み込み，書き出し
	- libpng : http://www.libpng.org/pub/png/libpng.html
	- zlib : http://www.zlib.net/
 
  @author Makoto Fujisawa
  @date 2011-06
*/
// FILE --rx_png.h--

#ifndef _RX_PNG_H_
#define _RX_PNG_H_



//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <string>

#include <png.h>


//-----------------------------------------------------------------------------
// PNGファイルの読み込みと書き込み
//-----------------------------------------------------------------------------
inline void ReadPngCallback(png_structp png_ptr, png_uint_32 row_number, int pass)
{
	
}

/*!
 * PNGファイルの読み込み
 * @param[in] fn ファイル名
 * @param[out] w,h 画像サイズ
 * @param[out] c 画像の色深度
 * @return 展開済み画像データ
 */
static unsigned char* ReadPngFile(const std::string &fn, int &w, int &h, int &c)
{
	png_FILE_p fp;
	unsigned char* img = 0;	// 出力データ

	// ファイルをバイナリモードで開く
	if((fp = fopen(fn.c_str(), "rb")) == NULL){
		fprintf(stderr, "png error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	png_structp read_ptr;
	png_infop read_info_ptr;

	// PNG読み込みオブジェクト生成
	read_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	read_info_ptr = png_create_info_struct(read_ptr);
	
	// 読み込むデータを登録
	png_init_io(read_ptr, fp);

	// 読み込み時のコールバックの設定(読み込み状況を表示したい場合など)
	png_set_read_status_fn(read_ptr, NULL);

	// ヘッダ情報の読み込み
	png_read_info(read_ptr, read_info_ptr);

	// 画像情報の取得
	png_uint_32 width, height;
	int bit_depth, color_type;
	int interlace_type, compression_type, filter_type;

	if(png_get_IHDR(read_ptr, read_info_ptr, &width, &height, &bit_depth,
					&color_type, &interlace_type, &compression_type, &filter_type))
	{
		w = (int)width;
		h = (int)height;
		c = 0;

		// チャンネル数
		c += (color_type & PNG_COLOR_MASK_COLOR) ? 3 : 0;
		c += (color_type & PNG_COLOR_MASK_ALPHA) ? 1 : 0;
		
		if(c && (color_type & PNG_COLOR_MASK_PALETTE)){
			png_set_palette_to_rgb(read_ptr);
			c = 4;
		}
		else if(!c){	// color_type == PNG_COLOR_TYPE_GRAY
			c = 1;
		}

		if(!c){
			png_destroy_read_struct(&read_ptr, &read_info_ptr, NULL);
			fclose(fp);
			return 0;
		}

		// 各行へのポインタを使って画像を読み込み
		img = new unsigned char[w*h*c];
		unsigned char **lines = new unsigned char*[h];
		for(int j = 0; j < h; ++j) lines[j] = &img[j*w*c];

		png_read_image(read_ptr, lines);

		delete [] lines;
	}
	
	png_destroy_read_struct(&read_ptr, &read_info_ptr, NULL);
	fclose(fp);

	return img;
}

/*!
 * PNGファイルの書き込み
 * @param[in] fn ファイル名
 * @param[in] img 画像データ
 * @param[in] w,h 画像サイズ
 * @param[in] c 画像の色深度
 * @param[in] quality 圧縮品質[0,100]
 */
static int WritePngFile(const std::string &fn, unsigned char *img, int w, int h, int c)
{
	png_FILE_p fp;

	// ファイルをバイナリモードで開く
	if((fp = fopen(fn.c_str(), "wb")) == NULL){
		fprintf(stderr, "png error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	png_structp write_ptr;
	png_infop write_info_ptr;

	// PNG読み込みオブジェクト生成
	write_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	write_info_ptr = png_create_info_struct(write_ptr);

	if(setjmp(png_jmpbuf(write_ptr))){
		fclose(fp);
		return 0;
	}
	
	// 書き込むファイルを登録
	png_init_io(write_ptr, fp);

	// 読み込み時のコールバックの設定(読み込み状況を表示したい場合など)
	png_set_write_status_fn(write_ptr, NULL);

	// カラータイプ
	int color_type = 0;
	if(c == 1) color_type = PNG_COLOR_TYPE_GRAY;
	else if(c == 3) color_type = PNG_COLOR_TYPE_RGB;
	else if(c == 4) color_type = PNG_COLOR_TYPE_RGB_ALPHA;
	
	// ヘッダ情報の登録
	png_set_IHDR(write_ptr, write_info_ptr, w, h, 8, color_type, 
				 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_BASE);

	png_color_8 sig_bit;
	sig_bit.red = 8;
	sig_bit.green = 8;
	sig_bit.blue = 8;
	sig_bit.alpha = 0;
	png_set_sBIT(write_ptr, write_info_ptr, &sig_bit);

	// PNGに書き込まれるコメント
	png_text text_ptr[1];
	text_ptr[0].key = "Description";
	text_ptr[0].text = "Saved by libpng";
	text_ptr[0].compression = PNG_TEXT_COMPRESSION_NONE;
	png_set_text(write_ptr, write_info_ptr, text_ptr, 1);

	// ヘッダ情報の書き込み
	png_write_info(write_ptr, write_info_ptr);

	//png_set_bgr(write_ptr);

	// 各行へのポインタを使って画像を読み込み
	unsigned char **lines = new unsigned char*[h];
	for(int j = 0; j < h; ++j) lines[j] = &img[j*w*c];

	png_write_image(write_ptr, lines);

	delete [] lines;

	png_write_end(write_ptr, write_info_ptr);
	png_destroy_write_struct(&write_ptr, &write_info_ptr);
	fclose(fp);

	return 1;
}




#endif // #ifdef _RX_PNG_H_
