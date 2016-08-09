/*! 
  @file rx_jpeg.h
	
  @brief JPEGファイル読み込み，書き出し
	- libjpegを使用
	- http://www.ijg.org/
 
  @author Makoto Fujisawa
  @date 2011-06
*/
// FILE --rx_jpeg.h--

#ifndef _RX_JPEG_H_
#define _RX_JPEG_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>

#include <vector>
#include <string>

#include <jpeglib.h>
#include <jerror.h>


//-----------------------------------------------------------------------------
// JPEGファイルの読み込みと書き込み
//-----------------------------------------------------------------------------
/*!
 * JPEGファイルの読み込み
 * @param[in] fn ファイル名
 * @param[out] w,h 画像サイズ
 * @param[out] c 画像の色深度
 * @return 展開済み画像データ
 */
static unsigned char* ReadJpegFile(const std::string &fn, int &w, int &h, int &c)
{
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;

	// JPEG解凍用オブジェクト生成
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);

	// ファイルをバイナリモードで開く
	FILE *fp;
	if((fp = fopen(fn.c_str(), "rb")) == NULL){
		fprintf(stderr, "jpeg error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	// 解凍するデータを指定
	jpeg_stdio_src(&cinfo, fp);

	// ファイルヘッダの読み込み
	jpeg_read_header(&cinfo, TRUE);

	// 画像色深度
	c = cinfo.num_components;
	if(!c){
		fprintf(stderr, "jpeg error : the number of color components is zero\n");
		return 0;
	}

	// 画像サイズ
	w = cinfo.image_width;
	h = cinfo.image_height;
	if(!w || !h){
		fprintf(stderr, "jpeg error : size of the image is zero\n");
		return 0;
	}

	// データを解凍
	jpeg_start_decompress(&cinfo);

	// データ
	JSAMPARRAY buf;
	buf = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, w*c, (JDIMENSION)1);

	// 出力データ
	unsigned char* img = new unsigned char[w*h*c];

	// 1行ずつコピー
	unsigned char* dst_ptr = img;
	unsigned char* src_ptr;
	while(cinfo.output_scanline < (unsigned int)h){
		jpeg_read_scanlines(&cinfo, buf, 1);
		src_ptr = buf[0];
		for(int i = 0; i < w*c; ++i){
			*dst_ptr++ = *src_ptr++;
		}
	}
	
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(fp);

	return img;
}

/*!
 * JPEGファイルの書き込み
 * @param[in] fn ファイル名
 * @param[in] img 画像データ
 * @param[in] w,h 画像サイズ
 * @param[in] c 画像の色深度
 * @param[in] quality 圧縮品質[0,100]
 */
static int WriteJpegFile(const std::string &fn, unsigned char *img, int w, int h, int c, int quality)
{
	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;

	// JPEG圧縮用オブジェクト生成
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);

	// ファイルをバイナリモードで開く
	FILE *fp;
	if((fp = fopen(fn.c_str(), "wb")) == NULL){
		fprintf(stderr, "jpeg error : cannot open %s file\n", fn.c_str());
		return 0;
	}

	// 出力ファイルを指定
	jpeg_stdio_dest(&cinfo, fp);

	// 画像ファイル情報
	cinfo.image_width = w;
	cinfo.image_height = h;
	cinfo.input_components = c;
	cinfo.in_color_space = (c == 3 ? JCS_RGB : JCS_GRAYSCALE);

	// 圧縮設定
	jpeg_set_defaults(&cinfo);	// デフォルトのパラメータをセット
	jpeg_set_quality(&cinfo, quality, TRUE);	// 画像品質を設定

	// データを圧縮
	jpeg_start_compress(&cinfo, TRUE);

	// 1行ずつ出力
	unsigned char* src_ptr = img;
	JSAMPARRAY dst_ptr = new JSAMPROW[w*c];
	while(cinfo.next_scanline < (unsigned int)h){
		jpeg_write_scanlines(&cinfo, &src_ptr, 1);
		src_ptr += w*c;
	}
	
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
	fclose(fp);

	return 1;
}




#endif // #ifdef _RX_JPEG_H_
