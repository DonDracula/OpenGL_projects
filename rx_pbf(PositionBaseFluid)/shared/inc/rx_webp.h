/*! 
  @file rx_webp.h
	
  @brief WebP画像ファイル読み込み，書き出し
	- libwebp : http://code.google.com/intl/ja/speed/webp/
 
  @author Makoto Fujisawa
  @date 2011-11
*/
// FILE --rx_webp.h--

#ifndef _RX_WEBP_H_
#define _RX_WEBP_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <cstdio>
#include <string>

#include "webp/decode.h"
#include "webp/encode.h"


//-----------------------------------------------------------------------------
// WebP画像ファイルの読み込み/書き込み
//-----------------------------------------------------------------------------
/*!
 * WebPファイルの読み込み
 * @param[in] fn ファイル名
 * @param[out] w,h 画像サイズ
 * @param[out] c 画像の色深度
 * @return 展開済み画像データ
 */
static unsigned char* ReadWebPFile(const std::string &fn, int &w, int &h, int &c)
{
	WebPDecoderConfig config;
	WebPDecBuffer* const output_buffer = &config.output;
	WebPBitstreamFeatures* const bitstream = &config.input;

	// デコード設定を初期化
	if(!WebPInitDecoderConfig(&config)){
		fprintf(stderr, "WebP error : library version mismatch.\n");
		return 0;
	}

	VP8StatusCode status = VP8_STATUS_OK;
	uint32_t data_size = 0;
	void* data = 0;
	FILE *fp = 0;
	unsigned char* img = 0;

	try{
		// ファイルをバイナリモードで開く
		if((fp = fopen(fn.c_str(), "rb")) == NULL){
			throw("WebP error : cannot open %s file.", fn.c_str());
		}

		fseek(fp, 0, SEEK_END);
		data_size = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		data = malloc(data_size);
		if(fread(data, data_size, 1, fp) != 1){
			throw("WebP error : could not read %d bytes of data from file %s .", data_size, fn.c_str());
		}

		// ビットストリームの特性を取得
		status = WebPGetFeatures((const uint8_t*)data, data_size, bitstream);
		if(status == VP8_STATUS_OK){
			// デコード設定を必要に応じて修正
			config.output.colorspace = MODE_RGBA;

			// デコード
			status = WebPDecode((const uint8_t*)data, data_size, &config);
		}
		else{
			throw("WebP error : decoding of %s failed.", fn.c_str());
		}

		// 出力データ
		w = output_buffer->width;
		h = output_buffer->height;
		c = (bitstream->has_alpha ? 4 : 3);

		const unsigned char* const rgb = output_buffer->u.RGBA.rgba;
		int size = output_buffer->u.RGBA.size;

		// 出力データへコピー
		img = new unsigned char[w*h*c];
		for(int i = 0; i < size; ++i){
			img[i] = rgb[i];
		}
	}
	catch(const char* str){
		fprintf(stderr, "%s\n", str);
	}

	if(fp) fclose(fp);
	if(data) free(data);

	// デコーダを解放
	WebPFreeDecBuffer(output_buffer);

	return img;
}

//! 圧縮設定
struct rxWebPConfig
{
	//! プリセット(default, photo, picture, drawing, icon, text)，プリセットが指定されていたら以下の設定は無視される
	string preset;	

	// 各種設定
	int method;				//!< 圧縮方法[0,6] (0:fast, 6:slowest)
	int target_size;		//!< ターゲットサイズ (byte)
	float target_PSNR;		//!< ターゲットPSNR (dB, 通常42ぐらい)
	int sns_strength;		//!< Spatial Noise Shaping [0,100]
	int filter_strength;	//!< フィルタ強度[0,100] (0でフィルタOFF)
	bool autofilter;		//!< 自動フィルタ設定
	int filter_sharpness;	//!< シャープネスフィルタ[0,7] (0:most, 7:least sharp)
	bool strong;			//!< simpleの代わりにstrongフィルタを用いる
	int pass;				//!< 解析パス数[1,10]
	int segments;			//!< セグメント数[1,4]
	int partition_limit;	//!< 最初のパーティションが512kにフィットするように画質を制限[0,100] (100でfull)
	int alpha_compression;	//!< 透明度(アルファ値)の圧縮を設定

	rxWebPConfig()
	{
		preset.clear();
		method = -1;
		target_size = -1;
		target_PSNR = -1;
		sns_strength = -1;
		filter_strength = -1;
		filter_sharpness = -1;
		autofilter = false;
		strong = false;
		pass = -1;
		segments = -1;
		partition_limit = -1;
		alpha_compression = -1;
	}
};

//! エラーメッセージ
static const char* RX_WEBP_ERROR_MESSAGES[] = 
{
	"OK",
	"OUT_OF_MEMORY: Out of memory allocating objects",
	"BITSTREAM_OUT_OF_MEMORY: Out of memory re-allocating byte buffer",
	"NULL_PARAMETER: NULL parameter passed to function",
	"INVALID_CONFIGURATION: configuration is invalid",
	"BAD_DIMENSION: Bad picture dimension. Maximum width and height is 16383 pixels.", 
	"PARTITION0_OVERFLOW: Partition #0 is too big to fit 512k.", 
	"PARTITION_OVERFLOW: Partition is too big to fit 16M",
	"BAD_WRITE: Picture writer returned an I/O error"
};

//! 画像書き込み
static int RxWebPWriter(const uint8_t* data, size_t data_size, const WebPPicture* const pic)
{
	FILE* const out = (FILE*)pic->custom_ptr;
	return data_size ? (fwrite(data, data_size, 1, out) == 1) : 1;
}


/*!
 * WebPファイルの書き込み
 * @param[in] fn ファイル名
 * @param[in] img 画像データ
 * @param[in] w,h 画像サイズ
 * @param[in] c 画像の色深度
 * @param[in] quality 圧縮品質[0,100]
 * @param[in] rxcfg 圧縮設定
 */
static int WriteWebPFile(const std::string &fn, unsigned char *img, int w, int h, int c, int q, rxWebPConfig rxcfg = rxWebPConfig())
{
	WebPPicture picture;
	WebPConfig config;
	WebPAuxStats stats;
	FILE *fp = 0;
	int rtn = 1;

	try{
		// 入力データの初期化
		if(!WebPPictureInit(&picture) || !WebPConfigInit(&config)){
			throw("WebP error : version mismatch.");
		}

		picture.width = w;
		picture.height = h;
		int stride = c*w*sizeof(*img);
		if(c == 4){
			WebPPictureImportRGBA(&picture, img, stride);
		}
		else{
			WebPPictureImportRGB(&picture, img, stride);
		}

		config.quality = (float)q;

		if(rxcfg.preset.empty()){
			if(rxcfg.method >= 0) config.method = rxcfg.method;
			if(rxcfg.target_size >= 0) config.target_size = rxcfg.target_size;
			if(rxcfg.target_PSNR >= 0) config.target_PSNR = rxcfg.target_PSNR;
			if(rxcfg.sns_strength >= 0) config.sns_strength = rxcfg.sns_strength;
			if(rxcfg.filter_strength >= 0) config.filter_strength = rxcfg.filter_strength;
			if(rxcfg.method >= 0) config.filter_sharpness = rxcfg.filter_sharpness;
			if(rxcfg.autofilter) config.autofilter = 1;
			if(rxcfg.strong) config.filter_type = 1;
			if(rxcfg.pass >= 0) config.pass = rxcfg.pass;
			if(rxcfg.segments >= 0) config.segments = rxcfg.segments;
			if(rxcfg.partition_limit >= 0) config.partition_limit = rxcfg.partition_limit;
			if(rxcfg.alpha_compression >= 0) config.alpha_compression = rxcfg.alpha_compression;
		}
		else{
			WebPPreset preset;
			if(rxcfg.preset == "default"){
				preset = WEBP_PRESET_DEFAULT;
			}
			else if(rxcfg.preset == "photo"){
				preset = WEBP_PRESET_PHOTO;
			}
			else if(rxcfg.preset == "picture"){
				preset = WEBP_PRESET_PICTURE;
			}
			else if(rxcfg.preset == "drawing"){
				preset = WEBP_PRESET_DRAWING;
			}
			else if(rxcfg.preset == "icon"){
				preset = WEBP_PRESET_ICON;
			}
			else if(rxcfg.preset == "text"){
				preset = WEBP_PRESET_TEXT;
			}
			else{
				preset = WEBP_PRESET_DEFAULT;
			}

			// プリセットを設定
			if(!WebPConfigPreset(&config, preset, config.quality)){
				throw("WebP error : preset error.");
			}
		}

		// 出力設定を検証
		if(!WebPValidateConfig(&config)){
			throw("WebP error : invalid configuration.");
		}

		// ファイルをバイナリモードで開く
		if((fp = fopen(fn.c_str(), "wb")) == NULL){
			throw("WebP error : cannot open %s file", fn.c_str());
		}

		picture.writer = RxWebPWriter;
		picture.custom_ptr = (void*)fp;
		picture.stats = &stats;

		// 圧縮
		if(!WebPEncode(&config, &picture)){
			throw("WebP error : can't encode picture (code : %d - %s).", 
				  picture.error_code, RX_WEBP_ERROR_MESSAGES[picture.error_code]);
		}
	}
	catch(const char* str){
		fprintf(stderr, "%s\n", str);
		rtn = 0;
	}

	free(picture.extra_info);
	if(fp) fclose(fp);

	// エンコーダを解放
	WebPPictureFree(&picture);
	
	return rtn;
}


#endif // #ifdef _RX_WEBP_H_
