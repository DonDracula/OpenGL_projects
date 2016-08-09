/*! 
  @file rx_directshow.h
	
  @brief DirectShowを使ったキャプチャ
   書籍「はじめての動画処理プログラミング」土井 滋貴 著 CQ出版
   http://www.cqpub.co.jp/hanbai/books/43/43001.htm
 
  @author Eiichiro Momma,Kenji Takahashi,Makoto Fujisawa
  @date 2007-07-04,2007-11-16,2011-12-12
*/

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <dshow.h>
#pragma include_alias( "dxtrans.h", "qedit.h" )
#define __IDxtCompositor_INTERFACE_DEFINED__
#define __IDxtAlphaSetter_INTERFACE_DEFINED__
#define __IDxtJpeg_INTERFACE_DEFINED__
#define __IDxtKey_INTERFACE_DEFINED__
#include <qedit.h>
#pragma comment(lib,"strmiids.lib")

#include <vector>
#include <string>

using namespace std;


//-----------------------------------------------------------------------------
// rxDirectShowクラス
//-----------------------------------------------------------------------------
class rxDirectShow
{
	IEnumMoniker *m_pClassEnum;
	IMoniker *m_pMoniker;

	IMediaControl *m_pMC;		// メディアコントロール
	ISampleGrabber *m_pGrab;
	BITMAPINFO m_BitmapInfo;

	ULONG m_cFetched;
	HRESULT m_HResult;
	bool m_bInit;

public:
	//! コンストラクタ
	rxDirectShow()
	{
		m_bInit = true;
		CoInitialize(NULL);	// COMの初期化

		m_pMC = 0;
		m_pGrab = 0;
		m_pClassEnum = 0;
		m_pMoniker = 0;

		//
		// キャプチャフィルタの準備
		// 

		// キャプチャデバイスを探す
		// デバイス列挙子を作成
		ICreateDevEnum *dev_enum;
		CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC,IID_ICreateDevEnum, (void**)&dev_enum);

		// ビデオキャプチャデバイス列挙子を作成
		dev_enum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &m_pClassEnum, 0);
		if(m_pClassEnum == NULL){
			printf("ビデオキャプチャデバイスは見つかりませんでした\n");
			CoUninitialize();
			m_bInit = false;
		}

		dev_enum->Release();
	}

	//! デストラクタ
	~rxDirectShow()
	{
		// インターフェースのリリース
		if(m_pMC) m_pMC->Release();     
		if(m_pGrab) m_pGrab->Release();
		if(m_pClassEnum) m_pClassEnum->Release();
		if(m_pMoniker) m_pMoniker->Release();

		// COMのリリース
		CoUninitialize();
	}

	//! デバイスリストの取得
	void GetDeviceList(vector<string> &list)
	{
		// リストクリア
		list.clear();

		if(m_pClassEnum == NULL) return;

		// EnumMonikerをResetする
		m_pClassEnum->Reset();

		// デバイス走査
		while(m_pClassEnum->Next(1, &m_pMoniker, &m_cFetched) == S_OK){
			IPropertyBag *pPropertyBag;
			char devname[256];

			// IPropertyBagにbindする
			m_pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pPropertyBag);

			VARIANT var;

			// FriendlyNameを取得
			var.vt = VT_BSTR;
			pPropertyBag->Read(L"FriendlyName", &var, 0);
			WideCharToMultiByte(CP_ACP, 0, var.bstrVal, -1, devname, sizeof(devname), 0, 0);
			VariantClear(&var);

			list.push_back(devname);
		}
	}

	// 初期化
	void Init(int dev = 0)
	{
		if(m_pClassEnum == NULL) return;

		// EnumMonikerをResetする
		m_pClassEnum->Reset();

		// デバイス走査
		int i = 0;
		while(m_pClassEnum->Next(1, &m_pMoniker, &m_cFetched) == S_OK){
			if(i == dev) break;
			i++;
		}

		// MonkierをFilterにBindする
		IBaseFilter *cfilter;		// キャプチャフィルタ
		m_pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&cfilter);


		//
		// フィルタグラフの準備
		//
	
		// フィルタグラフを作り、インターフェースを得る
		IGraphBuilder *fgraph;	// フィルタグラフ
		CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC, IID_IGraphBuilder, (void**)&fgraph);
		fgraph->QueryInterface(IID_IMediaControl, (LPVOID *) &m_pMC);

		// キャプチャフィルタをフィルタグラフに追加
		fgraph->AddFilter(cfilter, L"Video Capture");


		//		
		// グラバフィルタの準備
		//
		AM_MEDIA_TYPE amt;

		// グラバフィルタを作る
		IBaseFilter *gfilter;			// サンプルグラバフィルタ
		CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (LPVOID*)&gfilter);
		gfilter->QueryInterface(IID_ISampleGrabber, (void**)&m_pGrab);

		// グラバフィルタの挿入場所の特定のための設定
		ZeroMemory(&amt, sizeof(AM_MEDIA_TYPE));
		amt.majortype  = MEDIATYPE_Video;
		amt.subtype    = MEDIASUBTYPE_RGB24;
		amt.formattype = FORMAT_VideoInfo; 
		m_pGrab->SetMediaType(&amt);

		// グラバフィルタをフィルタグラフに追加
		fgraph->AddFilter(gfilter, L"SamGra");


		//
		// キャプチャグラフの準備
		//
		// キャプチャグラフを作る   
		ICaptureGraphBuilder2 *cgraph;      // キャプチャグラフ
		CoCreateInstance(CLSID_CaptureGraphBuilder2 , NULL, CLSCTX_INPROC, IID_ICaptureGraphBuilder2, (void**)&cgraph);

		// フィルタグラフをキャプチャグラフに組み込む
		cgraph->SetFiltergraph(fgraph);
		

		//
		// 解像度の設定
		//
		IAMStreamConfig *sconfig = NULL;
		cgraph->FindInterface(&PIN_CATEGORY_CAPTURE, 0, cfilter, IID_IAMStreamConfig, (void**)&sconfig);

		// ピンがサポートするフォーマット機能の数を取得
		int pin_count, pin_size;
		sconfig->GetNumberOfCapabilities(&pin_count, &pin_size);

		// 画面解像度の設定
		AM_MEDIA_TYPE *media_type;	// メディア サンプルのフォーマット構造体
		VIDEO_STREAM_CONFIG_CAPS vsconfig;				// ビデオ フォーマット構造体
		if(pin_size == sizeof(VIDEO_STREAM_CONFIG_CAPS)){
			for(int i = 0; i < pin_count; ++i){
				HRESULT hr = sconfig->GetStreamCaps(i, &media_type, reinterpret_cast<BYTE*>(&vsconfig));
				if(SUCCEEDED(hr)){
					if ((media_type->formattype == FORMAT_VideoInfo) && 
						(media_type->cbFormat >= sizeof(VIDEOINFOHEADER)) &&
						(media_type->pbFormat != NULL) ) 
					{
						// VIDEOINFOHEADERの取得
						hr = sconfig->GetFormat(&media_type);
						VIDEOINFOHEADER *video_info = reinterpret_cast<VIDEOINFOHEADER*>(media_type->pbFormat);
					
						// 解像度の変更
						video_info->bmiHeader.biWidth = 1920;
						video_info->bmiHeader.biHeight = 1080;

						//RXCOUT << video_info->bmiHeader.biWidth << " x ";
						//RXCOUT << video_info->bmiHeader.biHeight << endl;

						// キャストして代入
						media_type->pbFormat = (BYTE*)video_info;

						// SetFormatでpbFormatの変更を適用
						sconfig->SetFormat(media_type);
					}
				}
			}
		}


		//
		// レンダリングの設定
		//
		// キャプチャグラフの設定、グラバをレンダリング出力に設定
		cgraph->RenderStream(&PIN_CATEGORY_PREVIEW, &MEDIATYPE_Video, cfilter, NULL, gfilter);

		// ビットマップ情報の取得  
		m_pGrab->GetConnectedMediaType(&amt); 

		// ビデオ ヘッダーへのポインタを獲得する。
		printf( "SampleSize = %d (byte)\n", amt.lSampleSize );
		VIDEOINFOHEADER *pVideoHeader = (VIDEOINFOHEADER*)amt.pbFormat;

		// ビデオ ヘッダーには、ビットマップ情報が含まれる。
		// ビットマップ情報を BITMAPINFO 構造体にコピーする。
		ZeroMemory(&m_BitmapInfo, sizeof(m_BitmapInfo) );
		CopyMemory(&m_BitmapInfo.bmiHeader, &(pVideoHeader->bmiHeader), sizeof(BITMAPINFOHEADER));

		if(cfilter) cfilter->Release();
		if(gfilter) gfilter->Release();
		if(fgraph) fgraph->Release();
		if(cgraph) cgraph->Release();      
	}

	// 使用可能かどうか問い合わせ
	bool Flag(void){ return m_bInit; }

	//! キャプチャスタート
	void StartCapture(void)
	{
		m_pMC->Run();						// レンダリング開始
		m_pGrab->SetBufferSamples(TRUE);	// グラブ開始
	}

	//! キャプチャストップ
	void StopCapture(void)
	{
		m_pMC->Stop();						// レンダリング停止
		m_pGrab->SetBufferSamples(FALSE);	// グラブ停止
	}

	//! フレーム画像取得
	inline void QueryFrame(long *imageData)
	{
		m_HResult = m_pGrab->GetCurrentBuffer((long*)&(m_BitmapInfo.bmiHeader.biSizeImage), (long*)imageData);
	}

	//! キャプチャ動画の情報
	int GetWidth(void){ return m_BitmapInfo.bmiHeader.biWidth; }
	int GetHeight(void){ return m_BitmapInfo.bmiHeader.biHeight; }
	int GetCount(void){ return m_BitmapInfo.bmiHeader.biBitCount; }
	int GetColor(void)
	{ 
		return (int)(m_BitmapInfo.bmiHeader.biSizeImage/(m_BitmapInfo.bmiHeader.biWidth*m_BitmapInfo.bmiHeader.biHeight));
	}
};