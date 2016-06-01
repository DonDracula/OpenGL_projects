/*!
  @file rx_trackball.h
	
  @brief 簡易トラックボール処理
 
  @author Makoto Fujisawa
  @date 2008
*/
// FILE --rx_trackball.h--

#ifndef _RX_TRACKBALL_H_
#define _RX_TRACKBALL_H_


//-----------------------------------------------------------------------------
// 四元数関数
//-----------------------------------------------------------------------------
void QuatLerp(double p[], const double q0[], const double q1[], const double t);
void QuatMultiLerp(double p[], const double t[], const double q[][4], const int n, const double u);
void SetQuat(double q[], const double ang, const double ax, const double ay, const double az);


//-----------------------------------------------------------------------------
// トラックボール関数
//-----------------------------------------------------------------------------
class rxTrackball
{
public:
	rxTrackball();
	~rxTrackball();

	// 初期化と設定
	void Init(void);			//!< トラックボール処理の初期化
	void SetRegion(int w, int h);	//!< 画面領域指定
	void SetSpace(double l);		//!< 3D空間の大きさ指定

	// マウスドラッグ
	void Start(int x, int y, int button);	//!< ドラッグ開始
	void Motion(int x, int y, bool last = false);				//!< ドラッグ中
	void Stop(int x, int y);				//!< ドラッグ停止

	void TrackballLastRot(void);
	void TrackballRot(double vx, double vy);

	// アクセスメソッド
	void GetQuaternion(double q[4]) const;
	double GetScaling(void) const;		//!< スケーリング量を返す
	void GetScaling(double &s) const;
	double GetTranslation(int i) const;
	void GetTranslation(double t[2]) const;

	double *GetRotation(void);		//!< 回転の変換行列を返す
	double *GetQuaternionR(void);	//!< 四元数を返す(q[4])
	double *GetTranslationR(void);	//!< 平行移動量を返す(t[2])
	double *GetScalingR(void);		//!< スケーリング量を返す(s)

	void GetTransform(double m[16]);

	void GetLastVeloc(int &vx, int &vy);

	void CalGlobalPos(double dst[4], const double src[4]);
	void CalLocalPos(double dst[4], const double src[4]);

	void CalGlobalRot(double dst[4], const double src[4]);
	void CalLocalRot(double dst[4], const double src[4]);

	void GetViewPosition(double pos[3]);
	void GetViewDirection(double dir[3]);

	// 描画設定
	void ApplyRotation(void);		//!< 回転をOpenGLに設定
	void ApplyScaling(void);		//!< スケーリングをOpenGLに設定
	void ApplyTranslation(void);	//!< 平行移動量をOpenGLに設定

	void Apply(void);

	void ApplyQuaternion(const double q[4]);		//!< 回転をOpenGLに設定


	void SetScaling(double z);					
	void SetTranslation(double x, double y);
	void SetQuaternion(double q[4]);
	void SetRotation(double ang, double x, double y, double z);
	void SetRotation(double rot[16]);

	void AddRotation(double ang, double x, double y, double z);
	void GetLastRotation(double &ang, double &x, double &y, double &z);

	void GetRayTo(int x, int y, double fov, double ray_to[3]);


private:
	double m_fTransScale;		//!< マウスの相対位置→スケーリング，平行移動量の換算係数

	int m_iSx, m_iSy;	//!< ドラッグ開始位置
	int m_iPx, m_iPy;	//!< ドラッグ開始位置
	double m_fW, m_fH;	//!< マウスの絶対位置→ウィンドウ内での相対位置の換算係数

	int m_iVeloc[2];

	double m_fQuatRot[4];		//!< 回転(クォータニオン)
	double m_fQuatIncRot[4];	//!< ドラッグ中の回転増分量(クォータニオン)
	double m_fMatRot[16];		//!< 回転の変換行列
	
	double m_fTransDist[2];		//!< カメラパン
	double m_fScaleDist;		//!< スケーリング
	int m_iDrag;				//!< ドラッグ中か否か(1:左ドラッグ, 2:右ドラッグ, 3:ミドルドラッグ)

	double m_fLastRot[4];		//!< 最後の回転

};


#endif // #ifndef _RX_TRACKBALL_H_