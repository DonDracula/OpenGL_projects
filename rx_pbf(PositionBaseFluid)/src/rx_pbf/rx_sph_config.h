/*!
  @file rx_sph_config.h
	
  @brief SPHのシーン設定をファイルから読み込む
 
  @author Makoto Fujisawa
  @date 2012-08
*/
// FILE --rx_sph_config.h--

#ifndef _RX_SPH_CONFIG_H_
#define _RX_SPH_CONFIG_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// STL
#include <vector>
#include <string>

// ユーティリティ
#include "rx_utility.h"

// シミュレーション
#include "rx_sph_commons.h"
#include "rx_sph.h"

// 3Dモデル
#include "rx_model.h"
#include "rx_fltk_widgets.h"

// 設定ファイル
#include "rx_atom_ini.h"

using namespace std;


//-----------------------------------------------------------------------------
// 文字列処理関数
//-----------------------------------------------------------------------------
/*!
 * 文字列から値(Vec3)を取得
 * @param[out] val 値
 * @param[in] str 文字列
 * @param[in] rel trueでシミュレーション空間の大きさに対する係数として計算
 * @param[in] cen シミュレーション空間の中心座標
 * @param[in] ext シミュレーション空間の大きさ(各辺の長さの1/2)
 */
inline void GetValueFromString(Vec3 &val, const string &str, bool rel = false, Vec3 cen = Vec3(0.0), Vec3 ext = Vec3(1.0))
{
	int n = StringToVec3(str, val);
	if(rel){
		val = cen+(n == 1 ? RXFunc::Min3(ext) : ext)*val;
	}
}

/*!
 * 文字列から値(double)を取得
 * @param[out] val 値
 * @param[in] str 文字列
 * @param[in] rel trueでシミュレーション空間の大きさに対する係数として計算
 * @param[in] cen シミュレーション空間の中心座標
 * @param[in] ext シミュレーション空間の大きさ(各辺の長さの1/2)
 */
inline void GetValueFromString(double &val, const string &str, bool rel = false, Vec3 cen = Vec3(0.0), Vec3 ext = Vec3(1.0))
{
	val = atof(str.c_str());
	if(rel){
		val = RXFunc::Min3(ext)*val;
	}
}



//-----------------------------------------------------------------------------
//! rxSceneConfigクラス - SPHのシーン設定をファイルから読み込む
//-----------------------------------------------------------------------------
class rxSceneConfig
{
protected:
	rxParticleSystemBase *m_pSolver;	//!< ソルバ
	rxEnviroment m_Env;					//!< シーン環境設定
	
	string m_strCurrentScene;			//!< 現在のシーンの名前
	vector<string> m_vSceneFiles;		//!< シーンファイルリスト
	int m_iSceneFileNum;				//!< シーンファイルの数

	vector<string> m_vSceneTitles;		//!< シーンファイルリスト
	int m_iCurrentSceneIdx;				//!< 現在のシーンファイル

	vector<rxPolygons> m_vSolidPoly;	//!< 固体メッシュ

public:
	//! デフォルトコンストラクタ
	rxSceneConfig() : m_pSolver(0)
	{
		m_vSceneFiles.resize(12, "");	// シーンファイルリスト
		m_vSceneTitles.resize(12, "");	// シーンタイトルリスト
		m_iCurrentSceneIdx = -1;		// 現在のシーンファイル
		m_strCurrentScene = "null";		// 現在のシーンの名前
		m_iSceneFileNum = 0;

		Clear();
	}

	//! デストラクタ
	~rxSceneConfig(){}
	
	//! 設定初期化
	void Clear(void)
	{
		m_pSolver = 0;
		m_vSolidPoly.clear();
	}


	//! シーンタイトルリスト
	vector<string> GetSceneTitles(void) const { return m_vSceneTitles; }

	//! 現在のシーン
	int GetCurrentSceneIdx(void) const { return m_iCurrentSceneIdx; }

	//! SPH環境
	rxEnviroment GetEnv(void) const { return m_Env; }

	//! SPHクラス
	void Set(rxParticleSystemBase *solver){ m_pSolver = solver; }

	//! 固体ポリゴン
	int GetSolidPolyNum(void) const { return (int)m_vSolidPoly.size(); }
	vector<rxPolygons>& GetSolidPolys(void){ return m_vSolidPoly; }

public:
	/*!
	 * 設定からパラメータの読み込み
	 * @param[in] names 項目名リスト
	 * @param[in] values 値リスト
	 * @param[in] n リストのサイズ
	 * @param[in] header ヘッダ名
	 */
	static void SetSpace_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetSpace(names, values, n, header);
	}
	void SetSpace(string *names, string *values, int n, string header)
	{
		rxEnviroment sph_env;
		sph_env.use_inlet = 0;
		int idr = 0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(sph_env.boundary_cen, values[i], false);
			else if(names[i] == "ext") GetValueFromString(sph_env.boundary_ext, values[i], false);
			else if(names[i] == "max_particle_num")  sph_env.max_particles = atoi(values[i].c_str());
			else if(names[i] == "density")			 sph_env.dens = atof(values[i].c_str());
			else if(names[i] == "mass")				 sph_env.mass = atof(values[i].c_str());
			else if(names[i] == "kernel_particles")  sph_env.kernel_particles = atof(values[i].c_str());
			else if(names[i] == "mesh_res_max")		 sph_env.mesh_max_n = atoi(values[i].c_str());
			else if(names[i] == "inlet_boundary")	 sph_env.use_inlet = atoi(values[i].c_str());
			else if(names[i] == "dt")				 sph_env.dt = atof(values[i].c_str());
			else if(names[i] == "viscosity")		 sph_env.viscosity = atof(values[i].c_str());
			else if(names[i] == "gas_stiffness")	 sph_env.gas_k = atof(values[i].c_str());
			else if(names[i] == "init_vertex_store") sph_env.mesh_vertex_store = atoi(values[i].c_str());
			else if(names[i] == "epsilon")			 sph_env.epsilon = atof(values[i].c_str());
			else if(names[i] == "dens_fluctuation")	 sph_env.eta = atof(values[i].c_str());
			else if(names[i] == "min_iterations")	 sph_env.min_iter = atoi(values[i].c_str());
			else if(names[i] == "max_iterations")	 sph_env.max_iter = atoi(values[i].c_str());
			else if(names[i] == "use_artificial_pressure") sph_env.use_ap = atoi(values[i].c_str());
			else if(names[i] == "ap_k")				 sph_env.ap_k = atof(values[i].c_str());
			else if(names[i] == "ap_n")				 sph_env.ap_n = atof(values[i].c_str());
			else if(names[i] == "ap_q")				 sph_env.ap_q = atof(values[i].c_str());
		}
		if(sph_env.mesh_vertex_store < 1) sph_env.mesh_vertex_store = 1;

		sph_env.mesh_boundary_ext = sph_env.boundary_ext;
		sph_env.mesh_boundary_cen = sph_env.boundary_cen;

		m_Env = sph_env;

	}

	//! 液体 : 箱形
	static void SetLiquidBox_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetLiquidBox(names, values, n, header);
	}
	void SetLiquidBox(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0), vel(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ext") GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
			else if(names[i] == "vel") GetValueFromString(vel, values[i], false);
		}
		//m_pSolver->AddBox(-1, cen, ext, vel, -1);
		m_pSolver->AddBox(-1, cen, ext, vel, 2*m_pSolver->GetParticleRadius());
		RXCOUT << "set liquid box : " << cen << ", " << ext << ", " << vel << endl;
	}

	//! 液体 : 球
	static void SetLiquidSphere_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetLiquidSphere(names, values, n, header);
	}
	void SetLiquidSphere(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), vel(0.0);
		double rad = 0.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "rad") GetValueFromString(rad, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
			else if(names[i] == "vel") GetValueFromString(vel, values[i], false);
		}
		//m_pSolver->AddSphere(-1, cen, rad, vel, -1);
		RXCOUT << "set liquid sphere : " << cen << ", " << rad << endl;
	}

	//! 液体流入 : 線分
	static void SetInletLine_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetInletLine(names, values, n, header);
	}
	void SetInletLine(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 pos1(0.0), pos2(0.0), vel(0.0), up(0.0, 1.0, 0.0);
		int  span = -1, accum = 1;
		double spacing = 1.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "pos1")      GetValueFromString(pos1, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "pos2") GetValueFromString(pos2, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "vel")  GetValueFromString(vel,  values[i], false);
			else if(names[i] == "up")   GetValueFromString(up,   values[i], false);
			else if(names[i] == "span") span = atoi(values[i].c_str());
			else if(names[i] == "accum") accum = atoi(values[i].c_str());
			else if(names[i] == "spacing") spacing = atof(values[i].c_str());
		}

		rxInletLine inlet;
		inlet.pos1 = pos1;
		inlet.pos2 = pos2;
		inlet.vel = vel;
		inlet.span = span;
		inlet.up = up;
		inlet.accum = accum;
		inlet.spacing = spacing;

		int num_of_inlets = m_pSolver->AddLine(inlet);
		//((RXSPH*)m_pSolver)->AddSubParticles(g_iInletStart, count);
		RXCOUT << "set inlet boundary : " << pos1 << "-" << pos2 << ", " << vel << endl;
		RXCOUT << "                     span=" << span << ", up=" << up << ", accum=" << accum << ", spacing=" << spacing << endl;
	}

	//! 固体 : 箱形
	static void SetSolidBox_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetSolidBox(names, values, n, header);
	}
	void SetSolidBox(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0), ang(0.0), vel(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ext") GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ang") GetValueFromString(ang, values[i], false);
			else if(names[i] == "vel") GetValueFromString(vel, values[i], false);
		}
		m_pSolver->SetBoxObstacle(cen, ext, ang, vel, 1);
		RXCOUT << "set solid box : " << cen << ", " << ext << ", " << ang << endl;
	}

	//! 固体 : 球
	static void SetSolidSphere_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetSolidSphere(names, values, n, header);
	}
	void SetSolidSphere(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), move_pos1(0.0), move_pos2(0.0), vel(0.0);
		int  move = 0, move_start = -1;
		double rad = 0.0, move_max_vel = 0.0, lap = 1.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "rad") GetValueFromString(rad, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
			else if(names[i] == "vel")  GetValueFromString(vel, values[i], false);
			else if(names[i] == "move_pos1") GetValueFromString(move_pos1, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "move_pos2") GetValueFromString(move_pos2, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "move") move = atoi(values[i].c_str());
			else if(names[i] == "move_start") move_start = atoi(values[i].c_str());
			else if(names[i] == "move_max_vel") move_max_vel = atof(values[i].c_str());
			else if(names[i] == "lap") lap = atof(values[i].c_str());
		}
		m_pSolver->SetSphereObstacle(cen, rad, vel, 1);
		RXCOUT << "set solid sphere : " << cen << ", " << rad << endl;
	}

	//! 固体 : ポリゴン
	static void SetSolidPolygon_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetSolidPolygon(names, values, n, header);
	}
	void SetSolidPolygon(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		string fn_obj;
		Vec3 cen(0.0), ext(0.0), ang(0.0), vel(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")       GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ext")  GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ang")  GetValueFromString(ang, values[i], false);
			else if(names[i] == "vel")  GetValueFromString(vel, values[i], false);
			else if(names[i] == "file") fn_obj = values[i];
		}
		if(!fn_obj.empty()){
			m_pSolver->SetPolygonObstacle(fn_obj, cen, ext, ang, vel);
			RXCOUT << "set solid polygon : " << fn_obj << endl;
			RXCOUT << "                  : " << cen << ", " << ext << ", " << ang << endl;
		}
	}


	//! メッシュ生成境界範囲
	static void SetMeshBoundary_s(string *names, string *values, int n, string header, void *x)
	{
		((rxSceneConfig*)x)->SetMeshBoundary(names, values, n, header);
	}
	void SetMeshBoundary(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")       GetValueFromString(cen, values[i], rel, m_pSolver->GetCen(), 0.5*m_pSolver->GetDim());
			else if(names[i] == "ext")  GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pSolver->GetDim());
		}

		m_Env.mesh_boundary_cen = cen;
		m_Env.mesh_boundary_ext = ext;
		RXCOUT << "boundary for mash : " << cen << ", " << ext << endl;
	}

	/*!
	 * シミュレーション空間の設定読み込み
	 */
	bool LoadSpaceFromFile(void)
	{
		bool ok = true;
		rxINI *cfg = new rxINI();
		cfg->SetHeaderFunc("space", &rxSceneConfig::SetSpace_s, this);
		if(!(cfg->Load(m_strCurrentScene))){
			RXCOUT << "Failed to open the " << m_strCurrentScene << " file!" << endl;
			ok = false;
		}
		delete cfg;
		return ok;
	}
	
	/*!
	 * パーティクルや固体オブジェクトの設定読み込み
	 */
	bool LoadSceneFromFile(void)
	{
		bool ok = true;
		rxINI *cfg = new rxINI();
		cfg->SetHeaderFunc("liquid box",		&rxSceneConfig::SetLiquidBox_s,    this);
		cfg->SetHeaderFunc("liquid box (r)",    &rxSceneConfig::SetLiquidBox_s,    this);
		cfg->SetHeaderFunc("liquid sphere",		&rxSceneConfig::SetLiquidSphere_s, this);
		cfg->SetHeaderFunc("liquid sphere (r)", &rxSceneConfig::SetLiquidSphere_s, this);
		cfg->SetHeaderFunc("solid box",			&rxSceneConfig::SetSolidBox_s,     this);
		cfg->SetHeaderFunc("solid box (r)",		&rxSceneConfig::SetSolidBox_s,     this);
		cfg->SetHeaderFunc("solid sphere",		&rxSceneConfig::SetSolidSphere_s,  this);
		cfg->SetHeaderFunc("solid sphere (r)",	&rxSceneConfig::SetSolidSphere_s,  this);
		cfg->SetHeaderFunc("solid polygon",		&rxSceneConfig::SetSolidPolygon_s, this);
		cfg->SetHeaderFunc("solid polygon (r)", &rxSceneConfig::SetSolidPolygon_s, this);
		cfg->SetHeaderFunc("inlet line",		&rxSceneConfig::SetInletLine_s,    this);
		cfg->SetHeaderFunc("inlet line (r)",	&rxSceneConfig::SetInletLine_s,    this);
		cfg->SetHeaderFunc("mesh grid",			&rxSceneConfig::SetMeshBoundary_s, this);
		cfg->SetHeaderFunc("mesh grid (r)",		&rxSceneConfig::SetMeshBoundary_s, this);
		if(!(cfg->Load(m_strCurrentScene))){
			RXCOUT << "Failed to open the " << m_strCurrentScene << " file!" << endl;
			ok = false;
		}
		delete cfg;
		return ok;
	}

	void LoadSpaceFromFile(const string input)
	{
		// SPH設定をファイルから読み込み
		ifstream fsin;
		fsin.open(input.c_str());

		Vec3 bmin, bmax;
		fsin >> m_Env.max_particles;
		fsin >> bmin[0] >> bmin[1] >> bmin[2];
		fsin >> bmax[0] >> bmax[1] >> bmax[2];
		fsin >> m_Env.dens;
		fsin >> m_Env.mass;
		fsin >> m_Env.kernel_particles;

		m_Env.boundary_ext = 0.5*(bmax-bmin);
		m_Env.boundary_cen = 0.5*(bmax+bmin);

		fsin.close();

		RXCOUT << "[SPH - " << input << "]" << endl;
		RXCOUT << " num. of particles : " << m_Env.max_particles << endl;
		RXCOUT << " boundary min      : " << bmin << endl;
		RXCOUT << " boundary max      : " << bmax << endl;
		RXCOUT << " boundary cen      : " << m_Env.boundary_cen << endl;
		RXCOUT << " boundary ext      : " << m_Env.boundary_ext << endl;
		RXCOUT << " density           : " << m_Env.dens << endl;
		RXCOUT << " mass              : " << m_Env.mass << endl;
		RXCOUT << " kernel particles  : " << m_Env.kernel_particles << endl;

		m_Env.mesh_boundary_cen = m_Env.boundary_cen;
		m_Env.mesh_boundary_ext = m_Env.boundary_ext;

	}

	/*!
	 * 指定したフォルダにある設定ファイルの数とシーンタイトルを読み取る
	 * @param[in] dir 設定ファイルがあるフォルダ(何も指定しなければ実行フォルダ)
	 */
	void ReadSceneFiles(string dir = "")
	{
		m_vSceneFiles.resize(12, "");	// シーンファイルリスト
		m_vSceneTitles.resize(12, "");	// シーンタイトルリスト

		ifstream scene_ifs;
		string scene_fn = "null";
		for(int i = 1; i <= 12; ++i){
			if(ExistFile((scene_fn = CreateFileName(dir+"sph_scene_", ".cfg", i, 1)))){
				RXCOUT << "scene " << i << " : " << scene_fn << endl;
				m_vSceneFiles[i-1] = scene_fn;
				m_vSceneTitles[i-1] = scene_fn.substr(0, 11);

				// シーンタイトルの読み取り
				scene_ifs.open(scene_fn.c_str(), ios::in);
				string title_buf;
				getline(scene_ifs, title_buf);
				if(!title_buf.empty() && title_buf[0] == '#'){
					m_vSceneTitles[i-1] = title_buf.substr(2, title_buf.size()-2);
				}
				scene_ifs.close();

				m_iSceneFileNum++;
			}
		}

		if(m_iSceneFileNum){
			SetCurrentScene(0);
		}
	}

	/*!
	 * カレントのシーン設定
	 * @param[in] シーンインデックス
	 */
	bool SetCurrentScene(int idx)
	{
		if(idx < 0 || idx >= m_iSceneFileNum || m_vSceneFiles[idx] == ""){
			cout << "There is no scene files!" << endl;
			return false;
		}
		m_iCurrentSceneIdx = idx;
		m_strCurrentScene = m_vSceneFiles[m_iCurrentSceneIdx];
		return true;
	}

	/*!
	 * タイトルからカレントのシーン設定
	 * @param[in] label シーンタイトル
	 */
	bool SetCurrentSceneFromTitle(const string label)
	{
		int scene = 0;
		while(label.find(m_vSceneTitles[scene]) == string::npos || RX_ABS((int)(m_vSceneTitles[scene].size()-label.size())) > 3) scene++;

		if(m_iCurrentSceneIdx != -1 && m_vSceneFiles[scene] != ""){
			RXCOUT << "scene " << scene+1 << " : " << m_vSceneFiles[scene] << endl;
			m_iCurrentSceneIdx = scene;
			m_strCurrentScene = m_vSceneFiles[scene];
			return true;
		}

		return false;
	}


	vector<string> GetSceneTitleList(void){ return m_vSceneTitles; }
};


#endif // #ifndef _RX_SPH_CONFIG_H_