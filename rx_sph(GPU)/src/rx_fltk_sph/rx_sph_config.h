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

// boost
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/function.hpp>
#include <boost/ref.hpp>

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



/*!
 * OBJファイル読み込み
 * @param[in] filename wrlファイルのパス
 */
inline static void ReadOBJ(const string filename, rxPolygons &polys, Vec3 cen, Vec3 ext, Vec3 ang)
{
	if(!polys.vertices.empty()){
		polys.vertices.clear();
		polys.normals.clear();
		polys.faces.clear();
		polys.materials.clear();
	}
	rxOBJ obj;
	if(obj.Read(filename, polys.vertices, polys.normals, polys.faces, polys.materials, true)){
		RXCOUT << filename << " have been read." << endl;

		if(polys.normals.empty()){
			CalVertexNormals(polys);
		}

		RXCOUT << " the number of vertex   : " << polys.vertices.size() << endl;
		RXCOUT << " the number of normal   : " << polys.normals.size() << endl;
		RXCOUT << " the number of polygon  : " << polys.faces.size() << endl;
		RXCOUT << " the number of material : " << polys.materials.size() << endl;

		//FitVertices(Vec3(0.0), Vec3(1.0), polys.vertices);
		AffineVertices(polys, cen, ext, ang);

		// テクスチャ読み込み
		if(!polys.materials.empty()){
			rxMTL::iterator iter = polys.materials.begin();
			for(; iter != polys.materials.end(); ++iter){
				if(iter->second.tex_file.empty()) continue;

				RXCOUT << iter->first << " : " << iter->second.tex_file;
				LoadGLTexture(iter->second.tex_file, iter->second.tex_name, true, false);

				RXCOUT << " : " << iter->second.tex_name << endl;
			}
		}

		polys.open = 1;
	}
}

//-----------------------------------------------------------------------------
//! rxSPHConfigクラス - SPHのシーン設定をファイルから読み込む
//-----------------------------------------------------------------------------
class rxSPHConfig
{
protected:
	rxParticleSystemBase *m_pPS;	//!< SPH
	rxSPHEnviroment m_SphEnv;		//!< SPH環境設定
	
	string m_strCurrentScene;			//!< 現在のシーンの名前
	vector<string> m_vSceneFiles;		//!< シーンファイルリスト
	int m_iSceneFileNum;				//!< シーンファイルの数

	vector<string> m_vSceneTitles;		//!< シーンファイルリスト
	int m_iCurrentSceneIdx;				//!< 現在のシーンファイル

	vector<rxPolygons> m_vSolidPoly;	//!< 固体メッシュ

public:
	//! デフォルトコンストラクタ
	rxSPHConfig() : m_pPS(0)
	{
		m_vSceneFiles.resize(12, "");	// シーンファイルリスト
		m_vSceneTitles.resize(12, "");	// シーンタイトルリスト
		m_iCurrentSceneIdx = -1;		// 現在のシーンファイル
		m_strCurrentScene = "null";		// 現在のシーンの名前
		m_iSceneFileNum = 0;

		Clear();
	}

	//! デストラクタ
	~rxSPHConfig(){}
	
	//! 設定初期化
	void Clear(void)
	{
		m_pPS = 0;
		m_vSolidPoly.clear();
	}


	//! シーンタイトルリスト
	vector<string> GetSceneTitles(void) const { return m_vSceneTitles; }

	//! 現在のシーン
	int GetCurrentSceneIdx(void) const { return m_iCurrentSceneIdx; }

	//! SPH環境
	rxSPHEnviroment GetSphEnv(void) const { return m_SphEnv; }

	//! SPHクラス
	void SetPS(rxParticleSystemBase *ps){ m_pPS = ps; }

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
	void SetSphSpace(string *names, string *values, int n, string header)
	{
		rxSPHEnviroment sph_env;
		sph_env.use_inlet = 0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(sph_env.boundary_cen, values[i], false);
			else if(names[i] == "ext") GetValueFromString(sph_env.boundary_ext, values[i], false);
			else if(names[i] == "max_particle_num") sph_env.max_particles = atoi(values[i].c_str());
			else if(names[i] == "density")			sph_env.dens = atof(values[i].c_str());
			else if(names[i] == "mass")				sph_env.mass = atof(values[i].c_str());
			else if(names[i] == "kernel_particles") sph_env.kernel_particles = atof(values[i].c_str());
			else if(names[i] == "mesh_res_max")		sph_env.mesh_max_n = atoi(values[i].c_str());
			else if(names[i] == "inlet_boundary")	sph_env.use_inlet = atoi(values[i].c_str());
			else if(names[i] == "dt")				sph_env.dt = atof(values[i].c_str());
			else if(names[i] == "init_vertex_store")sph_env.mesh_vertex_store = atoi(values[i].c_str());
		}
		if(sph_env.mesh_vertex_store < 1) sph_env.mesh_vertex_store = 1;

		sph_env.mesh_boundary_ext = sph_env.boundary_ext;
		sph_env.mesh_boundary_cen = sph_env.boundary_cen;

		m_SphEnv = sph_env;

	}

	//! 液体 : 箱形
	void SetLiquidBox(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0), vel(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "ext") GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
			else if(names[i] == "vel") GetValueFromString(vel, values[i], false);
		}
		//m_pPS->AddBox(-1, cen, ext, vel, -1);
		m_pPS->AddBox(-1, cen, ext, vel, m_pPS->GetParticleRadius());
		RXCOUT << "set liquid box : " << cen << ", " << ext << ", " << vel << endl;
	}

	//! 液体 : 球
	void SetLiquidSphere(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), vel(0.0);
		double rad = 0.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "rad") GetValueFromString(rad, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
			else if(names[i] == "vel") GetValueFromString(vel, values[i], false);
		}
		//m_pPS->AddSphere(-1, cen, rad, vel, -1);
		RXCOUT << "set liquid sphere : " << cen << ", " << rad << endl;
	}

	//! 液体流入 : 線分
	void SetInletLine(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 pos1(0.0), pos2(0.0), vel(0.0), up(0.0, 1.0, 0.0);
		int  span = -1, accum = 1;
		double spacing = 1.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "pos1")      GetValueFromString(pos1, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "pos2") GetValueFromString(pos2, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
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

		int num_of_inlets = m_pPS->AddLine(inlet);
		//((RXSPH*)m_pPS)->AddSubParticles(g_iInletStart, count);
		RXCOUT << "set inlet boundary : " << pos1 << "-" << pos2 << ", " << vel << endl;
		RXCOUT << "                     span=" << span << ", up=" << up << ", accum=" << accum << ", spacing=" << spacing << endl;
	}

	//! 固体 : 箱形
	void SetSolidBox(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0), ang(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "ext") GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
			else if(names[i] == "ang") GetValueFromString(ang, values[i], false);
		}
		m_pPS->SetBoxObstacle(cen, ext, ang, 1);
		RXCOUT << "set solid box : " << cen << ", " << ext << ", " << ang << endl;
	}

	//! 固体 : 球
	void SetSolidSphere(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), move_pos1(0.0), move_pos2(0.0);
		int  move = 0, move_start = -1;
		double rad = 0.0, move_max_vel = 0.0, lap = 1.0;
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")      GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "rad") GetValueFromString(rad, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
			else if(names[i] == "move_pos1") GetValueFromString(move_pos1, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "move_pos2") GetValueFromString(move_pos2, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "move") move = atoi(values[i].c_str());
			else if(names[i] == "move_start") move_start = atoi(values[i].c_str());
			else if(names[i] == "move_max_vel") move_max_vel = atof(values[i].c_str());
			else if(names[i] == "lap") lap = atof(values[i].c_str());
		}
		m_pPS->SetSphereObstacle(cen, rad, 1);
		//if(move){
		//	m_vMovePos[0] = move_pos1;
		//	m_vMovePos[1] = move_pos2;
		//	m_fMoveMaxVel = move_max_vel;

		//	//m_bMoveSolid = false;
		//	if(move_start >= 0){
		//		m_iMoveStart = move_start;
		//	}
		//}
		RXCOUT << "set solid sphere : " << cen << ", " << rad << endl;
	}

	//! 固体 : ポリゴン
	void SetSolidPolygon(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		string fn_obj;
		Vec3 cen(0.0), ext(0.0), ang(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")       GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "ext")  GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
			else if(names[i] == "ang")  GetValueFromString(ang, values[i], false);
			else if(names[i] == "file") fn_obj = values[i];
		}
		if(!fn_obj.empty()){
			rxPolygons poly;
			ReadOBJ(fn_obj, poly, cen, ext, ang);
			m_vSolidPoly.push_back(poly);
			vector< vector<int> > tris;
			int pn = poly.faces.size();
			tris.resize(pn);
			for(int i = 0; i < pn; ++i){
				tris[i].resize(3);
				for(int j = 0; j < 3; ++j){
					tris[i][j] = poly.faces[i][j];
				}
			}
			m_pPS->SetPolygonObstacle(poly.vertices, poly.normals, tris);
			RXCOUT << "set solid polygon : " << fn_obj << endl;
			RXCOUT << "                  : " << cen << ", " << ext << ", " << ang << endl;
		}
	}

	//! メッシュ生成境界範囲
	void SetMeshBoundary(string *names, string *values, int n, string header)
	{
		bool rel = (header.find("(r)") != string::npos);
		Vec3 cen(0.0), ext(0.0);
		for(int i = 0; i < n; ++i){
			if(names[i] == "cen")       GetValueFromString(cen, values[i], rel, m_pPS->GetCen(), 0.5*m_pPS->GetDim());
			else if(names[i] == "ext")  GetValueFromString(ext, values[i], rel, Vec3(0.0), 0.5*m_pPS->GetDim());
		}

		m_SphEnv.mesh_boundary_cen = cen;
		m_SphEnv.mesh_boundary_ext = ext;
		RXCOUT << "boundary for mash : " << cen << ", " << ext << endl;
	}

	/*!
	 * シミュレーション空間の設定読み込み
	 */
	bool LoadSpaceFromFile(void)
	{
		bool ok = true;
		rxINI *cfg = new rxINI();
		cfg->SetHeaderFunc("space", boost::bind(&rxSPHConfig::SetSphSpace, this, _1, _2, _3, _4));
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
		cfg->SetHeaderFunc("liquid box",		boost::bind(&rxSPHConfig::SetLiquidBox, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("liquid box (r)",    boost::bind(&rxSPHConfig::SetLiquidBox, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("liquid sphere",		boost::bind(&rxSPHConfig::SetLiquidSphere, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("liquid sphere (r)", boost::bind(&rxSPHConfig::SetLiquidSphere, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid box",			boost::bind(&rxSPHConfig::SetSolidBox, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid box (r)",		boost::bind(&rxSPHConfig::SetSolidBox, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid sphere",		boost::bind(&rxSPHConfig::SetSolidSphere, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid sphere (r)",	boost::bind(&rxSPHConfig::SetSolidSphere, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid polygon",		boost::bind(&rxSPHConfig::SetSolidPolygon, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("solid polygon (r)", boost::bind(&rxSPHConfig::SetSolidPolygon, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("inlet line",		boost::bind(&rxSPHConfig::SetInletLine, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("inlet line (r)",	boost::bind(&rxSPHConfig::SetInletLine, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("mesh grid",			boost::bind(&rxSPHConfig::SetMeshBoundary, this, _1, _2, _3, _4));
		cfg->SetHeaderFunc("mesh grid (r)",		boost::bind(&rxSPHConfig::SetMeshBoundary, this, _1, _2, _3, _4));
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
		fsin >> m_SphEnv.max_particles;
		fsin >> bmin[0] >> bmin[1] >> bmin[2];
		fsin >> bmax[0] >> bmax[1] >> bmax[2];
		fsin >> m_SphEnv.dens;
		fsin >> m_SphEnv.mass;
		fsin >> m_SphEnv.kernel_particles;

		m_SphEnv.boundary_ext = 0.5*(bmax-bmin);
		m_SphEnv.boundary_cen = 0.5*(bmax+bmin);

		fsin.close();

		RXCOUT << "[SPH - " << input << "]" << endl;
		RXCOUT << " num. of particles : " << m_SphEnv.max_particles << endl;
		RXCOUT << " boundary min      : " << bmin << endl;
		RXCOUT << " boundary max      : " << bmax << endl;
		RXCOUT << " boundary cen      : " << m_SphEnv.boundary_cen << endl;
		RXCOUT << " boundary ext      : " << m_SphEnv.boundary_ext << endl;
		RXCOUT << " density           : " << m_SphEnv.dens << endl;
		RXCOUT << " mass              : " << m_SphEnv.mass << endl;
		RXCOUT << " kernel particles  : " << m_SphEnv.kernel_particles << endl;

		m_SphEnv.mesh_boundary_cen = m_SphEnv.boundary_cen;
		m_SphEnv.mesh_boundary_ext = m_SphEnv.boundary_ext;

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
		while(label.find(m_vSceneTitles[scene]) == string::npos) scene++;

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