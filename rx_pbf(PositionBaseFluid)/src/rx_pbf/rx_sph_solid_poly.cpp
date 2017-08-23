/*!
  @file rx_sph_solid_poly.cpp
	
  @brief SPH用固体定義
		 ポリゴン+陰関数(OpenVDBで作成)
 
  @author Makoto Fujisawa
  @date 2014-02
*/
//#define RX_USE_OPENVDB

#ifdef RX_USE_OPENVDB
#pragma comment(lib, "Half.lib")
#pragma comment(lib, "tbb.lib")
#pragma comment(lib, "openvdb.lib")
//#pragma comment(lib, "zlib.lib")
#endif

//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------

#ifdef RX_USE_OPENVDB
// OpenVDB
#include <openvdb/openvdb.h>
#include <openvdb/Exceptions.h>

#include <openvdb/tools/MeshToVolume.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/util/Util.h>

#include <openvdb/tools/Interpolation.h>
#endif


#include "rx_sph_solid.h"

// 表面パーティクル配置
#include "rx_particle_on_surf.h"

// 3Dメッシュ
#include "rx_model.h"

#ifdef RX_USE_OPENVDB
openvdb::FloatGrid::Ptr g_pGrid = 0;

/*!
 * ポリゴンから符号付き距離場の生成
 * @param[in] poly ポリゴンデータ
 * @param[in] gw グリッド分割幅
 * @return 符号付き距離場(グリッド)
 */
static openvdb::FloatGrid::Ptr MakeSDF(const rxPolygons &poly, double gw)
{
	using namespace openvdb;

	std::vector<Vec3s> points;
	std::vector<Vec3I> triangles;
	std::vector<Vec4I> quads;

	for(int i = 0; i < (int)poly.vertices.size(); ++i){
		Vec3 v = poly.vertices[i];
		points.push_back(Vec3s(v[0], v[1], v[2]));
	}

	for(int i = 0; i < (int)poly.faces.size(); ++i){
		rxFace f = poly.faces[i];
		triangles.push_back(Vec3I(f[0], f[1], f[2]));
	}

	float bandwidth[2] = {0.2f, 0.2f};	// exterior/interior narrow-band width;
	bandwidth[0] = gw*10;
	bandwidth[1] = gw*10;
	openvdb::math::Transform xform = *math::Transform::createLinearTransform(gw);

	return tools::meshToSignedDistanceField<FloatGrid>(xform, points, triangles, quads, bandwidth[0], bandwidth[1]);
}
#endif




//-----------------------------------------------------------------------------
// rxSolidPolygonクラスの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 * @param[in] filename ポリゴンファイル名(OBJ,WRLなど)
 * @param[in] cen 固体オブジェクト中心座標
 * @param[in] ext 固体オブジェクトの大きさ(辺の長さの1/2)
 * @param[in] ang 固体オブジェクトの角度(オイラー角)
 * @param[in] h 有効半径(分割セル幅)
 * @param[in] aspect 元ポリゴンのアスペクト比を保つかどうか(0でextに合わせて変更)
 */
rxSolidPolygon::rxSolidPolygon(const string &fn, Vec3 cen, Vec3 ext, Vec3 ang, double h, int aspect)
{
	m_vMin = -ext;
	m_vMax =  ext;
	m_vMassCenter = cen;

	m_hVrts = 0; 
	m_hTris = 0; 
	m_iNumTris = 0;

	m_iSgn = 1;
	m_iName = RXS_POLYGON;

	// ポリゴン初期化
	if(!m_Poly.vertices.empty()){
		m_Poly.vertices.clear();
		m_Poly.normals.clear();
		m_Poly.faces.clear();
		m_Poly.materials.clear();
	}
	
	// HACK:ポリゴンモデル読み込み
	readFile(fn, m_Poly);
	
	if(m_Poly.vertices.empty()){
		return;
	}

	// ポリゴン頂点をAABBにフィット
	if(RXFunc::IsZeroVec(ang)){
		fitVertices(cen, ext, m_Poly.vertices);
	}
	else{
		AffineVertices(m_Poly, cen, ext, ang);
	}

	// AABBの探索
	FindBBox(m_vMin, m_vMax, m_Poly.vertices);
	m_vMassCenter = 0.5*(m_vMin+m_vMax);
	Vec3 dim = m_vMax-m_vMin;
	m_vMin = -0.5*dim;
	m_vMax =  0.5*dim;

	cout << "solid min-max : " << m_vMin << " - " << m_vMax << endl;
	cout << "      mc      : " << m_vMassCenter << endl;


	// ポリゴン頂点法線の計算
	if(m_Poly.normals.empty()){
		CalVertexNormals(m_Poly);
	}

	int vn = (int)m_Poly.vertices.size();
	int n = (int)m_Poly.faces.size();

	if(m_hVrts) delete [] m_hVrts;
	if(m_hTris) delete [] m_hTris;
	m_hVrts = new RXREAL[vn*3];
	m_hTris = new int[n*3];

	for(int i = 0; i < vn; ++i){
		for(int j = 0; j < 3; ++j){
			m_hVrts[3*i+j] = m_Poly.vertices[i][j];
		}
	}

	for(int i = 0; i < n; ++i){
		for(int j = 0; j < 3; ++j){
			m_hTris[3*i+j] = m_Poly.faces[i][j];
		}
	}

	m_iNumVrts = vn;
	m_iNumTris = n;
	RXCOUT << "the number of triangles : " << m_iNumTris << endl;

	// 分割セル生成用に1%だけ拡張したBBoxを生成
	Vec3 minp = m_vMassCenter+m_vMin;
	Vec3 maxp = m_vMassCenter+m_vMax;
	CalExtendedBBox(minp, maxp, 0.05);


	// 近傍探索セル
	m_pNNGrid = new rxNNGrid(DIM);

	// 分割セル設定
	m_pNNGrid->Setup(minp, maxp, h, n);

	// 分割セルにポリゴンを登録
	m_pNNGrid->SetPolygonsToCell(m_hVrts, m_iNumVrts, m_hTris, m_iNumTris);
	
	// レイトレなどでの描画用に固体オブジェクトメッシュをファイル保存しておく
	string outfn = RX_DEFAULT_MESH_DIR+"solid_boundary.obj";
	rxOBJ saver;
	rxMTL mtl;	// 材質はemptyのまま
	if(saver.Save(outfn, m_Poly.vertices, m_Poly.normals, m_Poly.faces, mtl)){
		RXCOUT << "saved the mesh to " << outfn << endl;
	}

#ifdef RX_USE_OPENVDB
	g_pGrid = 0;
	openvdb::initialize();
	g_pGrid = MakeSDF(m_Poly, 0.02);
#endif
}

/*!
 * デストラクタ
 */
rxSolidPolygon::~rxSolidPolygon()
{
	if(m_pNNGrid) delete m_pNNGrid;
	if(m_hVrts) delete [] m_hVrts;
	if(m_hTris) delete [] m_hTris;
}

/*!
 * 距離値計算(点との距離)
 * @param[in] pos グローバル座標での位置
 * @param[out] col 距離などの情報(衝突情報)
 */
bool rxSolidPolygon::GetDistance(const Vec3 &pos, rxCollisionInfo &col)
{
	return GetDistanceR(pos, 0.0, col);
}
bool rxSolidPolygon::GetDistance(const Vec3 &pos0, const Vec3 &pos1, rxCollisionInfo &col)
{
	return GetDistanceR(pos0, pos1, 0.0, col);
}



/*!
 * 距離値計算
 * @param[in] pos グローバル座標での位置
 * @param[in] r 球の半径
 * @param[out] col 距離などの情報(衝突情報)
 */
bool rxSolidPolygon::GetDistanceR(const Vec3 &pos, const double &r, rxCollisionInfo &col)
{
#ifdef RX_USE_OPENVDB
	if(g_pGrid){
		openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> sampler(*g_pGrid);
		openvdb::Vec3R ijk(pos[0], pos[1], pos[2]);
		openvdb::FloatGrid::ValueType v0;
		v0 = sampler.wsSample(ijk);
		col.Penetration() = -((RXREAL)(v0)+m_fOffset*2);

		openvdb::FloatGrid::Accessor accessor = g_pGrid->getAccessor();
		openvdb::Coord nijk;
		openvdb::Vec3d npos(pos[0], pos[1], pos[2]);
		npos = g_pGrid->worldToIndex(npos);
		nijk[0] = int(npos[0]);
		nijk[1] = int(npos[1]);
		nijk[2] = int(npos[2]);
		accessor.getValue(nijk);
		openvdb::Vec3d nrm = openvdb::math::ISGradient<openvdb::math::CD_2ND>::result(accessor, nijk);
		double length = nrm.length();
		if(length > 1.0e-6){
			nrm *= 1.0/length;
		}
		col.Normal() = Vec3(nrm.x(), nrm.y(), nrm.z());
		return true;
	}
	else{
		return GetDistanceR(pos, pos, r, col);
	}
#else
	return GetDistanceR(pos, pos, r, col);
#endif
}

/*!
 * 距離値計算
 * @param[in] pos グローバル座標での位置
 * @param[in] r 球の半径
 * @param[out] col 距離などの情報(衝突情報)
 */
bool rxSolidPolygon::GetDistanceR(const Vec3 &pos0, const Vec3 &pos1, const double &r, rxCollisionInfo &col)
{
	Vec3 dir = Unit(pos1-pos0);
	int c = 0;
	double min_d = RX_FEQ_INF;
	set<int> polys_in_cell;	// 重複なしコンテナ
	int cn = m_pNNGrid->GetNNPolygons(pos0, polys_in_cell, r);	// r=0でもposが含まれるセル+1周囲は最低限探索する
	cn    += m_pNNGrid->GetNNPolygons(pos1+dir*r, polys_in_cell, r);
	if(cn){ 
		set<int>::iterator p = polys_in_cell.begin();
		for(; p != polys_in_cell.end(); ++p){
			int pidx = *p;

			// 三角形ポリゴンを構成する頂点インデックス
			int vidx[3];
			vidx[0] = m_hTris[3*pidx+0];
			vidx[1] = m_hTris[3*pidx+1];
			vidx[2] = m_hTris[3*pidx+2];

			// 三角形ポリゴンの頂点座標
			Vec3 vrts[3];
			vrts[0] = Vec3(m_hVrts[3*vidx[0]], m_hVrts[3*vidx[0]+1], m_hVrts[3*vidx[0]+2]);
			vrts[1] = Vec3(m_hVrts[3*vidx[1]], m_hVrts[3*vidx[1]+1], m_hVrts[3*vidx[1]+2]);
			vrts[2] = Vec3(m_hVrts[3*vidx[2]], m_hVrts[3*vidx[2]+1], m_hVrts[3*vidx[2]+2]);

			Vec3 cp, n;
			double d;
			//n = Unit(cross(vrts[1]-vrts[0], vrts[2]-vrts[0]));

			if(intersectSegmentTriangle(pos0, pos1+dir*r, vrts[0], vrts[1], vrts[2], cp, n) == 1){
				d = length(pos0-cp);
				if(d < min_d){
					col.Penetration() = d;
					col.Normal() = n;
					//double l = r/(dot(-n, dir));
					col.Contact() = cp;//-l*dir;
					min_d = d;
					c++;
				}
			}
		}
	}
	
	return (c == 0 ? false : true);
}


/*!
 * ポリゴン平面までの距離を計算
 * @param[in] x 計算座標
 * @param[in] p ポリゴン番号
 * @param[out] dist 符号付き距離
 */
int rxSolidPolygon::getDistanceToPolygon(Vec3 x, int p, double &dist)
{
	vector<int> &vs = m_Poly.faces[p].vert_idx;
	Vec3 pnrm = Unit(cross(m_Poly.vertices[vs[1]]-m_Poly.vertices[vs[0]], m_Poly.vertices[vs[2]]-m_Poly.vertices[vs[0]]));
	dist = dot(x-m_Poly.vertices[vs[0]], pnrm);
	return 0;
}



/*!
 * 表面曲率計算
 *  - 単に陰関数の空間2階微分を計算しているだけ
 * @param[in] pos 計算位置
 * @param[out] k 曲率
 */
bool rxSolidPolygon::GetCurvature(const Vec3 &pos, double &k)
{
	return CalCurvature(pos, k, &rxSolidPolygon::GetDistance_s, this);
}

/*!
 * OpenGL変換行列を設定
 */
void rxSolidPolygon::SetGLMatrix(void)
{
	glTranslatef(m_vMassCenter[0], m_vMassCenter[1], m_vMassCenter[2]);
	glMultMatrixd(m_matRot.GetValue());
}

/*!
 * OpenGLによる描画
 * @param[in] drw 描画フラグ(drw&2 == 1でワイヤフレーム描画)
 */
void rxSolidPolygon::Draw(int drw)
{
	glPushMatrix();

	//SetGLMatrix();
	
	// メッシュ
	glEnable(GL_LIGHTING);
	glColor4d(0.0, 1.0, 0.0, 1.0);
	m_Poly.Draw(drw & 14);

	glPopMatrix();
}

/*!
 * 分割セルに格納されたポリゴン情報を取得
 * @param[in] gi,gj,gk 対象分割セル
 * @param[out] polys ポリゴン
 * @return 格納ポリゴン数
 */
int rxSolidPolygon::GetPolygonsInCell(int gi, int gj, int gk, set<int> &polys)
{
	return m_pNNGrid->GetPolygonsInCell(gi, gj, gk, polys);
}

/*!
 * 分割セル内のポリゴンの有無を調べる
 * @param[in] gi,gj,gk 対象分割セル
 * @return ポリゴンが格納されていればtrue
 */
bool rxSolidPolygon::IsPolygonsInCell(int gi, int gj, int gk)
{
	return m_pNNGrid->IsPolygonsInCell(gi, gj, gk);
}



/*!
 * OBJファイル読み込み
 * @param[in] filename wrlファイルのパス
 */
void rxSolidPolygon::readOBJ(const string filename, rxPolygons &polys)
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

		RXCOUT << " the number of vertex   : " << polys.vertices.size() << endl;
		RXCOUT << " the number of normal   : " << polys.normals.size() << endl;
		RXCOUT << " the number of polygon  : " << polys.faces.size() << endl;
		RXCOUT << " the number of material : " << polys.materials.size() << endl;

		polys.open = 1;
	}
}


int rxSolidPolygon::readFile(const string filename, rxPolygons &polys)
{
	string ext = GetExtension(filename);
	if(ext == "obj"){
		readOBJ(filename, polys);
	}

	return polys.open;
}



/*!
 * 頂点列をAABBに合うようにFitさせる(元の形状の縦横比は維持)
 * @param[in] ctr AABB中心座標
 * @param[in] sl  AABBの辺の長さ(1/2)
 * @param[in] vec_set 頂点列
 * @param[in] start_index,end_index 頂点列の検索範囲
 */
bool rxSolidPolygon::fitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set, int aspect)
{
	if(vec_set.empty()) return false;

	int n = (int)vec_set.size();

	// 現在のBBoxの大きさを調べる
	Vec3 maxp = vec_set[0];
	Vec3 minp = vec_set[0];

	for(int i = 1; i < n; ++i){
		if(vec_set[i][0] > maxp[0]) maxp[0] = vec_set[i][0];
		if(vec_set[i][1] > maxp[1]) maxp[1] = vec_set[i][1];
		if(vec_set[i][2] > maxp[2]) maxp[2] = vec_set[i][2];
		if(vec_set[i][0] < minp[0]) minp[0] = vec_set[i][0];
		if(vec_set[i][1] < minp[1]) minp[1] = vec_set[i][1];
		if(vec_set[i][2] < minp[2]) minp[2] = vec_set[i][2];
	}

	Vec3 ctr0, sl0;
	sl0  = (maxp-minp)/2.0;
	ctr0 = (maxp+minp)/2.0;

	Vec3 size_conv;
	
	if(aspect){
		int max_axis = ( ( (sl0[0] > sl0[1]) && (sl0[0] > sl0[2]) ) ? 0 : ( (sl0[1] > sl0[2]) ? 1 : 2 ) );
		int min_axis = ( ( (sl0[0] < sl0[1]) && (sl0[0] < sl0[2]) ) ? 0 : ( (sl0[1] < sl0[2]) ? 1 : 2 ) );
		size_conv = Vec3(sl[max_axis]/sl0[max_axis]);
	}
	else{
		size_conv = sl/sl0;
	}

	// 全ての頂点をbboxにあわせて変換
	for(int i = 0; i < n; ++i){
		vec_set[i] = (vec_set[i]-ctr0)*size_conv+ctr;
	}

	return true;
}


/*!
 * レイ/線分と三角形の交差
 * @param[in] P0,P1 レイ/線分の端点orレイ上の点
 * @param[in] V0,V1,V2 三角形の頂点座標
 * @param[out] I 交点座標
 * @retval 1 交点Iで交差 
 * @retval 0 交点なし
 * @retval 2 三角形の平面内
 * @retval -1 三角形が"degenerate"である(面積が0，つまり，線分か点になっている)
 */
int rxSolidPolygon::intersectSegmentTriangle(Vec3 P0, Vec3 P1,			// Segment
									Vec3 V0, Vec3 V1, Vec3 V2,	// Triangle
									Vec3 &I, Vec3 &n)			// Intersection point (return)
{
	// 三角形のエッジベクトルと法線
	Vec3 u = V1-V0;
	Vec3 v = V2-V0;
	n = Unit(cross(u, v));
	if(RXFunc::IsZeroVec(n)){
		return -1;	// 三角形が"degenerate"である(面積が0)
	}

	// 線分
	Vec3 dir = P1-P0;
	double a = dot(n, P0-V0);
	double b = dot(n, dir);
	if(fabs(b) < 1e-10){	// 線分と三角形平面が平行
		if(a == 0){
			return 2;	// 線分が平面上
		}
		else{
			return 0;	// 交点なし
		}
	}

	// 交点計算

	// 2端点がそれぞれ異なる面にあるかどうかを判定
	float r = -a/b;
	if(a < 0){
		return 0;
	}

	if(r < 0.0){
		return 0;
	}
	else{
		if(fabs(a) > fabs(b)){
			return 0;
		}
		else{
			if(b > 0){
				return 0;
			}
		}
	}

	// 線分と平面の交点
	I = P0+r*dir;

	// 交点が三角形内にあるかどうかの判定
	double uu, uv, vv, wu, wv, D;
	uu = dot(u, u);
	uv = dot(u, v);
	vv = dot(v, v);
	Vec3 w = I-V0;
	wu = dot(w, u);
	wv = dot(w, v);
	D = uv*uv-uu*vv;

	double s, t;
	s = (uv*wv-vv*wu)/D;
	if(s < 0.0 || s > 1.0){
		return 0;
	}
	
	t = (uv*wu-uu*wv)/D;
	if(t < 0.0 || (s+t) > 1.0){
		return 0;
	}

	return 1;
}