/*! 
  @file rx_mesh.h

  @brief メッシュ構造の定義
 
  @author Makoto Fujisawa
  @date 2010
*/
// FILE --rx_mesh.h--

#ifndef _RX_MESH_H_
#define _RX_MESH_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <sstream>

#include <string>
#include <vector>
#include <map>
#include <set>

#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"

using namespace std;

//-----------------------------------------------------------------------------
// 材質クラス
//-----------------------------------------------------------------------------
class rxMaterialOBJ
{
public:
	string name;		//!< 材質の名前

	Vec4 diffuse;		//!< 拡散色(GL_DIFFUSE)
	Vec4 specular;		//!< 鏡面反射色(GL_SPECULAR)
	Vec4 ambient;		//!< 環境光色(GL_AMBIENT)

	Vec4 color;			//!< 色(glColor)
	Vec4 emission;		//!< 放射色(GL_EMISSION)

	double shininess;	//!< 鏡面反射指数(GL_SHININESS)

	int illum;
	string tex_file;	//!< テクスチャファイル名
	unsigned int tex_name;	//!< テクスチャオブジェクト
};


typedef map<string, rxMaterialOBJ> rxMTL;


//-----------------------------------------------------------------------------
// メッシュクラス
//-----------------------------------------------------------------------------
// ポリゴン
class rxFace
{
public:
	vector<int> vert_idx;	//!< 頂点インデックス
	string material_name;	//!< 材質名
	vector<Vec2> texcoords;	//!< テクスチャ座標
	int attribute;			//!< 属性

public:
	rxFace() : attribute(0) {}

public:
	// オペレータによる頂点アクセス
	inline int& operator[](int i){ return vert_idx[i]; }
	inline int  operator[](int i) const { return vert_idx[i]; }

	// 関数による頂点アクセス
	inline int& at(int i){ return vert_idx.at(i); }
	inline int  at(int i) const { return vert_idx.at(i); }

	//! ポリゴン頂点数の変更
	void resize(int size)
	{
		vert_idx.resize(size);
	}

	//! ポリゴン頂点数の変更
	int size(void) const
	{
		return (int)vert_idx.size();
	}

	//! 初期化
	void clear(void)
	{
		vert_idx.clear();
		material_name = "";
		texcoords.clear();
	}
};

// 三角形ポリゴン
class rxTriangle : public rxFace
{
public:
	rxTriangle()
	{
		vert_idx.resize(3);
	}
};

// ポリゴンオブジェクト
class rxPolygons
{
public:
	vector<Vec3> vertices;	//!< 頂点座標
	vector<Vec3> normals;	//!< 頂点法線
	vector<rxFace> faces;	//!< ポリゴン
	rxMTL materials;		//!< 材質
	int open;				//!< ファイルオープンフラグ

public:
	//! コンストラクタ
	rxPolygons() : open(0) {}
	//! デストラクタ
	~rxPolygons(){}

	//! 描画
	void Draw(int draw = 0x04, double dn = 0.02);

protected:
	//! double版の材質設定
	void glMaterialdv(GLenum face, GLenum pname, const GLdouble *params);
};



//-----------------------------------------------------------------------------
// メッシュデータの読み込み，保存に便利な関数
//-----------------------------------------------------------------------------

//! 座標の回転
Vec3 Rotate3(const Vec3 &pos0, const Vec3 &rot);

//! "文字列 数値"から数値部分の文字列のみを取り出す
bool StringToString(const string &buf, const string &head, string &sub);
bool StringToDouble(const string &buf, const string &head, double &val);

//! "x y z"の形式の文字列からVec2型へ変換
int StringToVec2s(const string &buf, const string &head, Vec2 &v);

//! "x y z"の形式の文字列からVec3型へ変換
int StringToVec3s(const string &buf, const string &head, Vec3 &v);

//! Vec3型から文字列に変換
string Vec3ToString(const Vec3 v);

//! 文字列からベクトル値を抽出
template<class T> 
inline int ExtractVector(string data, const string &end_str, const string &brk_str, int min_elems, vector< vector<T> > &vecs);

//! 先頭の空白(スペース，タブ)を削除
string GetDeleteSpace(const string &buf);

//! 先頭の空白(スペース，タブ)を削除
void DeleteHeadSpace(string &buf);

//! 空白(スペース，タブ)を削除
void DeleteSpace(string &buf);

//! 文字列に含まれる指定された文字列の数を返す
int CountString(string &s, int offset, string c);

//! ポリゴンを三角形に分解
int PolyToTri(vector<rxFace> &plys, const vector<int> &vidxs, const vector<int> &tidxs, const vector<Vec2> &vtc, string mat_name);

//! ファイル名からフォルダパスのみを取り出す
string ExtractDirPath(const string &fn);

//! ファイル名から拡張子を削除
string ExtractPathWithoutExt(const string &fn);

//! 文字列小文字化
void StringToLower(string &str);

//! 文字列が整数値を表しているかを調べる
bool IsInteger(const string &str);

//! 文字列が実数値を表しているかを調べる
bool IsNumeric(const string &str);

//! 頂点列からのAABBの検索
bool FindBBox(Vec3 &minp, Vec3 &maxp, const vector<Vec3> &vec_set, const int start_index, const int end_index);

//! 頂点列からのAABBの検索
bool FindBBox(Vec3 &minp, Vec3 &maxp, const vector<Vec3> &vec_set);

//! 頂点列をAABBに合うようにFitさせる
bool FitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set, const int start_index, const int end_index);

//! 頂点列をAABBに合うようにFitさせる
bool FitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set);

//! バイナリファイルから型指定でデータを読み込む
template<class T> 
inline T ReadBinary(ifstream &file);

//! バイナリファイルから型・サイズ指定でデータを読み込む
template<class T> 
inline T ReadBinary(ifstream &file, int byte);

//! バイナリファイルから2バイト分読み込んでint型に格納
inline int ReadInt(ifstream &file);

//! バイナリファイルから文字列を読み取る
inline bool ReadString(ifstream &file, string &name, int max_size = -1);


//-----------------------------------------------------------------------------
// Geometry processing
//-----------------------------------------------------------------------------
//! 法線反転
void ReverseNormals(const vector<Vec3> &vs, vector<int> &ps, vector<Vec3> &ns);

//! メッシュ法線計算
void CalNormals(const vector<Vec3> &vs, vector<int> &ps, vector<Vec3> &ns);

//! 頂点法線計算
void CalVertexNormals(const vector<Vec3> &vrts, int nvrts, vector<rxTriangle> &tris, int ntris, vector<Vec3> &vnrms);
void CalVertexNormals(const vector<Vec3> &vrts, int nvrts, vector<rxFace> &tris, int ntris, vector<Vec3> &vnrms);

//! 頂点法線計算
void CalVertexNormals(rxPolygons &polys);

//! 頂点法線計算
// void CalVertexNormalsFromVBO(GLuint vrts_vbo, GLuint tris_vbo, GLuint nrms_vbo, uint nvrts, uint ntris);

//! 幾何情報を持たないポリゴン頂点列とポリゴン法線から頂点法線を計算
void CalVertexNormalWithoutGeometry(const vector<Vec3> &vrts, vector<Vec3> &nrms);

//! 幾何情報を持たないポリゴン頂点列から幾何情報を生成
void CalVertexGeometry(vector<Vec3> &vrts, vector< vector<int> > &idxs);

//! 頂点列をAABBに合うようにFitさせる
bool AffineVertices(rxPolygons &polys, Vec3 cen, Vec3 ext, Vec3 ang);

//! シミュレーション空間を覆うグリッドの算出
int CalMeshDiv(Vec3 &minp, Vec3 maxp, int nmax, double &h, int n[3], double extend = 0.05);

//! メッシュ描画用のVBOを確保
bool AssignArrayBuffers(int max_verts, int dim, GLuint &uVrtVBO, GLuint &uNrmVBO, GLuint &uTriVBO);

//! FBOにデータを設定
bool SetArrayFromFBO(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face, int nvrts, int ntris);

//! ホスト側配列にデータを設定
bool SetFBOFromArray(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO, 
					 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face);




//-----------------------------------------------------------------------------
// MARK:ポリゴンの描画
//-----------------------------------------------------------------------------
inline void rxPolygons::glMaterialdv(GLenum face, GLenum pname, const GLdouble *params)
{
	GLfloat col[4];
	col[0] = (GLfloat)params[0];
	col[1] = (GLfloat)params[1];
	col[2] = (GLfloat)params[2];
	col[3] = (GLfloat)params[3];
	glMaterialfv(face, pname, col);
}

/*!
 * ポリゴンの描画
 * @param[in] polys ポリゴンデータ
 * @param[in] draw 描画フラグ(下位ビットから頂点,エッジ,面,法線 - 1,2,4,8)
 */
inline void rxPolygons::Draw(int draw, double dn)
{
	// 頂点数とポリゴン数
	int vn = (int)vertices.size();
	int pn = (int)faces.size();
		
	if(draw & 0x02){
		// エッジ描画における"stitching"をなくすためのオフセットの設定
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.0, 1.0);
	}

	if(draw & 0x04){
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_LIGHTING);
		glColor3d(0.0, 0.0, 1.0);
		bool use_mat = true;
		if(materials.empty()){
			// 材質が設定されていない，もしくは，エッジのみ描画の場合，GL_COLOR_MATERIALを用いる
			use_mat = false;

			glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
			glEnable(GL_COLOR_MATERIAL);
		}
		else{
			glDisable(GL_COLOR_MATERIAL);
		}


		bool tc = false;
		rxMaterialOBJ *cur_mat = NULL, *pre_mat = NULL;

		// すべてのポリゴンを描画
		for(int i = 0; i < pn; ++i){
			rxFace *face = &(faces[i]);
			int n = (int)face->vert_idx.size();

			if(use_mat){
				// 材質の設定
				cur_mat = &materials[face->material_name];

				// 現在の材質が前のと異なる場合のみ，OpenGLの材質情報を更新
				if(cur_mat != pre_mat){
					glColor4dv(cur_mat->color);
					glMaterialdv(GL_FRONT_AND_BACK, GL_DIFFUSE,  cur_mat->diffuse.data);
					glMaterialdv(GL_FRONT_AND_BACK, GL_SPECULAR, cur_mat->specular.data);
					glMaterialdv(GL_FRONT_AND_BACK, GL_AMBIENT,  cur_mat->ambient.data);
					glMaterialdv(GL_FRONT_AND_BACK, GL_EMISSION, cur_mat->emission.data);
					glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, (GLfloat)cur_mat->shininess);

					// テクスチャがあればバインド
					if(cur_mat->tex_name){
						glEnable(GL_TEXTURE_2D);
						glBindTexture(GL_TEXTURE_2D, cur_mat->tex_name);
						glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
					}
					else{
						glBindTexture(GL_TEXTURE_2D, 0);
						glDisable(GL_TEXTURE);
					}

					pre_mat = cur_mat;
				}
			}

			// ポリゴン描画
			glBegin(GL_POLYGON);
			for(int j = 0; j < n; ++j){
				int idx = face->vert_idx[j];
				if(idx >= 0 && idx < vn){
					glNormal3dv(normals[idx].data);
					if(!face->texcoords.empty()) glTexCoord2dv(face->texcoords[j].data);
					glVertex3dv((vertices[idx]).data);
				}
			}
			glEnd();
		}

		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
	}


	// 頂点描画
	if(draw & 0x01){
		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.0, 1.0);

		glDisable(GL_LIGHTING);
		glColor3d(1.0, 0.0, 0.0);
		glPointSize(5.0);
		glBegin(GL_POINTS);
		for(int i = 0; i < vn; ++i){
			glVertex3dv(vertices[i].data);
		}
		glEnd();
	}

	// エッジ描画
	if(draw & 0x02){
		glDisable(GL_LIGHTING);
		glColor3d(0.3, 1.0, 0.0);
		glLineWidth(1.0);
		for(int i = 0; i < pn; ++i){
			rxFace *face = &faces[i];
			int n = (int)face->vert_idx.size();
			
			glBegin(GL_LINE_LOOP);
			for(int j = 0; j < n; ++j){
				int idx = face->vert_idx[j];
				if(idx >= 0 && idx < vn){
					glVertex3dv(vertices[idx].data);
				}
			}
			glEnd();
		}
	}
	//glDisable(GL_POLYGON_OFFSET_FILL);

	// 法線描画
	if(draw & 0x08){
		glDisable(GL_LIGHTING);
		glColor3d(0.0, 0.9, 0.0);
		glLineWidth(1.0);
		for(int i = 0; i < vn; ++i){
			glBegin(GL_LINES);
			glVertex3dv(vertices[i].data);
			glVertex3dv((vertices[i]+dn*normals[i]).data);
			glEnd();
		}
	}
}


//-----------------------------------------------------------------------------
// MARK:関数の実装
//-----------------------------------------------------------------------------

/*!
 * 座標の回転
 * @param[in] pos0 回転したい座標値
 * @param[in] rot0 x,y,z軸周りの回転角(deg)
 * @return 開店後の座標値
 */
inline Vec3 Rotate3(const Vec3 &pos0, const Vec3 &rot)
{
	Vec3 S, C;
	S = Vec3(sin(rot[0]), sin(rot[1]), sin(rot[2]));
	C = Vec3(cos(rot[0]), cos(rot[1]), cos(rot[2]));

	Vec3 pos1;
	pos1[0] = C[1]*C[2]*pos0[0]-C[1]*S[2]*pos0[1]+S[1]*pos0[2];
	pos1[1] = (S[0]*S[1]*C[2]+C[0]*S[2])*pos0[0]+(-S[0]*S[1]*S[2]+C[0]*C[2])*pos0[1]-S[0]*C[1]*pos0[2];
	pos1[2] = (-C[0]*S[1]*C[2]+S[0]*S[2])*pos0[0]+(C[0]*S[1]*S[2]+S[0]*C[2])*pos0[1]+C[0]*C[1]*pos0[2];
	return pos1;
}

/*!
 * "文字列 数値"から数値部分の文字列のみを取り出す
 * @param[in] buf 元の文字列
 * @param[in] head 文字部分
 * @param[out] sub 数値部分の文字列
 */
inline bool StringToString(const string &buf, const string &head, string &sub)
{
	size_t pos = 0;
	if((pos = buf.find(head)) == string::npos) return false;
	pos += head.size();

	if((pos = buf.find_first_not_of(" 　\t", pos)) == string::npos) return false;
	
	sub = buf.substr(pos);
	return true;
}

/*!
 * "文字列 数値"から数値部分のみを取り出す
 * @param[in] buf 元の文字列
 * @param[in] head 文字部分
 * @param[out] val 数値
 */
inline bool StringToDouble(const string &buf, const string &head, double &val)
{
	string sub;
	if(StringToString(buf, head, sub)){
		sscanf(&sub[0], "%lf", &val);
		return true;
	}
	else{
		return false;
	}
}

/*!
 * "文字列 数値"から数値部分のみを取り出す
 * @param[in] buf 元の文字列
 * @param[in] head 文字部分
 * @param[out] val 数値
 */
inline bool StringToInt(const string &buf, const string &head, int &val)
{
	string sub;
	if(StringToString(buf, head, sub)){
		sscanf(&sub[0], "%d", &val);
		return true;
	}
	else{
		return false;
	}
}


/*!
 * "x y z"の形式の文字列からVec2型へ変換
 * @param[in] s 文字列
 * @param[out] v 値
 * @return 要素記述数
 */
inline int StringToVec2s(const string &buf, const string &head, Vec2 &v)
{
	string sub;
	StringToString(buf, head, sub);

	Vec2 tmp;

	if(sscanf(&sub[0], "%lf %lf", &tmp[0], &tmp[1]) != 2){
		return 0;
	}
	v = tmp;

	return 2;
}

/*!
 * "x y z"の形式の文字列からVec3型へ変換
 * @param[in] s 文字列
 * @param[out] v 値
 * @return 要素記述数
 */
inline int StringToVec3s(const string &buf, const string &head, Vec3 &v)
{
	string sub;
	StringToString(buf, head, sub);

	Vec3 tmp;
	double tmp_w;

	if(sscanf(&sub[0], "%lf %lf %lf %lf", &tmp[0], &tmp[1], &tmp[2], &tmp_w) != 4){
		if(sscanf(&sub[0], "%lf %lf %lf", &tmp[0], &tmp[1], &tmp[2]) != 3){
			return 0;
		}
	}
	v = tmp;

	return 3;
}

inline string Vec3ToString(const Vec3 v)
{
	stringstream ss;
	ss << v[0] << " " << v[1] << " " << v[2];
	return ss.str();
	//string s;
	//sscanf(&s[0], "%f %f %f", v[0], v[1], v[2]);
	//return s;
}

/*!
 * 文字列からベクトル値を抽出
 *  - たとえば，"0, 1, 2"だと，"がend_str, ","がbrk_str
 * @param[in] data 元の文字列
 * @param[in] end_str ベクトルの区切り文字
 * @param[in] brk_str ベクトル要素間の区切り文字
 * @param[in] min_elem 最小要素数
 * @param[out] vecs ベクトル値
 * @return 見つかったベクトルの数
 */
template<class T> 
inline int ExtractVector(string data, const string &end_str, const string &brk_str, int min_elems, 
						 vector< vector<T> > &vecs)
{
	data += end_str;	// 後の処理のために区切り文字列を最後に足しておく
	int n = 0;
	size_t cpos[2] = {0, 0};
	while((cpos[1] = data.find(end_str, cpos[0])) != string::npos){
		// 区切り文字が見つかったら，前回の区切り文字位置との間の文字列を抜き出す
		string sub = data.substr(cpos[0], cpos[1]-cpos[0]);
		if(sub.empty()){
			cpos[0] = cpos[1]+end_str.size();
			break;
		}

		// 抜き出した文字列を各ベクトル要素に分割
		sub += brk_str;
		vector<T> val;
		size_t spos[2] = {0, 0};
		while((spos[1] = sub.find(brk_str, spos[0])) != string::npos){
			string val_str = sub.substr(spos[0], spos[1]-spos[0]);
			DeleteSpace(val_str);
			if(val_str.empty()){
				spos[0] = spos[1]+brk_str.size();
				continue;
			}

			val.push_back((T)atof(val_str.c_str()));
			spos[0] = spos[1]+brk_str.size();
		}
		if((int)val.size() >= min_elems){
			vecs.push_back(val);
			n++;
		}
		cpos[0] = cpos[1]+end_str.size();
	}

	return n;
}

/*!
 * 先頭の空白(スペース，タブ)を削除
 * @param[in] buf 元の文字列
 * @return 空白削除後の文字列
 */
inline string GetDeleteSpace(const string &buf)
{
	string buf1 = buf;

	size_t pos;
	while((pos = buf1.find_first_of(" 　\t")) == 0){
		buf1.erase(buf1.begin());
		if(buf1.empty()) break;
	}

	return buf1;
}

/*!
 * 先頭の空白(スペース，タブ)を削除
 * @param[inout] buf 処理文字列
 */
inline void DeleteHeadSpace(string &buf)
{
	size_t pos;
	while((pos = buf.find_first_of(" 　\t")) == 0){
		buf.erase(buf.begin());
		if(buf.empty()) break;
	}
}

/*!
 * 空白(スペース，タブ)を削除
 * @param[inout] buf 処理文字列
 */
inline void DeleteSpace(string &buf)
{
	size_t pos;
	while((pos = buf.find_first_of(" 　\t")) != string::npos){
		buf.erase(pos, 1);
	}
}

/*!
 * 文字列に含まれる指定された文字列の数を返す
 * @param[in] s 元の文字列
 * @param[in] c 検索文字列
 * @return 含まれる数
 */
inline int CountString(string &s, int offset, string c)
{
	int count = 0;
	size_t pos0 = offset, pos = 0;
	int n = (int)c.size();

	while((pos = s.find(c, pos0)) != string::npos){
		if(pos != pos0){
			count++;
		}
		else{
			s.erase(s.begin()+pos);
		}
		pos0 = pos+n;
	}

	// 最後の文字列除去
	if(s.rfind(c) == s.size()-n){
		count--;
	}

	return count;
}

/*!
 * ポリゴンを三角形に分解
 * @param[out] ply 三角形に分解されたポリゴン列
 * @param[in] vidx ポリゴン(頂点数4以上)の頂点インデックス
 * @param[in] mat_name 材質名
 * @return 分解された三角形数
 */
inline int PolyToTri(vector<rxFace> &plys, 
					 const vector<int> &vidxs, 
					 string mat_name)
{
	int n = (int)vidxs.size();

	if(n <= 3) return 0;

	rxFace face;
	face.resize(3);
	face.material_name = mat_name;

	int num_tri = n-2;
	for(int i = 0; i < num_tri; ++i){
		face[0] = vidxs[0];
		face[1] = vidxs[i+1];
		face[2] = vidxs[i+2];

		if(face[1] != face[2] && face[0] != face[1] && face[2] != face[0]){
			plys.push_back(face);
		}
	}

	return num_tri;
}

/*!
 * ポリゴンを三角形に分解(with テクスチャ座標)
 * @param[out] ply 三角形に分解されたポリゴン列
 * @param[in] vidx ポリゴン(頂点数4以上)の頂点インデックス
 * @param[in] tidx ポリゴン(頂点数4以上)のテクスチャ座標インデックス
 * @param[in] vtc  テクスチャ座標ベクトル
 * @param[in] mat_name 材質名
 * @return 分解された三角形数
 */
inline int PolyToTri(vector<rxFace> &plys, 
					 const vector<int> &vidxs, 
					 const vector<int> &tidxs, 
					 const vector<Vec2> &vtc, 
					 string mat_name)
{
	int n = (int)vidxs.size();

	if(n <= 3) return 0;

	rxFace face;
	face.vert_idx.resize(3);
	face.material_name = mat_name;
	face.texcoords.resize(3);

	bool tc = !vtc.empty();

	int num_tri = n-2;
	for(int i = 0; i < num_tri; ++i){
		face[0] = vidxs[0];
		face[1] = vidxs[i+1];
		face[2] = vidxs[i+2];

		if(tc){
			face.texcoords[0] = vtc[tidxs[0]];
			face.texcoords[1] = vtc[tidxs[i+1]];
			face.texcoords[2] = vtc[tidxs[i+2]];
		}
		else{
			face.texcoords[0] = Vec2(0.0);
			face.texcoords[1] = Vec2(0.0);
			face.texcoords[2] = Vec2(0.0);
		}

		plys.push_back(face);
	}

	return num_tri;
}


/*!
 * ファイル名からフォルダパスのみを取り出す
 * @param[in] fn ファイル名(フルパス or 相対パス)
 * @return フォルダパス
 */
inline string ExtractDirPath(const string &fn)
{
	string::size_type pos;
	if((pos = fn.find_last_of("/")) == string::npos){
		if((pos = fn.find_last_of("\\")) == string::npos){
			return "";
		}
	}

	return fn.substr(0, pos);
}

/*!
 * ファイル名から拡張子を削除
 * @param[in] fn ファイル名(フルパス or 相対パス)
 * @return フォルダパス
 */
inline string ExtractPathWithoutExt(const string &fn)
{
	string::size_type pos;
	if((pos = fn.find_last_of(".")) == string::npos){
		return fn;
	}

	return fn.substr(0, pos);
}


/*!
 * 文字列小文字化
 * @param[inout] str 文字列
 */
inline void StringToLower(string &str)
{
	string::size_type i, size;

	size = str.size();

	for(i = 0; i < size; i++){
		if(str[i] >= 'A' && str[i] <= 'Z') str[i] += 32;
	}

	return;
}

/*!
 * 文字列が整数値を表しているかを調べる
 * @param[inout] str 文字列
 * @return 整数値ならtrue
 */
inline bool IsInteger(const string &str)
{
	if(str.find_first_not_of("-0123456789 \t") != string::npos) {
		return false;
	}

	return true;
}

/*!
 * 文字列が実数値を表しているかを調べる
 * @param[inout] str 文字列
 * @return 実数値ならtrue
 */
inline bool IsNumeric(const string &str)
{
	if(str.find_first_not_of("-0123456789. Ee\t") != string::npos) {
		return false;
	}

	return true;
}


/*!
 * バイナリファイルから型指定でデータを読み込む
 * @param[inout] file ファイル入力ストリーム
 * @return 読み込んだ数値
 */
template<class T> 
inline T ReadBinary(ifstream &file)
{
	T data;
	file.read((char*)&data, sizeof(T));
	return data;
}

/*!
 * バイナリファイルから型・サイズ指定でデータを読み込む
 * @param[inout] file ファイル入力ストリーム
 * @param[inout] byte 読み込むバイト数
 * @return 読み込んだ数値
 */
template<class T> 
inline T ReadBinary(ifstream &file, int byte)
{
	T data = 0;
	file.read((char*)&data, byte);
	return data;
}

/*!
 * バイナリファイルから2バイト分読み込んでint型に格納
 * @param[inout] file ファイル入力ストリーム
 * @return 読み込んだ数値
 */
inline int ReadInt(ifstream &file)
{
	int data = 0;
	file.read((char*)&data, 2);
	return data;
}

/*!
 * バイナリファイルから文字列を読み取る
 * @param[inout] file ファイル入力ストリーム
 * @param[out] name 文字列(何もなければ空)
 * @param[in] max_size 読み込む文字数(-1なら\0まで読み込む)
 * @return 文字列がない(ストリームの最初が\0)ならfalseを返す
 */
inline bool ReadString(ifstream &file, string &name, int max_size)
{
	char c = ReadBinary<char>(file);
	if((int)c == 0) return false;

	name.clear();
	name.push_back(c);
	do{
		c = ReadBinary<char>(file);
		name.push_back(c);
	}while(((int)c != 0) && (max_size == -1 || (int)name.size() < max_size));
	name.push_back((char)0);

	return true;
}


//-----------------------------------------------------------------------------
// MARK:Geometry processing
//-----------------------------------------------------------------------------
/*!
 * 法線反転
 * @param[in] vs 頂点リスト
 * @param[inout] ps メッシュリスト
 * @param[inout] ns 法線リスト
 */
static void ReverseNormals(const vector<Vec3> &vs, vector<int> &ps, vector<Vec3> &ns)
{
	if(ps.empty()) return;

	int nvp = 3;
	int pn = (int)ps.size()/nvp;

	vector<int> idx(nvp);
	for(int i = 0; i < pn; ++i){
		for(int j = 0; j < nvp; ++j){
			idx[j] = ps[nvp*i+j];
		}

		for(int j = 0; j < nvp; ++j){
			ps[nvp*i+j] = idx[nvp-j-1];
		}
	}

	if(ns.empty() || (int)ns.size() != pn){
		// 法線再計算
		ns.resize(pn);
		for(int i = 0; i < pn; ++i){
			ns[i] = Unit(cross(vs[ps[nvp*i+nvp-1]]-vs[ps[nvp*i+0]], vs[ps[nvp*i+1]]-vs[ps[nvp*i+0]]));
		}
	}
	else{
		// 法線反転
		for(int i = 0; i < pn; ++i){
			ns[i] = -ns[i];
		}
	}
}

/*!
 * メッシュ法線計算
 * @param[in] vs 頂点リスト
 * @param[inout] ps メッシュリスト
 * @param[inout] ns 法線リスト
 */
static void CalNormals(const vector<Vec3> &vs, vector<int> &ps, vector<Vec3> &ns)
{
	if(ps.empty()) return;

	int nvp = 3;
	int pn = (int)ps.size()/nvp;
	if(ns.empty() || (int)ns.size() != pn){
		// 法線再計算
		ns.resize(pn);
	}

	for(int i = 0; i < pn; ++i){
		int *idx = &ps[nvp*i];
		ns[i] = Unit(cross(vs[idx[1]]-vs[idx[0]], vs[idx[nvp-1]]-vs[idx[0]]));
		//ns[i] *= -1;
	}
}


/*!
 * 頂点法線計算
 * @param[in] vrts 頂点座標
 * @param[in] nvrts 頂点数
 * @param[in] tris 三角形ポリゴン幾何情報
 * @param[in] ntris 三角形ポリゴン数
 * @param[out] nrms 法線
 * @param[out] nnrms 法線数(=頂点数)
 */
static void CalVertexNormals(const vector<Vec3> &vrts, int nvrts, vector<rxTriangle> &tris, int ntris, 
							 vector<Vec3> &vnrms)
{
	int nnrms = nvrts;
	vnrms.resize(nnrms);
	
	// 頂点法線の初期化
	for(int i = 0; i < nnrms; i++){
		vnrms[i][0] = 0;
		vnrms[i][1] = 0;
		vnrms[i][2] = 0;
	}

	// 法線計算
	for(int i = 0; i < ntris; i++){
		Vec3 edge_vec1, edge_vec2, face_normal;

		// 面法線を計算
		edge_vec1 = vrts[tris[i][1]]-vrts[tris[i][0]];
		edge_vec2 = vrts[tris[i][2]]-vrts[tris[i][0]];
		face_normal = Unit(cross(edge_vec1, edge_vec2));

		// ポリゴンに所属する頂点の法線に積算
		vnrms[tris[i][0]] += face_normal;
		vnrms[tris[i][1]] += face_normal;
		vnrms[tris[i][2]] += face_normal;
	}

	// 頂点法線を正規化
	for(int i = 0; i < nnrms; i++){
		normalize(vnrms[i]);
	}

}

/*!
 * 頂点法線計算
 * @param[in] vrts 頂点座標
 * @param[in] nvrts 頂点数
 * @param[in] tris 三角形ポリゴン幾何情報
 * @param[in] ntris 三角形ポリゴン数
 * @param[out] nrms 法線
 * @param[out] nnrms 法線数(=頂点数)
 */
static void CalVertexNormals(const vector<Vec3> &vrts, int nvrts, vector<rxFace> &tris, int ntris, 
							 vector<Vec3> &vnrms)
{
	int nnrms = nvrts;
	vnrms.resize(nnrms);
	
	// 頂点法線の初期化
	for(int i = 0; i < nnrms; i++){
		vnrms[i][0] = 0;
		vnrms[i][1] = 0;
		vnrms[i][2] = 0;
	}

	// 法線計算
	for(int i = 0; i < ntris; i++){
		Vec3 edge_vec1, edge_vec2, face_normal;
		int n = (int)tris[i].size();

		// 面法線を計算
		edge_vec1 = vrts[tris[i][1]]-vrts[tris[i][0]];
		edge_vec2 = vrts[tris[i][n-1]]-vrts[tris[i][0]];
		face_normal = Unit(cross(edge_vec1, edge_vec2));

		// ポリゴンに所属する頂点の法線に積算
		for(int j = 0; j < n; ++j){
			vnrms[tris[i][j]] += face_normal;
		}
	}

	// 頂点法線を正規化
	for(int i = 0; i < nnrms; i++){
		normalize(vnrms[i]);
	}
}
/*!
 * 頂点法線計算
 * @param[in] polys ポリゴン
 */
static void CalVertexNormals(rxPolygons &polys)
{
	//(const vector<Vec3> &vrts, uint nvrts, vector<rxTriangle> &tris, uint ntris, vector<Vec3> &nrms)
	int pn = (int)polys.faces.size();
	int vn = (int)polys.vertices.size();

	vector<Vec3> fnrms;
	fnrms.resize(pn);

	int max_n = 0;

	// 面法線の計算
	for(int i = 0; i < pn; ++i){
		int n = (int)polys.faces[i].vert_idx.size()-1;
		fnrms[i] = Unit(cross(polys.vertices[polys.faces[i][1]]-polys.vertices[polys.faces[i][0]], 
							  polys.vertices[polys.faces[i][n]]-polys.vertices[polys.faces[i][0]]));

		n = n+1;
		if(n > max_n) max_n = n;
	}

	polys.normals.clear();
	polys.normals.resize(vn);
	for(int i = 0; i < vn; ++i){
		polys.normals[i] = Vec3(0.0);
	}

	// 頂点法線の計算
	vector<int> id;
	id.resize(max_n);
	for(int i = 0; i < pn; ++i){
		int n = (int)polys.faces[i].vert_idx.size();
		for(int j = 0; j < n; ++j){
			polys.normals[polys.faces[i][j]] += fnrms[i];
		}
	}

	// 法線正規化
	for(int i = 0; i < vn; i++){
		normalize(polys.normals[i]);
	}
}


/*!
 * 幾何情報を持たないポリゴン頂点列とポリゴン法線から頂点法線を計算
 * @param[in] vrts 幾何情報無しのポリゴン頂点列
 * @param[inout] nrms ポリゴン法線がそれぞれの頂点法線として格納されている
 */
static void CalVertexNormalWithoutGeometry(const vector<Vec3> &vrts, vector<Vec3> &nrms)
{
	int n = (int)vrts.size();

	int same_vrts[256];

	vector<int> find_vrts;
	find_vrts.resize(n);
	for(int i = 0; i < n; ++i){
		find_vrts[i] = 0;
	}

	int nvrts = 0;
	int avg_cnt = 0;

	double eps2 = 1e-4;
	for(int i = 0; i < n; ++i){
		if(find_vrts[i]) continue;
		
		Vec3 pos0 = vrts[i];
		Vec3 nrm0 = nrms[i];
		same_vrts[0] = i;
		int cnt = 1;
		find_vrts[i] = 1;
		for(int j = 0; j < n; ++j){
			if(find_vrts[j]) continue;

			Vec3 pos1 = vrts[j];

			if(norm2(pos0-pos1) < eps2){
				find_vrts[j] = 1;
				nrm0 += nrms[j];
				same_vrts[cnt] = j;
				cnt++;
			}
		}

		nrm0 /= (double)cnt;

		for(int j = 0; j < cnt; ++j){
			nrms[same_vrts[j]] = nrm0;
		}

		avg_cnt += cnt;
		nvrts++;
	}

	avg_cnt /= nvrts;
	//cout << avg_cnt << ", " << nvrts << endl;
}


/*!
 * 幾何情報を持たないポリゴン頂点列から幾何情報を生成
 * @param[inout] vrts 幾何情報無しのポリゴン頂点列
 * @param[out] idxs 幾何情報
 */
static void CalVertexGeometry(vector<Vec3> &vrts, vector< vector<int> > &idxs)
{
	int nv = (int)vrts.size();
	int np = nv/3;	// 三角形ポリゴン数

	// 幾何情報
	idxs.resize(np);
	for(int i = 0; i < np; ++i) idxs[i].resize(3);

	int same_vrts[256];	// 重複頂点のインデックスを格納

	vector<Vec3> compacted_vrts;	// 重複なし頂点列を格納するコンテナ

	// 重複除去処理済みフラグ格納コンテナの確保と初期化
	vector<int> find_vrts;
	find_vrts.resize(nv);
	for(int i = 0; i < nv; ++i){
		find_vrts[i] = 0;
	}

	int nvrts = 0;
	int avg_cnt = 0;

	double eps2 = 1e-4;
	for(int i = 0; i < nv; ++i){
		if(find_vrts[i]) continue;
		
		Vec3 pos0 = vrts[i];

		// まだ重複検索されていない頂点を格納
		compacted_vrts.push_back(pos0);
		idxs[i/3][i%3] = nvrts;

		same_vrts[0] = i;
		int cnt = 1;
		find_vrts[i] = 1;
		for(int j = 0; j < nv; ++j){
			if(find_vrts[j]) continue;

			Vec3 pos1 = vrts[j];

			// 距離が近い点を重複頂点とする
			if(norm2(pos0-pos1) < eps2){
				find_vrts[j] = 1;
				same_vrts[cnt] = j;

				idxs[j/3][j%3] = nvrts;	// compacted_vrts内のインデックス

				cnt++;	// 重複数をカウント
			}
		}

		avg_cnt += cnt;
		nvrts++;
	}

	avg_cnt /= nvrts;

	vrts = compacted_vrts;
}

/*!
 * オイラー角から回転行列に変換
 * @param[in] ang オイラー角
 * @param[out] mat 回転行列
 */
inline void EulerToMatrix(Vec3 ang, double mat[9])
{
	ang[1] = RX_TO_RADIANS(ang[1]);
	ang[0] = RX_TO_RADIANS(ang[0]);
	ang[2] = RX_TO_RADIANS(ang[2]);

	double cy = cos(ang[1]); 
	double sy = sin(ang[1]); 
	double cp = cos(ang[0]); 
	double sp = sin(ang[0]); 
	double cr = cos(ang[2]);
	double sr = sin(ang[2]);

	double cc = cy*cr; 
	double cs = cy*sr; 
	double sc = sy*cr; 
	double ss = sy*sr;

	mat[0] = cc+sp*ss;
	mat[1] = cs-sp*sc;
	mat[2] = -sy*cp;

	mat[3] = -cp*sr;
	mat[4] = cp*cr;
	mat[5] = -sp;

	mat[6] = sc-sp*cs;
	mat[7] = ss+sp*cc;
	mat[8] = cy*cp;
}

/*!
 * オイラー角から回転行列に変換
 * @param[in] ang オイラー角
 * @param[out] mat 回転行列
 */
inline void EulerToMatrix16(Vec3 ang, double mat[16])
{
	ang[1] = RX_TO_RADIANS(ang[1]);
	ang[0] = RX_TO_RADIANS(ang[0]);
	ang[2] = RX_TO_RADIANS(ang[2]);

	double cy = cos(ang[1]); 
	double sy = sin(ang[1]); 
	double cp = cos(ang[0]); 
	double sp = sin(ang[0]); 
	double cr = cos(ang[2]);
	double sr = sin(ang[2]);

	double cc = cy*cr; 
	double cs = cy*sr; 
	double sc = sy*cr; 
	double ss = sy*sr;

	mat[0]  = cc+sp*ss;
	mat[1]  = cs-sp*sc;
	mat[2]  = -sy*cp;

	mat[4]  = -cp*sr;
	mat[5]  = cp*cr;
	mat[6]  = -sp;

	mat[8]  = sc-sp*cs;
	mat[9]  = ss+sp*cc;
	mat[10] = cy*cp;
}

/*!
 * 頂点列をAABBに合うようにFitさせる(回転有り，アスペクト比無視)
 * @param[inout] poly ポリゴン
 * @param[in] cen AABB中心座標
 * @param[in] ext AABBの辺の長さ(1/2)
 * @param[in] ang 回転ベクトル
 */
static bool AffineVertices(rxPolygons &polys, Vec3 cen, Vec3 ext, Vec3 ang)
{
	int vn = (int)polys.vertices.size();
	if(vn <= 1) return false;

	// 現在のBBoxの大きさを調べる
	Vec3 minp, maxp;
	minp = maxp = polys.vertices[0];
	for(int i = 1; i < vn; ++i){
		Vec3 pos = polys.vertices[i];
		for(int i = 0; i < 3; ++i){
			if(pos[i] > maxp[i]) maxp[i] = pos[i];
			if(pos[i] < minp[i]) minp[i] = pos[i];
		}
	}
	
	Vec3 scale = (maxp-minp);
	Vec3 trans = (maxp+minp)/2.0;

	for(int i = 0; i < 3; ++i){
		if(fabs(scale[i]) < RX_FEQ_EPS){
			scale[i] = 1.0;
		}
	}


	double mat[9];
	EulerToMatrix(-ang, mat);
	for(int i = 0; i < vn; ++i){
		Vec3 pos = polys.vertices[i];
		Vec3 nrm = polys.normals[i];

		Vec3 pos1 = ((pos-trans)/scale)*2.0*ext;
		Vec3 nrm1 = nrm;

		pos[0] = mat[0]*pos1[0]+mat[1]*pos1[1]+mat[2]*pos1[2];
		pos[1] = mat[3]*pos1[0]+mat[4]*pos1[1]+mat[5]*pos1[2];
		pos[2] = mat[6]*pos1[0]+mat[7]*pos1[1]+mat[8]*pos1[2];
		nrm[0] = mat[0]*nrm1[0]+mat[1]*nrm1[1]+mat[2]*nrm1[2];
		nrm[1] = mat[3]*nrm1[0]+mat[4]*nrm1[1]+mat[5]*nrm1[2];
		nrm[2] = mat[6]*nrm1[0]+mat[7]*nrm1[1]+mat[8]*nrm1[2];

		pos += cen;

		polys.vertices[i] = pos;
		polys.normals[i]  = nrm;
	}

	return true;
}


/*!
 * 頂点列からのAABBの検索
 * @param[out] minp,maxp AABBの最大座標，最小座標
 * @param[in] vec_set 頂点列
 * @param[in] start_index,end_index 頂点列の検索範囲
 * @return 検索できたらtrue
 */
inline bool FindBBox(Vec3 &minp, Vec3 &maxp, 
					 const vector<Vec3> &vec_set, 
					 const int start_index, const int end_index)
{
	if((int)vec_set.size() == 0) return false;

	maxp = vec_set[start_index];
	minp = vec_set[start_index];

	for(int i = start_index+1; i < end_index; ++i){
		if(vec_set[i][0] > maxp[0]) maxp[0] = vec_set[i][0];
		if(vec_set[i][1] > maxp[1]) maxp[1] = vec_set[i][1];
		if(vec_set[i][2] > maxp[2]) maxp[2] = vec_set[i][2];
		if(vec_set[i][0] < minp[0]) minp[0] = vec_set[i][0];
		if(vec_set[i][1] < minp[1]) minp[1] = vec_set[i][1];
		if(vec_set[i][2] < minp[2]) minp[2] = vec_set[i][2];
	}

	return true;
}
/*!
 * 頂点列からのAABBの検索
 * @param[out] minp,maxp AABBの最大座標，最小座標
 * @param[in] vec_set 頂点列
 * @return 検索できたらtrue
 */
inline bool FindBBox(Vec3 &minp, Vec3 &maxp, const vector<Vec3> &vec_set)
{
	return FindBBox(minp, maxp, vec_set, 0, (int)vec_set.size());
}

/*!
 * 頂点列をAABBに合うようにFitさせる
 * @param[in] ctr AABB中心座標
 * @param[in] sl  AABBの辺の長さ(1/2)
 * @param[in] vec_set 頂点列
 * @param[in] start_index,end_index 頂点列の検索範囲
 */
inline bool FitVertices(const Vec3 &ctr, const Vec3 &sl, 
						vector<Vec3> &vec_set, 
						const int start_index, const int end_index)
{
	Vec3 ctr0, sl0, maxp, minp;

	// 現在のBBoxの大きさを調べる
	FindBBox(minp, maxp, vec_set, start_index, end_index);
			
	sl0  = (maxp-minp)/2.0;
	ctr0 = (maxp+minp)/2.0;

	int max_axis = ( ( (sl0[0] > sl0[1]) && (sl0[0] > sl0[2]) ) ? 0 : ( (sl0[1] > sl0[2]) ? 1 : 2 ) );
	int min_axis = ( ( (sl0[0] < sl0[1]) && (sl0[0] < sl0[2]) ) ? 0 : ( (sl0[1] < sl0[2]) ? 1 : 2 ) );
	double size_conv = sl[max_axis]/sl0[max_axis];

	// 全ての頂点をbboxにあわせて変換
	for(int i = start_index; i < end_index; ++i){
		vec_set[i] = (vec_set[i]-ctr0)*size_conv+ctr;
	}

	return true;
}

/*!
 * 頂点列をAABBに合うようにFitさせる
 * @param[in] ctr AABB中心座標
 * @param[in] sl  AABBの辺の長さ(1/2)
 * @param[in] vec_set 頂点列
 */
inline bool FitVertices(const Vec3 &ctr, const Vec3 &sl, vector<Vec3> &vec_set)
{
	FitVertices(ctr, sl, vec_set, 0, (int)vec_set.size());
	return true;
}

/*!
 * 頂点法線計算
 * @param[in] vrts 頂点座標
 * @param[in] nvrts 頂点数
 * @param[in] tris 三角形ポリゴン幾何情報
 * @param[in] ntris 三角形ポリゴン数
 * @param[out] nrms 法線
 * @param[out] nnrms 法線数(=頂点数)
static void CalVertexNormalsFromVBO(GLuint vrts_vbo, GLuint tris_vbo, GLuint nrms_vbo, uint nvrts, uint ntris)
{
	vector<Vec3> nrms_temp;
	nrms_temp.resize(nvrts);

	glBindBuffer(GL_ARRAY_BUFFER, vrts_vbo);
	float *vrts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, tris_vbo);
    uint *tris = (uint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_ONLY);

	// Set all normals to 0.
	for(uint i = 0; i < nvrts; i++){
		nrms_temp[i][0] = 0;
		nrms_temp[i][1] = 0;
		nrms_temp[i][2] = 0;
	}

	// Calculate normals.
	for(uint i = 0; i < ntris; i++){
		Vec3 vec1, vec2, normal;
		uint id0, id1, id2;
		id0 = tris[3*i+0];
		id1 = tris[3*i+1];
		id2 = tris[3*i+2];

		for(int j = 0; j < 3; ++j){
			vec1[j] = vrts[3*id1+j]-vrts[3*id0+j];
			vec2[j] = vrts[3*id2+j]-vrts[3*id0+j];
		}
		normal = Unit(cross(vec2, vec1));

		nrms_temp[id0] += normal;
		nrms_temp[id1] += normal;
		nrms_temp[id2] += normal;
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, nrms_vbo);
	float *nrms = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	// Normalize normals.
	for(uint i = 0; i < nvrts; i++){
		normalize(nrms_temp[i]);

		for(int j = 0; j < 3; ++j){
			nrms[3*i+j] = nrms_temp[i][j];
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}
 */


/*!
 * シミュレーション空間を覆うグリッドの算出
 *  - 最大グリッド数nmaxを指定して，最大辺がそのグリッド数になるように他の辺を分割する．
 *  - グリッドは立方体セルで構成される．
 * @param[in] minp,maxp シミュレーション空間の範囲
 * @param[in] nmax 最大グリッド数
 * @param[out] h 立方体セルの幅
 * @param[out] n 各軸のセル数
 * @param[in] extend 空間を完全に覆うための拡張幅[0,1]
 * @return 成功ならば0
 */
inline int CalMeshDiv(Vec3 &minp, Vec3 maxp, int nmax, double &h, int n[3], double extend)
{
	Vec3 l = maxp-minp;
	minp -= 0.5*extend*l;
	l *= 1.0+extend;

	double max_l = 0;
	int max_axis = 0;
	for(int i = 0; i < 3; ++i){
		if(l[i] > max_l){
			max_l = l[i];
			max_axis = i;
		}
	}

	h = max_l/nmax;
	for(int i = 0; i < 3; ++i){
		n[i] = (int)(l[i]/h)+1;
	}

	return 0;
}


/*!
 * メッシュ描画用のVBOを確保
 * @param[in] max_verts 最大頂点数
 * @param[in] uVrtVBO 頂点FBO
 * @param[in] uNrmVBO 法線FBO
 * @param[in] uTriVBO メッシュFBO
 */
inline bool AssignArrayBuffers(int max_verts, int dim, GLuint &uVrtVBO, GLuint &uNrmVBO, GLuint &uTriVBO)
{
	// 頂点VBO
	if(!uVrtVBO) glGenBuffers(1, &uVrtVBO);
	glBindBuffer(GL_ARRAY_BUFFER, uVrtVBO);
	glBufferData(GL_ARRAY_BUFFER, max_verts*dim*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 法線VBO
	if(!uNrmVBO) glGenBuffers(1, &uNrmVBO);
	glBindBuffer(GL_ARRAY_BUFFER, uNrmVBO);
	glBufferData(GL_ARRAY_BUFFER, max_verts*dim*sizeof(float), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// メッシュVBO
	if(!uTriVBO) glGenBuffers(1, &uTriVBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uTriVBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, max_verts*3*3*sizeof(unsigned int), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}


/*!
 * FBOにデータを設定
 * @param[in] uVrtVBO 頂点FBO
 * @param[in] uNrmVBO 法線FBO
 * @param[in] uTriVBO メッシュFBO
 */
inline bool SetFBOFromArray(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO, 
							vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face)
{
	int nv = (int)vrts.size();
	int nm = (int)face.size();
	int nn = nv;

	// 頂点アレイに格納
	glBindBuffer(GL_ARRAY_BUFFER, uVrtVBO);
	float *vrt_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for(int i = 0; i < nv; ++i){
		for(int j = 0; j < 3; ++j){
			vrt_ptr[3*i+j] = (float)vrts[i][j];
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 法線情報の取得
	glBindBuffer(GL_ARRAY_BUFFER, uNrmVBO);
	float *nrm_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	for(int i = 0; i < nn; ++i){
		for(int j = 0; j < 3; ++j){
			nrm_ptr[3*i+j] = (float)nrms[i][j];
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 接続情報の取得
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uTriVBO);
	unsigned int *tri_ptr = (unsigned int*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

	for(int i = 0; i < nm; ++i){
		for(int j = 0; j < 3; ++j){
			tri_ptr[3*i+j] = face[i][j];
		}
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}

/*!
 * ホスト側配列にデータを設定
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 */
inline bool SetArrayFromFBO(GLuint uVrtVBO, GLuint uNrmVBO, GLuint uTriVBO, 
							vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face, int nvrts, int ntris)
{
	// 頂点アレイに格納
	glBindBuffer(GL_ARRAY_BUFFER, uVrtVBO);
	float *vrt_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	vrts.resize(nvrts);
	for(int i = 0; i < nvrts; ++i){
		vrts[i][0] = vrt_ptr[4*i];
		vrts[i][1] = vrt_ptr[4*i+1];
		vrts[i][2] = vrt_ptr[4*i+2];
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 法線情報の取得
	glBindBuffer(GL_ARRAY_BUFFER, uNrmVBO);
	float *nrm_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

	nrms.resize(nvrts);
	for(int i = 0; i < nvrts; ++i){
		nrms[i][0] = nrm_ptr[4*i];
		nrms[i][1] = nrm_ptr[4*i+1];
		nrms[i][2] = nrm_ptr[4*i+2];
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// 接続情報の取得
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, uTriVBO);
	unsigned  *tri_ptr = (unsigned int*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_ONLY);

	face.resize(ntris);
	for(int i = 0; i < ntris; ++i){
		face[i].vert_idx.resize(3);
		face[i][0] = tri_ptr[3*i];
		face[i][1] = tri_ptr[3*i+1];
		face[i][2] = tri_ptr[3*i+2];
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return true;
}




#endif // #ifndef _RX_MESH_H_
