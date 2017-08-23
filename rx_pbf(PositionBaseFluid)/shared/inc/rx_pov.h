/*!
  @file rx_pov.h
	
  @brief POV/INC File Output
 
  @author Makoto Fujisawa
  @date   2011
*/
// FILE --rx_pov.h--

#ifndef _RX_POV_H_
#define _RX_POV_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_mesh.h"


//-----------------------------------------------------------------------------
// 名前空間
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxPOVクラスの宣言 - POV形式の出力
//-----------------------------------------------------------------------------
class rxPOV
{
public:
	//! コンストラクタ
	rxPOV(){}

	//! デストラクタ
	~rxPOV(){}


	/*!
	 * POV-Ray形式で保存(rxTriangle)
	 * @param[in] file_name 保存ファイル名
	 * @param[in] verts 頂点座標
	 * @param[in] norms 頂点法線
	 * @param[in] polys 三角形メッシュ
	 * @param[in] obj_name POV-Rayオブジェクト名
	 * @param[in] add 追記フラグ
	 */
	bool SaveListData(const string &file_name, vector<Vec3> &verts, vector<Vec3> &norms, vector<rxTriangle> &polys, 
					  const string &obj_name = "PolygonObject", bool add = false)
	{
		return SaveListData(file_name, verts, norms, polys, Vec3(0.0), Vec3(1.0), Vec3(0.0), obj_name, add);
	}

	/*!
	 * POV-Ray形式で保存(rxTriangle,変換付き)
	 * @param[in] file_name 保存ファイル名
	 * @param[in] verts 頂点座標
	 * @param[in] norms 頂点法線
	 * @param[in] polys 三角形メッシュ
	 * @param[in] trans,scale,rot 平行移動，スケーリング，回転
	 * @param[in] obj_name POV-Rayオブジェクト名
	 * @param[in] add 追記フラグ
	 */
	bool SaveListData(const string &file_name, vector<Vec3> &verts, vector<Vec3> &norms, vector<rxTriangle> &polys, 
					  const Vec3 &trans, const Vec3 &scale, const Vec3 &rot, 
					  const string &obj_name = "PolygonObject", bool add = false)
	{
		if((int)polys.size() == 0) return false;

		// INCファイル作成
		FILE* fp;

		if(add){
			if((fp = fopen(file_name.c_str(),"a")) == NULL) return false;
		}
		else{
			if((fp = fopen(file_name.c_str(),"w")) == NULL) return false;
		}

		fprintf(fp, "#declare %s = mesh2 {\n", obj_name.c_str());

		// 頂点データ出力
		fprintf(fp, "	vertex_vectors {\n");
		fprintf(fp, "		%d,\n", verts.size());

		int i;
		Vec3 v;
		for(i = 0; i < (int)verts.size(); ++i){
			v = verts[i];
			v *= scale;				// スケーリング
			v = Rotate3(v, rot);	// 回転
			v += trans;				// 平行移動

			fprintf(fp, "		<%f,%f,%f>\n", v[0], v[1], v[2]);
		}
		fprintf(fp, "	}\n");


		// 頂点法線出力
		fprintf(fp, "	normal_vectors {\n");
		fprintf(fp, "		%d,\n",verts.size());

		Vec3 normal;
		for(i = 0; i < (int)verts.size(); ++i){
			normal = norms[i];
			normal = Rotate3(normal, rot);	// 回転
			fprintf(fp, "		<%f,%f,%f>\n", normal[0], normal[1], normal[2]);
		}
		fprintf(fp, "	}\n");


		// ポリゴンインデックス出力
		fprintf(fp, "	face_indices {\n");
	
		int np = (int)polys.size();

		fprintf(fp, "		%d,\n", np);

		for(i = 0; i < np; ++i){
			// 三角形ポリゴン
			fprintf(fp, "		<%d,%d,%d>,\n", polys[i][0], polys[i][1], polys[i][2]);
		}

	/*
		int j, n, np = 0;
		for(i = 0; i < (int)polys.size(); ++i){
			n = polys[i].size();
			np += n-2;
		}

		fprintf(fp, "		%d,\n", np);

		for(i = 0; i < (int)polys.size(); ++i){
			n = polys[i].size();

			// 三角形ポリゴンに分割
			for(j = 0; j < n-2; ++j){
				fprintf(fp, "		<%d,%d,%d>,\n", polys[i][0], polys[i][j+1], polys[i][j+2]);
			}
		}
	*/
		fprintf(fp, "	}\n");
		fprintf(fp, "	inside_vector <0.0, 0.0, 0.0>\n");
		fprintf(fp, "}   //#declare %s\n\n", obj_name.c_str());

		fclose(fp);

		return true;
	}

	/*!
	 * POV-Ray形式で保存(rxFace)
	 * @param[in] file_name 保存ファイル名
	 * @param[in] verts 頂点座標
	 * @param[in] norms 頂点法線
	 * @param[in] polys ポリゴンメッシュ
	 * @param[in] obj_name POV-Rayオブジェクト名
	 * @param[in] add 追記フラグ
	 */
	bool SaveListData(const string &file_name, vector<Vec3> &verts, vector<Vec3> &norms, vector<rxFace> &polys, 
					  const string &obj_name = "PolygonObject", bool add = false)
	{
		return SaveListData(file_name, verts, norms, polys, Vec3(0.0), Vec3(1.0), Vec3(0.0), obj_name, add);
	}


	/*!
	 * POV-Ray形式で保存(rxFace,変換付き)
	 * @param[in] file_name 保存ファイル名
	 * @param[in] verts 頂点座標
	 * @param[in] norms 頂点法線
	 * @param[in] polys ポリゴンメッシュ
	 * @param[in] trans,scale,rot 平行移動，スケーリング，回転
	 * @param[in] obj_name POV-Rayオブジェクト名
	 * @param[in] add 追記フラグ
	 */
	bool SaveListData(const string &file_name, vector<Vec3> &verts, vector<Vec3> &norms, vector<rxFace> &polys, 
					  const Vec3 &trans, const Vec3 &scale, const Vec3 &rot, 
					  const string &obj_name = "PolygonObject", bool add = false)
	{
		if((int)polys.size() == 0) return false;

		// INCファイル作成
		FILE* fp;

		if(add){
			if((fp = fopen(file_name.c_str(),"a")) == NULL) return false;
		}
		else{
			if((fp = fopen(file_name.c_str(),"w")) == NULL) return false;
		}

		fprintf(fp, "#declare %s = mesh2 {\n", obj_name.c_str());

		//
		// 頂点データ出力
		fprintf(fp, "	vertex_vectors {\n");
		fprintf(fp, "		%d,\n", verts.size());

		int i;
		Vec3 v;
		for(i = 0; i < (int)verts.size(); ++i){
			v = verts[i];
			v *= scale;				// スケーリング
			v = Rotate3(v, rot);	// 回転
			v += trans;				// 平行移動

			fprintf(fp, "		<%f,%f,%f>\n", v[0], v[1], v[2]);
		}
		fprintf(fp, "	}\n");

		//
		// 頂点法線出力
		fprintf(fp, "	normal_vectors {\n");
		fprintf(fp, "		%d,\n",verts.size());

		Vec3 normal;
		for(i = 0; i < (int)verts.size(); ++i){
			normal = norms[i];
			normal = Rotate3(normal, rot);	// 回転
			fprintf(fp, "		<%f,%f,%f>\n", normal[0], normal[1], normal[2]);
		}
		fprintf(fp, "	}\n");

		//
		// ポリゴンインデックス出力
		fprintf(fp, "	face_indices {\n");

		// 総三角形数の算出と出力
		int np = 0;
		for(i = 0; i < (int)polys.size(); ++i){
			np += polys[i].vert_idx.size()-2;
		}
		fprintf(fp, "		%d,\n", np);

		for(i = 0; i < (int)polys.size(); ++i){
			int n = polys[i].vert_idx.size();

			// 三角形ポリゴンに分割
			for(int j = 0; j < n-2; ++j){
				fprintf(fp, "		<%d,%d,%d>,\n", polys[i][j+2], polys[i][j+1], polys[i][0]);
			}
		}

		fprintf(fp, "	}\n");
		fprintf(fp, "	inside_vector <0.0, 0.0, 0.0>\n");
		fprintf(fp, "}   //#declare %s\n\n", obj_name.c_str());

		fclose(fp);

		return true;
	}

	/*!
	 * POV-Ray形式で保存(rxPolygons)
	 * @param[in] file_name 保存ファイル名
	 * @param[in] polys ポリゴンオブジェクト
	 * @param[in] obj_name POV-Rayオブジェクト名
	 * @param[in] add 追記フラグ
	 */
	bool SaveListData(const string &file_name, rxPolygons &polys, 
					  const string &obj_name = "PolygonObject", bool add = false)
	{
		return SaveListData(file_name, polys.vertices, polys.normals, polys.faces, obj_name, add);
	}

};




#endif // #ifndef _RX_POV_H_