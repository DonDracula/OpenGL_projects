/*! 
  @file rx_mesh_e.h

  @brief 拡張メッシュ構造の定義
  		 - エッジ情報を追加
  		 - 頂点，エッジ，面の追加・削除に対応
 
  @author Makoto Fujisawa
  @date 2012-03
*/
// FILE --rx_mesh_e.h--

#ifndef _RX_MESH_E_H_
#define _RX_MESH_E_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_mesh.h"


//-----------------------------------------------------------------------------
// メッシュクラス
//-----------------------------------------------------------------------------
//! エッジ
struct rxEdge
{
	int v[2];		//!< エッジ両端の頂点インデックス
	set<int> f;		//!< エッジにつながるポリゴン情報
	double len;		//!< エッジ長さ
	int attribute;	//!< 属性
};

//! 拡張ポリゴンオブジェクト(エッジ情報追加)
class rxPolygonsE : public rxPolygons
{
public:
	vector<rxEdge> edges;		//!< エッジ
	vector< set<int> > vedges;	//!< 頂点につながるエッジ情報
	vector< set<int> > vfaces;	//!< 頂点につながるポリゴン情報
	vector< set<int> > fedges;	//!< ポリゴンに含まれるエッジ情報

	vector<Vec3> vcol;			//!< 頂点色

public:
	//! コンストラクタ
	rxPolygonsE() : rxPolygons() {}
	//! デストラクタ
	virtual ~rxPolygonsE(){}

	//! 初期化
	void Clear(void)
	{
		vertices.clear();
		normals.clear();
		faces.clear();
		materials.clear();
		edges.clear();
		vedges.clear();
		vfaces.clear();
		fedges.clear();
	}

};


//-----------------------------------------------------------------------------
// メッシュデータ編集関数
//-----------------------------------------------------------------------------

/*!
 * 削除フラグ付きのポリゴンを削除
 * @param[inout] src 変更するポリゴン
 * @param[in] del_attr 削除する面に登録された属性
 */
static int DeletePolygon(rxPolygonsE *src, int del_attr)
{
	vector<rxFace>::iterator iter = src->faces.begin();
	int d = 0;
	while(iter != src->faces.end()){
		if(iter->attribute == del_attr){
			iter = src->faces.erase(iter);
			d++;
		}
		else{
			++iter;
		}
	}
	return d;
}


/*!
 * アンブレラオペレータで頂点をFairing
 * @param[inout] obj 拡張ポリゴンオブジェクト
 * @return 成功の場合1
 */
static int VertexFairingByUmbrella(rxPolygonsE &obj)
{
	int vnum = (int)obj.vertices.size();
	if(!vnum || obj.vedges.empty()) return 0;

	for(int i = 0; i < vnum; ++i){
		Vec3 newpos = obj.vertices[i];

		// 頂点につながるエッジを走査
		int en = 0;
		set<int>::iterator eiter = obj.vedges[i].begin();
		do{
			int v1, v2;
			v1 = obj.edges[*eiter].v[0];
			v2 = obj.edges[*eiter].v[1];

			newpos += obj.vertices[(v1 != i ? v1 : v2)];
			en++;
		}while(++eiter != obj.vedges[i].end());

		newpos /= (double)(en+1);

		obj.vertices[i] = newpos;
	}

	return 1;
}


/*!
 * 頂点に接続するポリゴンを探索
 * @param[inout] obj 拡張ポリゴンオブジェクト
 * @return 成功の場合1
 */
static int SearchVertexFace(rxPolygonsE &obj)
{
	int vnum = (int)obj.vertices.size();
	if(!vnum) return 0;

	obj.vfaces.clear();
	obj.vfaces.resize(vnum);

	rxFace* face;
	int pnum = (int)obj.faces.size();

	// 全ポリゴンを探索
	for(int i = 0; i < pnum; ++i){
		face = &obj.faces[i];
		for(int j = 0; j < face->size(); ++j){
			int v = face->at(j);
			obj.vfaces[v].insert(i);
		}
	}

	return 1;
}

/*!
 * エッジデータを作成
 * @param[inout] obj 拡張ポリゴンオブジェクト
 * @return エッジ数
 */
static int SearchEdge(rxPolygonsE &obj)
{
	int edge_count = 0;
	int vnum = (int)obj.vertices.size();
	int pnum = (int)obj.faces.size();

	obj.vedges.clear();
	obj.vedges.resize(vnum);
	obj.fedges.clear();
	obj.fedges.resize(pnum);
	obj.edges.clear();

	rxFace* face;
	int vert_idx[2];


	// 全ポリゴンを探索
	for(int i = 0; i < pnum; ++i){
		face = &obj.faces[i];
		vnum = face->size();

		for(int j = 0; j < vnum; ++j){
			// エッジ頂点1
			vert_idx[0] = face->at(j);
			// エッジ頂点2
			vert_idx[1] = (j == vnum-1) ? face->at(0) : face->at(j+1);

			// 重複する稜線のチェック
			bool overlap = false;
			set<int>::iterator viter;

			// エッジ頂点1のチェック
			if((int)obj.vedges[vert_idx[0]].size() > 0){
				// 頂点につながっているエッジ(既に登録されたもの)を調べる
				viter = obj.vedges[vert_idx[0]].begin();
				do{
					int v1, v2;
					v1 = obj.edges[*viter].v[0];
					v2 = obj.edges[*viter].v[1];

					if( (vert_idx[0] == v1 && vert_idx[1] == v2) || (vert_idx[1] == v1 && vert_idx[0] == v2) ){
						overlap = true;
						obj.edges[*viter].f.insert(i);	// 面をエッジに加える
						obj.fedges[i].insert(*viter);	// エッジを面に加える
					}
				}while(++viter != obj.vedges[vert_idx[0]].end());
			}

			// エッジ頂点2のチェック
			if((int)obj.vedges[vert_idx[1]].size() > 0){
				// 頂点につながっているエッジ(既に登録されたもの)を調べる
				viter = obj.vedges[vert_idx[1]].begin();
				do{
					int v1, v2;
					v1 = obj.edges[*viter].v[0];
					v2 = obj.edges[*viter].v[1];

					if( (vert_idx[0] == v1 && vert_idx[1] == v2) || (vert_idx[1] == v1 && vert_idx[0] == v2) ){
						overlap = true;
						obj.edges[*viter].f.insert(i);	// 面をエッジに加える
						obj.fedges[i].insert(*viter);	// エッジを面に加える
					}
				}while(++viter != obj.vedges[vert_idx[1]].end());
			}

			// 重複する稜線なしの場合，新規に追加
			if(!overlap){
				obj.edges.push_back(rxEdge());
				obj.edges[edge_count].v[0] = vert_idx[0];
				obj.edges[edge_count].v[1] = vert_idx[1];
				obj.edges[edge_count].f.insert(i);

				obj.vedges[vert_idx[0]].insert(edge_count);
				obj.vedges[vert_idx[1]].insert(edge_count);
				obj.fedges[i].insert(edge_count);

				edge_count++;
			}

		}
	}

	return edge_count;
}


/*!
 * エッジ長さを計算
 * @param[inout] obj 拡張ポリゴンオブジェクト
 * @return エッジ数
 */
static int CalEdgeLength(rxPolygonsE &obj)
{
	int en = (int)obj.edges.size();

	if(en == 0) return 0;

	int vn = (int)obj.vertices.size();

	rxEdge *edge;
	for(int i = 0; i < en; ++i){
		edge = &obj.edges[i];

		int v0 = edge->v[0];
		int v1 = edge->v[1];

		edge->len = norm(obj.vertices[v0]-obj.vertices[v1]);
	}

	return 1;
}


#endif // #ifndef _RX_MESH_E_H_
