/*!
  @file rx_dxf.cpp

  @brief DXF File Input
		- 3DFACEのみ
 
  @author Makoto Fujisawa
  @date   2011
*/

#ifndef _RX_DXF_H_
#define _RX_DXF_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include "rx_mesh.h"

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxDXFクラスの宣言 - DXF形式の読み込み
//-----------------------------------------------------------------------------
class rxDXF
{
public:
	//! コンストラクタ
	rxDXF();
	//! デストラクタ
	~rxDXF();

	/*!
	 * DXFファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] vrts 頂点座標
	 * @param[out] vnms 頂点法線
	 * @param[out] poly ポリゴン
	 * @param[out] mats 材質情報
	 * @param[in] triangle ポリゴンの三角形分割フラグ
	 */
	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, bool triangle = true);

	/*!
	 * DXFファイル書き込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[in] vrts 頂点座標
	 * @param[in] plys ポリゴン
	 */
	bool Save(string file_name, const vector<Vec3> &vrts, const vector<rxFace> &plys);

private:
	// セクション
	enum{
		RX_DXF_SECTION_HEADER = 1, 
		RX_DXF_SECTION_CLASSES, 
		RX_DXF_SECTION_TABLES, 
		RX_DXF_SECTION_BLOCKS, 
		RX_DXF_SECTION_ENTITES, 
		RX_DXF_SECTION_OBJECTS, 
		RX_DXF_SECTION_THUMBNAILIMAGE, 
	};

	// エンティティ
	enum{
		RX_DXF_ENTITY_3DFACE = 1, 
		RX_DXF_ENTITY_3DSOLID, 
		RX_DXF_ENTITY_CIRCLE, 
		RX_DXF_ENTITY_ELLIPSE, 
		RX_DXF_ENTITY_LINE, 
		RX_DXF_ENTITY_MESH, 
		RX_DXF_ENTITY_POINT, 
		RX_DXF_ENTITY_POLYLINE, 
		RX_DXF_ENTITY_SPLINE, 
		RX_DXF_ENTITY_TEXT, 
		RX_DXF_ENTITY_VERTEX, 
	};

	map<string, int> m_mapSection;
	map<string, int> m_mapEntity;
};




#endif // _RX_VRML_H_
