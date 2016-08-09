/*!
  @file rx_vrml.h

  @brief VRML Input/Output
	- ShapeのIndexedFaceSetのみ対応

  @author Makoto Fujisawa
  @date   2011
*/

#ifndef _RX_VRML_H_
#define _RX_VRML_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include "rx_mesh.h"

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// VRMLクラスの宣言 - VRML形式の読み込み
//-----------------------------------------------------------------------------
class rxVRML
{
public:
	//! コンストラクタ
	rxVRML();
	//! デストラクタ
	~rxVRML();

	/*!
	 * VRMLファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] vrts 頂点座標
	 * @param[out] vnms 頂点法線
	 * @param[out] poly ポリゴン
	 * @param[out] mats 材質情報
	 * @param[in] triangle ポリゴンの三角形分割フラグ
	 */
	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, rxMTL &mats, bool tri = true);

	/*!
	 * VRMLファイル書き込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[in] vrts 頂点座標
	 * @param[in] vnms 頂点法線
	 * @param[in] plys ポリゴン
	 * @param[in] mats 材質情報
	 */
	bool Save(string file_name, const vector<Vec3> &vrts, const vector<Vec3> &vnms, const vector<rxFace> &plys, const rxMTL &mats);

};




#endif // _RX_VRML_H_
