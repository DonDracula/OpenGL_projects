/*!
  @file rx_obj.h
	
  @brief OBJ/MTL File Input/Output
 
  @author Makoto Fujisawa
  @date   2011
*/

#ifndef _RX_OBJ_H_
#define _RX_OBJ_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include "rx_mesh.h"

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxOBJクラスの宣言 - OBJ形式の読み込み
//-----------------------------------------------------------------------------
class rxOBJ
{
	rxMTL m_mapMaterials;	//!< ラベルとデータのマップ
	string m_strCurrentMat;	//!< 現在のデータを示すラベル

	//vector<rxMaterialOBJ> m_vMaterials;
	//int m_iCurrentMat;

public:
	//! コンストラクタ
	rxOBJ();
	//! デストラクタ
	~rxOBJ();

	/*!
	 * OBJファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] vrts 頂点座標
	 * @param[out] vnms 頂点法線
	 * @param[out] poly ポリゴン
	 * @param[out] mats 材質情報
	 * @param[in] triangle ポリゴンの三角形分割フラグ
	 */
	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, rxMTL &mats, bool triangle = true);

	/*!
	 * OBJファイル書き込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[in] vrts 頂点座標
	 * @param[in] vnms 頂点法線
	 * @param[in] plys ポリゴン
	 * @param[in] mats 材質情報
	 */
	bool Save(string file_name, const vector<Vec3> &vrts, const vector<Vec3> &vnms, const vector<rxFace> &plys, const rxMTL &mats);

	//! 材質リストの取得
	rxMTL GetMaterials(void){ return m_mapMaterials; }

private:
	int loadFace(string &buf, vector<int> &vidxs, vector<int> &nidxs, vector<int> &tidxs);
	int loadMTL(const string &mtl_fn);
	int saveMTL(const string &mtl_fn, const rxMTL &mats);
};



#endif // _RX_OBJ_H_
