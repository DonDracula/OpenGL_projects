/*!
  @file rx_stl.h
	
  @brief STL File Input/Output
 
  @author Makoto Fujisawa
  @date   2011
*/

#ifndef _RX_STL_H_
#define _RX_STL_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include "rx_mesh.h"

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxSTLクラスの宣言 - STL形式の読み込み
//-----------------------------------------------------------------------------
class rxSTL
{
public:
	//! コンストラクタ
	rxSTL();
	//! デストラクタ
	~rxSTL();

	/*!
	 * STLファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] vrts 頂点座標
	 * @param[out] vnms 頂点法線
	 * @param[out] poly ポリゴン
	 * @param[in] vertex_integration 頂点統合フラグ
	 * @param[in] vertex_normal 頂点法線計算フラグ
	 */
	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, 
			  bool vertex_integration = true, bool vertex_normal = true);

	/*!
	 * STLファイル書き込み(未実装)
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[in] vrts 頂点座標
	 * @param[in] plys ポリゴン
	 * @param[in] binary バイナリフォーマットでの保存フラグ
	 */
	bool Save(string file_name, const vector<Vec3> &vrts, const vector<rxFace> &plys, 
			  bool binary = false);

protected:
	//! ASCIIフォーマットのSTLファイル読み込み
	bool readAsciiData(ifstream &file, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys);

	//! バイナリフォーマットのSTLファイル読み込み
	bool readBinaryData(ifstream &file, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys);

	//! ASCIIフォーマットのSTLファイル読み込み
	bool saveAsciiData(ofstream &file, const vector<Vec3> &vrts, const vector<rxFace> &plys);

	//! バイナリフォーマットのSTLファイル読み込み
	bool saveBinaryData(ofstream &file, const vector<Vec3> &vrts, const vector<rxFace> &plys);

};



#endif // _RX_OBJ_H_
