/*!
  @file rx_off.h
	
  @brief OFF File Input
 
  @author Makoto Fujisawa
  @date   2012
*/

#ifndef _RX_OFF_H_
#define _RX_OFF_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include "rx_mesh.h"

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxOFFクラスの宣言 - OFF形式の読み込み
//-----------------------------------------------------------------------------
class rxOFF
{
public:
	//! コンストラクタ
	rxOFF();
	//! デストラクタ
	~rxOFF();

	/*!
	 * OFFファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] vrts 頂点座標
	 * @param[out] vnms 頂点法線
	 * @param[out] poly ポリゴン
	 * @param[in] triangle ポリゴンの三角形分割フラグ
	 */
	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, bool triangle = false);

private:
	int loadFace(string &buf, vector<int> &vidxs, vector<int> &nidxs, vector<int> &tidxs);
};



#endif // _RX_OBJ_H_
