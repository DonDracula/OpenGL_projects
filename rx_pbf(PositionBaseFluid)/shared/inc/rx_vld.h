/*!
  @file rx_vld.h
	
  @brief ボリュームデータの入出力
 
  @author Makoto Fujisawa
  @date   2013-10
*/

#ifndef _RX_VLD_H_
#define _RX_VLD_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
 
#include <cstdlib>

//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;


//-----------------------------------------------------------------------------
// rxVLDクラスの宣言 - rxVLD形式の読み込み
//-----------------------------------------------------------------------------
template<class T>
class rxVLD
{
	// メンバ関数
	inline int ix(const int &i, const int &j, const int &k, const int &nx, const int &ny)
	{
		return (i+nx*(j+ny*k));
	}

public:
	//! コンストラクタ
	rxVLD(){}
	//! デストラクタ
	~rxVLD(){}

	/*!
	 * VLDファイル読み込み
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[out] nx,ny,nz セル数
	 * @return ボリュームデータ
	 */
	T* Read(string file_name, int &nx, int &ny, int &nz);

	/*!
	 * VLDファイル書き込み(未実装)
	 * @param[in] file_name ファイル名(フルパス)
	 * @param[in] nx,ny,nz セル数
	 * @param[in] ボリュームデータ
	 */
	bool Save(string file_name, int nx, int ny, int nz, T* data);
};

template<class T> 
T* rxVLD<T>::Read(string file_name, int &nx, int &ny, int &nz)
{
	// ファイルをバイナリモードで開く
	ifstream fin;
	fin.open(file_name.c_str(), ios::in|ios::binary);
	if(!fin){
		cout << file_name << " could not open!" << endl;
		return 0;
	}

	// セル数の読み込み
	fin.read((char*)&nx, sizeof(int));
	fin.read((char*)&ny, sizeof(int));
	fin.read((char*)&nz, sizeof(int));

	if(nx == 0 || ny == 0 || nz == 0) return 0;

	T *data = new T[nx*ny*nz];

	// ボリュームデータの読み込み
	for(int k = 0; k < nz; ++k){
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				int idx = ix(i, j, k, nx, ny);
				fin.read((char*)&data[idx], sizeof(T));
			}
		}
	}

	fin.close();

	return data;
}

template<class T> 
bool rxVLD<T>::Save(string file_name, int nx, int ny, int nz, T* data)
{
	// ファイルをバイナリモードで開く
	ofstream fout;
	fout.open(file_name.c_str(), ios::out|ios::binary);
	if(!fout){
		cout << file_name << " could not open!" << endl;
		return false;
	}

	// セル数の書き込み
	fout.write((char*)&nx, sizeof(int));
	fout.write((char*)&ny, sizeof(int));
	fout.write((char*)&nz, sizeof(int));

	// ボリュームデータの書き込み
	for(int k = 0; k < nz; ++k){
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				int idx = ix(i, j, k, nx, ny);
				fout.write((char*)&data[idx], sizeof(T));
			}
		}
	}

	fout.close();

	return true;
}



#endif // _RX_VLD_H_
