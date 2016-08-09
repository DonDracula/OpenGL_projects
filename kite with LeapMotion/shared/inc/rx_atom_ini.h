/*! @file rx_atom_ini.h
	
	@brief 設定ファイル
 
	@author Makoto Fujisawa
	@date 2009-04
*/

#ifndef _RX_ATOM_INI_H_
#define _RX_ATOM_INI_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <string>
#include <map>
#include <algorithm>

#ifdef RX_USE_BOOST
//#include <boost/typeof/typeof.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/function.hpp>
#endif

using namespace std;


//-----------------------------------------------------------------------------
// 設定ファイル項目情報
//-----------------------------------------------------------------------------
struct rxINIPair
{
	string Header;		//!< ヘッダ([Header])
	string Name, Value;	//!< 名前と値(Name=Value)
	string Type;
	void* Var;
};

#define RX_INIFOR(x, y) for(vector<rxINIPair>::iterator x = y.begin(); x < y.end(); ++x)

/*!
 * 様々な型のstringへの変換(stringstreamを使用)
 * @param[in] x 入力
 * @return string型に変換したもの
 */
template<typename T>
inline std::string RX_TO_STRING_INI(const T &x)
{
	stringstream ss;
	ss << x;
	return ss.str();
}

//-----------------------------------------------------------------------------
// クラスの宣言
//-----------------------------------------------------------------------------
class rxINI
{
#ifdef RX_USE_BOOST
	typedef boost::function<void (string*, string*, int, string)> RX_HEADER_FUNC;
#else
	typedef void (*RX_HEADER_FUNC)(string*, string*, int, string, void *x);

	struct FuncPointer
	{
		RX_HEADER_FUNC Func;
		void *Pointer;
	};
#endif
	
protected:
	vector<rxINIPair> m_vContainer;
	string m_strPath;

	//! ヘッダ名と関数を関連づけるマップ
#ifdef RX_USE_BOOST
	map<string, RX_HEADER_FUNC> m_mapHeaderFunc;
#else
	map<string, FuncPointer> m_mapHeaderFunc;
#endif

public:
	//! デフォルトコンストラクタ
	rxINI(){}

	/*!
	 * コンストラクタ
	 * @param[in] path 設定ファイルパス
	 */
	rxINI(const string &path)
	{
		Load(path);
	}

	//! デストラクタ
	~rxINI(){}

	/*!
	 * INI項目の作成
	 * @param[in] header 項目ヘッダ
	 * @param[in] name 項目名
	 * @param[in] value 項目の値
	 * @return INI項目
	 */
	rxINIPair CreatePair(const string& header = "", const string& name = "", const string& value = "")
	{
		rxINIPair p = {header, name, value, "", NULL};
		return p;
	}

	/*!
	 * 項目の登録
	 *  - 各項目と変数が一対一で対応する
	 * @param[in] header ヘッダ名
	 * @param[in] name 項目名
	 * @param[in] var 項目に対応する値(変数)
	 * @param[in] def 値のデフォルト値(項目が見つからなかったら代入される)
	 */
	template<class T> 
	void Set(const string& header, const string& name, T* var, const T def)
	{
		string type = typeid(T).name();
		string value = RX_TO_STRING_INI(*var);

		RX_INIFOR(i, m_vContainer){
			if(i->Name == name && i->Header == header){
				cout << name << "had been registered." << endl;
				return;
			}
		}

		*var = def;

		rxINIPair pr = {header, name, value, type, (void*)var};
		m_vContainer.push_back(pr);
	}

	/*!
	 * ヘッダ関数の登録
	 *  - ヘッダが見つかったらヘッダ内の項目をすべて読み込み，これを引数として関数を呼び出す
	 * @param[in] header ヘッダ名
	 * @param[in] hfunc ヘッダ関数ポインタ
	 * @param[in] x 関数呼び出し用ポインタ(boost::function使用時は必要なし)
	 */
	// 項目の登録
	void SetHeaderFunc(string header, RX_HEADER_FUNC hfunc, void* x = 0)
	{
		string::size_type breakpoint = 0;
		while((breakpoint = header.find(" ")) != string::size_type(-1)){
			header.erase(breakpoint, 1);
		}
#ifdef RX_USE_BOOST
		m_mapHeaderFunc[header] = hfunc;
#else
		FuncPointer fp;
		fp.Func = hfunc;
		fp.Pointer = x;
		m_mapHeaderFunc[header] = fp;
#endif

	}


	// 設定ファイル項目のstring valueを更新
	int UpdateValues(void);

	// 名前から値を取得
	string GetValueByName(const string& name);

	// 値から名前を取得
	string GetNameByValue(const string& value);

	// 名前からヘッダを取得
	string GetHeaderByName(const string& name);

	// 値からヘッダを取得
	string GetHeaderByValue(const string& value);

	// 設定ファイル読み込み・書き込み
	int Load(const string& path);
	int Save(const string& path = "");

	// 項目リストを指定して設定保存
	int SaveList(vector<rxINIPair> *pairs, const string& path = "");
	int SaveList(rxINIPair *pairs, unsigned int size, const string& path = "");

	/*!
	 * 文字列小文字化
	 * @param[inout] str 文字列
	 */
	static inline void StringToLower(string &str)
	{
		string::size_type i, size;
 
		size = str.size();
 
		for(i = 0; i < size; i++){
			if(str[i] >= 'A' && str[i] <= 'Z') str[i] += 32;
		}
 
		return;
	}
};



/*!
 * 名前から値を取得
 * @param[in] name 項目名
 * @return 項目の値(見つからなかったら空string""を返す)
 */
inline string rxINI::GetValueByName(const string& name)
{
	for(vector<rxINIPair>::iterator i = m_vContainer.begin(); i < m_vContainer.end(); ++i)
		if(i->Name == name)
			return i->Value;

	// This value was not found.
	return "";
}

/*!
 * 値から名前を取得
 * @param[in] value 項目の値
 * @return 項目名(見つからなかったら空string""を返す)
 */
inline string rxINI::GetNameByValue(const string& value)
{
	for(vector<rxINIPair>::iterator i = m_vContainer.begin(); i < m_vContainer.end(); ++i)
		if(i->Value == value)
			return i->Name;

	return "";
}

/*!
 * 名前からヘッダを取得
 * @param[in] name 項目名
 * @return ヘッダ名(見つからなかったら空string""を返す)
 */
inline string rxINI::GetHeaderByName(const string& name)
{
	for(vector<rxINIPair>::iterator i = m_vContainer.begin(); i < m_vContainer.end(); ++i)
		if(i->Name == name)
			return i->Header;

	return "";
}

/*!
 * 値からヘッダを取得
 * @param[in] value 項目の値
 * @return ヘッダ名(見つからなかったら空string""を返す)
 */
inline string rxINI::GetHeaderByValue(const string& value)
{
	for(vector<rxINIPair>::iterator i = m_vContainer.begin(); i < m_vContainer.end(); ++i)
		if(i->Value == value)
			return i->Header;

	return "";
}


/*!
 * 設定ファイルからデータ取得
 * @param[in] path 設定ファイルパス
 * @return 
 */
inline int rxINI::Load(const string& path)
{
	ifstream in(path.c_str(), ios::in);

	if(!in || !in.is_open() || in.bad() || in.fail()){
		cout << "[rxINI::Load] Invalid or corrupted file specified" << endl;
		return 0;
	}

	m_strPath = path;

	string buf;
	string cur_header;
	string name = "", value = "";

	string::size_type header_end = 0;
	string::size_type comment_start = 0;
	string::size_type breakpoint = 0;
	string::size_type equal_sign_pos = 0;

	bool use_header_func = false;
	vector<string> names, values;
	while(!in.eof()){
		getline(in, buf);

		// ';'以降はコメントとして無視
		if( (comment_start = buf.find(';')) != string::size_type(-1) )
			buf = buf.substr(0, comment_start);

		// '#'以降はコメントとして無視
		if( (comment_start = buf.find('#')) != string::size_type(-1) )
			buf = buf.substr(0, comment_start);

		// '//'以降はコメントとして無視
		if( (comment_start = buf.find("//")) != string::size_type(-1) )
			buf = buf.substr(0, comment_start);

		while((breakpoint = buf.find("\t")) != string::size_type(-1)){
			buf.erase(breakpoint, 1);
		}
		while((breakpoint = buf.find(" ")) != string::size_type(-1)){
			buf.erase(breakpoint, 1);
		}

		if(buf.empty())
			continue;

		// ヘッダ行のチェック
		if( buf.at(0) == '[' ){
			if(use_header_func && !names.empty()){
#ifdef RX_USE_BOOST
				m_mapHeaderFunc[cur_header](&names[0], &values[0], (int)names.size(), cur_header);
#else
				m_mapHeaderFunc[cur_header].Func(&names[0], &values[0], (int)names.size(), cur_header, m_mapHeaderFunc[cur_header].Pointer);
#endif
			}

			if( (header_end = buf.find(']')) == string::size_type(-1) ){
				cout << "[rxINI::Load] Header tag opened with '[' but not closed with ']'" << endl;
				return 0;
			}

			cur_header = buf.substr(1, header_end-1);

			if(!m_mapHeaderFunc.empty()){
#ifdef RX_USE_BOOST
				map<string, RX_HEADER_FUNC>::iterator i = m_mapHeaderFunc.find(cur_header);
#else
				map<string, FuncPointer>::iterator i = m_mapHeaderFunc.find(cur_header);
#endif
				if(i == m_mapHeaderFunc.end()){
					use_header_func = false;
				}
				else{
					use_header_func = true;
					names.clear();
					values.clear();
				}
			}
		
			continue;
		}

		// 項目行
		if( (equal_sign_pos = buf.find('=')) == string::size_type(-1) ){
			cout << "[rxINI::Load] Value '" << buf;
			cout << "' is not a header and not a valid value, missing '='." << endl;
			return 0;
		}

		name = buf.substr(0, equal_sign_pos);
		value = buf.substr(equal_sign_pos + 1);

		if(use_header_func){
			names.push_back(name);
			values.push_back(value);
		}

		bool preset = false;
		RX_INIFOR(i, m_vContainer){
			if(i->Name == name){
				i->Value = value;
				if(i->Var != NULL){
					if(i->Type == typeid(float).name()){
#ifdef RX_USE_BOOST
						*((float*)i->Var) = boost::lexical_cast<float>(value);
#else
						*((float*)i->Var) = (float)atof(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (float : " << *((float*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(double).name()){
#ifdef RX_USE_BOOST
						*((double*)i->Var) = boost::lexical_cast<double>(value);
#else
						*((double*)i->Var) = (double)atof(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (double : " << *((double*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(double*).name()){
#ifdef RX_USE_BOOST
						//*((double**)i->Var) = boost::lexical_cast<double>(value);
#else
						//*((double**)i->Var) = (double)atof(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (double* : " << (*((double**)i->Var))[0] << ")" << endl;
					}
					else if(i->Type == typeid(long double).name()){
#ifdef RX_USE_BOOST
						*((long double*)i->Var) = boost::lexical_cast<long double>(value);
#else
						*((long double*)i->Var) = (long double)atof(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (long double : " << *((long double*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(int).name()){
#ifdef RX_USE_BOOST
						*((int*)i->Var) = boost::lexical_cast<int>(value);
#else
						*((int*)i->Var) = (int)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (int : " << *((int*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(unsigned int).name()){
#ifdef RX_USE_BOOST
						*((unsigned int*)i->Var) = boost::lexical_cast<unsigned int>(value);
#else
						*((unsigned int*)i->Var) = (unsigned int)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (unsigned int : " << *((unsigned int*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(short).name()){
#ifdef RX_USE_BOOST
						*((short*)i->Var) = boost::lexical_cast<short>(value);
#else
						*((short*)i->Var) = (short)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (short : " << *((short*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(unsigned short).name()){
#ifdef RX_USE_BOOST
						*((unsigned short*)i->Var) = boost::lexical_cast<unsigned short>(value);
#else
						*((unsigned short*)i->Var) = (unsigned short)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (unsigned short : " << *((unsigned short*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(long).name()){
#ifdef RX_USE_BOOST
						*((long*)i->Var) = boost::lexical_cast<long>(value);
#else
						*((long*)i->Var) = (long)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (long : " << *((long*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(unsigned long).name()){
#ifdef RX_USE_BOOST
						*((unsigned long*)i->Var) = boost::lexical_cast<unsigned long>(value);
#else
						*((unsigned long*)i->Var) = (unsigned long)atoi(value.c_str());
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (unsigned long : " << *((unsigned long*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(string).name()){
#ifdef RX_USE_BOOST
						*((string*)i->Var) = boost::lexical_cast<string>(value);
#else
						*((string*)i->Var) = value;
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (string : " << *((string*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(char).name()){
#ifdef RX_USE_BOOST
						*((char*)i->Var) = boost::lexical_cast<char>(value);
#else
						*((char*)i->Var) = value[0];
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (char : " << *((char*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(unsigned char).name()){
#ifdef RX_USE_BOOST
						*((unsigned char*)i->Var) = boost::lexical_cast<unsigned char>(value);
#else
						*((unsigned char*)i->Var) = (unsigned char)value.c_str()[0];
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (unsigned char : " << *((unsigned char*)i->Var) << ")" << endl;
					}
					else if(i->Type == typeid(bool).name()){
#ifdef RX_USE_BOOST
						*((bool*)i->Var) = boost::lexical_cast<bool>(value);
#else
						StringToLower(value);
						*((bool*)i->Var) = (value == "true" ? true : false);
#endif
						cout << "[setting] " << name << " : " << value;
						cout << " (bool : " << *((bool*)i->Var) << ")" << endl;
					}
				}
				preset = true;
				break;
			}
		}

		if(!preset) m_vContainer.push_back(CreatePair(cur_header, name, value));
	}

	return 1;
}

/*!
 * 設定ファイル項目のstring valueを更新
 */
inline int rxINI::UpdateValues(void)
{
	RX_INIFOR(i, m_vContainer){
		if(i->Var != NULL){
			if(i->Type == typeid(float).name()){
				i->Value = RX_TO_STRING_INI(*((float*)i->Var));
			}
			else if(i->Type == typeid(double).name()){
				i->Value = RX_TO_STRING_INI(*((double*)i->Var));
			}
			else if(i->Type == typeid(long double).name()){
				i->Value = RX_TO_STRING_INI(*((long double*)i->Var));
			}
			else if(i->Type == typeid(int).name()){
				i->Value = RX_TO_STRING_INI(*((int*)i->Var));
			}
			else if(i->Type == typeid(unsigned int).name()){
				i->Value = RX_TO_STRING_INI(*((unsigned int*)i->Var));
			}
			else if(i->Type == typeid(short).name()){
				i->Value = RX_TO_STRING_INI(*((short*)i->Var));
			}
			else if(i->Type == typeid(unsigned short).name()){
				i->Value = RX_TO_STRING_INI(*((unsigned short*)i->Var));
			}
			else if(i->Type == typeid(long).name()){
				i->Value = RX_TO_STRING_INI(*((long*)i->Var));
			}
			else if(i->Type == typeid(unsigned long).name()){
				i->Value = RX_TO_STRING_INI(*((unsigned long*)i->Var));
			}
			else if(i->Type == typeid(string).name()){
				i->Value = RX_TO_STRING_INI(*((string*)i->Var));
			}
			else if(i->Type == typeid(char).name()){
				i->Value = RX_TO_STRING_INI(*((char*)i->Var));
			}
			else if(i->Type == typeid(unsigned char).name()){
				i->Value = RX_TO_STRING_INI(*((unsigned char*)i->Var));
			}
			else if(i->Type == typeid(bool).name()){
				i->Value = RX_TO_STRING_INI(*((bool*)i->Var));
			}
		}
	}

	return 1;
}

/*!
 * 設定保存
 * @param[in] path  設定ファイルパス(""なら入力を使用)
 * @return -1でエラー
 */
inline int rxINI::Save(const string& path)
{
	return SaveList(&m_vContainer, path);
}

/*!
 * 項目リストを指定して設定保存
 * @param[in] pairs 項目リスト
 * @param[in] path  設定ファイルパス(""なら入力を使用)
 * @return -1でエラー
 */
inline int rxINI::SaveList(vector<rxINIPair> *pairs, const string& path)
{
	if(!pairs){
		cout << "[rxINI::Save] Null-pointer specified for 'pairs' argument." << endl;
		return 0;
	}

	string opath = path;
	if(path == "") opath = m_strPath;

	cout << "saving the settings to " << opath << "..." << endl;

	ofstream out(opath.c_str(), ios::out);
	if(!out || !out.is_open() || out.bad() || out.fail()){
		cout << "[rxINI::Save] Cannot save file at path " << opath << endl;
		return 0;
	}

	string current_header = "";

	UpdateValues();
	for(vector<rxINIPair>::iterator i = pairs->begin(); i < pairs->end(); ++i){
		if(i->Header != current_header){
			out << '[' << i->Header << ']' << '\n';
			current_header = i->Header;
		}

		// Write value
		out << i->Name << '=' << i->Value << '\n';
	}

	return 1;
}

/*!
 * 項目リストを指定して設定保存
 * @param[in] pairs 項目リスト
 * @param[in] size  項目リストのサイズ
 * @param[in] path  設定ファイルパス(""なら入力を使用)
 * @return -1でエラー
 */
inline int rxINI::SaveList(rxINIPair *pairs, unsigned int size, const string& path)
{
	vector<rxINIPair> vpairs(pairs, pairs+size);
	return SaveList(&vpairs, path);
}

#endif //_RX_ATOM_INI_H_
