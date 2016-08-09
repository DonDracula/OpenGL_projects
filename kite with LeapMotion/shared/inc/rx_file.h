/*! @file rx_file.h
	
	@brief ファイル操作(boost使用)
 
	@author Makoto Fujisawa
	@date 2009-02
*/

#ifndef _RX_COMMON_FILE_H_
#define _RX_COMMON_FILE_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <vector>
#include <string>

#include "boost/filesystem/path.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/fstream.hpp"

#include "boost/bind.hpp"
#include "boost/function.hpp"


//-----------------------------------------------------------------------------
// MARK:ファイル・フォルダ処理
//-----------------------------------------------------------------------------
namespace RXFile
{
	typedef boost::filesystem::path fspath;

	/*!
	 * 再帰的に全ファイルを取り出す
	 * @param[in] fpath フォルダパス
	 * @param[out] paths 見つかったファイル一覧
	 */
	static void Search(const fspath &dpath, std::vector<std::string> &paths)
	{
		// カレントディレクトリのファイル一覧
		boost::filesystem::directory_iterator end; 
		for(boost::filesystem::directory_iterator it(dpath); it != end; ++it){
			if(boost::filesystem::is_directory(*it)){
				Search(it->path(), paths);
			}
			else{
				paths.push_back(it->path().generic_string());	// nativeフォーマットの場合は単にstring()
			}
		} 
	}
	static void Search(const std::string &dir, std::vector<std::string> &paths)
	{
		fspath dpath(dir);
		if(boost::filesystem::is_directory(dpath)){
			Search(dpath, paths);
		}
	}

	/*!
	 * 再帰的に全ファイルを取り出す
	 * @param[in] fpath フォルダパス
	 * @param[out] paths 見つかったファイル一覧
	 * @param[in] fpComp 検索条件
	 */
	static void Search(const fspath &dpath, 
					   std::vector<std::string> &paths, 
					   boost::function<bool (std::string)> fpComp)
	{
		// カレントディレクトリのファイル一覧
		boost::filesystem::directory_iterator end; 
		for(boost::filesystem::directory_iterator it(dpath); it!=end; ++it){
			if(boost::filesystem::is_directory(*it)){
				Search(it->path(), paths);
			}
			else{
				std::string fpath = it->path().generic_string();
				if(fpComp(fpath)){
					paths.push_back(fpath);
				}
			}
		}
	}
	static void Search(const std::string &dir, 
		               std::vector<std::string> &paths, 
					   boost::function<bool (std::string)> fpComp)
	{
		fspath dpath(dir);
		if(boost::filesystem::is_directory(dpath)){
			Search(dpath, paths, fpComp);
		}
	}

	/*!
	 * ファイル名比較関数(拡張子)
	 * @param[in] fn 比較したいファイル名
	 * @param[in] ext 拡張子
	 * @return fnの拡張子がextと同じならtrue
	 */
	inline bool SearchCompExt(std::string fn, std::string ext)
	{
		return (fn.find(ext, 0) != std::string::npos);
	}

	/*!
	 * ファイル名比較関数(複数拡張子)
	 * @param[in] fn 比較したいファイル名
	 * @param[in] ext 拡張子
	 * @return fnの拡張子がextと同じならtrue
	 */
	inline bool SearchCompExts(std::string fn, vector<std::string> exts)
	{
		vector<std::string>::iterator i;
		for(i = exts.begin(); i != exts.end(); ++i){
			if(fn.find(*i, 0) != std::string::npos) break;
		}

		return (i != exts.end());
	}

	/*!
	 * 再帰的に全ファイルを取り出す(子フォルダの階層最大値指定)
	 * @param[in] fpath フォルダパス
	 * @param[out] paths 見つかったファイル一覧
	 * @param[inout] d 現在の階層数
	 * @param[in] n 最大階層数
	 */
	static void Search(const fspath &dpath, std::vector<std::string> &paths, int d, const int n)
	{
		// カレントディレクトリのファイル一覧
		boost::filesystem::directory_iterator end; 
		for(boost::filesystem::directory_iterator it(dpath); it!=end; ++it){
			if(boost::filesystem::is_directory(*it) && (n == -1 || d < n)){
				Search(it->path(), paths, d+1, n);
			}
			else{
				paths.push_back(it->path().generic_string());
			}
		}
	}

	/*!
	 * 再帰的に全ファイルを取り出す
	 * @param[in] fpath フォルダパス
	 * @param[out] paths 見つかったファイル一覧
	 * @param[inout] d 現在の階層数
	 * @param[in] n 最大階層数
	 * @param[in] fpComp 検索条件
	 */
	static void Search(const fspath &dpath, std::vector<std::string> &paths, int d, const int n, 
					   boost::function<bool (std::string)> fpComp)
	{
		// カレントディレクトリのファイル一覧
		boost::filesystem::directory_iterator end; 
		for(boost::filesystem::directory_iterator it(dpath); it!=end; ++it){
			if(boost::filesystem::is_directory(*it) && (n == -1 || d < n)){
				Search(it->path(), paths, d+1, n, fpComp);
			}
			else{
				std::string fpath = it->path().generic_string();
				if(fpComp(fpath)){
					paths.push_back(fpath);
				}
			}
		}
	}

	static void Search(const std::string &dir, std::vector<std::string> &paths, const int n)
	{
		fspath dir_path(dir);
		if(boost::filesystem::is_directory(dir_path)){
			int d = 0;
			Search(dir_path, paths, d, n);
		}
	}

	static void Search(const std::string &dir, std::vector<std::string> &paths, std::string ext, const int n)
	{
		fspath dir_path(dir);
		if(boost::filesystem::is_directory(dir_path)){
			boost::function<bool (std::string)> fpComp;
			fpComp = boost::bind(SearchCompExt, _1, ext);

			int d = 0;
			Search(dir_path, paths, d, n, fpComp);
		}
	}

	static void Search(const std::string &dir, std::vector<std::string> &paths, 
					   boost::function<bool (std::string)> fpComp, const int n)
	{
		fspath dir_path(dir);
		if(boost::filesystem::is_directory(dir_path)){
			int d = 0;
			Search(dir_path, paths, d, n, fpComp);
		}
	}

	static void Search(const std::string &dir, std::vector<std::string> &paths, vector<std::string> exts, const int n)
	{
		fspath dir_path(dir);
		if(boost::filesystem::is_directory(dir_path)){
			boost::function<bool (std::string)> fpComp;
			fpComp = boost::bind(SearchCompExts, _1, exts);

			int d = 0;
			Search(dir_path, paths, d, n, fpComp);
		}
	}


	/*!
	 * ファイル/フォルダの作成日時確認
	 * @param[in] path ファイルパス
	 * @return 作成日時，存在しなければ-1を返す
	 */
	inline long WriteTime(const std::string &path)
	{
		fspath file_path(path);
		if(boost::filesystem::exists(file_path)){
			long t = (long)boost::filesystem::last_write_time(file_path);
			return t;
		}
		else{
			return -1;
		}
	}

	/*!
	 * ファイルのサイズ確認
	 * @param[in] path ファイルパス
	 * @return サイズ(バイト)，存在しなければ-1を返す
	 */
	inline long Size(const std::string &path)
	{
		fspath file_path(path);
		if(boost::filesystem::exists(file_path)){
			long s = (long)boost::filesystem::file_size(file_path);
			return s;
		}
		else{
			return -1;
		}
	}

	/*!
	 * ファイル/フォルダ削除(ファイルと空フォルダのみ)
	 * @param[in] path ファイル・フォルダパス
	 */
	inline void Remove(const std::string &path)
	{
		fspath file_path(path);
		boost::filesystem::remove(file_path);
	}

	/*!
	 * ファイル/フォルダ削除(すべて)
	 * @param[in] path ファイル・フォルダパス
	 * @return 削除ファイル数
	 */
	inline unsigned long RemoveAll(const std::string &path)
	{
		fspath file_path(path);
		return (unsigned long)boost::filesystem::remove_all(file_path);
	}

	/*!
	 * ファイル/フォルダの存在確認
	 * @param[in] path ファイル・フォルダパス
	 * @return 存在するかどうか
	 */
	inline bool Exist(const std::string &path)
	{
		fspath file_path(path);
		return boost::filesystem::exists(file_path);
		//FILE *fp;
		//if( (fp = fopen(path.c_str(), "r")) == NULL ){
		//	return false;
		//}
		//fclose(fp);
		//return true;
	}

	/*!
	 * リネーム／移動
	 *  - 複数ルートのあるファイルシステム(Windows含む)では異なるルート間の移動は不可
	 *  - リネーム/移動先に同名ファイルがあったらキャンセルする
	 * @param[in] path ファイル・フォルダパス
	 * @return 
	 */
	inline bool Rename(const std::string &from, const std::string &to)
	{
		fspath fp0(from);
		fspath fp1(to);

		if(!boost::filesystem::exists(fp0) || boost::filesystem::exists(fp1)){
			return false;
		}

		boost::filesystem::rename(fp0, fp1);

		return true;
	}

	/*!
	 * ファイルコピー(フォルダ不可)
	 *  - コピー先に同名ファイルがあったらキャンセルする
	 * @param[in] path ファイルパス
	 * @return 
	 */
	inline bool Copy(const std::string &from, const std::string &to)
	{
		fspath fp0(from);
		fspath fp1(to);

		if(!boost::filesystem::exists(fp0) || boost::filesystem::exists(fp1)){
			return false;
		}

		boost::filesystem::copy_file(fp0, fp1);

		return true;
	}

	/*!
	 * 空テキスト作成
	 *  - フォルダがなければ作成
	 * @param[in] path ファイルパス
	 */
	inline bool ZeroText(const std::string &path)
	{
		fspath file_path(path);

		// フォルダパスの抽出
		fspath dir_path = file_path.branch_path();

		// フォルダ存在確認
		if(!boost::filesystem::exists(dir_path)){
			// フォルダ作成
			boost::filesystem::create_directory(dir_path);

			if(!boost::filesystem::exists(dir_path)){
				return false;
			}
		}

		FILE *fp = fopen(path.c_str(), "r");
		if(fp){
			fclose(fp);
			return false;
		}
		else{
			fp = fopen(path.c_str(), "w");
			if(fp){
				fclose(fp);
				return true;
			}
			else{
				return false;
			}
		}
	}

	/*!
	 * ファイルパスからファイル名のみ抽出
	 * @param[in] path ファイルパス
	 */
	inline std::string FileName(const std::string &path)
	{
		fspath file_path(path);
		return file_path.filename().generic_string();
	}

	/*!
	 * ファイルパスからファイル名をのぞいたものを抽出
	 * @param[in] path ファイルパス
	 */
	inline std::string DirName(const std::string &path)
	{
		fspath file_path(path);
		return file_path.branch_path().string();
	}

	/*!
	 * ファイルパスから親フォルダ名を抽出
	 * @param[in] path ファイルパス
	 */
	inline std::string ParentDirName(const std::string &path)
	{
		std::string::size_type pos1, pos0;
		pos1 = path.find_last_of("\\/");
		pos0 = path.find_last_of("\\/", pos1-1);

		if(pos0 != std::string::npos && pos1 != std::string::npos){
			return path.substr(pos0+1, pos1-pos0-1);
		}
		else{
			return "";
		}
	}

	/*!
	 * フォルダ区切りの検索
	 * @param[in] str ファイル・フォルダパス
	 * @param[out] pos 見つかった位置
	 */
	inline bool FindPathBound(const string &str, std::string::size_type &pos)
	{
		std::string::size_type pos0, pos1;
		pos0 = str.find_last_of("\\");
		pos1 = str.find_last_of("/");

		if(pos0 == std::string::npos){
			if(pos1 == std::string::npos){
				return false;
			}
			else{
				pos = pos1;
			}
		}
		else{
			if(pos1 == std::string::npos){
				pos = pos0;
			}
			else{
				pos = (pos0 < pos1) ? pos0 : pos1;
			}
		}

		return true;
	}

	/*!
	 * フォルダの存在確認と作成
	 * @param[in] path_str ファイル・フォルダパス
	 */
	inline bool CheckAndMakeDir(const std::string &path_str)
	{
		using namespace boost::filesystem;

		if(!exists(path_str)){
			vector<string> tmp_paths;
			tmp_paths.push_back(path_str);

			// パスを分解
			for(;;){
				std::string::size_type pos;
				if(!FindPathBound(tmp_paths.back(), pos)){
					break;
				}
				else{
					string str = tmp_paths.back().substr(0, pos);
					tmp_paths.push_back(str);
				}
			}

			// 分解したパスそれぞれ存在チェックし，なければ作成
			vector<std::string>::iterator itr = tmp_paths.end()-1;
			for( ; itr != tmp_paths.begin(); --itr){
				//cout << *itr << endl;
				if(!exists(*itr)){
					create_directory(*itr);
				}
			}
		}

		return true;
	}

	/*!
	 * ファイルパスから拡張子抽出(.(ドット)付)
	 * @param[in] str ファイルパス
	 * @return 拡張子(.(ドット)付)
	 */
	inline string GetExtensionWithDot(const std::string &str)
	{
		std::string::size_type pos = str.find_last_of(".");
		return str.substr(pos, str.size());
	}

	/*!
	 * ファイルパスから拡張子抽出
	 * @param[in] str ファイルパス
	 * @return 拡張子
	 */
	inline string GetExtension(const std::string &str)
	{
		std::string::size_type pos = str.find_last_of(".");
		return str.substr(pos+1, str.size());
	}


};	// namespace RXFile



#endif // #ifndef _RX_COMMON_FILE_H_