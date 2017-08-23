/*! 
  @file rx_timer.h
	
  @brief 時間計測
 
  @author Makoto Fujisawa
  @date 2012-02
*/
// FILE -- rx_timer.h --

#ifndef _RX_TIMER_H_
#define _RX_TIMER_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <iostream>
#include <sstream>

#include <vector>
#include <map>
#include <string>

#define RX_USE_QPC	// Windowsでの時間計測にQueryPerformanceCounterを用いる
//#define RX_USE_MM
#ifdef WIN32
	#include <windows.h>
 
	#ifdef RX_USE_MM
	#include <mmsystem.h>
	#pragma comment (lib, "winmm.lib")
	#endif
#else
	#include <ctime>
#endif
 
#ifdef WIN32

#ifdef RX_USE_QPC
	// QueryPerformanceCounterでベースクロックにCPUが使われル場合，周波数が可変のCPU(TurboBoostなど)を使っていると
	// 計測時の負荷状態によって得られる値が変化してしまうので注意．
	// Sleep(1000);などで計測値を環境上であらかじめチェックしておくこと！
	// Core i7 3.2GHz, Win7 x64環境ではQueryPerformanceFrequencyが3124873となったので，CPUクロックではない模様．
	#define RXTIME LONGLONG
	inline RXTIME RX_GET_TIME(void)
	{
		LARGE_INTEGER t;
		QueryPerformanceCounter((LARGE_INTEGER*)&t);
		return t.QuadPart;
	}
	inline double RX_GET_TIME2SEC(void)
	{
		LARGE_INTEGER f;
		QueryPerformanceFrequency((LARGE_INTEGER*)&f);
		return 1.0/(double)(f.QuadPart);
	}
#else
	#ifdef RX_USE_MM
		#define RXTIME DWORD
		inline RXTIME RX_GET_TIME(void){ return timeGetTime(); }
		inline double RX_GET_TIME2SEC(void){ return 1.0e-3; }
	#else
		#define RXTIME DWORD
		inline RXTIME RX_GET_TIME(void){ return GetTickCount(); }
		inline double RX_GET_TIME2SEC(void){ return 1.0e-3; }
	#endif
#endif

#else
	#define RXTIME clock_t
	inline RXTIME RX_GET_TIME(void){ return clock(); }
	inline double RX_GET_TIME2SEC(void){ return (1.0/CLOCKS_PER_SEC); }
#endif
 

using namespace std;



//-----------------------------------------------------------------------------
// 時間計測クラス
//-----------------------------------------------------------------------------
class rxTimer
{
	RXTIME m_tStart, m_tEnd;
	vector<double> m_vTimes;		//!< 計測された時間
	vector<string> m_vComments;		//!< 各計測データのラベル (Split, Stopの引数で指定)
	double m_fT2S;					//!< 計測された時間単位から秒単位への変換係数
 
public:
	//! コンストラクタ
	rxTimer()
	{
		m_fT2S = RX_GET_TIME2SEC();
	}
 
	//! デストラクタ
	~rxTimer(){}
 
	//! 計測開始
	void Start(void)
	{
		m_tStart = RX_GET_TIME();
	}
 
	//! 計測
	void Split(const string &cmnt = "", bool restart = false)
	{
		m_tEnd = RX_GET_TIME();
 
		double time = (double)(m_tEnd-m_tStart)*m_fT2S;
		m_vTimes.push_back(time);
		m_vComments.push_back(cmnt);
 
		if(restart) m_tStart = RX_GET_TIME();
	}
 
	//! 計測終了
	void Stop(const string &cmnt = "")
	{
		m_tEnd = RX_GET_TIME();
 
		double time = (double)(m_tEnd-m_tStart)*m_fT2S;
		m_vTimes.push_back(time);
		m_vComments.push_back(cmnt);
 
		m_tStart = m_tEnd = 0;
	}
 
	//! リセット
	void Reset(void)
	{
		m_vTimes.clear();
		m_vComments.clear();
		m_tStart = m_tEnd = 0;
	}
 
	// 最後に記録された時間を削除
	void PopBackTime(void)
	{
		m_vTimes.pop_back();
		m_vComments.pop_back();
	}
 
	//! 時間をセット(他の計測方法で計測した結果など)
	void SetTime(const double &t, const string &cmnt = "")
	{
		m_vTimes.push_back(t);
		m_vComments.push_back(cmnt);
	}
 
	//! 時間の取得
	double GetTime(int i)
	{
		if(i >= (int)m_vTimes.size()) return 0.0;
 
		return m_vTimes[i];
	}
 
	//! 記録された時間数の取得
	int GetTimeNum(void)
	{
		return (int)m_vTimes.size();
	}
 
	//! 記録された時間を画面出力
	double Print(void)
	{
		int m = 0, mi;
		if(m_vTimes.empty()){
			return 0.0;
		}
		else{
			// 総計測時間を計算
			double total = 0.0;
			for(int i = 0; i < (int)m_vTimes.size(); ++i){
				mi = (int)m_vComments[i].size();
				if(mi > m) m = mi;
 
				total += m_vTimes[i];
			}
 
			SetTime(total, "total");
		}
 
		int cur_p = cout.precision();
		cout.precision(3);
		cout.setf(ios::fixed);
		for(int i = 0; i < (int)m_vTimes.size(); ++i){
			string spc;
			for(int k = 0; k < m-(int)m_vComments[i].size(); ++k) spc += " ";
			cout << m_vComments[i] << spc << " : " << m_vTimes[i] << endl;
		}
		cout.unsetf(ios::fixed);
		cout.precision(cur_p);
 
		double t = m_vTimes.back();
 
		PopBackTime(); // 格納した合計時間を次の計算に備えて削除
 
		return t;
	}
 
	//! 記録された時間を文字列に出力
	double PrintToString(string &str)
	{
		int m = 0, mi;
		if(m_vTimes.empty()){
			return 0.0;
		}
		else{
			// 総計測時間を計算
			double total = 0.0;
			for(int i = 0; i < (int)m_vTimes.size(); ++i){
				mi = (int)m_vComments[i].size();
				if(mi > m) m = mi;
 
				total += m_vTimes[i];
			}
 
			SetTime(total, "total");
		}
 
		stringstream ss;
		ss.precision(3);
		ss.setf(ios::fixed);
		
		int n = (int)m_vTimes.size();
		for(int i = 0; i < n; ++i){
			string spc;
			for(int k = 0; k < m-(int)m_vComments[i].size(); ++k) spc += " ";
			ss << m_vComments[i] << spc << " : " << m_vTimes[i] << "\n";
		}
 
		ss << "\n";
		str = ss.str();
 
		double t = m_vTimes.back();
 
		PopBackTime(); // 格納した合計時間を次の計算に備えて削除
 
		return t;
	}
};


//! 平均時間計測クラス
class rxTimerAvg
{
public:
	// 時間と計測回数
	struct rxTimeAndCount
	{
		double time;
		int count;
		int idx;
	};

	// 時間と計測回数を文字列と関連づけるマップ
	typedef map<string, rxTimeAndCount> RXMAPTC;

private:
	rxTimer m_Tmr;		//!< 時間計測クラス
	RXMAPTC m_TimeMap;	//!< 時間と計測回数を文字列と関連づけるマップ

public:
	//! コンストラクタ
	rxTimerAvg()
	{
		Clear();
		ResetTime();
		ClearTime();
	}

	/*!
	 * すべてクリア
	 */
	void Clear(void)
	{
		m_TimeMap.clear();
	}

	/*!
	 * 蓄積時間の初期化
	 */
	void ClearTime(void)
	{
		for(RXMAPTC::iterator it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
			it->second.time = 0.0;
			it->second.count = 0;
			//it->second.idx = -1;
		}
	}

	/*!
	 * リセット
	 */
	void ResetTime(void)
	{
		m_Tmr.Reset();
		m_Tmr.Start();
	}

	/*!
	 * 計測
	 * @param[in] cmnt 時間蓄積用の名前
	 */
	void Split(const string &cmnt)
	{
		RXMAPTC::iterator i = m_TimeMap.find(cmnt);
		
		m_Tmr.Stop();
		if(i == m_TimeMap.end()){
			m_TimeMap[cmnt].time = m_Tmr.GetTime(0);
			m_TimeMap[cmnt].count = 1;
			m_TimeMap[cmnt].idx = m_TimeMap.size()-1;
		}
		else{
			m_TimeMap[cmnt].time += m_Tmr.GetTime(0);
			m_TimeMap[cmnt].count++;
		}
		m_Tmr.Reset();
		m_Tmr.Start();
	}

	/*!
	 * 総時間の取得
	 * @return 総時間
	 */
	double GetTotalTime(void)
	{
		if(m_TimeMap.empty()){
			m_Tmr.Stop();
			return m_Tmr.GetTime(0);
		}
		else{
			double total = 0.0;
			for(RXMAPTC::iterator it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
				total += it->second.time/it->second.count;
			}

			return total;
		}
	}

	/*!
	 * 記録された時間を画面出力
	 */
	void Print(void)
	{
		int m = 0, mi;
		double total = 0.0;
		for(RXMAPTC::iterator it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
			mi = (int)it->first.size();
			if(mi > m) m = mi;

			total += it->second.time/it->second.count;
		}

		int cur_p = cout.precision();
		cout.precision(3);
		cout.setf(ios::fixed);
		for(RXMAPTC::iterator it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
			string spc;
			for(int k = 0; k < m-(int)(it->first.size()); ++k) spc += " ";
			cout << it->first << spc << " : " << (it->second.time/it->second.count) << "[s]" << endl;
		}
		cout.unsetf(ios::fixed);
		cout.precision(cur_p);

		string spc;
		for(int k = 0; k < m-5; ++k) spc += " ";
		cout << "total" << spc << " : " << total << endl;
	}

	/*!
	 * 記録された時間を文字列に出力
	 * @param[out] str 出力文字列
	 */
	void PrintToString(string &str)
	{
		int m = 0, mi;
		double total = 0.0;
		for(RXMAPTC::iterator it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
			mi = (int)it->first.size();
			if(mi > m) m = mi;

			total += it->second.time/it->second.count;
		}

		stringstream ss;
		ss.precision(3);
		ss.setf(ios::fixed);
		for(int i = 0; i < (int)m_TimeMap.size(); ++i){
			RXMAPTC::iterator it = m_TimeMap.begin();

			for(it = m_TimeMap.begin(); it != m_TimeMap.end(); ++it){
				if(it->second.idx == i){
					break;
				}
			}

			string spc;
			for(int k = 0; k < m-(int)(it->first.size()); ++k) spc += " ";

			ss << it->first << spc << " : " << (it->second.time/it->second.count) << "[s]\n";
		}

		string spc;
		for(int k = 0; k < m-5; ++k) spc += " ";
		ss << "total" << spc << " : " << total << "[s]\n";

		str = ss.str();
	}
};




#endif // #ifdef _RX_TIMER_H_