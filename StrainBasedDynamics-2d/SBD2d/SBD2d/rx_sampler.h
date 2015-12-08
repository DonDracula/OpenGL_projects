/*! 
  @file rx_sampler.h
	
  @brief サンプル点生成
 
  @author Makoto Fujisawa
  @date 2013-06
*/

#ifndef _RX_SAMPLER_
#define _RX_SAMPLER_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// STL
#include <vector>
#include <list>

#include "rx_utility.h"

using namespace std;


//-----------------------------------------------------------------------------
// 他のサンプラの親クラス
//-----------------------------------------------------------------------------
class rxSampler
{
protected:
	Vec2 m_v2MinPos, m_v2MaxPos;	//!< サンプリング範囲
	Vec2 m_v2Dim;					//!< サンプリング範囲の大きさ
	int m_iNumMax;					//!< 最大サンプリング数

	double m_fDmin;					//!< サンプリング点間の最小距離
	int m_iNumAround;				//!< 点周囲のサンプリング数

public:
	/*!
	 * コンストラクタ
	 * @param[in] minp,maxp 生成範囲
	 * @param[in] num_max 最大点数
	 */
	rxSampler(Vec2 minp, Vec2 maxp, int num_max)
	{
		m_v2MinPos = minp;
		m_v2MaxPos = maxp;
		m_v2Dim = maxp-minp;
		m_iNumMax = num_max;
	}

	//! デストラクタ
	virtual ~rxSampler()
	{
	}
	
	//! 点のサンプリング
	virtual void Generate(vector<Vec2> &points) = 0;

protected:
	/*!
	 * 座標値からその点が含まれるセル位置を返す
	 * @param[in] p 座標値
	 * @param[out] i,j セル位置
	 * @param[in] h グリッド幅
	 */
	void calGridPos(Vec2 p, int &i, int &j, double h)
	{
		Vec2 dp = (p-m_v2MinPos)/h;
		i = (int)(dp[0]);
		j = (int)(dp[1]);
	}

	/*!
	 * 中心点から一定距離内のランダム点を生成
	 * @param[in] cen 中心点座標
	 * @param[in] d_min 最小距離(2*d_minが最大)
	 * @return ランダム点
	 */
	Vec2 genRandomAroundPoint(Vec2 cen, double d_min)
	{
		// 追加する点までの距離(半径)と角度を乱数で設定
		double rad = d_min*(1.0+RXFunc::Frand());
		double ang = 2*RX_PI*(RXFunc::Frand());

		// cenから距離[d_min, 2*d_min]内の点
		Vec2 new_p = cen+rad*Vec2(sin(ang), cos(ang));

		return new_p;
	}

	inline int IDX(int i, int j, int nx) const { return i+j*nx; }
};


//-----------------------------------------------------------------------------
// ランダムサンプラ
//  - 乱数を使ったサンプル点の生成
//-----------------------------------------------------------------------------
class rxRandomSampler : public rxSampler
{
protected:
	rxRandomSampler() : rxSampler(Vec2(0.0), Vec2(1.0), 100){}

public:
	/*!
	 * コンストラクタ
	 * @param[in] minp,maxp 生成範囲
	 * @param[in] num_max 最大点数
	 */
	rxRandomSampler(Vec2 minp, Vec2 maxp, int num_max) : rxSampler(minp, maxp, num_max)
	{
	}

	//! デストラクタ
	virtual ~rxRandomSampler(){}

	/*!
	 * サンプリング点の生成
	 * @param[out] points サンプリングされた点
	 */
	virtual void Generate(vector<Vec2> &points)
	{
		points.resize(m_iNumMax);
		for(int i = 0; i < m_iNumMax; ++i){
			points[i] = RXFunc::Rand(m_v2MaxPos, m_v2MinPos);
		}
	}
};


//-----------------------------------------------------------------------------
// Uniform Poisson Disk Sampling
//  - R. Bridson, "Fast Poisson Disk Sampling in Arbitrary Dimensions", SIGGRAPH2007 sketches. 
//  - http://devmag.org.za/2009/05/03/poisson-disk-sampling/
//-----------------------------------------------------------------------------
class rxUniformPoissonDiskSampler : public rxSampler
{
	// サンプル点を格納するグリッドセル情報
	struct rxPointGrid
	{
		Vec2 point;		//!< 格納された点座標
		int num;		//!< 格納された点の数(0 or 1)
		rxPointGrid() : num(0){}
	};

	double m_fH;					//!< グリッド幅
	int m_iNx, m_iNy;				//!< グリッド数

	vector<rxPointGrid> m_vGrid;	//!< 点を格納するグリッド

protected:
	rxUniformPoissonDiskSampler() : rxSampler(Vec2(0.0), Vec2(1.0), 100){}

public:
	/*!
	 * コンストラクタ
	 * @param[in] minp,maxp 生成範囲
	 * @param[in] num_max 最大点数
	 * @param[in] num_around 点周囲のサンプリング数
	 * @param[in] d_min サンプリング点間の最小距離
	 */
	rxUniformPoissonDiskSampler(Vec2 minp, Vec2 maxp, int num_max, int num_around, double d_min)
		: rxSampler(minp, maxp, num_max)
	{
		// 点を格納するグリッド
		// 格子幅をr/√dとすることで1点/グリッドになるようにする
		m_fH = d_min/RX_ROOT2;
		m_iNx = (int)(m_v2Dim[0]/m_fH)+1;
		m_iNy = (int)(m_v2Dim[1]/m_fH)+1;

		m_vGrid.resize(m_iNx*m_iNy);

		m_fDmin = d_min;
		m_iNumAround = num_around;
	}

	//! デストラクタ
	virtual ~rxUniformPoissonDiskSampler(){}

protected:
	/*!
	 * 最初のサンプリング点の追加
	 * @param[in] grid 点を格納するグリッド
	 * @param[out] active サンプリング位置決定用基準点
	 * @param[out] point サンプリング点
	 */	
	void addFirstPoint(list<Vec2> &active, vector<Vec2> &point)
	{
		// ランダムな位置に点を生成
		Vec2 p = RXFunc::Rand(m_v2MaxPos, m_v2MinPos);

		// 点を含むグリッド位置
		int i, j;
		calGridPos(p, i, j, m_fH);
		int g = IDX(i, j, m_iNx);
		m_vGrid[g].point = p;
		m_vGrid[g].num = 1;

		active.push_back(p);
		point.push_back(p);
	}


	/*!
	 * サンプリング点の追加
	 * @param[in] grid 点を格納するグリッド
	 * @param[out] active サンプリング位置決定用基準点
	 * @param[out] point サンプリング点
	 */	
	void addPoint(list<Vec2> &active, vector<Vec2> &point, Vec2 p)
	{
		// 点を含むグリッド位置
		int i, j;
		calGridPos(p, i, j, m_fH);
		int g = IDX(i, j, m_iNx);
		m_vGrid[g].point = p;
		m_vGrid[g].num = 1;

		active.push_back(p);
		//point.push_back(p);
	}


	/*!
	 * サンプリング点の追加
	 * @param[in] grid 点を格納するグリッド
	 * @param[out] active サンプリング位置決定用基準点
	 * @param[out] point サンプリング点
	 */	
	bool addNextPoint(list<Vec2> &active, vector<Vec2> &point, Vec2 p)
	{
		bool found = false;

		Vec2 q = genRandomAroundPoint(p, m_fDmin);

		if(RXFunc::InRange(q, m_v2MinPos, m_v2MaxPos)){
			int x, y;	// qが含まれるグリッドセル位置
			calGridPos(q, x, y, m_fH);

			bool close = false;

			// すでに追加された点の周囲に逐次的に点を追加していく
			// 周囲のグリッドを探索して，近すぎる点(距離がd_minより小さい)がないか調べる
			for(int i = RX_MAX(0, x-2); i < RX_MIN(m_iNx, x+3) && !close; ++i){
				for(int j = RX_MAX(0, y-2); j < RX_MIN(m_iNy, y+3) && !close; ++j){
					int g = IDX(i, j, m_iNx);
					if(m_vGrid[g].num){	// グリッドにすでに点が含まれている場合
						double dist = norm(m_vGrid[g].point-q);
						if(dist < m_fDmin){
							close = true;
						}
					}
				}
			}

			// 近すぎる点がなければ新しい点として追加
			if(!close){
				found = true;
				active.push_back(q);
				point.push_back(q);
				int g = IDX(x, y, m_iNx);
				m_vGrid[g].point = q;
				m_vGrid[g].num = 1;
			}

		}

		return found;
	}

public:
	/*!
	 * サンプリング
	 * @param[out] points サンプリングされた点
	 */
	virtual void Generate(vector<Vec2> &points)
	{
		// 点を格納するグリッドの初期化
		int gnum = m_iNx*m_iNy;
		for(int k = 0; k < gnum; ++k){
			m_vGrid[k].point = Vec2(0.0);
			m_vGrid[k].num = 0;
		}

		list<Vec2> active;
		list<Vec2>::const_iterator itr;

		if(points.empty()){
			// 最初の点の追加(ランダムな位置)
			addFirstPoint(active, points);
		}
		else{
			// 事前に追加された点
			for(vector<Vec2>::iterator i = points.begin(); i != points.end(); ++i){
				addPoint(active, points, *i);
			}
		}

		// これ以上点が追加できない or 最大点数 までサンプリング
		while((!active.empty()) && ((int)points.size() <= m_iNumMax)){
			// アクティブリストからランダムに1点を取り出す
			int idx = RXFunc::Nrand(active.size());
			itr = active.begin();
			for(int i = 0; i < idx; ++i) itr++;
			Vec2 p = *(itr);

			// 点pから[d_min,2*d_min]の位置でかつ他の点に近すぎない位置に新しい点を追加
			bool found = false;
			for(int i = 0; i < m_iNumAround; ++i){
				found |= addNextPoint(active, points, p);
			}

			// 点pの周りにこれ以上点を追加できない場合はactiveリストから削除
			if(!found){
				active.erase(itr);
			}
		}
	}
};


//-----------------------------------------------------------------------------
// Poisson Disk Sampling with a distribution function
//  - R. Bridson, "Fast Poisson Disk Sampling in Arbitrary Dimensions", SIGGRAPH2007 sketches. 
//  - http://devmag.org.za/2009/05/03/poisson-disk-sampling/
//  - 分布関数の値により部分的に点の密度を変える
//-----------------------------------------------------------------------------
class rxPoissonDiskSampler : public rxSampler
{
	// サンプル点を格納するグリッドセル情報
	struct rxPointGrid
	{
		vector<Vec2> points;		//!< 格納された点座標
		int num;		//!< 格納された点の数(0 or 1)
		rxPointGrid() : num(0){}
	};

	double m_fH;					//!< グリッド幅
	int m_iNx, m_iNy;				//!< グリッド数

	vector<rxPointGrid> m_vGrid;	//!< 点を格納するグリッド

	double (*m_fpDist)(Vec2);		//!< 分布関数

protected:
	rxPoissonDiskSampler() : rxSampler(Vec2(0.0), Vec2(1.0), 100){}

public:
	/*!
	 * コンストラクタ
	 * @param[in] minp,maxp 生成範囲
	 * @param[in] num_max 最大点数
	 * @param[in] num_around 点周囲のサンプリング数
	 * @param[in] d_min サンプリング点間の最小距離
	 * @param[in] dist 分布関数
	 */
	rxPoissonDiskSampler(Vec2 minp, Vec2 maxp, int num_max, int num_around, double d_min, double (*dist)(Vec2))
		: rxSampler(minp, maxp, num_max)
	{
		// 点を格納するグリッド
		// 格子幅をr/√dとすることで1点/グリッドになるようにする
		m_fH = d_min/RX_ROOT2;
		m_iNx = (int)(m_v2Dim[0]/m_fH)+1;
		m_iNy = (int)(m_v2Dim[1]/m_fH)+1;

		m_vGrid.resize(m_iNx*m_iNy);

		m_fDmin = d_min;
		m_iNumAround = num_around;

		m_fpDist = dist;
	}

	//! デストラクタ
	virtual ~rxPoissonDiskSampler(){}

protected:
	/*!
	 * 最初のサンプリング点の追加
	 * @param[in] grid 点を格納するグリッド
	 * @param[out] active サンプリング位置決定用基準点
	 * @param[out] point サンプリング点
	 */	
	void addFirstPoint(list<Vec2> &active, vector<Vec2> &point)
	{
		// ランダムな位置に点を生成
		Vec2 p = RXFunc::Rand(m_v2MaxPos, m_v2MinPos);

		// 点を含むグリッド位置
		int i, j;
		calGridPos(p, i, j, m_fH);
		int g = IDX(i, j, m_iNx);
		m_vGrid[g].points.push_back(p);
		m_vGrid[g].num++;

		active.push_back(p);
		point.push_back(p);
	}

	/*!
	 * サンプリング点の追加
	 * @param[in] grid 点を格納するグリッド
	 * @param[out] active サンプリング位置決定用基準点
	 * @param[out] point サンプリング点
	 */	
	bool addNextPoint(list<Vec2> &active, vector<Vec2> &point, Vec2 p)
	{
		bool found = false;

		double fraction = m_fpDist(p);
		Vec2 q = genRandomAroundPoint(p, m_fDmin);

		if(RXFunc::InRange(q, m_v2MinPos, m_v2MaxPos)){
			int x, y;	// qが含まれるグリッドセル位置
			calGridPos(q, x, y, m_fH);

			bool close = false;

			// すでに追加された点の周囲に逐次的に点を追加していく
			// 周囲のグリッドを探索して，近すぎる点(距離がd_minより小さい)がないか調べる
			for(int i = RX_MAX(0, x-2); i < RX_MIN(m_iNx, x+3) && !close; ++i){
				for(int j = RX_MAX(0, y-2); j < RX_MIN(m_iNy, y+3) && !close; ++j){
					int g = IDX(i, j, m_iNx);

					// グリッド内の点を調べていく
					for(int l = 0; l < m_vGrid[g].num; ++l){
						double dist = norm(m_vGrid[g].points[l]-q);
						if(dist < m_fDmin*fraction){
							close = true;
						}
					}
				}
			}

			// 近すぎる点がなければ新しい点として追加
			if(!close){
				found = true;
				active.push_back(q);
				point.push_back(q);
				int g = IDX(x, y, m_iNx);
				m_vGrid[g].points.push_back(q);
				m_vGrid[g].num++;
			}

		}

		return found;
	}

public:
	/*!
	 * サンプリング
	 * @param[out] points サンプリングされた点
	 */
	virtual void Generate(vector<Vec2> &points)
	{
		// 点を格納するグリッドの初期化
		int gnum = m_iNx*m_iNy;
		for(int k = 0; k < gnum; ++k){
			m_vGrid[k].points.clear();
			m_vGrid[k].num = 0;
		}

		list<Vec2> active;
		list<Vec2>::const_iterator itr;

		// 最初の点の追加(ランダムな位置)
		addFirstPoint(active, points);

		// これ以上点が追加できない or 最大点数 までサンプリング
		while((!active.empty()) && ((int)points.size() <= m_iNumMax)){
			// アクティブリストからランダムに1点を取り出す
			int idx = RXFunc::Nrand(active.size());
			itr = active.begin();
			for(int i = 0; i < idx; ++i) itr++;
			Vec2 p = *(itr);

			// 点pから[d_min,2*d_min]の位置でかつ他の点に近すぎない位置に新しい点を追加
			bool found = false;
			for(int i = 0; i < m_iNumAround; ++i){
				found |= addNextPoint(active, points, p);
			}

			// 点pの周りにこれ以上点を追加できない場合はactiveリストから削除
			if(!found){
				active.erase(itr);
			}
		}
	}
};

#endif // _RX_SAMPLER_