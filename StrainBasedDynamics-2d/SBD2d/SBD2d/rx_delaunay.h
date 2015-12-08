/*! 
  @file rx_delaunay.h
	
  @brief ドロネー三角形分割
 
  @author Makoto Fujisawa
  @date 2014-07
*/

#ifndef _RX_DELAUNAY_H_
#define _RX_DELAUNAY_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
// C標準
#include <iostream>

// STL
#include <vector>

// Vec2,Vec3など
#include "rx_utility.h"


//-----------------------------------------------------------------------------
// Delaunay三角形分割クラス
//  - http://tercel-sakuragaoka.blogspot.jp/2011/06/processingdelaunay.html
//    のコードをC++にして，三角形を頂点+位相構造に変えただけ
//-----------------------------------------------------------------------------
class rxDelaunayTriangles
{  
	struct rxDTriangle
	{
		int idx[3];
		//rxTriangle triangle;
		bool enable;

		rxDTriangle(){}
		rxDTriangle(int a, int b, int c, bool e) : enable(e)
		{
			idx[0] = a; idx[1] = b; idx[2] = c;
		}

		//! アクセスオペレータ
		inline int& operator[](int i){ return idx[i]; }
		inline int operator[](int i) const { return idx[i]; }

		//! ==オペレータ
		inline bool operator==(const rxDTriangle &a) const
		{
			return ((idx[0] == a[0] && idx[1] == a[1] && idx[2] == a[2]) || 
					(idx[0] == a[0] && idx[1] == a[2] && idx[2] == a[1]) || 
					(idx[0] == a[1] && idx[1] == a[0] && idx[2] == a[2]) || 
					(idx[0] == a[1] && idx[1] == a[2] && idx[2] == a[0]) || 
					(idx[0] == a[2] && idx[1] == a[0] && idx[2] == a[1]) || 
					(idx[0] == a[2] && idx[1] == a[1] && idx[2] == a[0]) );
		}
	};
	typedef vector<rxDTriangle> TriList;
	TriList m_vTriangles;

	vector<Vec2> m_vVertices;

public:
	/*!
	 * コンストラクタ
	 */
	rxDelaunayTriangles()
	{ 
	}

	/*!
	 * Delaunay三角形分割
	 * @param[in] point_array 点群
	 * @param[in] minp,maxp 三角形を生成する領域
	 * @return 作成された三角形数
	 */
	int DelaunayTriangulation(vector<Vec2> &point_array, Vec2 minp, Vec2 maxp)
	{  
		// 三角形リストを初期化 
		m_vTriangles.clear();

		// 全体を囲む三角形を作成してリストに追加
		rxDTriangle first_triangle = getCircumscribedTriangle(minp, maxp);
		m_vTriangles.push_back(first_triangle);

		// 点を逐次添加し、反復的に三角分割を行う  
		vector<Vec2>::iterator itr = point_array.begin();
		for(; itr != point_array.end(); ++itr){
			Vec2 p = *itr;
			TriList tmpTriangles;

			int pn = (int)m_vVertices.size();
			m_vVertices.push_back(p);

			// 現在の三角形リストから要素を一つずつ取り出して、  
			// 与えられた点が各々の三角形の外接円の中に含まれるかどうか判定  
			TriList::iterator jtr = m_vTriangles.begin();
			while(jtr != m_vTriangles.end()){
				rxDTriangle &t = *jtr;

				// 外接円を求める  
				Vec2 c;
				double r;
				getCircumscribedCircles(t, c, r);

				// 追加された点が外接円内部に存在する場合は三角形を分割
				if(norm(c-p) < r){
					// 新しい三角形を作成し，重複チェックして格納
					addTriangle(tmpTriangles, rxDTriangle(pn, t[0], t[1], true));  
					addTriangle(tmpTriangles, rxDTriangle(pn, t[1], t[2], true));  
					addTriangle(tmpTriangles, rxDTriangle(pn, t[2], t[0], true)); 

					// 外接円判定に使った三角形をリストから削除  
					jtr = m_vTriangles.erase(jtr);
				}
				else{
					jtr++;
				}
			}  

			// 一時ハッシュのうち、重複のないものを三角形リストに追加   
			jtr = tmpTriangles.begin();
			for(; jtr != tmpTriangles.end(); ++jtr){
				if(jtr->enable){
					m_vTriangles.push_back(*jtr);
				}
			}  
		}  

		// 外部三角形と頂点を共有する三角形を削除  
		TriList::iterator jtr = m_vTriangles.begin();
		while(jtr != m_vTriangles.end()){
			rxDTriangle t = *jtr;
			if(hasCommonVertex(first_triangle, t)){
				jtr = m_vTriangles.erase(jtr);
			}
			else{
				jtr++;
			}
		}

		// 外部三角形用に追加した頂点を削除(三角形の頂点インデックスも修正)
		itr = m_vVertices.begin();
		itr = m_vVertices.erase(itr);
		itr = m_vVertices.erase(itr);
		itr = m_vVertices.erase(itr);
		if(!m_vTriangles.empty()){
			jtr = m_vTriangles.begin();
			while(jtr != m_vTriangles.end()){
				rxDTriangle &t = *jtr;
				for(int i = 0; i < 3; ++i){
					t[i] -= 3;
				}
				jtr++;
			}
		}


		return (int)m_vTriangles.size();
	}  


	//! アクセスメソッド
	int GetTriangleNum(void) const { return (int)m_vTriangles.size(); }
	int GetVertexNum(void) const { return (int)m_vVertices.size(); }

	Vec2 GetVertex(int i) const { return m_vVertices[i]; }
	int  GetTriangle(int t, int v) const { return m_vTriangles[t][v]; }
	void GetTriangle(int t, int *idx) const
	{
		idx[0] = m_vTriangles[t][0];
		idx[1] = m_vTriangles[t][1];
		idx[2] = m_vTriangles[t][2];
	}


private:
	/*!
	 * 三角形リスト中に重複があれば元の三角形をfalseにして，なければ新しい三角形tをリストに追加
	 * @param[in] tri_list 三角形リスト
	 * @param[in] t 追加三角形
	 */
	void addTriangle(TriList &tri_list, rxDTriangle t)
	{  
		TriList::iterator itr = tri_list.begin();
		for(; itr != tri_list.end(); ++itr){
			if(*itr == t){
				itr->enable = false;
				break;
			}
		}
				
		if(itr == tri_list.end()){
			t.enable = true;
			tri_list.push_back(t);
		}
	} 

	/*!
	 * 共通点を持つかどうかの判定
	 */
	bool hasCommonVertex(const rxDTriangle &t1, const rxDTriangle &t2)
	{
		return (t1[0] == t2[0] || t1[0] == t2[1] || t1[0] == t2[2] ||  
				t1[1] == t2[0] || t1[1] == t2[1] || t1[1] == t2[2] ||  
				t1[2] == t2[0] || t1[2] == t2[1] || t1[2] == t2[2] );  
	}


	/*!
	 * 矩形を内部に含む正三角形の算出
	 * @param[in] start,end 矩形の対角線を構成する2端点座標
	 * @return 三角形
	 */
	rxDTriangle getCircumscribedTriangle(Vec2 start, Vec2 end)
	{  
		if(end[0] < start[0]){  
			double tmp = start[0];  
			start[0] = end[0];  
			end[0] = tmp;  
		}  
		if(end[1] < start[1]){  
			double tmp = start[1];  
			start[1] = end[1];  
			end[1] = tmp;  
		}  

		// 矩形を包含する円  
		Vec2 cen = 0.5*(end+start); 
		double rad = norm(cen-start)+0.01; 

		// 円に外接する正三角形(円の中心=重心，辺長2√3 r
		const double s3 = sqrt(3.0);
		Vec2 p1(cen[0]-s3*rad, cen[1]-rad);  
		Vec2 p2(cen[0]+s3*rad, cen[1]-rad);  
		Vec2 p3(cen[0], cen[1]+2*rad);

		int v1, v2, v3;
		v1 = (int)m_vVertices.size(); m_vVertices.push_back(p1);
		v2 = (int)m_vVertices.size(); m_vVertices.push_back(p2);
		v3 = (int)m_vVertices.size(); m_vVertices.push_back(p3);

		return rxDTriangle(v1, v2, v3, false);
	}  


	/*!
	 * 三角形の外接円を求める
	 * @param[in] t 三角形
	 * @param[out] c,r 
	 */
	void getCircumscribedCircles(rxDTriangle t, Vec2 &c, double &r)
	{  
		double x1 = m_vVertices[t[0]][0];  
		double y1 = m_vVertices[t[0]][1];  
		double x2 = m_vVertices[t[1]][0];  
		double y2 = m_vVertices[t[1]][1];  
		double x3 = m_vVertices[t[2]][0];  
		double y3 = m_vVertices[t[2]][1];  

		double e = 2.0*((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1));  
		double x = ((y3-y1)*(x2*x2-x1*x1+y2*y2-y1*y1) + (y1-y2)*(x3*x3-x1*x1+y3*y3-y1*y1))/e;
		double y = ((x1-x3)*(x2*x2-x1*x1+y2*y2-y1*y1) + (x2-x1)*(x3*x3-x1*x1+y3*y3-y1*y1))/e;
		c = Vec2(x, y);

		// 外接円の半径は中心から三角形の任意の頂点までの距離に等しい 
		r = norm(c-m_vVertices[t[0]]);  
	}  
};


/*!
 * Delaunay三角形分割を実行
 * @param[in] points 元になる点群
 * @param[out] tris 三角形(元の点群リストに対する位相構造)
 * @return 作成された三角形の数
 */
inline static int CreateDelaunayTriangles(vector<Vec2> &points, vector< vector<int> > &tris)
{
	if(points.empty()) return 0;
	
	// 点群のAABBを求める
	Vec2 minp, maxp;
	minp = maxp = points[0];
	vector<Vec2>::iterator itr = points.begin();
	for(; itr != points.end(); ++itr){
		if((*itr)[0] < minp[0]) minp[0] = (*itr)[0];
		if((*itr)[1] < minp[1]) minp[1] = (*itr)[1];
		if((*itr)[0] > maxp[0]) maxp[0] = (*itr)[0];
		if((*itr)[1] > maxp[1]) maxp[1] = (*itr)[1];
	}

	// Delaunay三角形分割
	rxDelaunayTriangles delaunay;
	delaunay.DelaunayTriangulation(points, minp, maxp);

	// 生成された三角形情報を格納
	int tn = delaunay.GetTriangleNum();
	tris.resize(tn);
	for(int i = 0; i < tn; ++i){
		tris[i].resize(3);
		delaunay.GetTriangle(i, &tris[i][0]);
	}

	return 1;
}



#endif // #ifdef _RX_DELAUNAY_H_
