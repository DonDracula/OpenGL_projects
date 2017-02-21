/*! @file rx_numerical.h
	
	@brief 数値計算ルーチン
 
	@author Makoto Fujisawa
	@date  
*/


#ifndef _RX_NUMERICAL_H_
#define _RX_NUMERICAL_H_

#pragma warning (disable: 4996)
#pragma warning (disable: 4819)
#pragma warning (disable: 4786)
#pragma warning (disable: 4244)


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <cmath>
#include <iostream>
#include <fstream>

#include <string>
#include <vector>

#include <cmath>

#include <algorithm>
#include <functional>

#include "boost/lexical_cast.hpp"

#include "boost/bind.hpp"
#include "boost/function.hpp"
#include "boost/mem_fn.hpp"

using namespace std;

namespace RXNumerical
{
	// 許容誤差など
	const double RX_FEQ_EPS = 1.0e-10;
	const double RX_EPS = 1.0e-8;

	/*!
	 * vectorコンテナ同士の内積計算
	 * @param 
	 * @return 
	 */
	inline float DotProduct(const vector<float> &a, const vector<float> &b)
	{
		float d = 0.0f;
		for(int i = 0; i < (int)a.size(); ++i){
			d += a[i]*b[i];
		}
		return d;
	}

	/*!
	 * 2次元vectorコンテナ同士の内積計算
	 * @param 
	 * @return 
	 */
	inline float DotProduct(const vector< vector<float> > &a, const vector< vector<float> > &b, const int &nx, const int &ny)
	{
		float d = 0.0f;
		for(int i = 0; i < nx; ++i){
			for(int j = 0; j < ny; ++j){
				d += a[i][j]*b[i][j];
			}
		}
		return d;
	}


	inline int GetIndex(const int &i, const int &j, const int &k, const int &nx, const int &ny){ return k*nx*ny+j*nx+i; }
	inline int GetIndex2(const int &i, const int &j, const int &nx){ return i+nx*j; }

	/*!
	 * ICCG法用近傍探索ルーチン
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors2D(vector< vector<int> > &neigh, const int nx, const int ny)
	{
		int i, j, idx, nx2, ny2;
		nx2 = nx+2;
		ny2 = ny+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				idx = GetIndex2(i, j, nx2);

				if(i == 0 || i == nx+1 || j == 0 || j == ny+1){
					neigh[idx][0] = -1;
					neigh[idx][1] = -1;
					neigh[idx][2] = -1;
					neigh[idx][3] = -1;
					neigh[idx][4] = -1;
				}
				else{
					neigh[idx][0] = 4;
					neigh[idx][1] = GetIndex2(i+1, j, nx2);
					neigh[idx][2] = GetIndex2(i-1, j, nx2);
					neigh[idx][3] = GetIndex2(i, j+1, nx2);
					neigh[idx][4] = GetIndex2(i, j-1, nx2);
				}
			}
		}

		return nx2*ny2;
	}
	/*!
	 * ICCG法用近傍探索ルーチン
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny,nz グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors3D(vector< vector<int> > &neigh, const int nx, const int ny, const int nz)
	{
		int i, j, k, idx, nx2, ny2, nz2;
		nx2 = nx+2;
		ny2 = ny+2;
		nz2 = nz+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				for(k = 0; k < nz2; ++k){
					idx = GetIndex(i, j, k, nx2, ny2);

					if(i == 0 || i == nx+1 || j == 0 || j == ny+1 || k == 0 || k == nz+1){
						neigh[idx][0] = -1;
						neigh[idx][1] = -1;
						neigh[idx][2] = -1;
						neigh[idx][3] = -1;
						neigh[idx][4] = -1;
						neigh[idx][5] = -1;
						neigh[idx][6] = -1;
					}
					else{
						neigh[idx][0] = 6;
						neigh[idx][1] = GetIndex(i+1, j, k, nx2, ny2);
						neigh[idx][2] = GetIndex(i-1, j, k, nx2, ny2);
						neigh[idx][3] = GetIndex(i, j+1, k, nx2, ny2);
						neigh[idx][4] = GetIndex(i, j-1, k, nx2, ny2);
						neigh[idx][5] = GetIndex(i, j, k+1, nx2, ny2);
						neigh[idx][6] = GetIndex(i, j, k-1, nx2, ny2);
					}
				}
			}
		}

		return nx2*ny2*nz2;
	}

	/*!
	 * 行列とベクトルの掛け算
	 * @param[out] y 結果を格納するコンテナ
	 * @param[in] mat 行列
	 * @param[in] x ベクトル
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void mulMatrixVector(vector<T> &y, vector< vector<T> > &mat, vector<T> x, 
								vector< vector<int> > neigh, int size_m)
	{
		int i, j, l;

		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			y[i] = mat[i][0]*x[i];
			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス
				y[i] = y[i]+mat[i][l]*x[j];
			}
		}
	}

	/*!
	 * vector同士の内積
	 * @param[out] mul 内積
	 * @param[in] x1 ベクトル1
	 * @param[in] x2 ベクトル2
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void mulVector(T &mul, const vector<T> &x1, const vector<T> &x2, 
						  vector< vector<int> > neigh, int size_m)
	{
		int i;

		mul = 0.0;
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] >= 0){
				mul += x1[i]*x2[i];
			}
		}
	}


	/*!
	 * ガウス消去法
	 * @param[out] y 解を入れるベクトル
	 * @param[in] mat 行列(左辺項)
	 * @param[in] xx 右辺項
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void solverLL(vector<T> &y, vector< vector<T> > &mat, vector<T> &xx, 
						 vector< vector<int> > neigh, int size_m)
	{
		int i, j, l;
		vector<T> x;
		x.resize(size_m);

		x = xx;

		// 前進代入
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス

				if(j > i) continue;
				if(neigh[j][0] < 0) continue;

				x[i] = x[i]-mat[i][l]*y[j];
			}
			y[i] = x[i]/mat[i][0];
		}

		x = y;

		// 後退代入
		for(i = size_m-1; i >= 0; --i){
			if(neigh[i][0] < 0) continue;

			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス

				if(j < i) continue;
				if(neigh[j][0] < 0) continue;

				x[i] = x[i]-mat[i][l]*y[j];
			}
			y[i] = x[i]/mat[i][0];
		}
	}

	/*!
	 * 共役勾配法用の収束チェッカ
	 * @param[in] r 残差ベクトル
	 * @param[inout] tol 収束条件
	 * @param[in] k 現在の反復回数
	 * @param[in] max_iter 最大反復回数
	 * @param[in] dt タイムステップ幅
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 * @return 収束したかどうか
	 */
	template<class T> 
	static bool checkConvergence(const vector<T> &r, T &tol, int k, int max_iter, T dt, 
								 vector< vector<int> > neigh, int size_m)
	{
		int i, j;
		T err1, err1sum;

		j = 0;
		err1sum = 0.0;
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			if(r[i] > 0){
				err1 = r[i];
			}
			else{
				err1 = -r[i];
			}
			
			if(err1 > tol){
				j++;
			}

			err1sum += err1;
		}

		if(j == 0){
			tol = err1sum;
			return true;
		}
		else{
			return false;
		}
	}



	/*!
	 * 不完全コレスキー分解(incomplete Cholesky decomposition)
	 * @param[in] poiss 元の行列
	 * @param[out] ic   不完全コレスキー分解後の行列
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void IcDecomp(const vector< vector<T> > &poiss, vector< vector<T> > &ic, 
							 vector< vector<int> > neigh, int size_m)
	{
		int i, j, k, mi, mj, ki, kj;
		double sum;

		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			sum = poiss[i][0];
			for(k = 1; k <= neigh[i][0]; ++k){
				j = neigh[i][k];	// iの近傍グリッドのインデックス
				if(j > i) continue;
				if(neigh[j][0] < 0) continue;

				sum = sum-ic[i][k]*ic[i][k];
			}
			ic[i][0] = sqrt(sum);

			for(k = 1; k <= neigh[i][0]; ++k){
				j = neigh[i][k];	// iの近傍グリッドのインデックス
				if(j < i) continue;
				if(neigh[j][0] < 0) continue;

				sum = poiss[i][k];
				for(mj = 1; mj <= neigh[j][0]; ++mj){
					kj = neigh[j][mj];	// jの近傍グリッドのインデックス
					if(kj >= i) continue;
					if(neigh[kj][0] < 0) continue;

					for(mi = 1; mi <= neigh[i][0]; ++mi){
						ki = neigh[i][mi];	// iの近傍パーティクルのインデックスmi

						if(ki == kj){
							sum = sum-ic[i][mi]*ic[j][mj];
							break;
						}
					}
				}
				ic[i][k] = sum/ic[i][0];

				for(mj = 1; mj <= neigh[j][0]; ++mj){
					kj = neigh[j][mj];	// jの近傍グリッドのインデックス

					if(i == kj){
						ic[j][mj] = ic[i][k];
						break;
					}
				}
			}
		}
	}

	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static int pcgSolver2(vector< vector<T> > &poiss, vector<T> &x, vector<T> &b, 
						  vector< vector<T> > &ic, vector< vector<int> > neigh, int size_m, T dt, 
						  boost::function<void (vector<T>&)> bcfunc, int &max_iter, T &tol)
	{
		int i, k = 0;
		vector<T> r, p, q, s;
		T aa, bb, rqo, rqn, ps;

		r.resize(size_m);
		p.resize(size_m);
		q.resize(size_m);
		s.resize(size_m);
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVector(s, poiss, x, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			r[i] = b[i]-s[i];
		}

		solverLL(q, ic, r, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			p[i] = q[i];
		}

		if(checkConvergence(r, tol, k, max_iter, dt, neigh, size_m)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
			max_iter = k;
			return k;
		}

		mulVector(rqo, r, q, neigh, size_m); // rqo = r*q
		for(k = 0; k < max_iter; k++){
			bcfunc(p);

			mulMatrixVector(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVector(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 0; i < size_m; ++i){
				x[i] = x[i]+aa*p[i];
				r[i] = r[i]-aa*s[i];
			}

			solverLL(q, ic, r, neigh, size_m);

			mulVector(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 0; i < size_m; ++i){
				p[i] = q[i]+bb*p[i];
			}

			if(checkConvergence(r, tol, k, max_iter, dt, neigh, size_m)){
				//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
				max_iter = k;
				return k;
			}
		}

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), k, tol);
		max_iter = k;
		return k;
	}


	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	static int pcgSolver(vector< vector<double> > &poiss, vector<double> &x, vector<double> &b, 
						 vector< vector<double> > &ic, vector< vector<int> > neigh, int size_m, double dt, 
						 boost::function<void (vector<double>*)> bcfunc, int &max_iter, double &tol)
	{
		int i, k = 0;
		vector<double> r, p, q, s;
		double aa, bb, rqo, rqn, ps;

		r.resize(size_m);
		p.resize(size_m);
		q.resize(size_m);
		s.resize(size_m);
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVector(s, poiss, x, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			r[i] = b[i]-s[i];
		}

		solverLL(q, ic, r, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			p[i] = q[i];
		}

		if(checkConvergence(r, tol, k, max_iter, dt, neigh, size_m)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
			max_iter = k;
			return k;
		}

		mulVector(rqo, r, q, neigh, size_m); // rqo = r*q
		for(k = 0; k < max_iter; k++){
			bcfunc(&p);

			mulMatrixVector(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVector(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 0; i < size_m; ++i){
				x[i] = x[i]+aa*p[i];
				r[i] = r[i]-aa*s[i];
			}

			solverLL(q, ic, r, neigh, size_m);

			mulVector(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 0; i < size_m; ++i){
				p[i] = q[i]+bb*p[i];
			}

			if(checkConvergence(r, tol, k, max_iter, dt, neigh, size_m)){
				//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
				max_iter = k;
				return k;
			}
		}

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), k, tol);
		max_iter = k;
		return k;
	}




	/*!
	 * ICCG法用近傍探索ルーチン(配列版)
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors2DP(int **neigh, const int nx, const int ny)
	{
		int i, j, idx, nx2, ny2;
		nx2 = nx+2;
		ny2 = ny+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				idx = GetIndex2(i, j, nx2);

				if(i == 0 || i == nx+1 || j == 0 || j == ny+1){
					neigh[idx][0] = -1;
					neigh[idx][1] = -1;
					neigh[idx][2] = -1;
					neigh[idx][3] = -1;
					neigh[idx][4] = -1;
				}
				else{
					neigh[idx][0] = 4;
					neigh[idx][1] = GetIndex2(i+1, j, nx2);
					neigh[idx][2] = GetIndex2(i-1, j, nx2);
					neigh[idx][3] = GetIndex2(i, j+1, nx2);
					neigh[idx][4] = GetIndex2(i, j-1, nx2);
				}
			}
		}

		return nx2*ny2;
	}
	/*!
	 * ICCG法用近傍探索ルーチン(配列版)
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny,nz グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors3DP(int **neigh, const int nx, const int ny, const int nz)
	{
		int i, j, k, idx, nx2, ny2, nz2;
		nx2 = nx+2;
		ny2 = ny+2;
		nz2 = nz+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				for(k = 0; k < nz2; ++k){
					idx = GetIndex(i, j, k, nx2, ny2);

					if(i == 0 || i == nx+1 || j == 0 || j == ny+1 || k == 0 || k == nz+1){
						neigh[idx][0] = -1;
						neigh[idx][1] = -1;
						neigh[idx][2] = -1;
						neigh[idx][3] = -1;
						neigh[idx][4] = -1;
						neigh[idx][5] = -1;
						neigh[idx][6] = -1;
					}
					else{
						neigh[idx][0] = 6;
						neigh[idx][1] = GetIndex(i+1, j, k, nx2, ny2);
						neigh[idx][2] = GetIndex(i-1, j, k, nx2, ny2);
						neigh[idx][3] = GetIndex(i, j+1, k, nx2, ny2);
						neigh[idx][4] = GetIndex(i, j-1, k, nx2, ny2);
						neigh[idx][5] = GetIndex(i, j, k+1, nx2, ny2);
						neigh[idx][6] = GetIndex(i, j, k-1, nx2, ny2);
					}
				}
			}
		}

		return nx2*ny2*nz2;
	}

	/*!
	 * ICCG法用近傍探索ルーチン(配列版)
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors2DPB(int **neigh, const int nx, const int ny)
	{
		int i, j, idx, nx2, ny2;
		nx2 = nx+2;
		ny2 = ny+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				idx = GetIndex2(i, j, nx2);

				neigh[idx][0] = -1;
				neigh[idx][1] = -1;
				neigh[idx][2] = -1;
				neigh[idx][3] = -1;
				neigh[idx][4] = -1;

				int ncount = 0;
				if(i != nx+1){
					neigh[idx][ncount+1] = GetIndex2(i+1, j, nx2);
					ncount++;
				}

				if(i != 0){
					neigh[idx][ncount+1] = GetIndex2(i-1, j, nx2);
					ncount++;
				}

				if(j != ny+1){
					neigh[idx][ncount+1] = GetIndex2(i, j+1, nx2);
					ncount++;
				}

				if(j != 0){
					neigh[idx][ncount+1] = GetIndex2(i, j-1, nx2);
					ncount++;
				}

				if(ncount == 0){
					neigh[idx][0] = -1;
				}
				else{
					neigh[idx][0] = ncount;
				}
			}
		}

		return nx2*ny2;
	}

	/*!
	 * ICCG法用近傍探索ルーチン(配列版)
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny,nz グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors3DPB(int **neigh, const int nx, const int ny, const int nz)
	{
		int i, j, k, idx, nx2, ny2, nz2;
		nx2 = nx+2;
		ny2 = ny+2;
		nz2 = nz+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				for(k = 0; k < nz2; ++k){
					idx = GetIndex(i, j, k, nx2, ny2);

					neigh[idx][0] = -1;
					neigh[idx][1] = -1;
					neigh[idx][2] = -1;
					neigh[idx][3] = -1;
					neigh[idx][4] = -1;
					neigh[idx][5] = -1;
					neigh[idx][6] = -1;

					int ncount = 0;
					if(i != nx+1){
						neigh[idx][ncount+1] = GetIndex(i+1, j, k, nx2, ny2);
						ncount++;
					}

					if(i != 0){
						neigh[idx][ncount+1] = GetIndex(i-1, j, k, nx2, ny2);
						ncount++;
					}

					if(j != ny+1){
						neigh[idx][ncount+1] = GetIndex(i, j+1, k, nx2, ny2);
						ncount++;
					}

					if(j != 0){
						neigh[idx][ncount+1] = GetIndex(i, j-1, k, nx2, ny2);
						ncount++;
					}

					if(k != nz+1){
						neigh[idx][ncount+1] = GetIndex(i, j, k+1, nx2, ny2);
						ncount++;
					}

					if(k != 0){
						neigh[idx][ncount+1] = GetIndex(i, j, k-1, nx2, ny2);
						ncount++;
					}

					if(ncount == 0){
						neigh[idx][0] = -1;
					}
					else{
						neigh[idx][0] = ncount;
					}
				}
			}
		}

		return nx2*ny2*nz2;
	}

	/*!
	 * ICCG法用近傍探索ルーチン(配列版,9グリッド版)
	 * @param[out] neigh 近傍位置を表す配列
	 * @param[in] nx,ny グリッド数
	 * @return 配列サイズ(size_m)
	 */
	static int SetNeighbors2D9(int **neigh, const int nx, const int ny)
	{
		int i, j, idx, nx2, ny2;
		nx2 = nx+2;
		ny2 = ny+2;
		for(i = 0; i < nx2; ++i){
			for(j = 0; j < ny2; ++j){
				idx = GetIndex2(i, j, nx2);

				if(i == 0 || i == nx+1 || j == 0 || j == ny+1){
					neigh[idx][0] = -1;
					neigh[idx][1] = -1;
					neigh[idx][2] = -1;
					neigh[idx][3] = -1;
					neigh[idx][4] = -1;
					neigh[idx][5] = -1;
					neigh[idx][6] = -1;
					neigh[idx][7] = -1;
					neigh[idx][8] = -1;
				}
				else{
					neigh[idx][0] = 8;
					neigh[idx][1] = GetIndex2(i+1, j, nx2);
					neigh[idx][2] = GetIndex2(i-1, j, nx2);
					neigh[idx][3] = GetIndex2(i, j+1, nx2);
					neigh[idx][4] = GetIndex2(i, j-1, nx2);
					neigh[idx][5] = GetIndex2(i-1, j-1, nx2);
					neigh[idx][6] = GetIndex2(i+1, j+1, nx2);
					neigh[idx][7] = GetIndex2(i-1, j+1, nx2);
					neigh[idx][8] = GetIndex2(i+1, j-1, nx2);
				}
			}
		}

		return nx2*ny2;
	}
	/*!
	 * 行列とベクトルの掛け算(配列版)
	 * @param[out] y 結果を格納するコンテナ
	 * @param[in] mat 行列
	 * @param[in] x ベクトル
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void mulMatrixVectorP(T *y, T **mat, T *x, int **neigh, int size_m)
	{
		int i, j, l;

		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			y[i] = mat[i][0]*x[i];
			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス
				y[i] = y[i]+mat[i][l]*x[j];
			}
		}
	}

	/*!
	 * vector同士の内積(配列版)
	 * @param[out] mul 内積
	 * @param[in] x1 ベクトル1
	 * @param[in] x2 ベクトル2
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void mulVectorP(T &mul, const T *x1, const T *x2, int **neigh, int size_m)
	{
		int i;

		mul = 0.0;
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] >= 0){
				mul += x1[i]*x2[i];
			}
		}
	}


	/*!
	 * ガウス消去法(配列版)
	 * @param[out] y 解を入れるベクトル
	 * @param[in] mat 行列(左辺項)
	 * @param[in] xx 右辺項
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void solverLLP(T *y, T **mat, T *xx, int **neigh, int size_m)
	{
		int i, j, l;
		T *x;
		x = new T[size_m];

		for(i = 0; i < size_m; ++i) x[i] = xx[i];


		// 前進代入
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス

				if(j > i) continue;
				if(neigh[j][0] < 0) continue;

				x[i] = x[i]-mat[i][l]*y[j];
			}
			y[i] = x[i]/mat[i][0];
		}

		for(i = 0; i < size_m; ++i) x[i] = y[i];

		// 後退代入
		for(i = size_m-1; i >= 0; --i){
			if(neigh[i][0] < 0) continue;

			for(l = 1; l <= neigh[i][0]; ++l){
				j = neigh[i][l];	// 近傍インデックス

				if(j < i) continue;
				if(neigh[j][0] < 0) continue;

				x[i] = x[i]-mat[i][l]*y[j];
			}
			y[i] = x[i]/mat[i][0];
		}

		delete [] x;
	}

	/*!
	 * 共役勾配法用の収束チェッカ(配列版)
	 * @param[in] r 残差ベクトル
	 * @param[inout] tol 収束条件
	 * @param[in] k 現在の反復回数
	 * @param[in] max_iter 最大反復回数
	 * @param[in] dt タイムステップ幅
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 * @return 収束したかどうか
	 */
	template<class T> 
	static bool checkConvergenceP(const T *r, T &tol, int k, int max_iter, T dt, int **neigh, int size_m)
	{
		int i, j;
		T err1, err1sum;

		j = 0;
		err1sum = 0.0;
		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			if(r[i] > 0){
				err1 = r[i];
			}
			else{
				err1 = -r[i];
			}
			
			if(err1 > tol){
				j++;
			}

			err1sum += err1;
		}

//		if(err1sum < tol){ 
		if(j == 0){
			tol = err1sum;
			return true;
		}
		else{
			return false;
		}
	}



	/*!
	 * 不完全コレスキー分解(incomplete Cholesky decomposition)(配列版)
	 * @param[in] poiss 元の行列
	 * @param[out] ic   不完全コレスキー分解後の行列
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static void IcDecompP(T **poiss, T **ic, int **neigh, int size_m)
	{
		int i, j, k, mi, mj, ki, kj;
		T sum;

		for(i = 0; i < size_m; ++i){
			if(neigh[i][0] < 0) continue;

			sum = poiss[i][0];
			for(k = 1; k <= neigh[i][0]; ++k){
				j = neigh[i][k];	// iの近傍グリッドのインデックス
				if(j > i) continue;
				if(neigh[j][0] < 0) continue;

				sum = sum-ic[i][k]*ic[i][k];
			}
			ic[i][0] = sqrt(sum);

			for(k = 1; k <= neigh[i][0]; ++k){
				j = neigh[i][k];	// iの近傍グリッドのインデックス
				if(j < i) continue;
				if(neigh[j][0] < 0) continue;

				sum = poiss[i][k];
				for(mj = 1; mj <= neigh[j][0]; ++mj){
					kj = neigh[j][mj];	// jの近傍グリッドのインデックス
					if(kj >= i) continue;
					if(neigh[kj][0] < 0) continue;

					for(mi = 1; mi <= neigh[i][0]; ++mi){
						ki = neigh[i][mi];	// iの近傍グリッドのインデックスmi

						if(ki == kj){
							sum = sum-ic[i][mi]*ic[j][mj];
							break;
						}
					}
				}
				ic[i][k] = sum/ic[i][0];

				for(mj = 1; mj <= neigh[j][0]; ++mj){
					kj = neigh[j][mj];	// jの近傍グリッドのインデックス

					if(i == kj){
						ic[j][mj] = ic[i][k];
						break;
					}
				}
			}
		}
	}

	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ(配列版)
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static int pcgSolverP(T **poiss, T *x, T *b, T **ic, int **neigh, int size_m, T dt, int &max_iter, T &tol)
	{
		int i, k = 0;
		T *r, *p, *q, *s;
		T aa, bb, rqo, rqn, ps;

		r = new T[size_m];
		p = new T[size_m];
		q = new T[size_m];
		s = new T[size_m];
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVectorP(s, poiss, x, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			r[i] = b[i]-s[i];
		}

		solverLLP(q, ic, r, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			p[i] = q[i];
		}

		if(checkConvergenceP(r, tol, k, max_iter, dt, neigh, size_m)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
			max_iter = k;
			return k;
		}

		mulVectorP(rqo, r, q, neigh, size_m); // rqo = r*q
		for(k = 0; k < max_iter; k++){
			mulMatrixVectorP(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVectorP(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 0; i < size_m; ++i){
				x[i] = x[i]+aa*p[i];
				r[i] = r[i]-aa*s[i];
			}

			solverLLP(q, ic, r, neigh, size_m);

			mulVectorP(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 0; i < size_m; ++i){
				p[i] = q[i]+bb*p[i];
			}

			if(checkConvergenceP(r, tol, k, max_iter, dt, neigh, size_m)){
				break;
			}
		}

		delete [] r;
		delete [] p;
		delete [] q;
		delete [] s;

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), k, tol);
		max_iter = k;
		return k;
	}

	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ(配列版)
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	template<class T> 
	static int pcgSolverP(T **poiss, T *x, T *b, T **ic, int **neigh, int size_m, T dt, 
						  boost::function<void (T*)> bcfunc, int &max_iter, T &tol)
	{
		int i, k = 0;
		T *r, *p, *q, *s;
		T aa, bb, rqo, rqn, ps;

		r = new T[size_m];
		p = new T[size_m];
		q = new T[size_m];
		s = new T[size_m];
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVectorP(s, poiss, x, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			r[i] = b[i]-s[i];
		}

		solverLLP(q, ic, r, neigh, size_m);

		for(i = 0; i < size_m; ++i){
			p[i] = q[i];
		}

		if(checkConvergenceP(r, tol, k, max_iter, dt, neigh, size_m)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
			max_iter = k;
			return k;
		}

		mulVectorP(rqo, r, q, neigh, size_m); // rqo = r*q
		for(k = 0; k < max_iter; k++){
			bcfunc(p);

			mulMatrixVectorP(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVectorP(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 0; i < size_m; ++i){
				x[i] = x[i]+aa*p[i];
				r[i] = r[i]-aa*s[i];
			}

			solverLLP(q, ic, r, neigh, size_m);

			mulVectorP(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 0; i < size_m; ++i){
				p[i] = q[i]+bb*p[i];
			}

			if(checkConvergenceP(r, tol, k, max_iter, dt, neigh, size_m)){
				//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), k, tol);
				break;
			}
		}

		delete [] r;
		delete [] p;
		delete [] q;
		delete [] s;

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), k, tol);
		max_iter = k;
		return k;
	}


	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ(配列版)
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	static int pcgSolverPB3(double **poiss, double *x, double *b, double **ic, int **neigh, int nx, int ny, int nz, double dt, 
						    boost::function<void (double*)> bcfunc, int &max_iter, double &tol)
	{
		int i, j, k, l = 0, idx;
		double *r, *p, *q, *s;
		double aa, bb, rqo, rqn, ps;

		int size_m = (nx+2)*(ny+2)*(nz+2);

		r = new double[size_m];
		p = new double[size_m];
		q = new double[size_m];
		s = new double[size_m];
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVectorP(s, poiss, x, neigh, size_m);

		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				for(k = 1; k <= nz; ++k){
					idx = GetIndex(i, j, k, nx+2, ny+2);
					r[idx] = b[idx]-s[idx];
				}
			}
		}

		solverLLP(q, ic, r, neigh, size_m);

		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				for(k = 1; k <= nz; ++k){
					idx = GetIndex(i, j, k, nx+2, ny+2);
					p[idx] = q[idx];
				}
			}
		}

		if(checkConvergenceP(r, tol, l, max_iter, dt, neigh, size_m)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), l, tol);
			max_iter = l;
			return l;
		}

		mulVectorP(rqo, r, q, neigh, size_m); // rqo = r*q
		for(l = 0; l < max_iter; l++){
			bcfunc(p);

			mulMatrixVectorP(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVectorP(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					for(k = 1; k <= nz; ++k){
						idx = GetIndex(i, j, k, nx+2, ny+2);
						x[idx] = x[idx]+aa*p[idx];
						r[idx] = r[idx]-aa*s[idx];
					}
				}
			}

			solverLLP(q, ic, r, neigh, size_m);

			mulVectorP(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					for(k = 1; k <= nz; ++k){
						idx = GetIndex(i, j, k, nx+2, ny+2);
						p[idx] = q[idx]+bb*p[idx];
					}
				}
			}

			if(checkConvergenceP(r, tol, l, max_iter, dt, neigh, size_m)){
				//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), l, tol);
				break;
			}
		}

		delete [] r;
		delete [] p;
		delete [] q;
		delete [] s;

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), l, tol);
		max_iter = l;
		return l;
	}

	/*!
	 * 共役勾配法用の収束チェッカ(配列版)
	 * @param[in] r 残差ベクトル
	 * @param[inout] tol 収束条件
	 * @param[in] k 現在の反復回数
	 * @param[in] max_iter 最大反復回数
	 * @param[in] dt タイムステップ幅
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 * @return 収束したかどうか
	 */
	static bool checkConvergencePB2(const double *r, const double &tol, double &err1sum, int l, int max_iter, double dt, int **neigh, int nx, int ny)
	{
		int i, j, m, idx;
		double err1;

		m = 0;
		err1sum = 0.0;
		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				idx = GetIndex2(i, j, nx+2);

				if(neigh[idx][0] < 0) continue;

				if(r[idx] > 0){
					err1 = r[idx];
				}
				else{
					err1 = -r[idx];
				}
				
				if(err1 > tol){
					m++;
				}

				err1sum += err1;
			}
		}

//		if(err1sum < tol){ 
		if(m == 0){
			return true;
		}
		else{
			return false;
		}
	}

	/*!
	 * 不完全コレスキー分解付共役勾配(ICCG)ソルバ(配列版)
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[in] ic 不完全コレスキー分解した行列
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	static int pcgSolverPB2(double **poiss, double *x, double *b, double **ic, int **neigh, int nx, int ny, double dt, 
						    boost::function<void (double*)> bcfunc, int &max_iter, double &tol)
	{
		int i, j, l = 0, idx;
		double *r, *p, *q, *s;
		double aa, bb, rqo, rqn, ps, tol_err;

		int size_m = (nx+2)*(ny+2);

		r = new double[size_m];
		p = new double[size_m];
		q = new double[size_m];
		s = new double[size_m];
			
		for(i = 0; i < size_m; ++i){
			s[i] = 0.0;
			r[i] = 0.0;
			q[i] = 0.0;
		}

		mulMatrixVectorP(s, poiss, x, neigh, size_m);

		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				idx = GetIndex2(i, j, nx+2);
				r[idx] = b[idx]-s[idx];
			}
		}

		solverLLP(q, ic, r, neigh, size_m);

		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				idx = GetIndex2(i, j, nx+2);
				p[idx] = q[idx];
			}
		}

		if(checkConvergencePB2(r, tol, tol_err, l, max_iter, dt, neigh, nx, ny)){
			//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), l, tol);
			tol = tol_err;
			max_iter = l;
			return l;
		}


		//ofstream fo;
		//fo.open("log\\_debug_iccg_bc.lgx", ios::out);

		//fo << setprecision(5);		// 精度
		//int width = 8;

		mulVectorP(rqo, r, q, neigh, size_m); // rqo = r*q
		for(l = 0; l < max_iter; l++){
			bcfunc(p);

			mulMatrixVectorP(s, poiss, p, neigh, size_m);	// s = poiss*p
			mulVectorP(ps, p, s, neigh, size_m);			// ps = p*s

			aa = rqo/ps;
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					idx = GetIndex2(i, j, nx+2);
					x[idx] = x[idx]+aa*p[idx];
					r[idx] = r[idx]-aa*s[idx];
				}
			}

			solverLLP(q, ic, r, neigh, size_m);

			mulVectorP(rqn, r, q, neigh, size_m);	// rqn = r*q

			bb = rqn/rqo;
			rqo = rqn;
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					idx = GetIndex2(i, j, nx+2);
					p[idx] = q[idx]+bb*p[idx];
				}
			}

			//fo << l << endl;
			//for(i = 0; i < nx+2; ++i){
			//	fo << setw(width) << i << " ";
			//}

			//fo << endl;
			//for(j = (nx+2)-1; j >= 0; --j){
			//	fo << setw(3) << j << " : ";
			//	for(i = 0; i < ny+2; ++i){
			//		double val = r[GetIndex2(i, j, nx+2)];

			//		fo << setw(width) << val << " ";

			//	}
			//	fo << endl;
			//}
			//fo << endl;
			//fo << endl;


			if(checkConvergencePB2(r, tol, tol_err, l, max_iter, dt, neigh, nx, ny)){
				//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)"), l, tol);
				break;
			}
		}

		//fo.close();

		delete [] r;
		delete [] p;
		delete [] q;
		delete [] s;

		//wxLogMessage(_T("ICCG iteration = %d, (tol = %f)(false)"), l, tol);
		max_iter = l;
		tol = tol_err;
		return l;
	}

	/*!
	 * 共役勾配(CG)ソルバ
	 * @param[in] poiss 元の行列
	 * @param[out] x 解を格納する
	 * @param[in] b 右辺項
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 * @param[in] neigh 近傍位置を表す配列
	 * @param[in] size_m 配列サイズ
	 */
	static int cgSolverP(double **poiss, double *x, double *b, int **neigh, int size_m, double dt, 
						  boost::function<void (double*)> bcfunc, int &max_iter, double &tol)
	{
		int i, k = 0;
		double tol2 = tol*tol;

		double *r, *p, *y;
		r = new double[size_m];
		p = new double[size_m];
		y = new double[size_m];

		// 第0近似解に対する残差の計算
		for(i = 0; i < size_m; ++i){
			x[i] = 0.0;
			r[i] = b[i];
			p[i] = b[i];
		}

		double resid, norm2_b;
		mulVectorP(norm2_b, b, b, neigh, size_m);		// b*b
		if(fabs(norm2_b) < RX_FEQ_EPS) norm2_b = 1.0;

		double rr0, rr1;
		mulVectorP(rr0, r, r, neigh, size_m);

		if((resid = rr0/norm2_b) <= tol2){
			tol = sqrt(resid);
			max_iter = 0;

			delete [] r;
			delete [] p;
			delete [] y;

			return 0;
		}

		double alpha0, beta0, py;
		for(k = 0; k < max_iter; k++){
			// y = AP の計算
			mulMatrixVectorP(y, poiss, p, neigh, size_m);	// s = poiss*p
			bcfunc(y);

			mulVectorP(py, p, y, neigh, size_m);			// py = p*y

			// alpha = r*r/(P*AP)の計算
			alpha0 = rr0/py;

			// 解x、残差rの更新
			for(i = 0; i < size_m; ++i){
				x[i] += alpha0*p[i];
				r[i] -= alpha0*y[i];
			}

			// r_(k+1)*r_(k+1)
			mulVectorP(rr1, r, r, neigh, size_m);

			if((resid = rr1/norm2_b) <= tol2){
				tol = sqrt(resid);
				max_iter = k+1;

				delete [] r;
				delete [] p;
				delete [] y;

				return 0;     
			}

			beta0 = rr1/rr0;
			for(i = 0; i < size_m; ++i){
				p[i] = r[i]+beta0*p[i];
			}

			rr0 = rr1;
		}

		delete [] r;
		delete [] p;
		delete [] y;

		tol = resid;
		max_iter = k;
		return k;
	}


	/*!
	 * vectorコンテナ同士の内積計算
	 * @param 
	 * @return 
	 */
	inline double DotProductV(const vector<double> &a, const vector<double> &b)
	{
		double d = 0.0f;
		for(int i = 0; i < (int)a.size(); ++i){
			d += a[i]*b[i];
		}
		return d;
	}

		
	/*!
	 * 共役勾配(CG)ソルバ
	 * @param[in] func y = APを計算する関数オブジェクト
	 * @param[in] b 右辺項
	 * @param[out] x 解を格納する
	 * @param[inout] max_iter 最大反復回数
	 * @param[inout] tol 許容誤差
	 */
	static int cgSolver(boost::function<void (vector<double>&, vector<double>)> func, 
						const vector<double> &b, vector<double> &x, int &max_iter, double &tol)
	{
		int k, idx;
		int size = (int)x.size();
		double tol2 = tol*tol;

		vector<double> r, p, y;
		r.resize(size);
		p.resize(size);
		y.resize(size);

		// 第0近似解に対する残差の計算
		for(idx = 0; idx < size ; ++idx){
			x[idx] = 0.0;
			r[idx] = b[idx];
			p[idx] = b[idx];
		}

		double resid;
		double norm2_b = DotProductV(b, b);
		if(fabs(norm2_b) < RX_FEQ_EPS) norm2_b = 1.0;
			
		double rr0 =DotProductV(r, r), rr1;
		if((resid = rr0/norm2_b) <= tol2){
			tol = sqrt(resid);
			max_iter = 0;

			return 0;
		}

		double alpha0, beta0;
		for(k = 0; k < max_iter; k++){
			// y = AP の計算
			func(y, p);

			// alpha = r*r/(P*AP)の計算
			alpha0 = rr0/DotProductV(p, y);

			// 解x、残差rの更新
			for(idx = 0; idx < size ; ++idx){
				x[idx] += alpha0*p[idx];
				r[idx] -= alpha0*y[idx];
			}

			// r_(k+1)*r_(k+1)
			rr1 = DotProductV(r, r);

			if((resid = rr1/norm2_b) <= tol2){
				tol = sqrt(resid);
				max_iter = k+1;

				return 0;     
			}

			beta0 = rr1/rr0;
			for(idx = 0; idx < size ; ++idx){
				p[idx] = r[idx]+beta0*p[idx];
			}

			rr0 = rr1;
		}

		tol = resid;

		return k;
	}



	/*!
	 * LU分解:行列a(n×n)をLU分解して返す
	 * @param[inout] a n×n正値対称行列
	 * @return 
	 */
	static bool LUDecomp(vector< vector<double> > &a, vector<int> &indx, double &d, const int &n)
	{
		int i, imax, j, k;
		float big, dum, sum, temp;
		vector<double> vv;

		vv.resize(n);
		d = 1.0;
		
		for(i = 0; i < n; i++){
			big = 0.0;
			for(j = 0; j < n; j++)
				if((temp = fabs(a[i][j])) > big) big = temp;

			if(big  ==  0.0) return false;
			
			vv[i] = 1.0f/big;
		}

		for(j = 0; j < n; j++){
			for(i = 0; i < j; i++){
				sum = a[i][j];
				for(k = 0; k < i; k++) sum -= a[i][k]*a[k][j];
				a[i][j] = sum;
			}

			big = 0.0;
			for(i = j; i < n; i++){
				sum = a[i][j];
				for(k = 0; k < j; k++) sum -= a[i][k]*a[k][j];
				a[i][j] = sum;
				if((dum = vv[i]*fabs(sum)) >= big){
					big = dum;
					imax = i;
				}
			}

			if(j != imax){
				for(k = 0; k < n; k++){
					dum = a[imax][k];
					a[imax][k] = a[j][k];
					a[j][k] = dum;
				}
				d = -d;
				vv[imax] = vv[j];
			}

			indx[j] = imax;
			if(a[j][j] == 0.0) a[j][j] = (double)RX_FEQ_EPS;
			if(j != n-1){
				dum = 1.0f/(a[j][j]);
				for(i = j+1; i < n; i++) a[i][j] *= dum;
			}
		}

		return true;
	}

	/*!
	 * LU分解した行列a(n×n)から前進代入・後退代入によりA・x=bを解く
	 * @param[in] a LU分解された行列
	 * @param[in] indx LUDecompルーチンが吐き出したピボット選択による行交換の履歴
	 * @param[inout] b   右辺ベクトル
	 * @param[in] n 
	 * @return 
	 */
	static bool LUbksb(const vector< vector<double> > &a, const vector<int> &indx, vector<double> &b, const int &n)
	{
		int i, ii=0, ip, j;
		double sum;

		// 前進代入
		for(i = 0; i < n; ++i){
			ip = indx[i];
			sum = b[ip];
			b[ip] = b[i];
			if(ii != 0){
				for(j = ii-1; j < i; ++j){
					sum -= a[i][j]*b[j];
				}
			}
			else if(sum != 0.0){
				ii = i+1;
			}

			b[i] = sum;
		}

		// 後退代入
		for(i = n-1; i >= 0; --i){
			sum = b[i];
			for(j = i+1; j < n; ++j){
				sum -= a[i][j]*b[j];
			}

			b[i] = sum/a[i][i];
		}

		return true;
	}

	/*!
	 * LU分解で逆行列を計算
	 * @param[in]  mat 計算したい行列
	 * @param[out] inv 逆行列
	 * @return 行列の大きさが0のときにfalseを返す
	 */
	static bool CalInverseMatrixByLU(vector< vector<double> > &mat, vector< vector<double> > &inv)
	{
		int n = (int)mat.size();

		if(n <= 0) return false;

		double d;
		vector<int> indx;
		indx.resize(n);

		// 行列を1回だけLU分解
		LUDecomp(mat, indx, d, n);


		vector<double> col;
		col.resize(n);

		// 行ごとに逆行列を算出
		for(int j = 0; j < n; ++j){
			for(int i = 0; i < n; ++i){
				col[i] = 0.0;
			}
			col[j] = 1.0;

			LUbksb(mat, indx, col, n);
			for(int i = 0; i < n; ++i){
				inv[i][j] = col[i];
			}
		}

		return true;
	}


	/*!
	 * 2次元vectorコンテナに格納された行列同士の掛け算
	 * @param[in] a,b 掛け算する行列(lxm, mxn)
	 * @param[out] y  結果の行列(lxn)
	 * @return 行列のサイズがおかしい時
	 */
	inline bool MulMatrix(const vector< vector<double> > &a, const vector< vector<double> > &b, vector< vector<double> > &y)
	{
		if((int)a.size() == 0 || a[0].size() != b.size() || (int)b.size() == 0) return false;

		int l, m, n;
		l = (int)a.size();
		m = (int)a[0].size();
		n = (int)b[0].size();

		for(int i = 0; i < l; ++i){
			for(int j = 0; j < n; ++j){
				y[i][j] = 0.0;

				for(int k = 0; k < m; ++k){
					y[i][j] += a[i][k]*b[k][j];
				}
			}
		}

		return true;
	}
	/*!
	 * 2次元vectorコンテナに格納された行列同士の掛け算
	 * @param[in] a,b 掛け算する行列とベクトル(lxm, mx1)
	 * @param[out] y  結果のベクトル(mx1)
	 * @return 行列のサイズがおかしい時
	 */
	inline bool MulMatrixVector(const vector< vector<double> > &a, const vector<double> &b, vector<double> &y)
	{
		if((int)a.size() == 0 || a[0].size() != b.size() || (int)b.size() == 0) return false;

		int l, m;
		l = (int)a.size();
		m = (int)a[0].size();

		for(int i = 0; i < l; ++i){
			y[i] = 0.0;
			for(int k = 0; k < m; ++k){
				y[i] += a[i][k]*b[k];
			}
		}

		return true;
	}

	// ガウス・ザイデル法による連立方程式ax = bの計算
	// a : n×n正値対称行列
	// b : n次元定数ベクトル
	// x : 引数として初期値を入れておき、ルーチン内で解を代入
	// max_iter : 最大反復回数
	static bool GaussSeidel(vector< vector<double> > &A, vector<double> &b, vector<double> &x, int n, int max_iter = 1000)
	{
		vector<double> x_old;
		x_old.resize(n);

		double denom, dev, devMax;
		int i, j, iter = 0;
		  
		// 行列AとベクトルBの正規化
		for(i = 0; i < n; i++){
			denom = A[i][i];
			if(denom < RX_FEQ_EPS) return false;

			b[i] /= denom;
			for(j = 0; j < n; j++) A[i][j] /= denom;
		}

		// ガウス・ザイデル反復の実行
		while(true){
			for(i = 0; i < n; i++){
				x_old[i] = x[i];
				x[i] = 0;
				for(j = 0; j < n; j++){
					if(j != i){
						x[i] -= A[i][j]*x[j];
					}
				}
				x[i] += b[i];
			}

			// 収束判定
			devMax = fabs(x_old[0]-x[0])/x[0];
			for(i = 1; i < n; i++){
				dev = fabs(x_old[i]-x[i])/x[i];
				devMax = (dev > devMax) ? dev : devMax;
			}

			if(devMax <= (double)RX_FEQ_EPS){
				return true;
			}
			else{
				iter++;
				if(iter > max_iter)
					return false;
			}
		}
	}

	// 3重対角行列解法(TDMA:Tri-Diagonal Matrix Algorithm)
	// a[i]T[i]=b[i]T[i+1]+c[i]T[i-1]+d[i]
	static void TDMA(const int &n, float *T, float *a, float *b, float *c, float *d)
	{
		int i;
		float *p = new float[n];
		float *q = new float[n];

		p[0] = b[0]/a[0];
		q[0] = d[0]/a[0];
		// 漸化式の係数の計算
		for(i = 1; i < n; ++i){
			float term = 1.0f/(a[i]-c[i]*p[i-1]);
			p[i] = b[i]*term;
			q[i] = (d[i]+c[i]*q[i-1])*term;
		}

		T[n-1] = q[n-1];
		for(i = n-2; i >= 0; --i){
			T[i] = p[i]*T[i+1]+q[i];
		}

		delete[] p;
		delete[] q;
	}

	// 3重対角行列解法(TDMA:Tri-Diagonal Matrix Algorithm)
	// a[i]T[i]=b[i]T[i+1]+c[i]T[i-1]+d[i]
	static void TDMA(const int &n, vector<float> T,
					const vector<float> &a, const vector<float> &b, const vector<float> &c, const vector<float> &d)
	{
		int i;
		vector<float> p, q;
		p.resize(n);
		q.resize(n);

		p[0] = b[0]/a[0];
		q[0] = d[0]/a[0];
		// 漸化式の係数の計算
		for(i = 1; i < n; ++i){
			float term = 1.0f/(a[i]-c[i]*p[i-1]);
			p[i] = b[i]*term;
			q[i] = (d[i]+c[i]*q[i-1])*term;
		}

		T[n-1] = q[n-1];
		for(i = n-2; i >= 0; --i){
			T[i] = p[i]*T[i+1]+q[i];
		}
	}

	// Poisson方程式の共役勾配法による解法
	// ▽^2 p = b を計算し、結果をxに格納する。
	static int PoissonCG(const vector< vector<float> > &b, vector< vector<float> > &x, const int &nx, const int &ny)
	{
		int i, j, k;
		vector< vector<float> > r, p, y;
		r.resize(nx);
		for(i = 0; i < nx; ++i) r[i].resize(ny);
		p.resize(nx);
		for(i = 0; i < nx; ++i) p[i].resize(ny);
		y.resize(nx);
		for(i = 0; i < nx; ++i) y[i].resize(ny);

		// 第0近似解に対する残差の計算
		for(i = 0; i < nx; ++i){
			for(j = 0; j < ny; ++j){
				x[i][j] = 0.0;
				r[i][j] = b[i][j];
				p[i][j] = b[i][j];
			}
		}

		float rr0 = DotProduct(r, r, nx, ny), rr1;
		float alpha, beta;
		k = 0;
		while(k < 1000){

			// y = AP の計算
			for(i = 0; i < nx; ++i){
				for(j = 0; j < ny; ++j){
					y[i][j] = -4.0f*p[i][j];
					if(i != 0) y[i][j] += p[i-1][j];
					if(j != 0) y[i][j] += p[i][j-1];
					if(i != nx-1) y[i][j] += p[i+1][j];
					if(j != ny-1) y[i][j] += p[i][j+1];
				}
			}

			// alpha = r*r/(P*AP)の計算
			alpha = rr0/DotProduct(p, y, nx, ny);

			// 解x、残差rの更新
			for(i = 0; i < nx; ++i){
				for(j = 0; j < ny; ++j){
					x[i][j] += alpha*p[i][j];
					r[i][j] -= alpha*y[i][j];
				}
			}

			// r_(k+1)*r_(k+1)
			rr1 = DotProduct(r, r, nx, ny);

			if(sqrt(rr1) < RX_EPS) break;

			beta = rr1/rr0;
			for(i = 0; i < nx; ++i){
				for(j = 0; j < ny; ++j){
					p[i][j] = r[i][j]+beta*p[i][j];
				}
			}

			rr0 = rr1;

			k++;
		}

		return k;

	}

	inline int getIndex(const int &i, const int &j, const int &nx){ return i+(nx+2)*j; }

	// Poisson方程式の共役勾配法による解法
	// ▽^2 p = b を計算し、結果をxに格納する。
	static int PoissonCGForFluid2D(const vector<float> &b, vector<float> &x, const int &nx, const int &ny, const float &a)
	{
		int i, j, k, index;
		int size = (nx+2)*(ny+2);

		vector<float> r, p, y;
		r.resize(size);
		p.resize(size);
		y.resize(size);

		// 第0近似解に対する残差の計算
		for(i = 1; i <= nx; ++i){
			for(j = 1; j <= ny; ++j){
				index = getIndex(i, j, nx);
				x[index] = 0.0;
				r[index] = b[index];
				p[index] = b[index];
			}
		}

		float rr0 = DotProduct(r, r), rr1;
		float alpha, beta;
		k = 0;
		while(k < 1000){

			// y = AP の計算
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					index = getIndex(i, j, nx);

					y[index] = a*(p[getIndex(i-1, j, nx)]+p[getIndex(i, j-1, nx)]
					            +p[getIndex(i+1, j, nx)]+p[getIndex(i, j+1, nx)]-4.0f*p[index]);
				}
			}

			// alpha = r*r/(P*AP)の計算
			alpha = rr0/DotProduct(p, y);

			// 解x、残差rの更新
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					index = getIndex(i, j, nx);

					x[index] += alpha*p[index];
					r[index] -= alpha*y[index];
				}
			}

			// r_(k+1)*r_(k+1)
			rr1 = DotProduct(r, r);

			if(sqrt(rr1) < RX_EPS) break;

			beta = rr1/rr0;
			for(i = 1; i <= nx; ++i){
				for(j = 1; j <= ny; ++j){
					index = getIndex(i, j, nx);

					p[index] = r[index]+beta*p[index];
				}
			}

			rr0 = rr1;

			k++;
		}

		return k;

	}

	// 共役勾配法による連立方程式ax = bの計算
	// a : n×n正値対称行列
	// b : n次元定数ベクトル
	// x : 引数として初期値を入れておき、ルーチン内で解を代入
	static int CG(const vector< vector<float> > &a, const vector<float> &b, const int &n, vector<float> &x)
	{
		int i, j, k;
		vector<float> r, p, y;
		r.resize(n);
		p.resize(n);
		y.resize(n);

		// 第0近似解に対する残差の計算
		float ax;
		for(i = 0; i < n; ++i){
			ax = 0.0f;
			for(j = 0; j < n; ++j){
				ax += a[i][j]*x[j];
			}
			r[i] = b[i]-ax;
			p[i] = r[i];
		}

		float rr0 = DotProduct(r, r), rr1;
		float alpha, beta;
		k = 0;
		while(k < 1000){

			// y = AP の計算
			for(i = 0; i < n; ++i){
				y[i] = DotProduct(a[i], p);
			}

			// alpha = r*r/(P*AP)の計算
			alpha = rr0/DotProduct(p, y);

			// 解x、残差rの更新
			for(i = 0; i < n; ++i){
				x[i] += alpha*p[i];
				r[i] -= alpha*y[i];
			}

			rr1 = DotProduct(r, r);

			if(sqrt(rr1) < RX_EPS) break;

			beta = rr1/rr0;
			for(i = 0; i < n; ++i){
				p[i] = r[i]+beta*p[i];
			}

			rr0 = rr1;

			k++;
		}

		return k;
	}

	// 4次ルンゲクッタによる連立一階常微分方程式の数値解法
	//  Numerical Recipes in C 日本語版 p533
	static void rk4(const vector<double> &y, const double &x, const double &h, vector<double> &y_out, 
					void derivs(const double x, const vector<double> &y, vector<double> &dydx))
	{
		int i;
		double xh, hh, h6;

		int n = (int)y.size();
		vector<double> dym, dyt, yt, dydx;
		
		dym.resize(n);
		dyt.resize(n);
		yt.resize(n);
		dydx.resize(n);

		hh = h*0.5;
		h6 = h/6.0;
		xh = x+hh;

		derivs(x, y, dydx);			// k1の計算
		for(i = 0; i < n; ++i){		// yn+k1/2の計算
			yt[i] = y[i]+hh*dydx[i];
		}

		derivs(xh, yt, dyt);		// k2の計算
		for(i = 0; i < n; ++i){
			yt[i] = y[i]+hh*dyt[i];	// yn+k2/2の計算
		}

		derivs(xh, yt, dym);		// k3の計算
		for(i = 0; i < n; ++i){
			yt[i] = y[i]+h*dym[i];	// yn+k3の計算
			dym[i] += dyt[i];		// k2+k3
		}

		derivs(x+h, yt, dyt);		// k4の計算
		for(i = 0; i < n; ++i){
			y_out[i] = y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
		}
	}

	// 2次ルンゲクッタ(中点法)による連立一階常微分方程式の数値解法
	//  Numerical Recipes in C 日本語版 p531
	static void rk2(const vector<double> &y, const double &x, const double &h, vector<double> &y_out, 
					void derivs(const double x, const vector<double> &y, vector<double> &dydx))
	{
		int i;
		double hh;

		int n = (int)y.size();
		vector<double> dyt, yt, dydx;
		
		dyt.resize(n);
		yt.resize(n);
		dydx.resize(n);

		hh = h*0.5;

		derivs(x, y, dydx);			// k1の計算
		for(i = 0; i < n; ++i){		// yn+k1/2の計算
			yt[i] = y[i]+hh*dydx[i];
		}

		derivs(x+hh, yt, dyt);		// k2の計算
		for(i = 0; i < n; ++i){
			y_out[i] = y[i]+hh*dyt[i];
		}
	}

	inline void rot(vector< vector<double> > &a, const double s, const double tau, const int i, const int j, const int k, const int l)
	{
		double g,h;

		g = a[i][j];
		h = a[k][l];
		a[i][j] = g-s*(h+g*tau);
		a[k][l] = h+s*(g-h*tau);
	}

	// 対称行列の固有値と固有ベクトル
	//  Numerical Recipes in C++ Chapter11.1
	//  a : n×nの対称行列
	//  d, v : 固有値と固有ベクトルを格納(返値)
	//  nrot, max_iter : JACOBI Rotationの回数，最大反復回数
	static bool eigenJacobi(vector< vector<double> > &a, int n, vector<double> &d, vector< vector<double> > &v, int &nrot, int &max_iter)
	{
		int i, j, iq, ip;
		double tresh, theta ,tau, t, sm, s, h, g, c;
		vector<double> b, z;

		b.resize(n);
		z.resize(n);

		// vの初期化
		for(ip = 0; ip < n; ++ip){
			for(iq = 0; iq < n; ++iq){
				v[ip][iq] = 0.0;
			}
			v[ip][ip] = 1.0;
		}

		// b, z, dの初期化
		for(ip = 0; ip < n; ++ip){
			b[ip] = d[ip] = a[ip][ip];
			z[ip] = 0.0;
		}

		nrot = 0;
		for(i = 0; i < max_iter; ++i){
			sm = 0.0;
			for(ip = 0; ip < n-1; ++ip){
				for(iq = ip+1; iq < n; ++iq){
					sm += fabs(a[ip][iq]);
				}
			}
			if(sm == 0.0){
				max_iter = i;
				return true;
			}

			if(i < 4){
				tresh = 0.2*sm/(n*n);
			}
			else{
				tresh = 0.0;
			}

			for(ip = 0; ip < n-1; ++ip){
				for(iq = ip+1; iq < n; ++iq){
					g = 100.0*fabs(a[ip][iq]);
					if(i > 4 && fabs(d[ip])+g == fabs(d[ip]) && fabs(d[iq])+g == fabs(d[iq])){
						a[ip][iq]=0.0;
					}
					else if(fabs(a[ip][iq]) > tresh){
						h = d[iq]-d[ip];
						if(fabs(h)+g == fabs(h)){
							t = (a[ip][iq])/h;
						}
						else{
							theta = 0.5*h/(a[ip][iq]);
							t = 1.0/(fabs(theta)+sqrt(1.0+theta*theta));
							if(theta < 0.0) t = -t;
						}

						c = 1.0/sqrt(1.0+t*t);
						s = t*c;
						tau = s/(1.0+c);
						h = t*a[ip][iq];

						z[ip] -= h;
						z[iq] += h;
						d[ip] -= h;
						d[iq] += h;

						a[ip][iq] = 0.0;
						for(j = 0;j < ip; ++j){
							rot(a, s, tau, j, ip, j, iq);
						}
						for(j = ip+1; j < iq; ++j){
							rot(a, s, tau, ip, j, j, iq);
						}
						for(j = iq+1; j < n; ++j){
							rot(a, s, tau, ip, j, iq, j);
						}
						for(j = 0; j < n; ++j){
							rot(v, s, tau, j, ip, j, iq);
						}
						++nrot;
					}
				}
			}
			for(ip = 0; ip < n; ++ip){
				b[ip] += z[ip];
				d[ip] = b[ip];
				z[ip] = 0.0;
			}
		}
		
		return false;
	}

	static double Newton1D(boost::function<void (const double, double&, double&)> funcd, 
						   const double &x1, const double &x2, const double xacc)
	{
		const int JMAX = 20;
		double df, dx, f, rtn;

		rtn= 0.5*(x1+x2);
		for(int j = 0; j < JMAX; ++j){
			funcd(rtn, f, df);
			dx = f/df;
			rtn -= dx;

			if((x1-rtn)*(rtn-x2) < 0.0){
				return 0.0;
			}

			if(fabs(dx) < xacc){
				return rtn;
			}
		}
		
		return 0.0;
	}

} // namespace numerical

#endif // _RX_NUMERICAL_H_