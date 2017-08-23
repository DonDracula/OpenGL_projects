/*!
  @file rx_particle_on_surf.cpp
	
  @brief 陰関数表面にパーティクルを配置

  @ref A. Witkin and P. Heckbert, "Using particles to sample and control implicit surfaces", SIGGRAPH1994.
 
  @author Makoto Fujisawa
  @date   2013-06
*/


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_particle_on_surf.h"


//-----------------------------------------------------------------------------
// rxParticleOnSurfクラスの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 */
rxParticleOnSurf::rxParticleOnSurf()
{
	m_uNumParticles = 0;

	// 近傍探索セル
	m_pNNGrid = new rxNNGrid(DIM);

	m_fAlpha = 6.0;
	m_fEh = 0.8*m_fAlpha;
	m_fRho = 15.0;
	m_fPhi = 15.0;
	m_fBeta = 10.0;

	m_fGamma = 4.0;
	m_fNu = 0.2;
	m_fDelta = 0.7;

	m_pFuncPtr = 0;
}

/*!
 * パーティクル
 * @param[in] vrts パーティクル初期位置
 * @param[in] rad パーティクル半径
 */
void rxParticleOnSurf::Initialize(const vector<Vec3> &vrts, double rad, Vec3 minp, Vec3 maxp, 
								   Vec4 (*func)(void*, double, double, double), void* func_ptr)
{
	m_fParticleRadius = rad;
	m_fEffectiveRadius = 3.0*m_fParticleRadius;
	m_fSh = m_fParticleRadius;
	m_fSmax = 1.5*m_fSh;

	m_uNumParticles = (uint)vrts.size();

	m_fpFunc = func;
	m_pFuncPtr = func_ptr;

	// メモリ確保
	m_vPos.resize(m_uNumParticles*DIM, 0.0);
	m_vPs.resize(m_uNumParticles);


	// 初期パーティクル位置
	for(uint i = 0; i < m_uNumParticles; ++i){
		for(int j = 0; j < 3; ++j){
			m_vPos[DIM*i+j]= vrts[i][j];
		}
		m_vPs[i].Sigma = m_fSh;
	}

	// 分割セル設定
	m_pNNGrid->Setup(minp, maxp, m_fEffectiveRadius, m_uNumParticles);
	m_vNeighs.resize(m_uNumParticles);

}

/*!
 * 確保したメモリの解放
 */
void rxParticleOnSurf::Finalize(void)
{
	m_vPos.clear();
	m_vPs.clear();
}

/*!
 * repulsion radius σ の更新
 *  - "Using particles to sample and control implicit surfaces"の式(10)～(13)
 * @param[in] dt 更新用時間ステップ幅
 */
void rxParticleOnSurf::updateSigma(double dt)
{
	RXREAL h = m_fEffectiveRadius;

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0(m_vPos[DIM*i+0], m_vPos[DIM*i+1], m_vPos[DIM*i+2]);
		RXREAL si = m_vPs[i].Sigma;

		double D = 0.0, Ds = 0.0;
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1(m_vPos[DIM*j+0], m_vPos[DIM*j+1], m_vPos[DIM*j+2]);

			Vec3 rij = pos1-pos0;

			RXREAL r = norm(rij);

			if(r <= h){
				double Eij = m_fAlpha*exp(-r*r/(2.0*si*si));
				D += Eij;			// 現在の反発エネルギ(式(10)の上)
				Ds += r*r*Eij;		// エネルギDのσによる微分(式(13))
			}
		}
		Ds /= si*si*si;

		double Dv = -m_fRho*(D-m_fEh);// ターゲットエネルギに近づけるための線形フィードバック(式(10))
		double sv = Dv/(Ds+m_fBeta);	// σの変化量(式(12))

		// σの更新
		m_vPs[i].Sigma += sv*dt;
	}
}

/*!
 * 反発による速度の計算
 *  - 適応的なパーティクルの追加/削除のためのσの更新を含むバージョン
 *  - "Using particles to sample and control implicit surfaces"の4.3節，式(9)
 * @param[in] dt 更新用時間ステップ幅
 */
void rxParticleOnSurf::applyRepulsion2(double dt)
{
	RXREAL h = m_fEffectiveRadius;

	// 近傍探索セルへパーティクルを格納 & 近傍探索
	SetParticlesToCell();

	// repulsion radius の更新
	updateSigma(dt);

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0(m_vPos[DIM*i+0], m_vPos[DIM*i+1], m_vPos[DIM*i+2]);
		RXREAL si = m_vPs[i].Sigma;

		Vec3 Ep(0.0);
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1(m_vPos[DIM*j+0], m_vPos[DIM*j+1], m_vPos[DIM*j+2]);

			Vec3 rij = pos1-pos0;

			RXREAL r = norm(rij);

			if(r <= h){
				double Eij = m_fAlpha*exp(-r*r/(2.0*si*si));
	
				RXREAL sj = m_vPs[j].Sigma;
				double Eji = m_fAlpha*exp(-r*r/(2.0*sj*sj));
				Vec3 rji = pos0-pos1;

				Ep += (rij/(si*si))*Eij-(rji/(sj*sj))*Eji;	// 式(9)
			}
		}
		Ep *= si*si;

		m_vPs[i].Ep = Ep;
	}
}

/*!
 * 反発による速度の計算
 *  - σの変更を含まないシンプルなバージョン
 *  - "Using particles to sample and control implicit surfaces"の4.1節
 * @param[in] dt 更新用時間ステップ幅
 */
void rxParticleOnSurf::applyRepulsion(double dt)
{
	RXREAL h = m_fEffectiveRadius;

	// 近傍探索セルへパーティクルを格納 & 近傍探索
	SetParticlesToCell();

	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 pos0(m_vPos[DIM*i+0], m_vPos[DIM*i+1], m_vPos[DIM*i+2]);
		RXREAL si = m_vPs[i].Sigma;

		Vec3 Ep(0.0);
		for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
			int j = itr->Idx;
			if(j < 0 || i == j) continue;

			Vec3 pos1(m_vPos[DIM*j+0], m_vPos[DIM*j+1], m_vPos[DIM*j+2]);

			Vec3 rij = pos1-pos0;

			RXREAL r = norm(rij);

			if(r <= h){
				double Eij = m_fAlpha*exp(-r*r/(2.0*si*si));	// 反発エネルギ
				Ep += rij*Eij;
			}
		}

		m_vPs[i].Ep = Ep;
	}
}

/*!
 * パーティクル位置更新
 *  - パーティクルが陰関数曲面上に載るように反発による速度を修正して，位置を更新
 * @param[in] dt 更新用時間ステップ幅
 * @param[out] v_avg 移動量の2乗平均値
 */
void rxParticleOnSurf::applyFloating(double dt, double &v_avg)
{
	v_avg = 0.0;
	if(!m_uNumParticles) return;

	RXREAL h = m_fEffectiveRadius;
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 p(m_vPos[DIM*i+0], m_vPos[DIM*i+1], m_vPos[DIM*i+2]);	// 現在のパーティクル座標
		Vec3 ve = m_vPs[i].Ep;	// 反発による速度

		Vec4 fv = m_fpFunc(m_pFuncPtr, p[0], p[1], p[2]);	// パーティクル座標における陰関数値とその勾配を取得
		Vec3 fx(fv[0], fv[1], fv[2]);	// 勾配
		RXREAL f = fv[3];				// 陰関数値

		// パーティクル移流速度
		Vec3 v = ve-((dot(fx, ve)+m_fPhi*f)/(dot(fx, fx)))*fx;

		m_vPos[DIM*i+0] -= v[0]*dt;
		m_vPos[DIM*i+1] -= v[1]*dt;
		m_vPos[DIM*i+2] -= v[2]*dt;

		m_vPs[i].Vel = v;

		v_avg += norm2(v*dt);
	}
	v_avg /= m_uNumParticles;
}

/*!
 * パーティクルの分裂と削除判定
 * @param[in] dt 更新用時間ステップ幅
 */
void rxParticleOnSurf::testFissionDeath(void)
{
	if(!m_uNumParticles) return;

	// 近傍探索セルへパーティクルを格納 & 近傍探索
	SetParticlesToCell();

	int num_remove = 0, num_fission = 0;
	RXREAL h = m_fEffectiveRadius;
	for(uint i = 0; i < m_uNumParticles; ++i){
		Vec3 p(m_vPos[DIM*i+0], m_vPos[DIM*i+1], m_vPos[DIM*i+2]);	// パーティクル座標
		Vec3 v = m_vPs[i].Vel;	// パーティクル速度
		RXREAL si = m_vPs[i].Sigma;	// 反発半径σ

		RXREAL vp = norm(v);

		// パーティクルが平衡状態に近いかどうか
		if(vp < m_fGamma*si){
			// 反発エネルギDの計算
			double D = 0.0;
			for(vector<rxNeigh>::iterator itr = m_vNeighs[i].begin() ; itr != m_vNeighs[i].end(); ++itr){
				int j = itr->Idx;
				if(j < 0 || i == j) continue;

				Vec3 pos1(m_vPos[DIM*j+0], m_vPos[DIM*j+1], m_vPos[DIM*j+2]);
				Vec3 rij = pos1-p;
				RXREAL r = norm(rij);
				if(r <= h){
					double Eij = m_fAlpha*exp(-r*r/(2.0*si*si));
					D += Eij;			// 反発エネルギ
				}
			}
			RXREAL R = RXFunc::Frand();	// [0,1]の乱数

			// σが大きすぎる or エネルギーが適切でσが規定値以上 → 分裂(パーティクル追加)
			if((si > m_fSmax) || (D > m_fNu*m_fEh && si > m_fSh)){
				m_vPs[i].Flag = 1;	// 追加フラグをON
				num_fission++;
			}
			// σが小さすぎる and 乱数を使ったテストを通った → 削除
			else if((si < m_fDelta*m_fSh) && (R > si/(m_fDelta*m_fSh))){
				m_vPs[i].Flag = 2;	// 削除フラグをON
				num_remove++;
			}

		}

		// 表面から離れすぎたパーティクルも削除
		Vec4 fv = m_fpFunc(m_pFuncPtr, p[0], p[1], p[2]);	// パーティクル座標における陰関数値とその勾配を取得
		RXREAL f = fv[3];				// 陰関数値
		if(fabs(f) > 2.0*m_fSmax){
			m_vPs[i].Flag = 2;	// 削除フラグをON
			num_remove++;
		}
	}

	// パーティクル削除
	if(num_remove){
		int cnt = 0;
		vector<rxSurfParticle>::iterator itr = m_vPs.begin();
		vector<RXREAL>::iterator jtr = m_vPos.begin();
		while(itr != m_vPs.end()){
			if(itr->Flag == 2){
				itr = m_vPs.erase(itr);
				jtr = m_vPos.erase(jtr, jtr+DIM);
				cnt++;
			}
			else{
				++itr;
				jtr += DIM;
			}
		}
		m_uNumParticles = (int)m_vPs.size();
		//cout << cnt << " particles are removed." << endl;
	}
}

/*!
 * パーティクル位置更新
 * @param[in] dt 更新用時間ステップ幅
 * @param[inout] num_iter 最大反復回数+実際の反復回数
 * @param[inout] eps 修飾判定用移動許容量+誤差
 */
void rxParticleOnSurf::Update(double dt, int &num_iter, RXREAL &eps)
{
	double v_avg;
	int k;
	for(k = 0; k < num_iter; ++k){
		// 反発による速度の計算
		applyRepulsion2(dt);

		// パーティクル位置の更新
		applyFloating(dt, v_avg);

		// パーティクル追加/削除
		testFissionDeath();

		v_avg = sqrt(v_avg);
		if(v_avg < eps) break;
	}

	eps = v_avg;
	num_iter = k;
}


//-----------------------------------------------------------------------------
// 近傍探索
//-----------------------------------------------------------------------------
/*!
 * 近傍粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[in] prts パーティクル位置
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
void rxParticleOnSurf::GetNearestNeighbors(int idx, RXREAL *prts, vector<rxNeigh> &neighs, RXREAL h)
{
	if(idx < 0 || idx >= (int)m_uNumParticles) return;

	Vec3 pos;
	pos[0] = prts[DIM*idx+0];
	pos[1] = prts[DIM*idx+1];
	pos[2] = prts[DIM*idx+2];

	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGrid->GetNN(pos, prts, m_uNumParticles, neighs, h);
}

/*!
 * 近傍粒子探索
 * @param[in] idx 探索中心パーティクルインデックス
 * @param[out] neighs 探索結果格納する近傍情報コンテナ
 * @param[in] h 有効半径
 */
void rxParticleOnSurf::GetNearestNeighbors(Vec3 pos, vector<rxNeigh> &neighs, RXREAL h)
{
	if(h < 0.0) h = m_fEffectiveRadius;

	m_pNNGrid->GetNN(pos, &m_vPos[0], m_uNumParticles, neighs, h);
}


/*!
 * 全パーティクルを分割セルに格納
 */
void rxParticleOnSurf::SetParticlesToCell(RXREAL *prts, int n, RXREAL h)
{
	// 分割セルに粒子を登録
	m_pNNGrid->SetObjectToCell(prts, n);

	// 近傍粒子探索
	if(h < 0.0) h = m_fEffectiveRadius;
	for(int i = 0; i < (int)m_uNumParticles; i++){
		m_vNeighs[i].clear();
		GetNearestNeighbors(i, prts, m_vNeighs[i], h);
	}
}
void rxParticleOnSurf::SetParticlesToCell(void)
{
	SetParticlesToCell(&m_vPos[0], m_uNumParticles, m_fEffectiveRadius);
}