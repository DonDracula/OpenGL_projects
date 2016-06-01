//-----------------------------------------------------------------------------------
// マクロ-------------string
//-----------------------------------------------------------------------------------
//凧糸関連
#define LINE_SP 40							//凧糸の分割数
#define LINE_K 10.0*((double)LINE_SP)		//ばね定数
#define LINE_D 1.0							//減衰係数
#define LINE_RHO 0.2						//質量密度
#define LINE_E 0.02							//抵抗係数
#define SEG_LENGTH 0.1f						//the initial length of two particles

#define RX_GOUND_HEIGHT  0.0		//ground height

//-----------------------------------------------------------------------------------
// マクロ----------------kite
//-----------------------------------------------------------------------------------
//凧関連



#define COL_NUM 13							//凧の横方向分割点数->(4の倍数+1)
#define ROW_NUM 13							//凧の縦方向分割点数

#define RHO 1.21							//空気密度(1.2kg/m^3)
#define G_ACCE -9.8							//重力加速度

#define KITE_RHO 0.2						//凧の密度? 基本的には扱いやすいよう200g/1m^2

//テーブル関係
#define TABLE_S 0.68						//アスペクト比
#define TABLE_L 1.48						//アスペクト比

//しっぽ関連
#define TAIL_NUM 2							//しっぽの数

#define TAIL_SP 10							//しっぽの分割数
#define TAIL_K 10.0*((double)TAIL_SP)		//ばね定数
#define TAIL_D 1.0							//減衰係数
#define TAIL_RHO 0.2						//質量密度
#define TAIL_E 0.02							//抵抗係数

//たわみ関連
#define BAR_SP (COL_NUM-1)					//骨の分割数
#define BAR_L 1.0							//骨の長さ
#define BAR_WIDTH 0.016						//骨断面の幅(正方形断面の場合,高さも兼ねる)
											//(およそ0.02mほどからはっきり変化がわかる)
#define INV_BAMBOO_E pow(10.0,-9.0)			//竹のヤング率の逆数

//-----------------------------------------------------------------------------------
// マクロ----fluid
//-----------------------------------------------------------------------------------
#define IX(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*(k))//!< 3次元の格納
#define SWAP(x0,x) {double * tmp=x0;x0=x;x=tmp;}//!< 配列の入れ替え

#define GRID 16			//!< 1次元毎のグリッド分割数
#define F_FORCE -1.0	//!< 外力
#define STEP 0.01		//!< タイムステップ幅
#define VISC 0.0007		//!< 動粘性係数
#define DIFF 0.0		//!< 拡散係数


