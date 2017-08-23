/*!
  @file rx_kernel.h
	
  @brief SPH法のカーネル計算
 
  @author Makoto Fujisawa
  @date 2012-12
*/

#ifndef _RX_SPH_KERNEL_H_
#define _RX_SPH_KERNEL_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_utility.h"

//-----------------------------------------------------------------------------
// Poly6カーネル
//-----------------------------------------------------------------------------
/*!
 * カーネル係数
 * @param[in] h 有効半径
 * @param[in] d 次元(1,2,3)
 * @param[in] type ノーマル:1，勾配:2，ラプラシアン:3
 * @return カーネル係数値
 */
inline static double KernelCoefPoly6(double h, int d, int type)
{
	double a = 1.0;
	if(d < 1) d = 1;
	if(d > 3) d = 3;
	switch(type){
	case 1:	// ノーマル
		switch(d){
		case 2: a = 4.0/(RX_PI*pow((double)h, (double)8.0));			break;
		case 3:	a = 315.0/(64.0*RX_PI*pow((double)h, (double)9.0));		break;
		}
		break;

	case 2:	// 勾配
		switch(d){
		case 2:	a = -24.0/(RX_PI*pow((double)h, (double)8.0));			break;
		case 3:	a = -945.0/(32.0*RX_PI*pow((double)h, (double)9.0));	break;
		}
		break;

	case 3:	// ラプラシアン
		switch(d){
		case 2: a = -24.0/(RX_PI*pow((double)h, (double)8.0));			break;
		case 3: a = -945.0/(32.0*RX_PI*pow((double)h, (double)9.0));	break;
		}
		break;

	default:
		break;
	}

	return a;
}


/*!
 * Poly6カーネル関数値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @return 関数値
 */
inline double KernelPoly6(double r, double h, double a)
{
	if(r >= 0.0 && r <= h){
		double q = h*h-r*r;
		return a*q*q*q;
	}
	else{
		return 0.0;
	}
}

/*!
 * Poly6カーネル関数勾配値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] rij 相対位置ベクトル
 * @param[in] a カーネル係数
 * @return 勾配値
 */
template<class T> 
inline T KernelPoly6G(double r, double h, double a, T rij)
{
	if(r >= 0.0 && r <= h){
		double q = h*h-r*r;
		return  a*q*q*rij;
	}
	else{
		return T(0.0);
	}
}
 
/*!
 * Splineカーネル関数ラプラシアンの計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] d 次元(1,2,3)
 * @return ラプラシアンの値
 */
inline double KernelPoly6L(double r, double h, double a, double d)
{
	if(r >= 0.0 && r <= h){
		double q = h*h-r*r;
		return a*(3.0*q*q-4.0*r*r*q);
	}
	else{
		return 0.0;
	}
}



//-----------------------------------------------------------------------------
// Spikyカーネル
//-----------------------------------------------------------------------------
/*!
 * カーネル係数
 * @param[in] h 有効半径
 * @param[in] d 次元(1,2,3)
 * @param[in] type ノーマル:1，勾配:2，ラプラシアン:3
 * @return カーネル係数値
 */
inline static double KernelCoefSpiky(double h, int d, int type)
{
	double a = 1.0;
	if(d < 1) d = 1;
	if(d > 3) d = 3;
	switch(type){
	case 1:	// ノーマル
		switch(d){
		case 2: a = 10.0/(RX_PI*pow((double)h, (double)5.0));	break;
		case 3:	a = 15.0/(RX_PI*pow((double)h, (double)6.0));	break;
		}
		break;

	case 2:	// 勾配
		switch(d){
		case 2:	a = -30.0/(RX_PI*pow((double)h, (double)5.0));	break;
		case 3:	a = -45.0/(RX_PI*pow((double)h, (double)6.0));	break;
		}
		break;

	case 3:	// ラプラシアン
		switch(d){
		case 2: a = -60.0/(RX_PI*pow((double)h, (double)5.0));	break;
		case 3: a = -90.0/(RX_PI*pow((double)h, (double)6.0));	break;
		}
		break;

	default:
		break;
	}

	return a;
}

/*!
 * Spikyカーネル関数値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @return 関数値
 */
inline double KernelSpiky(double r, double h, double a)
{
	if(r >= 0.0 && r <= h){
		double q = h-r;
		return a*q*q*q;
	}
	else{
		return 0.0;
	}
}

/*!
 * Spikyカーネル関数勾配値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] rij 相対位置ベクトル
 * @return 勾配値
 */
template<class T> 
inline T KernelSpikyG(double r, double h, double a, T rij)
{
	if(r > 0.0 && r <= h){
		double q = h-r;
		return  a*q*q*rij/r;
	}
	else{
		return T(0.0);
	}
}
 
/*!
 * Splineカーネル関数ラプラシアンの計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] d 次元(1,2,3)
 * @return ラプラシアンの値
 */
inline double KernelSpikyL(double r, double h, double a, double d)
{
	if(r > 0.0 && r <= h){
		double q = h-r;
		return a*(q*q/r-q);
	}
	else{
		return 0.0;
	}
}



//-----------------------------------------------------------------------------
// Viscカーネル
//-----------------------------------------------------------------------------
/*!
 * カーネル係数
 * @param[in] h 有効半径
 * @param[in] d 次元(1,2,3)
 * @param[in] type ノーマル:1，勾配:2，ラプラシアン:3
 * @return カーネル係数値
 */
inline static double KernelCoefVisc(double h, int d, int type)
{
	double a = 1.0;
	if(d < 1) d = 1;
	if(d > 3) d = 3;
	switch(type){
	case 1:	// ノーマル
		switch(d){
		case 2: a = 10.0/(3.0*RX_PI*pow((double)h, (double)2.0));	break;
		case 3:	a = 15.0/(2.0*RX_PI*pow((double)h, (double)3.0));	break;
		}
		break;

	case 2:	// 勾配
		switch(d){
		case 2:	a = 10.0/(3.0*RX_PI*pow((double)h, (double)4.0));	break;
		case 3:	a = 15.0/(2.0*RX_PI*pow((double)h, (double)5.0));	break;
		}
		break;

	case 3:	// ラプラシアン
		switch(d){
		case 2: a = 20.0/(3.0*RX_PI*pow((double)h, (double)5.0));	break;
		case 3: a = 45.0/(RX_PI*pow((double)h, (double)6.0));		break;
		}
		break;

	default:
		break;
	}

	return a;
}

/*!
 * Viscカーネル関数値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @return 関数値
 */
inline double KernelVisc(double r, double h, double a)
{
	if(r > 0.0 && r <= h){
		double q = r/h;
		return a*(-q*q*q/2.0+q*q+2.0/q-1.0);
	}
	else{
		return 0.0;
	}
}

/*!
 * Viscカーネル関数勾配値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] rij 相対位置ベクトル
 * @return 勾配値
 */
template<class T> 
inline T KernelViscG(double r, double h, double a, T rij)
{
	if(r > 0.0 && r <= h){
		double q = r/h;
		return  a*(-1.5/q+2.0-q*q*q/2.0)*rij;
	}
	else{
		return T(0.0);
	}
}
 
/*!
 * Splineカーネル関数ラプラシアンの計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] d 次元(1,2,3)
 * @return ラプラシアンの値
 */
inline double KernelViscL(double r, double h, double a, double d)
{
	if(r > 0.0 && r <= h){
		return a*(h-r);
	}
	else{
		return 0.0;
	}
}


//-----------------------------------------------------------------------------
// Splineカーネル
//-----------------------------------------------------------------------------
/*!
 * カーネル係数
 * @param[in] h 有効半径
 * @param[in] d 次元(1,2,3)
 * @param[in] type ノーマル:1，勾配:2，ラプラシアン:3
 * @return カーネル係数値
 */
inline static double KernelCoefSpline(double h, int d, int type)
{
	double a = 1.0;
	if(d < 1) d = 1;
	if(d > 3) d = 3;
	switch(type){
	case 1:	// ノーマル
		switch(d){
		case 1: a = 2.0/(3.0*h);				break;
		case 2: a = 10.0/(7.0*RX_PI*h*h);		break;
		case 3:	a = 1.0/(RX_PI*h*h*h);			break;
		}
		break;

	case 2:	// 勾配
		switch(d){
		case 1:	a = 3.0/(2.0*h*h*h);			break;
		case 2:	a = 45.0/(14.0*RX_PI*h*h*h*h);	break;
		case 3:	a = 9.0/(4.0*RX_PI*h*h*h*h*h);	break;
		}
		break;

	case 3:	// ラプラシアン
		switch(d){
		case 1: a = 1.0/(2.0*RX_PI*h*h*h);		break;
		case 2: a = 45.0/(42.0*RX_PI*h*h*h*h);	break;
		case 3: a = 3.0/(4.0*RX_PI*h*h*h*h*h);	break;
		}
		break;

	default:
		break;
	}

	return a;
}


/*!
 * Splineカーネル関数値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @return 関数値
 */
inline double KernelSpline(double r, double h, double a)
{
	double q = r/h;
	if(q >= 0.0 && q <= 1.0){
		return a*(1.0-1.5*q*q+0.75*q*q*q);
	}
	else if(q > 1.0 && q <= 2.0){
		return a*0.25*(2.0-q)*(2.0-q)*(2.0-q);
	}
	else{
		return 0.0;
	}
}

/*!
 * Splineカーネル関数勾配値の計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] rij 相対位置ベクトル
 * @param[in] a カーネル係数
 * @return 勾配値
 */
template<class T> 
inline T KernelSplineG(double r, double h, double a, T rij)
{
	double q = r/h;
	if(q >= 0.0 && q < 1.0){
		return  a*(q-4.0/3.0)*rij;
	}
	else if(q >= 1.0 && q < 2.0){
		return -a*(2.0-q)*(2.0-q)*rij/q/3.0;
	}
	else{
		return T(0.0);
	}
}
 
/*!
 * Splineカーネル関数ラプラシアンの計算
 * @param[in] r 距離
 * @param[in] h 有効半径
 * @param[in] a カーネル係数
 * @param[in] d 次元(1,2,3)
 * @return ラプラシアンの値
 */
inline double KernelSplineL(double r, double h, double a, double d)
{
	double q = r/h;
	if(q >= 0.0 && q < 1.0){
		return a*(3.0*(d+1.0)*q-4.0*d);
	}
	else if(q >= 1.0 && q < 2.0){
		return a*((1.0-d)*(2.0-q)*(2.0-q)/q+2.0*(2.0-q));
	}
	else{
		return 0.0;
	}
}



#endif	// _RX_KERNEL_H_