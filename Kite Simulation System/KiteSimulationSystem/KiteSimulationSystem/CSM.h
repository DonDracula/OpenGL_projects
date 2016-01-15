/*
	file shape_matching.h
	brief Chain Shape Matching function
	data 2015-06
*/
#ifndef _CSM_H
#define _CSM_H

#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <vector>
#include <string>

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

#define LINE_SP 20
#define LINE_K 5.0*((double)LINE_SP)		//ばね定数
#define LINE_D 1.0							//減衰係数
#define LINE_RHO 0.2						//質量密度

using namespace std;

typedef void (*CollisionFunc)(Vec3&,Vec3&,Vec3&,int);

class CSM
{
public:
	struct Cluster
	{
		vector<int> Node;			//the list number of cluster note
		int NumNode;				//the count of the note
		vector<Vec3> Disp;			// the positon of the note
	};

public:
	Vec3 m_vWindSpeed;							//speed of wind
	float segmentLength;						//the initial length of two particles
protected:
	//shape data
	int m_iNumOfVertices;							// number of all the particles 

	vector<Vec3> m_vOrgPos;							//original position of the vertice in space 
	vector<Vec3> m_vCurPos;							//current position of it
	vector<Vec3> m_vNewPos;							//next positon of it
	vector<Vec3> m_vGoalPos;							//goal positon of the vertice
	vector<double> m_vMass;							//mass value
	vector<Vec3> m_vVel;							//velocity 
	vector<bool> m_vFix;							//a flag to judge whether the vertice is fixed or not 
	vector<int> m_vNumCluster;						//the number of the cluster
	vector<Cluster> m_vCluster;						//cluster

	//simulation parameters
	double m_fDt;									//time step
	Vec3 m_vMin,m_vMax;								//simutation space
	Vec3 m_vGraviity;								//gravity accelaration



	double m_fAlpha;								//stiffness param (0~1)
	double m_fBeta;									//deformation param(between 0 and 1)

	bool m_bLinearDeformation;						
	bool m_bVolumeConservation;						//volume conservation

	//int m_iObjectNum;								//number of the object rigons 
	CollisionFunc m_Collision;



public :
	CSM();
	~CSM();

	void Clear();
	void AddVertex(const Vec3 &pos, double mass);
	void AddCluster(const vector<int> &list);

	void Update();

	void SetTimeStep(double dt){m_fDt = dt;};
	void SetSimulationSpace(Vec3 minp,Vec3 maxp){m_vMin = minp; m_vMax= maxp;};
	void SetStiffness(double alpha, double beta){m_fAlpha = alpha; m_fBeta = beta;};

	void SetCollisionFunc(CollisionFunc func){m_Collision = func;};

	int GetNumOfVertices() const{return m_iNumOfVertices;}
	const Vec3& GetVertexPos(int i) {return m_vCurPos[i];}
	const Vec3& GetVertexVel(int i) {return m_vVel[i];}
	double GetMass(int i) {return m_vMass[i];}

	void FixVertex(int i,const Vec3 &pos);
	void UnFixVertex(int i);
	//bool IsFixed(int i){ return m_vFix[i];}

protected:
	//initialize the vector positoin
	void initialize(void);

	//shape matching method
	void calExternalForces(double dt);
	void calCollision(double dt);
	void shapeMatchingFun(Cluster &cl, double dt);
	void integrate(double dt);

	void clamp(Vec3 &pos) const
	{
		if(pos[0] < m_vMin[0]) pos[0] = m_vMin[0];
		if(pos[0] > m_vMax[0]) pos[0] = m_vMax[0];
		if(pos[1] < m_vMin[1]) pos[1] = m_vMin[1];
		if(pos[1] > m_vMax[1]) pos[1] = m_vMax[1];
		if(pos[2] < m_vMin[2]) pos[2] = m_vMin[2];
		if(pos[2] > m_vMax[2]) pos[2] = m_vMax[2];
	}
};

/*!
 * Jacobi法による固有値の算出
 * @param[inout] a 実対称行列．計算後，対角要素に固有値が入る
 * @param[out] v 固有ベクトル(aと同じサイズ)
 * @param[in] n 行列のサイズ(n×n)
 * @param[in] eps 収束誤差
 * @param[in] iter_max 最大反復回数
 * @return 反復回数
 */
inline int EigenJacobiMethod(double *a, double *v, int n, double eps = 1e-8, int iter_max = 100)
{
	double *bim, *bjm;
	double bii, bij, bjj, bji;
 
	bim = new double[n];
	bjm = new double[n];
 
	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			v[i*n+j] = (i == j) ? 1.0 : 0.0;
		}
	}
 
	int cnt = 0;
	for(;;){
		int i = -1, j = -1;
 
		double x = 0.0;
		for(int ia = 0; ia < n; ++ia){
			for(int ja = 0; ja < n; ++ja){
				int idx = ia*n+ja;
				if(ia != ja && fabs(a[idx]) > x){
					i = ia;
					j = ja;
					x = fabs(a[idx]);
				}
			}
		}

		if(i == -1 || j == -1) return 0;
 
		double aii = a[i*n+i];
		double ajj = a[j*n+j];
		double aij = a[i*n+j];
 
		double m_fAlpha, m_fBeta;
		m_fAlpha = (aii-ajj)/2.0;
		m_fBeta  = sqrt(m_fAlpha*m_fAlpha+aij*aij);
 
		double st, ct;
		ct = sqrt((1.0+fabs(m_fAlpha)/m_fBeta)/2.0);    // sinθ
		st = (((aii-ajj) >= 0.0) ? 1.0 : -1.0)*aij/(2.0*m_fBeta*ct);    // cosθ
 
		// A = PAPの計算
		for(int m = 0; m < n; ++m){
			if(m == i || m == j) continue;
 
			double aim = a[i*n+m];
			double ajm = a[j*n+m];
 
			bim[m] =  aim*ct+ajm*st;
			bjm[m] = -aim*st+ajm*ct;
		}
 
		bii = aii*ct*ct+2.0*aij*ct*st+ajj*st*st;
		bij = 0.0;
 
		bjj = aii*st*st-2.0*aij*ct*st+ajj*ct*ct;
		bji = 0.0;
 
		for(int m = 0; m < n; ++m){
			a[i*n+m] = a[m*n+i] = bim[m];
			a[j*n+m] = a[m*n+j] = bjm[m];
		}
		a[i*n+i] = bii;
		a[i*n+j] = bij;
		a[j*n+j] = bjj;
		a[j*n+i] = bji;
 
		// V = PVの計算
		for(int m = 0; m < n; ++m){
			double vmi = v[m*n+i];
			double vmj = v[m*n+j];
 
			bim[m] =  vmi*ct+vmj*st;
			bjm[m] = -vmi*st+vmj*ct;
		}
		for(int m = 0; m < n; ++m){
			v[m*n+i] = bim[m];
			v[m*n+j] = bjm[m];
		}
 
		double e = 0.0;
		for(int ja = 0; ja < n; ++ja){
			for(int ia = 0; ia < n; ++ia){
				if(ia != ja){
					e += fabs(a[ja*n+ia]);
				}
			}
		}
		if(e < eps) break;
 
		cnt++;
		if(cnt > iter_max) break;
	}
 
	delete [] bim;
	delete [] bjm;
 
	return cnt;
}


/*!
 * 極分解で回転行列と対称行列に分解 A=RS
 * @param[in] A 入力行列
 * @param[out] R 回転行列(直交行列 R^-1 = R^T)
 * @param[out] S 対称行列
 */
inline void PolarDecomposition(const rxMatrix3 &A, rxMatrix3 &R, rxMatrix3 &S)
{
	R.makeIdentity();

	// S = (A^T A)^(1/2)を求める
	rxMatrix3 ATA;
	// (A^T A)の計算
	ATA = A.Transpose()*A;

	rxMatrix3 U;
	R.makeIdentity();
	U.makeIdentity();

	// (A^T A)を固有値分解して対角行列と直交行列を求める
	// M^(1/2) = U^T M' U 
	//  M = (A^T A), M':対角行列の平方根を取ったもの, U:直交行列, 
	EigenJacobiMethod(&ATA, &U, 3);

	// 対角行列の平方根をとって，逆行列計算のために逆数にしておく
	real l0 = (ATA(0,0) <= 0.0) ? 0.0 : 1.0/sqrt(ATA(0,0));
	real l1 = (ATA(1,1) <= 0.0) ? 0.0 : 1.0/sqrt(ATA(1,1));
	real l2 = (ATA(2,2) <= 0.0) ? 0.0 : 1.0/sqrt(ATA(2,2));

	// U^T M' U の逆行列計算
	rxMatrix3 S1;
	S1(0,0) = l0*U(0,0)*U(0,0) + l1*U(0,1)*U(0,1) + l2*U(0,2)*U(0,2);
	S1(0,1) = l0*U(0,0)*U(1,0) + l1*U(0,1)*U(1,1) + l2*U(0,2)*U(1,2);
	S1(0,2) = l0*U(0,0)*U(2,0) + l1*U(0,1)*U(2,1) + l2*U(0,2)*U(2,2);
	S1(1,0) = S1(0,1);
	S1(1,1) = l0*U(1,0)*U(1,0) + l1*U(1,1)*U(1,1) + l2*U(1,2)*U(1,2);
	S1(1,2) = l0*U(1,0)*U(2,0) + l1*U(1,1)*U(2,1) + l2*U(1,2)*U(2,2);
	S1(2,0) = S1(0,2);
	S1(2,1) = S1(1,2);
	S1(2,2) = l0*U(2,0)*U(2,0) + l1*U(2,1)*U(2,1) + l2*U(2,2)*U(2,2);

	R = A*S1;	// R = A S^-1
	S = R.Transpose()*A; // S = R^-1 A = R^T A
}

#endif //_CSM_H