#include "math2d.h"

// --------------------------------------------------
void m2Matrix::jacobiRotate(m2Matrix &A, m2Matrix &R)
{
	// rotates A through phi in 01-plane to set A(0,1) = 0
	// rotation stored in R whose columns are eigenvectors of A
	float d = (A(0,0) - A(1,1))/(2.0f*A(0,1));
	float t = 1.0f / (fabs(d) + sqrt(d*d + 1.0f));
	if (d < 0.0f) t = -t;
	float c = 1.0f/sqrt(t*t + 1);
	float s = t*c;
	A(0,0) += t*A(0,1);
	A(1,1) -= t*A(0,1);
	A(0,1) = A(1,0) = 0.0f;
	// store rotation in R
	for (int k = 0; k < 2; k++) {
		float Rkp = c*R(k,0) + s*R(k,1);
		float Rkq =-s*R(k,0) + c*R(k,1);
		R(k,0) = Rkp;
		R(k,1) = Rkq;
	}
}


// --------------------------------------------------
void m2Matrix::eigenDecomposition(m2Matrix &A, m2Matrix &R)
{
	// only for symmetric matrices!
	// A = R A' R^T, where A' is diagonal and R orthonormal

	R.id();	// unit matrix
	jacobiRotate(A, R);
}


// --------------------------------------------------
void m2Matrix::polarDecomposition(const m2Matrix &A, m2Matrix &R, m2Matrix &S)
{
	// A = RS, where S is symmetric and R is orthonormal
	// -> S = (A^T A)^(1/2)

	R.id();	// default answer

	m2Matrix ATA;
	ATA.multiplyTransposedLeft(A, A);

	m2Matrix U;
	R.id();
	eigenDecomposition(ATA, U);

	float l0 = ATA(0,0); if (l0 <= 0.0f) l0 = 0.0f; else l0 = 1.0f / sqrt(l0);
	float l1 = ATA(1,1); if (l1 <= 0.0f) l1 = 0.0f; else l1 = 1.0f / sqrt(l1);

	m2Matrix S1;
	S1.r00 = l0*U.r00*U.r00 + l1*U.r01*U.r01;
	S1.r01 = l0*U.r00*U.r10 + l1*U.r01*U.r11;
	S1.r10 = S1.r01;
	S1.r11 = l0*U.r10*U.r10 + l1*U.r11*U.r11;
	R.multiply(A, S1);
	S.multiplyTransposedLeft(R, A);
}


