#include "m5Matrix.h"

// --------------------------------------------------

#define EPSILON 1e-8
#define JACOBI_ITERATIONS 20


//---------------------------------------------------------------------
void m5Matrix::jacobiRotate(m5Matrix &A, m5Matrix &R, int p, int q)
//---------------------------------------------------------------------
{
	// rotates A through phi in pq-plane to set A(p,q) = 0
	// rotation stored in R whose columns are eigenvectors of A
	float d = (A(p,p) - A(q,q))/(2.0f*A(p,q));
	float t = 1.0f / (fabs(d) + sqrt(d*d + 1.0f));
	if (d < 0.0f) t = -t;
	float c = 1.0f/sqrt(t*t + 1);
	float s = t*c;
	A(p,p) += t*A(p,q);
	A(q,q) -= t*A(p,q);
	A(p,q) = A(q,p) = 0.0f;
	// transform A
	int k;
	for (k = 0; k < 5; k++) {
		if (k != p && k != q) {
			float Akp = c*A(k,p) + s*A(k,q);
			float Akq =-s*A(k,p) + c*A(k,q);
			A(k,p) = A(p,k) = Akp;
			A(k,q) = A(q,k) = Akq;
		}
	}
	// store rotation in R
	for (k = 0; k < 5; k++) {
		float Rkp = c*R(k,p) + s*R(k,q);
		float Rkq =-s*R(k,p) + c*R(k,q);
		R(k,p) = Rkp;
		R(k,q) = Rkq;
	}
}


//---------------------------------------------------------------------
void m5Matrix::eigenDecomposition(m5Matrix &A, m5Matrix &R)
//---------------------------------------------------------------------
{
	// only for symmetric matrices!
	// A = R A' R^T, where A' is diagonal and R orthonormal

	R.id();	// unit matrix
	int iter = 0;
	while (iter < JACOBI_ITERATIONS) {	// 10 off diagonal elements
		// find off diagonal element with maximum modulus
		int p,q;
		float a,max;
		max = -1.0f;
		for (int i = 0; i < 4; i++) {
			for (int j = i+1; j < 5; j++) {
				a = fabs(A(i,j));
				if (max < 0.0f || a > max) {
					p = i; q = j; max = a;
				}
			}
		}
		// all small enough -> done
//		if (max < EPSILON) break;  debug
		if (max <= 0.0f) break;
		// rotate matrix with respect to that element
		jacobiRotate(A, R, p,q);
		iter++;
	}
}


//----------------------------------------------------------------------------
void m5Matrix::invert()
//----------------------------------------------------------------------------
{
	m5Matrix R, A = *this;
	eigenDecomposition(A, R);

	float d[5];
	int i,j,k;

	for (i = 0; i < 5; i++) {
		d[i] = A(i,i);
		if (d[i] != 0.0f) d[i] = 1.0f / d[i];
	}

	for (i = 0; i < 5; i++) {
		for (j = 0; j < 5; j++) {
			m2Real &a = (&r00)[i*5+j];
			a = 0.0f;
			for (k = 0; k < 5; k++)
				a += d[k]*R(i,k)*R(j,k);
		}
	}
}



