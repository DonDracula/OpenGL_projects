#ifndef M5_MATRIX_H
#define M5_MATRIX_H
//---------------------------------------------------------------------------

#include "math2d.h"

//---------------------------------------------------------------------------
class m5Matrix {
//---------------------------------------------------------------------------
public:
	m5Matrix() {}
	m5Matrix(const m5Matrix &m) { *this = m; }
	m2Real & m5Matrix::operator()(int i, int j) {
		assert(i >= 0 && i <= 4);
		assert(j >= 0 && i <= 4);
		return (&r00)[i*5+j];
	}
	const m2Real & m5Matrix::operator()(int i, int j) const {
		assert(i >= 0 && i <= 4);
		assert(j >= 0 && i <= 4);
		return (&r00)[i*5+j];
	}
	void zero() { 
		r00 = 0.0; r01 = 0.0; r02 = 0.0; r03 = 0.0; r04 = 0.0;
		r10 = 0.0; r11 = 0.0; r12 = 0.0; r13 = 0.0; r14 = 0.0;
		r20 = 0.0; r21 = 0.0; r22 = 0.0; r23 = 0.0; r24 = 0.0;
		r30 = 0.0; r31 = 0.0; r32 = 0.0; r33 = 0.0; r34 = 0.0;
		r40 = 0.0; r41 = 0.0; r42 = 0.0; r43 = 0.0; r44 = 0.0;
	}
	void id() {
		r00 = 1.0; r01 = 0.0; r02 = 0.0; r03 = 0.0; r04 = 0.0;
		r10 = 0.0; r11 = 1.0; r12 = 0.0; r13 = 0.0; r14 = 0.0;
		r20 = 0.0; r21 = 0.0; r22 = 1.0; r23 = 0.0; r24 = 0.0;
		r30 = 0.0; r31 = 0.0; r32 = 0.0; r33 = 1.0; r34 = 0.0;
		r40 = 0.0; r41 = 0.0; r42 = 0.0; r43 = 0.0; r44 = 1.0;
	}
	static void jacobiRotate(m5Matrix &A, m5Matrix &R, int p, int q);
	static void eigenDecomposition(m5Matrix &A, m5Matrix &R);
	void invert();

// --------------------------------------------------
	m2Real r00,r01,r02,r03,r04;
	m2Real r10,r11,r12,r13,r14;
	m2Real r20,r21,r22,r23,r24;
	m2Real r30,r31,r32,r33,r34;
	m2Real r40,r41,r42,r43,r44;
};


#endif