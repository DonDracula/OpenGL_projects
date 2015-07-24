#ifndef M2_MATRIX_H
#define M2_MATRIX_H
//---------------------------------------------------------------------------

#include "math2d.h"

//---------------------------------------------------------------------------
class m2Matrix {
//---------------------------------------------------------------------------
public:
	m2Matrix() {}
	m2Matrix(const m2Matrix &m) { *this = m; }
	m2Matrix(const m2Vector &col0, const m2Vector &col1, const m2Vector &col2) {
		r00 = col0.x; r01 = col1.x;
		r10 = col0.y; r11 = col1.y;
	}
	m2Matrix(m2Real c00, m2Real c01, m2Real c10, m2Real c11, m2Real c20, m2Real c21) {
		r00 = c00; r01 = c01;
		r10 = c10; r11 = c11;
	}
	m2Real & m2Matrix::operator()(int i, int j) {
		assert(i >= 0 && i <= 1);
		assert(j >= 0 && i <= 1);
		return (&r00)[i*2+j];
	}
	const m2Real & m2Matrix::operator()(int i, int j) const {
		assert(i >= 0 && i <= 1);
		assert(j >= 0 && i <= 1);
		return (&r00)[i*2+j];
	}
	void zero() { 
		r00 = 0.0; r01 = 0.0; 
		r10 = 0.0; r11 = 0.0; 
	}
	void id() {
		r00 = 1.0; r01 = 0.0; 
		r10 = 0.0; r11 = 1.0; 
	}
	void setColumns(const m2Vector &col0, const m2Vector &col1) {
		r00 = col0.x; r01 = col1.x;
		r10 = col0.y; r11 = col1.y;
	}
	void setColumn(int i, const m2Vector &col) {
		if (i == 0) { r00 = col.x; r10 = col.y; }
		if (i == 1) { r01 = col.x; r11 = col.y; }
	}
	void setRows(const m2Vector &row0, const m2Vector &row1) {
		r00 = row0.x; r01 = row0.y;
		r10 = row1.x; r11 = row1.y;
	}
	void setRow(int i, const m2Vector &row) {
		if (i == 0) { r00 = row.x; r01 = row.y; }
		if (i == 1) { r10 = row.x; r11 = row.y; }
	}
	void getColumns(m2Vector &col0, m2Vector &col1) const {
		col0.x = r00; col1.x = r01; 
		col0.y = r10; col1.y = r11; 
	}
	void getColumn(int i, m2Vector &col) const {
		if (i == 0) { col.x = r00; col.y = r10; }
		if (i == 1) { col.x = r01; col.y = r11; }
	}
	m2Vector getColumn(int i) const {
		m2Vector r; getColumn(i, r);
		return r;
	}
	void getRows(m2Vector &row0, m2Vector &row1) const {
		row0.x = r00; row0.y = r01; 
		row1.x = r10; row1.y = r11; 
	}
	void getRow(int i, m2Vector &row) const {
		if (i == 0) { row.x = r00; row.y = r01; }
		if (i == 1) { row.x = r10; row.y = r11; }
	}
	m2Vector getRow(int i) const {
		m2Vector r; getRow(i, r);
		return r;
	}
	bool operator == (const m2Matrix &m) const {
		return 
			(r00 == m.r00) && (r01 == m.r01) &&
			(r10 == m.r10) && (r11 == m.r11);
	}
	m2Matrix operator + (const m2Matrix &m) const {
		m2Matrix res; 
		res.r00 = r00 + m.r00; res.r01 = r01 + m.r01;
		res.r10 = r10 + m.r10; res.r11 = r11 + m.r11;
		return res;
	}
	m2Matrix operator - (const m2Matrix &m) const {
		m2Matrix res; 
		res.r00 = r00 - m.r00; res.r01 = r01 - m.r01; 
		res.r10 = r10 - m.r10; res.r11 = r11 - m.r11; 
		return res;
	}
	void operator += (const m2Matrix &m) {
		r00 += m.r00; r01 += m.r01; 
		r10 += m.r10; r11 += m.r11; 
	}
	void operator -= (const m2Matrix &m) {
		r00 -= m.r00; r01 -= m.r01; 
		r10 -= m.r10; r11 -= m.r11; 
	}
	m2Matrix operator * (const m2Real f) const {
		m2Matrix res;
		res.r00 = r00*f; res.r01 = r01*f; 
		res.r10 = r10*f; res.r11 = r11*f; 
		return res;
	}
	m2Matrix operator / (const m2Real f) const {
		m2Matrix res;
		res.r00 = r00*f; res.r01 = r01*f; 
		res.r10 = r10*f; res.r11 = r11*f; 
		return res;
	}
	void operator *=(m2Real f) { 
		r00 *= f; r01 *= f; 
		r10 *= f; r11 *= f; 
	}
	void operator /=(m2Real f) { 
		r00 /= f; r01 /= f; 
		r10 /= f; r11 /= f; 
	}
	m2Vector operator * (const m2Vector &v) const {
		return multiply(v);
	}
	m2Vector multiply(const m2Vector &v) const {
		m2Vector res;
		res.x = r00 * v.x + r01 * v.y;
		res.y = r10 * v.x + r11 * v.y;
		return res;
	}
	m2Vector multiplyTransposed(const m2Vector &v) const {
		m2Vector res;
		res.x = r00 * v.x + r10 * v.y;
		res.y = r01 * v.x + r11 * v.y;
		return res;
	}
	void multiply(const m2Matrix &left, const m2Matrix &right) {
		m2Matrix res;
		res.r00 = left.r00*right.r00 + left.r01*right.r10;
		res.r01 = left.r00*right.r01 + left.r01*right.r11;
		res.r10 = left.r10*right.r00 + left.r11*right.r10;
		res.r11 = left.r10*right.r01 + left.r11*right.r11;
		*this = res;
	}
	void multiplyTransposedLeft(const m2Matrix &left, const m2Matrix &right) {
		m2Matrix res;
		res.r00 = left.r00*right.r00 + left.r10*right.r10;
		res.r01 = left.r00*right.r01 + left.r10*right.r11;
		res.r10 = left.r01*right.r00 + left.r11*right.r10;
		res.r11 = left.r01*right.r01 + left.r11*right.r11;
		*this = res;
	}	
	void multiplyTransposedRight(const m2Matrix &left, const m2Matrix &right) {
		m2Matrix res;
		res.r00 = left.r00*right.r00 + left.r01*right.r01;
		res.r01 = left.r00*right.r10 + left.r01*right.r11;
		res.r10 = left.r10*right.r00 + left.r11*right.r01;
		res.r11 = left.r10*right.r10 + left.r11*right.r11;
		*this = res;
	}
	m2Matrix operator * (const m2Matrix &m) const {
		m2Matrix res;
		res.multiply(*this, m);
		return res;
	}
	bool invert() {
		m2Real d = r00*r11 - r01*r10;
		if (d == 0.0)
			return false;
		d = (m2Real)1.0/d;

		m2Real d00 =  r11*d;
		m2Real d01 = -r01*d;
		m2Real d10 = -r10*d;
		m2Real d11 =  r00*d;

		r00 = d00;
		r01 = d01;
		r10 = d10;
		r11 = d11;

		return true;
	}
	bool setInverse(m2Matrix &m) {
		m = *this;
		return m.invert();
	}
	void transpose() {
		m2Real r;
		r = r01; r01 = r10; r10 = r;
	}
	void setTransposed(m2Matrix &m) {
		m = *this;
		m.transpose();
	}
	m2Real magnitudeSquared() { 
		return r00*r00+r01*r01 + r10*r10+r11*r11; 
	}
	m2Real magnitude() { return sqrt(magnitudeSquared()); }

	void gramSchmidt() {
		m2Vector c0,c1;
		getColumn(0,c0); c0.normalize();
		c1.x = -c0.y;
		c1.y = c0.x;
		setColumns(c0, c1);
	}

	void rot(m2Real angle)
	{
		m2Real cosAngle = cos(angle);
		m2Real sinAngle = sin(angle);
		r00 = r11 = cosAngle;
		r01 = -sinAngle;
		r10 = sinAngle;
	}

    m2Real determinant() {
    	return r00*r11 - r01*r10;
    }

// --------------------------------------------------
	m2Real r00,r01;
	m2Real r10,r11;

	static void jacobiRotate(m2Matrix &A, m2Matrix &R);
	static void eigenDecomposition(m2Matrix &A, m2Matrix &R);
	static void polarDecomposition(const m2Matrix &A, m2Matrix &R, m2Matrix &S);
};


#endif