#ifndef M2_VECTOR_H
#define M2_VECTOR_H

//---------------------------------------------------------------------------

#include "math2d.h"
#include <assert.h>

//---------------------------------------------------------------------------
class m2Vector
//---------------------------------------------------------------------------
{	
public:
	inline m2Vector() {}
	inline m2Vector(m2Real x0, m2Real y0) { x = x0; y = y0; }

	inline void set(m2Real x0, m2Real y0) { x = x0; y = y0; }
	inline void zero() { x = 0.0; y = 0.0; }
	inline bool isZero() { return x == 0.0 && y == 0.0; }

	m2Real & operator[] (int i) {
		assert(i >= 0 && i <= 2);
		return (&x)[i];
	}
	bool operator == (const m2Vector &v) const {
		return (x == v.x) && (y == v.y);
	}
	m2Vector operator + (const m2Vector &v) const {
		m2Vector r; r.x = x + v.x; r.y = y + v.y;
		return r;
	}
	m2Vector operator - (const m2Vector &v) const {
		m2Vector r; r.x = x - v.x; r.y = y - v.y;
		return r;
	}
	void operator += (const m2Vector &v) {
		x += v.x; y += v.y; 
	}
	void operator -= (const m2Vector &v) {
		x -= v.x; y -= v.y; 
	}
	void operator *= (const m2Vector &v) {
		x *= v.x; y *= v.y; 
	}
	void operator /= (const m2Vector &v) {
		x /= v.x; y /= v.y;
	}
	m2Vector operator -() const {
		m2Vector r; r.x = -x; r.y = -y;
		return r;
	}
	m2Vector operator * (const m2Real f) const {
		m2Vector r; r.x = x*f; r.y = y*f;
		return r;
	}
	m2Vector operator / (const m2Real f) const {
		m2Vector r; r.x = x/f; r.y = y/f;
		return r;
	}
	m2Real cross(const m2Vector &v1, const m2Vector &v2) const {
		return v1.x*v2.y-v2.x*v1.y;
	}
	inline m2Real dot(const m2Vector &v) const {
		return x*v.x + y*v.y;
	}

	inline void minimum(const m2Vector &v) {
		if (v.x < x) x = v.x;
		if (v.y < y) y = v.y;
	}
	inline void maximum(const m2Vector &v) {
		if (v.x > x) x = v.x;
		if (v.y > y) y = v.y;
	}

	inline m2Real magnitudeSquared() const { return x*x + y*y; }
	inline m2Real magnitude() const { return sqrt(x*x + y*y); }
	inline m2Real distanceSquared(const m2Vector &v) const {
		m2Real dx,dy; dx = v.x - x; dy = v.y - y;
		return dx*dx + dy*dy;
	}
	inline m2Real distance(const m2Vector &v) const {
		m2Real dx,dy; dx = v.x - x; dy = v.y - y;
		return sqrt(dx*dx + dy*dy);
	}
	void operator *=(m2Real f) { x *=f; y *=f; }
	void operator /=(m2Real f) { x /=f; y /=f; }
	m2Real normalize() {
		m2Real l = magnitude();
		if (l == 0.0f) { x = 1.0f; y = 0.0f; }
		m2Real l1 = 1.0f/l; x *= l1; y *= l1;
		return l;
	}

// ------------------------------
	m2Real x,y;
};


#endif