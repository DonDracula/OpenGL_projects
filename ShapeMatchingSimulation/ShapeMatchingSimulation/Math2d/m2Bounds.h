#ifndef M2_BOUNDS_H
#define M2_BOUNDS_H

//---------------------------------------------------------------------------

#include "math2d.h"

//---------------------------------------------------------------------------
class m2Bounds
//---------------------------------------------------------------------------
{	
public:
	inline m2Bounds() { setEmpty(); }
	inline m2Bounds(const m2Vector &min0, const m2Vector &max0) { min = min0; max = max0; }

	inline void set(const m2Vector &min0, const m2Vector &max0) { min = min0; max = max0; }
	inline void setEmpty() {
		set(m2Vector(m2RealMax, m2RealMax),
			m2Vector(m2RealMin, m2RealMin));
	}
	inline void setInfinite() {
		set(m2Vector(m2RealMin, m2RealMin),
			m2Vector(m2RealMax, m2RealMax));
	}
	inline bool isEmpty() const { 
		if (min.x > max.x) return true;
		if (min.y > max.y) return true;
		return false;
	}

	bool operator == (const m2Bounds &b) const {
		return (min == b.min) && (max == b.max);
	}
	void combine(const m2Bounds &b) {
		min.minimum(b.min);
		max.maximum(b.max);
	}
	void operator += (const m2Bounds &b) {
		combine(b);
	}
	m2Bounds operator + (const m2Bounds &b) const {
		m2Bounds r = *this;
		r.combine(b);
		return r;
	}
	bool intersects(const m2Bounds &b) const {
		if ((b.min.x > max.x) || (min.x > b.max.x)) return false;
		if ((b.min.y > max.y) || (min.y > b.max.y)) return false;
		return true;
	}
	void intersect(const m2Bounds &b) {
		min.maximum(b.min);
		max.minimum(b.max);
	}
	void include(const m2Vector &v) {
		max.maximum(v);
		min.minimum(v);
	}
	bool contain(const m2Vector &v) const {
		return 
			min.x <= v.x && v.x <= max.x && 
			min.y <= v.y && v.y <= max.y;
	}
	void operator += (const m2Vector &v) {
		include(v);
	}
	void getCenter(m2Vector &v) const {
		v = min + max; v *= 0.5f;
	}
	void clamp(m2Vector &pos) const {
		if (isEmpty()) return;
		pos.maximum(min);
		pos.minimum(max);
	}
	void clamp(m2Vector &pos, m2Real offset) const {
		if (isEmpty()) return;
		if (pos.x < min.x + offset) pos.x = min.x + offset;
		if (pos.x > max.x - offset) pos.x = max.x - offset;
		if (pos.y < min.y + offset) pos.y = min.y + offset;
		if (pos.y > max.y - offset) pos.y = max.y - offset;
	}
//--------------------------------------------
	m2Vector min, max;
};


#endif