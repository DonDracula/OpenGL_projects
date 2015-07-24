#ifndef M2_REAL_H
#define M2_REAL_H

//---------------------------------------------------------------------------

#include "math2d.h"
#include <float.h>

#define m2Pi     3.1415926535897932f
#define m2HalfPi 1.5707963267948966f
#define m2TwoPi  6.2831853071795865f
#define m2RealMax FLT_MAX
#define m2RealMin FLT_MIN
#define m2RadToDeg 57.295779513082321f
#define m2DegToRad 0.0174532925199433f 

typedef float m2Real;

//---------------------------------------------------------------------------

inline m2Real m2Clamp(m2Real &r, m2Real min, m2Real max)
{
	if (r < min) return min;
	else if (r > max) return max;
	else return r;
}

//---------------------------------------------------------------------------

inline m2Real m2Min(m2Real r1, m2Real r2)
{
	if (r1 <= r2) return r1;
	else return r2;
}

//---------------------------------------------------------------------------

inline m2Real m2Max(m2Real r1, m2Real r2)
{
	if (r1 >= r2) return r1;
	else return r2;
}

//---------------------------------------------------------------------------

inline m2Real m2Abs(m2Real r)
{
	if (r < 0.0f) return -r;
	else return r;
}

//---------------------------------------------------------------------------

inline m2Real m2Random(m2Real min, m2Real max)
{
	return min + ((m2Real)rand() / RAND_MAX) * (max - min);
}

//---------------------------------------------------------------------------

inline m2Real m2Acos(m2Real r)
{
	// result is between 0 and pi
	if (r < -1.0f) r = -1.0f;
	if (r >  1.0f) r =  1.0f;
	return acos(r);
}


#endif