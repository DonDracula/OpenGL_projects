#ifndef _VEC_H_
#define _VEC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cmath>
#include <iostream>
#include <fstream>

#include <string>
#include <vector>

//-----------------------------------------------------------------------------
// 定義
//-----------------------------------------------------------------------------
#ifndef VEC_RX_FEQ_INF
#define VEC_RX_FEQ_INF 1.0e10
#endif

#ifndef VEC_RX_FEQ_EPS
#define VEC_RX_FEQ_EPS	1e-10
#endif

//-----------------------------------------------------------------------------
// Vec3クラスの宣言 - 3次元ベクトルを扱うクラス
//-----------------------------------------------------------------------------
class Vec3
{
public:
	double data[3];

	inline Vec3()
	{
		data[0] = 0.0;
		data[1] = 0.0;
		data[2] = 0.0;
	}

	inline Vec3(const double a, const double b, const double c)
	{
		data[0] = a;
		data[1] = b;
		data[2] = c;
	}

	inline Vec3(const double a, const double b)
	{
		data[0] = a;
		data[1] = b;
		data[2] = 0;
	}

	inline Vec3(const double &a)
	{
		data[0] = a;
		data[1] = a;
		data[2] = a;
	}

	inline Vec3(const Vec3 &v)
	{
		data[0] = v[0];
		data[1] = v[1];
		data[2] = v[2];
	}

	inline Vec3(const double* v)
	{
		memcpy(data, v, 3*sizeof(double));
	}

	void SetValue(const double *v) 
	{
		data[0] = v[0]; 
		data[1] = v[1]; 
		data[2] = v[2];
	}

	void SetValue(const double& x, const double& y, const double& z)
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}

	void GetValue(double *v) const 
	{
		v[0] = data[0];
		v[1] = data[1];
		v[2] = data[2];
	}

	inline operator const double*() const
	{
		return data;
	}

	inline operator double*()
	{
		return data;
	}

	inline double& operator[](int i)
	{
		return data[i];
	}

	inline double operator[](int i) const
	{
		return data[i];
	}

	inline Vec3& operator=(const Vec3 &a)
	{
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
		return *this;
	}

	inline Vec3& operator=(const double &a)
	{
		data[0] = a;
		data[1] = a;
		data[2] = a;
		return *this;
	}

	inline Vec3& operator+=(const Vec3 &a)
	{
		data[0] += a[0];
		data[1] += a[1];
		data[2] += a[2];
		return *this;
	}

	inline Vec3& operator-=(const Vec3 &a)
	{
		data[0] -= a[0];
		data[1] -= a[1];
		data[2] -= a[2];
		return *this;
	}

	inline Vec3& operator*=(const double &a)
	{
		data[0] *= a;
		data[1] *= a;
		data[2] *= a;
		return *this;
	}

	inline Vec3& operator*=(const Vec3 &a)
	{
		data[0] *= a[0];
		data[1] *= a[1];
		data[2] *= a[2];
		return *this;
	}

	inline Vec3& operator/=(const double &a)
	{
		double inv = 1.0/a;
		data[0] *= inv;
		data[1] *= inv;
		data[2] *= inv;
		return *this;
	}

	inline bool operator==(const Vec3 &a)
	{
		return (a[0] == data[0] && a[1] == data[1] && a[2] == data[2]);
	}

	inline bool operator>(const Vec3 &v);
	inline bool operator<(const Vec3 &v);
	inline bool operator>=(const Vec3 &v);
	inline bool operator<=(const Vec3 &v);

};


//-----------------------------------------------------------------------------
// Vec2クラスの宣言 - 2次元ベクトルを扱うクラス
//-----------------------------------------------------------------------------
class Vec2
{
public:
	double data[2];

	inline Vec2() { }

	inline Vec2(const double a, const double b)
	{
		data[0] = a;
		data[1] = b;
	}

	inline Vec2(const double &a)
	{
		data[0] = a;
		data[1] = a;
	}

	inline Vec2(const Vec2 &v)
	{
		data[0] = v[0];
		data[1] = v[1];
	}

	inline Vec2(const double* v)
	{
		memcpy(data,v,2*sizeof(double));
	}

	inline operator const double*() const
	{
		return data;
	}

	inline operator double*()
	{
		return data;
	}

	inline double& operator[](int i)
	{
		return data[i];
	}

	inline double operator[](int i) const
	{
		return data[i];
	}

	inline Vec2& operator=(const Vec2 &a)
	{
		data[0] = a[0];
		data[1] = a[1];
		return *this;
	}

	inline Vec2& operator=(const double &a)
	{
		data[0] = a;
		data[1] = a;
		return *this;
	}

	inline Vec2& operator+=(const Vec2 &a)
	{
		data[0] += a[0];
		data[1] += a[1];
		return *this;
	}

	inline Vec2& operator-=(const Vec2 &a)
	{
		data[0] -= a[0];
		data[1] -= a[1];
		return *this;
	}

	inline Vec2& operator*=(const double &a)
	{
		data[0] *= a;
		data[1] *= a;
		return *this;
	}

	inline Vec2& operator*=(const Vec2 &a)
	{
		data[0] *= a[0];
		data[1] *= a[1];
		return *this;
	}

	inline Vec2& operator/=(const double &a)
	{
		double inv = 1.0/a;
		data[0] *= inv;
		data[1] *= inv;
		return *this;
	}

	inline bool operator==(const Vec2 &a)
	{
		return (a[0] == data[0] && a[1] == data[1]);
	}

	inline bool operator>(const Vec2 &v);
	inline bool operator<(const Vec2 &v);
	inline bool operator>=(const Vec2 &v);
	inline bool operator<=(const Vec2 &v);
};

//-----------------------------------------------------------------------------
// Vec4クラスの宣言 - 4次元ベクトルを扱うクラス
//-----------------------------------------------------------------------------
class Vec4
{
public:
	double data[4];

	Vec4()
	{
		*this = 0.0;
	}

	Vec4(double a, double b, double c, double d)
	{
		data[0] = a;
		data[1] = b;
		data[2] = c;
		data[3] = d;
	}

	Vec4(double a)
	{
		data[0] = a;
		data[1] = a;
		data[2] = a;
		data[3] = a;
	}

	Vec4(const Vec3& v)
	{
		data[0] = v[0];
		data[1] = v[1];
		data[2] = v[2];
		data[3] = 0.0;
	}

	Vec4(const Vec3& v, const double &a)
	{
		data[0] = v[0];
		data[1] = v[1];
		data[2] = v[2];
		data[3] = a;
	}

	Vec4(const Vec4& v)
	{
		(*this) = v;
	}

	Vec4(const double* v)
	{
		memcpy(data,v,4*sizeof(double));
	}

	operator const double*() const
	{
		return data;
	}

	operator double*()
	{
		return data;
	}

	double& operator[](int i)
	{
		return data[i];
	}

	double operator[](int i) const
	{
		return data[i];
	}

	inline Vec4& operator=(const Vec4& a)
	{
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
		data[3] = a[3];
		return *this;
	}

	inline Vec4& operator=(const Vec3& a)
	{
		data[0] = a[0];
		data[1] = a[1];
		data[2] = a[2];
		data[3] = 0.0;
		return *this;
	}
		
	inline Vec4& operator=(const double a)
	{
		data[0] = a;
		data[1] = a;
		data[2] = a;
		data[3] = a;
		return *this;
	}

	inline Vec4& operator+=(const Vec4& a)
	{
		data[0] += a[0];
		data[1] += a[1];
		data[2] += a[2];
		data[3] += a[3];
		return *this;
	}

	inline Vec4& operator-=(const Vec4& a)
	{
		data[0] -= a[0];
		data[1] -= a[1];
		data[2] -= a[2];
		data[3] -= a[3];
		return *this;
	}

	inline Vec4& operator*=(double a)
	{
		data[0] *= a;
		data[1] *= a;
		data[2] *= a;
		data[3] *= a;
		return *this;
	}

	inline Vec4& operator/=(double a)
	{
		data[0] /= a;
		data[1] /= a;
		data[2] /= a;
		data[3] /= a;
		return *this;
	}
};


//-----------------------------------------------------------------------------
// Vec3クラスの実装
//-----------------------------------------------------------------------------
static int dCmp(double a, double b)
{
	if(fabs(a-b) < VEC_RX_FEQ_EPS){
		return 0;
	}

	if(a > b) return 1;
	return -1;
}

inline bool Vec3::operator>(const Vec3 &v)
{
	int status;

	status = dCmp(data[0], v[0]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	status = dCmp(data[1], v[1]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	status = dCmp(data[2], v[2]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	return false;
}

inline bool Vec3::operator<(const Vec3 &v)
{
	int status;

	status = dCmp(data[0], v[0]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	status = dCmp(data[1], v[1]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	status = dCmp(data[2], v[2]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	return false;
}

inline bool Vec3::operator>=(const Vec3 &v)
{
	int status;

	status = dCmp(data[0], v[0]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	status = dCmp(data[1], v[1]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	status = dCmp(data[2], v[2]);
	switch(status) {
	  case 1:
		return true;
	  case -1:
		return false;
	}

	return true;
}

inline bool Vec3::operator<=(const Vec3 &v)
{
	int status;

	status = dCmp(data[0], v[0]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	status = dCmp(data[1], v[1]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	status = dCmp(data[2], v[2]);
	switch(status) {
	  case -1:
		return true;
	  case 1:
		return false;
	}

	return true;
}


//-----------------------------------------------------------------------------
// Vec2クラスの実装
//-----------------------------------------------------------------------------
inline bool Vec2::operator>(const Vec2 &v)
{
	return data[0]*data[0]+data[1]*data[1] > v[0]*v[0]+v[1]*v[1];
}

inline bool Vec2::operator<(const Vec2 &v)
{
	return data[0]*data[0]+data[1]*data[1] < v[0]*v[0]+v[1]*v[1];
}

inline bool Vec2::operator>=(const Vec2 &v)
{
	return data[0]*data[0]+data[1]*data[1] >= v[0]*v[0]+v[1]*v[1];
}

inline bool Vec2::operator<=(const Vec2 &v)
{
	return data[0]*data[0]+data[1]*data[1] <= v[0]*v[0]+v[1]*v[1];
}


//-----------------------------------------------------------------------------
// Vec3クラスに関連した関数
//-----------------------------------------------------------------------------
inline Vec3 operator*(const double &s, const Vec3 &v)
{
	return Vec3(v[0]*s, v[1]*s, v[2]*s);
}

inline Vec3 operator*(const Vec3 &v, const double &s)
{
	return Vec3(v[0]*s, v[1]*s, v[2]*s);
}

inline Vec3 operator*(const int &s, const Vec3 &v)
{
	return Vec3(v[0]*s, v[1]*s, v[2]*s);
}

inline Vec3 operator*(const Vec3 &v, const int &s)
{
	return Vec3(v[0]*s, v[1]*s, v[2]*s);
}

inline Vec3 operator/(const Vec3 &v, const double &s)
{
	double inv = 1.0/s;		
	return Vec3(v[0]*inv, v[1]*inv, v[2]*inv);
}

inline Vec3 operator-(const Vec3& a, const Vec3& b)
{
	return Vec3(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
}

inline Vec3 operator-(const Vec3& a)
{
	return Vec3(-a[0], -a[1], -a[2]);
}

inline Vec3 operator+(const Vec3& a, const Vec3& b)
{
	return Vec3(a[0]+b[0], a[1]+b[1], a[2]+b[2]);
}

inline Vec3 operator*(const Vec3& a, const Vec3& b)
{
	return Vec3(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}

inline Vec3 operator/(const Vec3& a, const Vec3& b)
{
	return Vec3(a[0]/b[0], a[1]/b[1], a[2]/b[2]);
}

inline double dot(const Vec3& a, const Vec3& b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline Vec3 cross(const Vec3& a, const Vec3& b)
{
	return Vec3(a[1]*b[2]-a[2]*b[1],
				a[2]*b[0]-a[0]*b[2],
				a[0]*b[1]-a[1]*b[0]);
}

inline double norm2(const Vec3& a)
{
	return dot(a,a);
}

inline double norm(const Vec3& a)
{
	return sqrt(dot(a,a));
}

inline double length(const Vec3& a)
{
	return sqrt(dot(a,a));
}


inline double unitize(Vec3& a)
{
	double l = norm2(a);
	if(l != 1.0 && l > 0.0)
	{
		l = sqrt(l);
		a /= l;
	}
	return l;
}

inline double normalize(Vec3& a)
{
	return unitize(a);
}


inline Vec3 remCompB(const Vec3 &A, const Vec3 &B)  // Remove component parallel to B.
{
	Vec3 C;					// Cumbersome due to compiler falure.
	double x = norm2(B);
	if(x > 0.0)
		C = (A-B) * (dot(A,B) / x);
	else
		C = A;

	return C;
}


inline void remCompB2(Vec3 &A, const Vec3 &B) // Remove component parallel to B.
{
	double x = norm2(B);
	if (x > 0.0)
		(A-=B)*(dot(A,B) / x);
}


inline Vec3 Unit(const Vec3 &A)
{
	//Vec3d d = A;
	//d = d/norm(d);
	//return d;
	double d = norm2(A);
	return d > 0.0 ? A * sqrt(1.0/d) : Vec3(0.0);
}


inline Vec3 Unit(const double &x, const double &y, const double &z)
{
	return Unit(Vec3(x,y,z));
}


inline std::ostream &operator<<(std::ostream &out, const Vec3 &a)
{
	return out << "(" << a[0] << ", " << a[1] << ", " << a[2] << ")" ;
}


inline std::istream &operator>>(std::istream &in, Vec3& a)
{
	return in >> a[0] >> a[1] >> a[2] ;
}

//typedef Vec3		Vec3d;



//-----------------------------------------------------------------------------
// Vec4クラスに関連した関数
//-----------------------------------------------------------------------------
inline Vec4 operator*(double s, const Vec4 &v)
{
	return Vec4(v[0]*s, v[1]*s, v[2]*s, v[3]*s);
}

inline Vec4 operator*(const Vec4 &v, double s)
{
	return s*v;
}


inline Vec4 operator/(const Vec4 &v, double s)
{
	double inv = 1.0/s;
	return Vec4(v[0]*inv, v[1]*inv, v[2]*inv, v[3]*inv);
}


inline Vec4 operator*(int s, const Vec4 &v)
{
	return Vec4(v[0]*s, v[1]*s, v[2]*s, v[3]*s);
}


inline Vec4 operator*(const Vec4 &v, int s)
{
	return s*v;
}


inline Vec4 operator/(const Vec4 &v, int s)
{
	double inv = 1.0/s;
	return Vec4(v[0]*inv, v[1]*inv, v[2]*inv, v[3]*inv);
}


inline Vec4 operator-(const Vec4& a, const Vec4& b)
{
	return Vec4(a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]);
}


inline Vec4 operator-(const Vec4& a)
{
	return Vec4(-a[0], -a[1], - a[2], - a[3]);
}


inline Vec4 operator+(const Vec4& a, const Vec4& b)
{
	return Vec4(a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3]);
}


inline Vec4 operator*(const Vec4& a, const Vec4& b)
{
	return Vec4(a[0]*b[0], a[1]*b[1], a[2]*b[2], a[3]*b[3]);
}


inline Vec4 operator/(const Vec4& a, const Vec4& b)
{
	double inv = 1.0/(b[0]*b[1]*b[2]*b[3]);
	return Vec4(a[0]*b[1]*b[2]*b[3]*inv, a[1]*b[0]*b[2]*b[3]*inv, a[2]*b[0]*b[1]*b[3]*inv, a[2]*b[0]*b[1]*b[2]*inv);
}


inline double dot(const Vec4& a, const Vec4& b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
}



inline double norm2(const Vec4& a)
{
	return dot(a,a);
}


inline double norm(const Vec4& a)
{
	return sqrt(dot(a,a));
}


inline double unitize(Vec4& a)
{
	double l = norm2(a);
	if(l != 1.0 && !(fabs(l) < VEC_RX_FEQ_EPS))
	{
		l = sqrt(l);
		a /= l;
	}
	return l;
}

inline double normalize(Vec4& a)
{
	return unitize(a);
}


inline Vec3 proj(const Vec4& a)
{
	Vec3 u(a[0], a[1], a[2]);
	if(a[3] != 1.0 && !(fabs(a[3]) < VEC_RX_FEQ_EPS))
		u /= a[3];
	return u;
}


inline std::ostream &operator<<(std::ostream &out, const Vec4 &a)
{
	return out << "(" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << ")" ;
}


inline std::istream &operator>>(std::istream &in, Vec4& a)
{
	return in >> a[0] >> a[1] >> a[2] >> a[3] ;
}

inline void vec4_to_array(const Vec4& a, double *b)
{
	b[0] = a[0]; b[1] = a[1];
	b[2] = a[2]; b[3] = a[3];
}



//-----------------------------------------------------------------------------
// Vec2クラスに関連した関数
//-----------------------------------------------------------------------------
inline Vec2 operator*(const double s, const Vec2 &v)
{
	return Vec2(v[0]*s, v[1]*s);
}

inline Vec2 operator*(const Vec2 &v, const double s)
{
	return s*v;
}

inline Vec2 operator/(const Vec2 &v, const double s)
{
	double inv = 1.0/s;		
	return Vec2(v[0]*inv, v[1]*inv);
}

inline Vec2 operator-(const Vec2& a, const Vec2& b)
{
	return Vec2(a[0]-b[0], a[1]-b[1]);
}

inline Vec2 operator-(const Vec2& a)
{
	return Vec2(-a[0], -a[1]);
}

inline Vec2 operator+(const Vec2& a, const Vec2& b)
{
	return Vec2(a[0]+b[0], a[1]+b[1]);
}

inline Vec2 operator*(const Vec2& a, const Vec2& b)
{
	return Vec2(a[0]*b[0], a[1]*b[1]);
}

inline Vec2 operator/(const Vec2& a, const Vec2& b)
{
	return Vec2(a[0]/b[0], a[1]/b[1]);
}

inline double dot(const Vec2& a, const Vec2& b)
{
	return a[0]*b[0] + a[1]*b[1];
}

inline double norm2(const Vec2& a)
{
	return dot(a,a);
}

inline double norm(const Vec2& a)
{
	return sqrt(dot(a,a));
}

inline Vec2 cross(const Vec2& a, const Vec2& b)
{
	return Vec2(-a[1]*b[0], a[0]*b[1]);
}


inline double unitize(Vec2& a)
{
	double l = norm2(a);
	if(l != 1.0 && l > 0.0)
	{
		l = sqrt(l);
		a /= l;
	}
	return l;
}

inline double normalize(Vec2& a)
{
	return unitize(a);
}


inline Vec2 remCompB(const Vec2 &A, const Vec2 &B)  // Remove component parallel to B.
{
	Vec2 C;					// Cumbersome due to compiler falure.
	double x = norm2(B);
	if(x > 0.0)
		C = (A-B) * (dot(A,B) / x);
	else
		C = A;

	return C;
}


inline void remCompB2(Vec2 &A, const Vec2 &B) // Remove component parallel to B.
{
	double x = norm2(B);
	if (x > 0.0)
		(A-=B)*(dot(A,B) / x);
}


inline Vec2 Unit(const Vec2 &A)
{
	//Vec2d d = A;
	//d = d/norm(d);
	//return d;
	double d = norm2(A);
	return d > 0.0 ? A * sqrt(1.0/d) : Vec2(0.0);
}


inline Vec2 Unit(const double &x, const double &y)
{
	return Unit(Vec2(x,y));
}


inline std::ostream &operator<<(std::ostream &out, const Vec2 &a)
{
	return out << "(" << a[0] << ", " << a[1] << ")" ;
}


inline std::istream &operator>>(std::istream &in, Vec2& a)
{
	return in >> a[0] >> a[1];
}




#endif // _VEC_H_