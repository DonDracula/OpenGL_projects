/*! @file rx_quaternion.h
	
	@brief ベクトル・行列ライブラリ-クオータニオン
 
	@author Makoto Fujisawa
	@date  
*/

#ifndef _QUATERNION_H_
#define _QUATERNION_H_


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <memory.h>
#include "rx_matrix.h"


//-----------------------------------------------------------------------------
//! クオータニオンクラス
//-----------------------------------------------------------------------------
class rxQuaternion
{
protected:
    union 
    {
        struct 
        {
            real q[4];
        };
        struct 
        {
            real x;
            real y;
            real z;
            real w;
        };
    };

    // renormalization counter
    unsigned char counter;

	void counterNormalize()
    {
		if(counter > QUATERNION_NORMALIZATION_THRESHOLD){
            normalize();
		}
    }

public:
	// コンストラクタ
	rxQuaternion()
	{
		*this = identity();
	}

	rxQuaternion(const real v[4])
	{
		setValue(v);
	}

	rxQuaternion(real q0, real q1, real q2, real q3)
	{
		setValue(q0, q1, q2, q3);
	}

	rxQuaternion(const rxMatrix4 & m)
	{
		setValue(m);
	}

	rxQuaternion(const Vec3 &axis, real radians)
	{
		setValue(axis, radians);
	}

	rxQuaternion(const Vec3 &rotateFrom, const Vec3 &rotateTo)
	{
		setValue(rotateFrom, rotateTo);
	}

	rxQuaternion(const Vec3 & from_look, const Vec3 & from_up,
			   const Vec3 & to_look, const Vec3& to_up)
	{
		setValue(from_look, from_up, to_look, to_up);
	}

	const real* getValue() const
	{
		return  &q[0];
	}

	void getValue(real &q0, real &q1, real &q2, real &q3) const
	{
		q0 = q[0];
		q1 = q[1];
		q2 = q[2];
		q3 = q[3];
	}

	rxQuaternion& setValue(real q0, real q1, real q2, real q3)
	{
		q[0] = q0;
		q[1] = q1;
		q[2] = q2;
		q[3] = q3;
		counter = 0;
		return *this;
	}

	void getValue(Vec3 &axis, real &radians) const
	{
		radians = (real)(2.0*acos(q[3]));
		if(radians == 0.0){
			axis = Vec3(0.0, 0.0, 1.0);
		}
		else{
			axis[0] = q[0];
			axis[1] = q[1];
			axis[2] = q[2];
			unitize(axis);
		}
	}

	real GetAngle(void)
	{
		return (real)(2.0*acos(q[3]));
	}

	Vec3 GetAxis(void)
	{
		Vec3 axis;
		real radians;

		getValue(axis, radians);

		return axis;
	}

	Vec3 GetVector(void)
	{
		return Vec3(q[0], q[1], q[2]);
	}

	void getValue(rxMatrix4 &m) const
	{
		real s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

		real norm = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];

		s = (equivalent(norm, 0.0)) ? 0.0 : (2.0/norm);

		xs = q[0]*s;
		ys = q[1]*s;
		zs = q[2]*s;

		wx = q[3]*xs;
		wy = q[3]*ys;
		wz = q[3]*zs;

		xx = q[0]*xs;
		xy = q[0]*ys;
		xz = q[0]*zs;

		yy = q[1]*ys;
		yz = q[1]*zs;
		zz = q[2]*zs;

		m(0,0) = (real)(1.0-(yy+zz));
		m(1,0) = (real)(xy+wz);
		m(2,0) = (real)(xz-wy);

		m(0,1) = (real)(xy-wz);
		m(1,1) = (real)(1.0-(xx+zz));
		m(2,1) = (real)(yz+wx);

		m(0,2) = (real)(xz+wy);
		m(1,2) = (real)(yz-wx);
		m(2,2) = (real)(1.0-(xx+yy));

		m(3,0) = m(3,1) = m(3,2) = m(0,3) = m(1,3) = m(2,3) = 0.0;
		m(3,3) = 1.0;
	}

	void GetValue(rxMatrix3 &m) const
	{
		real s, xs, ys, zs, wx, wy, wz, xx, xy, xz, yy, yz, zz;

		real norm = q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];

		s = (equivalent(norm, 0.0)) ? 0.0 : (2.0/norm);

		xs = q[0]*s;
		ys = q[1]*s;
		zs = q[2]*s;

		wx = q[3]*xs;
		wy = q[3]*ys;
		wz = q[3]*zs;

		xx = q[0]*xs;
		xy = q[0]*ys;
		xz = q[0]*zs;

		yy = q[1]*ys;
		yz = q[1]*zs;
		zz = q[2]*zs;

		m(0,0) = (real)(1.0-(yy+zz));
		m(1,0) = (real)(xy+wz);
		m(2,0) = (real)(xz-wy);

		m(0,1) = (real)(xy-wz);
		m(1,1) = (real)(1.0-(xx+zz));
		m(2,1) = (real)(yz+wx);

		m(0,2) = (real)(xz+wy);
		m(1,2) = (real)(yz-wx);
		m(2,2) = (real)(1.0-(xx+yy));
	}


	/*!
	 * 四元数からオイラー角を取得
	 * @return オイラー角(Vec3,Degree)
	 */
	Vec3 GetEulerAngles(void)
	{
		real r11, r21, r31, r32, r33, r12, r13;
		real q00, q11, q22, q33;
		real tmp;
		Vec3 u;

		q00 = q[3]*q[3];
		q11 = q[0]*q[0];
		q22 = q[1]*q[1];
		q33 = q[2]*q[2];

		r11 = q00+q11-q22-q33;
		r21 = 2*(q[0]*q[1]+q[3]*q[2]);
		r31 = 2*(q[0]*q[2]-q[3]*q[1]);
		r32 = 2*(q[1]*q[2]+q[3]*q[0]);
		r33 = q00-q11-q22+q33;

		tmp = fabs(r31);
		if(tmp > 0.999999){
			r12 = 2*(q[0]*q[1]-q[3]*q[2]);
			r13 = 2*(q[0]*q[2]+q[3]*q[1]);

			u[0] = RX_TO_DEGREES(0.0f);							//roll
			u[1] = RX_TO_DEGREES((-(RX_PI/2)*r31/tmp));		// pitch
			u[2] = RX_TO_DEGREES(atan2(-r12, -r31*r13));	// yaw
			return u;
		}

		u[0] = RX_TO_DEGREES(atan2(r32, r33));	// roll
		u[1] = RX_TO_DEGREES(asin(-r31));		// pitch
		u[2] = RX_TO_DEGREES(atan2(r21, r11));	// yaw

		return u;
	}

	/*!
	 * オイラー角から四元数を設定
	 * @param e オイラー角(Vec3,Degree)
	 */
	void SetEulerAngles(Vec3 e)
	{
		double roll  = RX_TO_DEGREES(e[0]);
		double pitch = RX_TO_DEGREES(e[1]);
		double yaw   = RX_TO_DEGREES(e[2]);
		
		double cyaw, cpitch, croll, syaw, spitch, sroll;
		double cyawcpitch, syawspitch, cyawspitch, syawcpitch;

		cyaw   = cos(0.5*yaw);
		cpitch = cos(0.5*pitch);
		croll  = cos(0.5*roll);
		syaw   = sin(0.5*yaw);
		spitch = sin(0.5*pitch);
		sroll  = sin(0.5*roll);

		cyawcpitch = cyaw*cpitch;
		syawspitch = syaw*spitch;
		cyawspitch = cyaw*spitch;
		syawcpitch = syaw*cpitch;

		q[3] = cyawcpitch*croll+syawspitch*sroll;
		q[0] = cyawcpitch*sroll-syawspitch*croll;
		q[1] = cyawspitch*croll+syawcpitch*sroll;
		q[2] = syawcpitch*croll-cyawspitch*sroll;
	}



	rxQuaternion& setValue(const real *qp)
	{
		memcpy(q,qp,sizeof(real)*4);

		counter = 0;
		return *this;
	}

	rxQuaternion& setValue(const rxMatrix4 &m)
	{
		real tr, s;
		int i, j, k;
		const int nxt[3] = { 1, 2, 0 };

		tr = m(0,0)+m(1,1)+m(2,2);

		if(tr > 0.0){
			s = real(sqrt(tr+m(3,3)));
			q[3] = real (s*0.5);
			s = real(0.5)/s;

			q[0] = real ((m(1,2)-m(2,1))*s);
			q[1] = real ((m(2,0)-m(0,2))*s);
			q[2] = real ((m(0,1)-m(1,0))*s);
		}
		else{
			i = 0;
			if(m(1,1) > m(0,0))
				i = 1;

			if(m(2,2) > m(i,i))
				i = 2;

			j = nxt[i];
			k = nxt[j];

			s = real(sqrt((m(i,j)-(m(j,j)+m(k,k)))+1.0));

			q[i] = real (s*0.5);
			s = real(0.5/s);

			q[3] = real((m(j,k)-m(k,j))*s);
			q[j] = real((m(i,j)+m(j,i))*s);
			q[k] = real((m(i,k)+m(k,i))*s);
		}

		counter = 0;
		return *this;
	}

	rxQuaternion& setValue(const rxMatrix3 &m)
	{
		real tr, s;
		int i, j, k;
		const int nxt[3] = { 1, 2, 0 };

		tr = m(0,0)+m(1,1)+m(2,2);

		if(tr > 0.0){
			s = real(sqrt(tr+1.0));
			q[3] = real(s*0.5);
			s = real(0.5)/s;

			q[0] = real((m(1,2)-m(2,1))*s);
			q[1] = real((m(2,0)-m(0,2))*s);
			q[2] = real((m(0,1)-m(1,0))*s);
		}
		else{
			i = 0;
			if(m(1,1) > m(0,0))
				i = 1;

			if(m(2,2) > m(i,i))
				i = 2;

			j = nxt[i];
			k = nxt[j];

			s = real(sqrt((m(i,j)-(m(j,j)+m(k,k)))+1.0));

			q[i] = real(s*0.5);
			s = real(0.5/s);

			q[3] = real((m(j,k)-m(k,j))*s);
			q[j] = real((m(i,j)+m(j,i))*s);
			q[k] = real((m(i,k)+m(k,i))*s);
		}

		counter = 0;
		return *this;
	}

	void SetMatrix3(const rxMatrix3 &m)
	{
		real trace = m(0, 0)+m(1, 1)+m(2, 2);
		
		if (trace > 0.0) 
		{
			double s = sqrt(trace+1.0);
			q[3] = (real)(s*0.5);
			s = 0.5/s;
			
			q[0] = (real)(m(2, 1)-m(1, 2))*s;
			q[1] = (real)(m(0, 2)-m(2, 0))*s;
			q[2] = (real)(m(1, 0)-m(0, 1))*s;
		} 
		else 
		{
			int i = m(0, 0) < m(1, 1) ? 
				(m(1, 1) < m(2, 2) ? 2 : 1) :
				(m(0, 0) < m(2, 2) ? 2 : 0); 
			int j = (i+1) % 3;  
			int k = (i+2) % 3;
			
			double s = sqrt(m(i, i)-m(j, j)-m(k, k)+1.0);
			q[i] = (real)(s*0.5);
			s = 0.5/s;
			
			q[3] = (real)(m(k, j)-m(j, k))*s;
			q[j] = (real)(m(j, i)+m(i, j))*s;
			q[k] = (real)(m(k, i)+m(i, k))*s;
		}
	}

	rxQuaternion& setValue(const Vec3 &axis, real theta)
	{
		real sqnorm = norm2(axis);

		if(sqnorm <= RX_EPSILON){
			// axis too small.
			x = y = z = 0.0;
			w = 1.0;
		} 
		else{
			theta *= real(0.5);
			real sin_theta = real(sin(theta));

			if(!equivalent(sqnorm,1.0)) 
				sin_theta /= real(sqrt(sqnorm));
			x = sin_theta*axis[0];
			y = sin_theta*axis[1];
			z = sin_theta*axis[2];
			w = real(cos(theta));
		}
		return *this;
	}

	rxQuaternion& setValue(const Vec3 & rotateFrom, const Vec3 & rotateTo)
	{
		Vec3 p1, p2;
		real alpha;

		p1 = rotateFrom; 
		unitize(p1);
		p2 = rotateTo;  
		unitize(p2);

		alpha = dot(p1, p2);

		if(equivalent(alpha,1.0)){ 
			*this = identity(); 
			return *this; 
		}

		// ensures that the anti-parallel case leads to a positive dot
		if(equivalent(alpha,-1.0)){
			Vec3 v;

			if(p1[0] != p1[1] || p1[0] != p1[2]){
    			v = Vec3(p1[1], p1[2], p1[0]);
			}
			else{
    			v = Vec3(-p1[0], p1[1], p1[2]);
			}

			v -= p1*dot(p1, v);
			unitize(v);

			setValue(v, RX_PI);
			return *this;
		}

		p1 = cross(p1, p2);  
		unitize(p1);
		setValue(p1,real(acos(alpha)));

		counter = 0;
		return *this;
	}

	rxQuaternion& setValue(const Vec3 & from_look, const Vec3 & from_up,
						 const Vec3 & to_look, const Vec3 & to_up)
	{
		rxQuaternion r_look = rxQuaternion(from_look, to_look);
		
		Vec3 rotated_from_up(from_up);
		r_look.multVec(rotated_from_up);
		
		rxQuaternion r_twist = rxQuaternion(rotated_from_up, to_up);
		
		*this = r_twist;
		*this *= r_look;
		return *this;
	}


	void normalize()
	{
		real rnorm = 1.0/real(sqrt(w*w+x*x+y*y+z*z));

		if(equivalent(rnorm, 0.0)){
			return;
		}

		x *= rnorm;
		y *= rnorm;
		z *= rnorm;
		w *= rnorm;
		counter = 0;
	}

	bool equals(const rxQuaternion &r, real tolerance) const
	{
		real t;
		t = ((q[0]-r.q[0])*(q[0]-r.q[0])+(q[1]-r.q[1])*(q[1]-r.q[1])+(q[2]-r.q[2])*(q[2]-r.q[2])+(q[3]-r.q[3])*(q[3]-r.q[3]));

		if(t > RX_EPSILON){
			return false;
		}
		return 1;
	}

	rxQuaternion& conjugate()
	{
		q[0] *= -1.0;
		q[1] *= -1.0;
		q[2] *= -1.0;
		return *this;
	}

	rxQuaternion& invert()
	{
		return conjugate();
	}

	rxQuaternion inverse() const
	{
		rxQuaternion r = *this;
		return r.invert();
	}

	//
	// rxQuaternion multiplication with cartesian vector
	// v' = q*v*q(star)
	//
	void multVec(const Vec3 &src, Vec3 &dst) const
	{
		real v_coef = w*w-x*x-y*y-z*z;                     
		real u_coef = 2.0*(src[0]*x+src[1]*y+src[2]*z);  
		real c_coef = 2.0*w;                                       

		dst[0] = v_coef*src[0]+u_coef*x+c_coef*(y*src[2]-z*src[1]);
		dst[1] = v_coef*src[1]+u_coef*y+c_coef*(z*src[0]-x*src[2]);
		dst[2] = v_coef*src[2]+u_coef*z+c_coef*(x*src[1]-y*src[0]);
	}

	void multVec(Vec3 & src_and_dst) const
	{
		multVec(Vec3(src_and_dst), src_and_dst);
	}

	void scaleAngle(real scaleFactor)
	{
		Vec3 axis;
		real radians;

		getValue(axis, radians);
		radians *= scaleFactor;
		setValue(axis, radians);
	}

	static rxQuaternion slerp(const rxQuaternion & p, const rxQuaternion & q, real alpha)
	{
		rxQuaternion r;

		real cos_omega = p.x*q.x+p.y*q.y+p.z*q.z+p.w*q.w;
		// ifB is on opposite hemisphere from A, use -B instead
	    
		int bflip;
		if((bflip = (cos_omega < 0.0))){
			cos_omega = -cos_omega;
		}

		// complementary interpolation parameter
		real beta = 1.0-alpha;     

		if(cos_omega >= 1.0-RX_EPSILON){
			return p;
		}

		real omega = real(acos(cos_omega));
		real one_over_sin_omega = 1.0/real(sin(omega));

		beta    = real(sin(omega*beta) *one_over_sin_omega);
		alpha   = real(sin(omega*alpha)*one_over_sin_omega);

		if(bflip){
			alpha = -alpha;
		}

		r.x = beta*p.q[0]+ alpha*q.q[0];
		r.y = beta*p.q[1]+ alpha*q.q[1];
		r.z = beta*p.q[2]+ alpha*q.q[2];
		r.w = beta*p.q[3]+ alpha*q.q[3];

		return r;
	}

	static rxQuaternion identity()
	{
		static rxQuaternion ident(Vec3(1.0, 0.0, 0.0), 0.0);
		return ident;
	}


	//
	// オペレータ
	// 
	rxQuaternion& operator*=(const rxQuaternion &qr)
	{
		rxQuaternion ql(*this);

		w = ql.w*qr.w-ql.x*qr.x-ql.y*qr.y-ql.z*qr.z;
		x = ql.w*qr.x+ql.x*qr.w+ql.y*qr.z-ql.z*qr.y;
		y = ql.w*qr.y+ql.y*qr.w+ql.z*qr.x-ql.x*qr.z;
		z = ql.w*qr.z+ql.z*qr.w+ql.x*qr.y-ql.y*qr.x;

		counter += qr.counter;
		counter++;
		counterNormalize();
		return *this;
	}

	rxQuaternion& operator+=(const rxQuaternion &qr)
	{
		q[0] += qr.q[0];
		q[1] += qr.q[1];
		q[2] += qr.q[2];
		q[3] += qr.q[3];

		return *this;
	}

	rxQuaternion& operator-=(const rxQuaternion &qr)
	{
		q[0] -= qr.q[0];
		q[1] -= qr.q[1];
		q[2] -= qr.q[2];
		q[3] -= qr.q[3];

		return *this;
	}

	rxQuaternion& operator*=(const real &s)
	{
		q[0] *= s;
		q[1] *= s;
		q[2] *= s;
		q[3] *= s;

		return *this;
	}

	rxQuaternion& operator/=(const real &s)
	{
		q[0] /= s;
		q[1] /= s;
		q[2] /= s;
		q[3] /= s;

		return *this;
	}

	real& operator [](int i)
	{
		assert(i < 4);
		return q[i];
	}

	const real& operator [](int i) const
	{
		assert(i < 4);
		return q[i];
	}

	friend bool operator==(const rxQuaternion &q1, const rxQuaternion &q2);      
	friend bool operator!=(const rxQuaternion &q1, const rxQuaternion &q2);
	friend rxQuaternion operator+(const rxQuaternion &q1, const rxQuaternion &q2);
	friend rxQuaternion operator-(const rxQuaternion &q1, const rxQuaternion &q2);
	friend rxQuaternion operator*(const rxQuaternion &q1, const rxQuaternion &q2);
	friend rxQuaternion operator*(const rxQuaternion &q, const real &s);
	friend rxQuaternion operator*(const real &s, const rxQuaternion &q);
	friend rxQuaternion operator*(const rxQuaternion &q, const Vec3 &v);
	friend rxQuaternion operator*(const Vec3 &v, const rxQuaternion &q);
	friend rxQuaternion operator/(const rxQuaternion &q, const real &s);

	friend real norm(const rxQuaternion &q);
};


inline bool operator==(const rxQuaternion & q1, const rxQuaternion & q2)
{
    return (equivalent(q1.x, q2.x) &&
		    equivalent(q1.y, q2.y) &&
		    equivalent(q1.z, q2.z) &&
		    equivalent(q1.w, q2.w));
}

inline bool operator!=(const rxQuaternion &q1, const rxQuaternion &q2)
{ 
    return !(q1 == q2); 
}

inline rxQuaternion operator*(const rxQuaternion &q1, const rxQuaternion &q2)
{	
    rxQuaternion r(q1); 
    r *= q2; 
    return r; 
}

inline rxQuaternion operator+(const rxQuaternion &q1, const rxQuaternion &q2)
{
	return rxQuaternion(q1.q[0]+q2.q[0], q1.q[1]+q2.q[1], q1.q[2]+q2.q[2], q1.q[3]+q2.q[3]);
}

inline rxQuaternion operator-(const rxQuaternion &q1, const rxQuaternion &q2)
{
	return rxQuaternion(q1.q[0]-q2.q[0], q1.q[1]-q2.q[1], q1.q[2]-q2.q[2], q1.q[3]-q2.q[3]);
}

inline rxQuaternion operator*(const rxQuaternion &q, const real &s)
{
	return	rxQuaternion(q.q[0]*s, q.q[1]*s, q.q[2]*s, q.q[3]*s);
}

inline rxQuaternion operator*(const real &s, const rxQuaternion &q)
{
	return	rxQuaternion(q.q[0]*s, q.q[1]*s, q.q[2]*s, q.q[3]*s);
}

inline rxQuaternion operator*(const rxQuaternion &q, const Vec3 &v)
{
	return	rxQuaternion(q.q[3]*v[0]+q.q[1]*v[2]-q.q[2]*v[1],
						 q.q[3]*v[1]+q.q[2]*v[0]-q.q[0]*v[2],
						 q.q[3]*v[2]+q.q[0]*v[1]-q.q[1]*v[0],
						 -(q.q[0]*v[0]+q.q[1]*v[1]+q.q[2]*v[2]));
}

inline rxQuaternion operator*(const Vec3 &v, const rxQuaternion &q)
{
	return	rxQuaternion(q.q[3]*v[0]+q.q[2]*v[1]-q.q[1]*v[2],
						 q.q[3]*v[1]+q.q[0]*v[2]-q.q[2]*v[0],
						 q.q[3]*v[2]+q.q[1]*v[0]-q.q[0]*v[1],
						 -(q.q[0]*v[0]+q.q[1]*v[1]+q.q[2]*v[2]));
}

inline rxQuaternion operator/(const rxQuaternion &q, const real &s)
{
	return rxQuaternion(q.q[0]/s, q.q[1]/s, q.q[2]/s, q.q[3]/s);
}

inline rxQuaternion QRotate(const rxQuaternion &q1, const rxQuaternion &q2)
{
	return q1*q2*(q1.inverse());
}

inline Vec3 QVRotate(const rxQuaternion &q, const Vec3 &v)
{
	rxQuaternion t;

	t = q*v*(q.inverse());

	return t.GetVector();
}

inline real norm(const rxQuaternion &q)
{
	return (real)sqrt((double)(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]));
}

inline real norm2(const rxQuaternion &q)
{
	return q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
}


#endif// _QUATERNION_H_
