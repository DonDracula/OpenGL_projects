#ifndef VECTOR_H
#define VECTOR_H

#include <cassert>
#include <cmath>
#include <cstring>
#include <memory>
#include <iostream>

#include "matrix.h"


template<int N>
class vector
{
public:
    vector()
    {
        for (int i = 0; i < N; ++i) {
            p[i] = 0;
        }
    }
    
    vector(const float u[N])
    {
        memcpy(p, u, N * sizeof(float));
    }

    float& operator[](int index)
    {
        assert(index >= 0 && index < N);

        return p[index];
    }

    const float& operator[](int index) const
    {
        assert(index >= 0 && index < N);

        return p[index];
    }

    void set(const vector<N>& u) 
    { 
        memcpy(p, u.p, N * sizeof(float));
    }
    
    void add(const vector<N>& v) 
    {
        for (int i = 0; i < N; ++i) {
            p[i] += v[i];
        }
    }
    
    void add(const vector<N>& u, const vector<N>& v) 
    {
        for (int i = 0; i < N; ++i) {
            p[i] = u[i] + v[i];
        }
    }

    void sub(const vector<N>& v) 
    {
        for (int i = 0; i < N; ++i) {
            p[i] -= v[i];
        }
    }
    
    void sub(const vector<N>& u, const vector<N>& v) 
    {
        for (int i = 0; i < N; ++i) {
            p[i] = u[i] - v[i];
        }
    }

    void scale(float s) 
    {
        for (int i = 0; i < N; ++i) {
            p[i] *= s;
        }
    }
    
    template<int M>
    void mult(const matrix<N,M>& A, const vector<M>& u)
    {
        for (int i = 0; i < M; ++i) {
            p[i] = 0;
            for (int j = 0; j < N; ++j) {
                p[i] += A(i,j) * u[j];
            }
        }
    }

    float mag() const 
    {
        float mag2 = 0;
        for (int i = 0; i < N; ++i) {
            mag2 += p[i]*p[i];
        }
        return sqrt(mag2);
    }

    void normalize() 
    {
        float m = mag();
        assert(m > 0);
        scale(1/m);
    }

    float* data()
    {
        return p;
    }
    
    vector<N> operator+(const vector<N>& v) const 
    {
        vector<N> u(*this);
        u.add(v);
        return u;
    }

    vector<N> operator-(const vector<N>& v) const 
    {
        vector<N> u(*this);
        u.sub(v);
        return u;
    }
    
    void operator+=(const vector<N>& v) 
    {
        add(v);
    }

    void operator-=(const vector<N>& v)
    {
        sub(v);
    }

    vector<N> operator*(float s) const 
    {
        vector<N> u(*this);
        u.scale(s);
        return u;
    }

    vector<N> operator/(float s) const 
    {
        vector<N> u(*this);
        assert(s != 0);
        u.scale(1.0/s);
        return u;
    }
    
protected:
    float p[N];
}; 

class vector3 : public vector<3>
{
public:
    vector3()
    {
    }

    vector3(float x, float y, float z) 
    {
        set(x, y, z);
    }
    
    void set(float x, float y, float z) 
    {
        p[0] = x; p[1] = y; p[2] = z;
    }
    
    void set(const vector3& u)
    {
        vector<3>::set(u);
    }

    void cross(const vector3& u, const vector3& v) 
    {
        p[0] = u[1]*v[2] - u[2]*v[1]; 
        p[1] = u[2]*v[0] - u[0]*v[2]; 
        p[2] = u[0]*v[1] - u[1]*v[0]; 
    }
    
    vector3 operator+(const vector3& v) const 
    {
        vector3 u(*this);
        u.add(v);
        return u;
    }

    vector3 operator-(const vector3& v) const 
    {
        vector3 u(*this);
        u.sub(v);
        return u;
    }
    
    void operator+=(const vector3& v) 
    {
        add(v);
    }

    void operator-=(const vector3& v)
    {
        sub(v);
    }

    vector3& operator=(const vector3& u) {
        set(u);
        return *this;
    }
    
    vector3 operator*(float f) {
        vector3 tmp(*this);
        tmp.scale(f);
        return tmp;
    }

};

typedef vector<2> vector2;
typedef vector<4> vector4;

template<int N>
float dot(const vector<N>& u, const vector<N>& v)
{
    float uv = 0;
    for (int i = 0; i < N; ++i) {
        uv += u[i] * v[i];
    }
    return uv;
}

template<int N>
vector<N> operator*(float s, const vector<N>& u) 
{
    vector<N> v;
    v.set(u);
    v.scale(s);
    return v;
}


template<int N>
std::ostream& operator<<(std::ostream& ostr, const vector<N>& u)
{
    for (int i = 0; i < N; ++i) {
        ostr << u[i] << " ";
    }
    return ostr;
}

#endif
