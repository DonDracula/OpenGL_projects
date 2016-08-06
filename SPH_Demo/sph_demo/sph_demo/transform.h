#ifndef TRANSFORM_H
#define TRANSFORM_H

#include <cmath>

inline void rotate_x(float angle, matrix44& T)
{
    T.zero();
    float cosa = cos(angle);
    float sina = sin(angle);
    T(0,0) = 1;
    T(1,1) = cosa; T(1,2) = -sina;
    T(2,1) = sina; T(2,2) = cosa;
    T(3,3) = 1;
}

inline void rotate_y(float angle, matrix44& T)
{
    T.zero();
    float cosa = cos(angle);
    float sina = sin(angle);
    T(0,0) = cosa; T(0,2) = sina;
    T(1,1) = 1;
    T(2,0) = -sina; T(2,2) = cosa;
    T(3,3) = 1;
}

inline void rotate_z(float angle, matrix44& T)
{
    T.zero();
    float cosa = cos(angle);
    float sina = sin(angle);
    T(0,0) = cosa; T(0,1) = -sina;
    T(1,0) = sina; T(1,1) = cosa;
    T(2,2) = 1;
    T(3,3) = 1;
}

inline void translate(float tx, float ty, float tz, matrix44& T)
{
    T.zero();
    T(0,0) = T(1,1) = T(2,2) = 1;
    T(0,3) = tx;
    T(1,3) = ty;
    T(2,3) = tz;
    T(3,3) = 1;
}

inline void scale(float sx, float sy, float sz, matrix44& T)
{
    T.zero();
    T(0,0) = sx;
    T(1,1) = sy;
    T(2,2) = sz;
    T(3,3) = 1;
}

inline void transform(const matrix44& T, vector3& u)
{
    vector4 uh, uht;
    for (int i = 0; i < 3; ++i) {
        uh[i] = u[i];
    }
    uh[3] = 1;

    uht.mult(T, uh);
    
    for (int i = 0; i < 3; ++i) {
        u[i] = uht[i];
    }
}

#endif
