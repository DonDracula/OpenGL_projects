#include "gl_camera.h"

#include <cmath>
#include <GL/gl.h>

#include "transform.h"

void gl_camera::set_elevation(float deg)
{
    float new_deg = deg;

    // get the remainder
    new_deg = new_deg - (int)(new_deg / 360.0) * 360.0;

    if (new_deg > 0)
    {
        if (new_deg > 180.0)
            new_deg = new_deg - 360;
    }
    else
    {
        if (new_deg < -180.0)
            new_deg =  new_deg + 360;

    }
    
    this->elevation = new_deg;
}

float gl_camera::get_elevation() const
{
    return this->elevation;
}

// the number of degrees about the z-axis
void gl_camera::set_twist(float deg)
{
    float new_deg = deg;

    // get the remainder
    new_deg = new_deg - (int)(new_deg / 360.0) * 360.0;

    if (new_deg > 0)
    {
        if (new_deg > 180.0)
            new_deg = new_deg - 360;
    }
    else
    {
        if (new_deg < -180.0)
            new_deg =  new_deg + 360;

    }
    
    this->twist = new_deg;
}

float gl_camera::get_twist() const
{
    return this->twist;
}

void gl_camera::set_distance(float distance)
{
    this->distance = distance;
}

float gl_camera::get_distance() const
{
    return distance;
}

void gl_camera::set_focal_point(const vector3& focal_point)
{
    this->focal_point = focal_point;
}

vector3 gl_camera::get_focal_point() const 
{
    return focal_point;
}
    
vector3 gl_camera::get_position() const
{
    
    vector3 p;
    
    float t1, t2;
    float b;
    
    p[2] = get_distance();
    p[0] = p[1] = 0.0;
   
    // rotate about x
    b = -get_elevation() / 180.0 * M_PI;
    t1 = cos(b)*p[1] - sin(b)*p[2];
    t2 = sin(b)*p[1] + cos(b)*p[2];

    p[1] = t1;
    p[2] = t2;
    
    // rotate about y
    b = get_twist() / 180.0 * M_PI;
    t1 = cos(b)*p[0] + sin(b)*p[2];
    t2 = -sin(b)*p[0] + cos(b)*p[2];
    
    p[0] = t1;
    p[2] = t2;
    
    // translate 
    p.add(get_focal_point());
    return p;
}

vector3 gl_camera::get_target() const
{
    vector3 tar;
    tar.set(get_focal_point());
    tar.sub(get_position());
    tar.normalize();
    return tar;
}
    
vector3 gl_camera::get_up() const
{
    vector3 up;
    up.cross(get_right(), get_target());
    return up;
}

vector3 gl_camera::get_right() const
{
    vector3 Y;
    Y[1] = 1;

    vector3 right;
    right.cross(get_target(), Y);
    right.normalize();
    return right;
}

float to_radian(float degree) {
    return degree*M_PI/180.0;
}

void gl_camera::get_model_transform(matrix44& M)
{
    matrix44 T_twist, T_elevate, T_focal, T_dist;
    matrix44 tmp1, tmp2;

    translate(
        0, 0, -distance, T_dist
        );
    rotate_x(
        to_radian(get_elevation()), T_elevate
        );
    rotate_y(
        -to_radian(get_twist()), T_twist
        );
    translate(
        -focal_point[0], -focal_point[1], -focal_point[2], T_focal
        );

    tmp1.mult(T_dist, T_elevate);
    tmp2.mult(T_twist, T_focal);
    M.mult(tmp1, tmp2);
}

void gl_camera::apply_gl_transform()
{

    glTranslated(0.0, 0.0, -distance);
    glRotated(get_elevation(), 1.0, 0.0, 0.0);
    glRotated(-get_twist(), 0.0, 1.0, 0.0);
    glTranslated(
        -focal_point[0], -focal_point[1], -focal_point[2]
    );
}
