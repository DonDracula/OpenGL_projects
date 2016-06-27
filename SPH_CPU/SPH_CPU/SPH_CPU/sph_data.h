/*
	include the definitions which will be used in the SPH method

*/

#ifndef _SPHDATA_H
#define _SPHDATA_H

#include "sph_header.h"
#include "sph_type.h"

float window_width=1000;
float window_height=1000;

float xRot = 15.0f;
float yRot = 0.0f;
float xTrans = 0.0f;
float yTrans = 0.0f;
float zTrans = -35.0;

int ox;
int oy;
int buttonState;
float xRotLength = 0.0f;
float yRotLength = 0.0f;

Vec3_f real_world_origin;
Vec3_f real_world_side;
Vec3_f sim_ratio;

//size of the world
float world_witdth;
float world_height;
float world_length;
	
#endif //_SPHDATA_H