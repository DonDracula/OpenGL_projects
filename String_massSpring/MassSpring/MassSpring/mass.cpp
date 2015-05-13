#include "mass.h"

Mass :: Mass(float m)
{
	this->m=m;
}

Mass::~Mass()
{
}

void Mass::applyForce(Vec3 f)
{
	this->force +=f;
}

void Mass::init()
{
	force.data[0]=0;
	force.data[1]=0;
	force.data[2]=0;
}

void Mass::simulate(float dt)
{
	vel+=(force/m)*dt;
	pos+=vel*dt;
}

Vec3 Mass::getPos()
{
	return pos;
}

void Mass::setPos(Vec3 p)
{
	pos = p;
}

void Mass::setVel(Vec3 v)
{
	this->vel =v;
}

float Mass::getM()
{
	return this->m;
}


