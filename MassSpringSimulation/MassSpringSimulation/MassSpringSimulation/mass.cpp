#include "mass.h"

Mass::Mass(float m)
{
	this->m =m;
}

Mass::~Mass()
{
}

void Mass::applyForce(Vec3 f)
{
	this->force+=f;								// teh externall force is added to the force of the mass
}

void Mass::init()
{
	force.data[0] = 0;
	force.data[1] =0;
	force.data[2] =0;
}

void Mass::simulate(float dt)
{
	vel +=(force/m)*dt;							//change in velocity is added toe the velocity
												//the change is proportinal with the acceleration(force/m) and change in time
	pos +=vel*dt;								//change in position is added to the position
												//change in positon is velocity times the change in time
}

Vec3 Mass::getPos()
{
	return pos;
}

void Mass::setPos(Vec3 p)
{
	pos=p;
}

void Mass::setVel(Vec3 v)
{
	this->vel = v;
}

Vec3 Mass::getVel()
{
	return this->vel;
}

float Mass::getM()
{
	return this->m;
}