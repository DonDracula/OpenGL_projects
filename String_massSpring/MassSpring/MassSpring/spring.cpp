#include "spring.h"

Spring:: Spring()
{
	//ctor
}

Spring ::Spring()
{
	//dtor
}

Spring::Spring(Mass*mass1,Mass*mass2,float springConstant,float springLength,float frictionConstant)
{
	this->springConstant =springConstant;				//set teh springConstant
	this->resLength =springLength;						//set the springLength
	this->frictionConstant =frictionConstant;			//set the frictionConstant

	this->mass1 =mass1;									//set mass1
	this->mass2 =mass2;
}

void Spring:: solve()
{
	Vec3 springVector =mass1->getPos() - mass2->getPos();
	float r =length(springVector);

	Vec3 force(0.0f);
	if(0!=r)
	{
		force+=(springVector/r)*(r-resLength)*(-springConstant);
	}
	force +=-(mass1->getVel()-mass2->getVel())*frictionConstant;

	mass1->applyForce(force);						//force is applied to mass1
	mass2->applyForce(force);
}