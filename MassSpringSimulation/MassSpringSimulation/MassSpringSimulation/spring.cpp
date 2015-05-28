#include "spring.h"

Spring::Spring()
{
}

Spring::~Spring()
{
}

Spring::Spring(Mass* mass1,Mass* mass2,float springConstant,
			   float springLength,float frictionConstant)
{
	this->springConstant = springConstant;									//set the springConstant
	this->restLength = springLength;									//set the springLength
	this->frictionConstant = frictionConstant;							//set the frictionConstant

	this->mass1 = mass1;												//set mass
	this->mass2 = mass2;

}

void Spring::solve()
{
	Vec3 springVector = mass1->getPos()-mass2->getPos();					//vector between the two masses
	float r = length(springVector);										//distance between the two masses

	Vec3 force(0.0);													//force initially has a zero value
	if(0!=r)															//to avoid a division by zero check if r is zero
	{
		force+=(springVector/r)*(r-restLength)*(-springConstant);		//the spring force is added to the force
	}
	force+=-(mass1->getVel()-mass2->getVel())*frictionConstant;				//the firiction force is added to the spring
																			//with this additoin we obtain the net force of the spring

	mass1->applyForce(force);											//force is applied to mass1
	mass2->applyForce(-force);											//the opposite of force to mass2
}