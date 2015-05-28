#include "RopeSimulator.h"

RopeSimulator::RopeSimulator()
{
}

RopeSimulator::~RopeSimulator()
{
}

RopeSimulator::RopeSimulator(									//constructor
		int numOfMasses,							//1.the number of masses
		float m,									//2.weight of each mass
		float springConstant,						//3.how stiff the springs are
		float springLength,							//4.the length that a spring does not exert any force
		float springFrictionConstant,				//5.inner friction constant of spring
		Vec3 g,										//6.gravitational acceleration
		float airFrictionConstant,					//7.air friction constant
		float groundRepulsionConstant,				//8.ground repulsion constant
		float groundFrictionConstant,				//9.ground friction constant
		float groundAbsorptionConstant,				//10.ground absorption constant
		float groundHeight							//11.height of the ground(y position)
		)
{
	this->numOfMasses = numOfMasses;
	this->gravitation = g;
	this->airFrictionConstant = airFrictionConstant;
	this->groundFrictionConstant = groundFrictionConstant;
	this->groundRepulsionConstant = groundRepulsionConstant;
	this->groundAbsorptionConstant = groundAbsorptionConstant;
	this->groundHeight = groundHeight;

	this->masses= new Mass*[numOfMasses];

	for(int count = 0;count<numOfMasses;++count)				//we will step to every pointer in the array
	{
		masses[count] = new Mass(m);							//create a mass as a pointer and put it in the array
		masses[count]->init();
	}

	for(int index = 0;index<numOfMasses;++index)				//to set initial position of masses loop with for(;;)
	{
		masses[index]->setPos(Vec3(index*springLength,0,0));
	}

	springs = new Spring*[numOfMasses-1];
	for(int index=0;index<numOfMasses-1;++index)				//to create each spring,start a loop
	{
		//create the spirng with "a" by the mass with index  "a" and another mass with index "a+1"
		springs[index] = new Spring(masses[index],masses[index+1],
			springConstant,springLength,springFrictionConstant);
	}
}

void RopeSimulator::release()
{
	for(int count=0;count<numOfMasses;++count)					//delete all the masses
	{
		delete(masses[count]);
		masses[count] = NULL;
	}
	delete(masses);
	masses = NULL;

	for(int index= 0;index<numOfMasses-1;++index)				//delete all springs,start a loop;
	{
		delete(springs[index]);
		springs[index] = NULL;
	}
	delete(springs);
	springs = NULL;
}

void RopeSimulator::simulate(float dt)							//iterate the masses and spring by the change in time
{
	for(int count = 0; count<numOfMasses;++count)				//we will iterate every mass
	{
		masses[count]->simulate(dt);							//iterate the mass and obtain new position and new velocity
	}

	ropeConnectionPos += ropeConnectionVel*dt;					//iterate the position of ropeConnectionPos

	if(ropeConnectionPos.data[1] < groundHeight)				//ropeConnectionPos shall not go under the ground
	{
		ropeConnectionPos.data[1] = groundHeight;
		ropeConnectionVel.data[1] = 0;
	}

	masses[0]->setPos(ropeConnectionPos);						//mass with index "0" shall position at ropeConnectionPos
	masses[0]->setVel(ropeConnectionVel);						//the mass's velocity is set to be equal to ropeConnectionVel
}

void RopeSimulator::setRopeConnectionPos(Vec3 p)
{
	this->ropeConnectionPos = p;
}

void RopeSimulator::setRopeConnectionVel(Vec3 v)
{
	this->ropeConnectionVel = v;
}

float RopeSimulator ::getGroundHeight()
{
	return this->groundHeight;
}

float RopeSimulator::getNumOfMasses()
{
	return this->numOfMasses;
}

Mass* RopeSimulator::getMass(int index)
{
	if(index<0||index>=numOfMasses)						//if the index is not in the array
		return NULL;									//then return NULL
	return masses[index];
}

void RopeSimulator::operate(float dt)
{
	this->resetMassesForce();
	this->solve();
	this->simulate(dt);
}

void RopeSimulator::solve()
{
	for(int index = 0; index<numOfMasses-1;++index)			//apply force of all springs
	{
		springs[index]->solve();							//spring with index "a" should apply its force
	}

	for(int index = 0;index<numOfMasses;++index)			//start a loop to apply forces which are common for all masses
	{
		masses[index]->applyForce(gravitation*masses[index]->getM());	//the gravitational force
		masses[index]->applyForce(-masses[index]->getVel()*airFrictionConstant);	//the air friction

		if(masses[index]->getPos().data[1] < groundHeight)				//forces from the ground are applied if a mass collides with the ground
		{
			Vec3 v;													//a temporary Vector3D

			v=masses[index]->getVel();								//get the velocity
			v.data[1] = 0;											//omit the velocity component in y direction
				
			//the velocity in y direction is omited because we will apply a friction force to ecreate 
			//a sliding effect.sliding is parallel to the ground.velocity in y direction will be used in 
			//the absorption effect
			masses[index]->applyForce(-v*groundFrictionConstant);			//ground friction force is applied

			v=masses[index]->getVel();								//get the velocity
			v.data[0] = 0;											//omit the x and z components of the velocity
			v.data[2] = 0;											//we will use v in the absorption effect

			//above, we obtained avelocity which is vertical to the ground and it will be used in the absorption force

			if(v.data[1]<0)											//let's absorb energy only when a mass collides towards the ground
				masses[index]->applyForce(-v*groundAbsorptionConstant);			//the absorption force is applied

			//The ground shall repel a mass like a spring.  
                //By "Vector3D(0, groundRepulsionConstant, 0)" we create a vector in the plane normal direction  
                //with a magnitude of groundRepulsionConstant.  
                //By (groundHeight - masses[a]->pos.y) we repel a mass as much as it crashes into the ground.  
			Vec3 force = Vec3(0,groundRepulsionConstant,0)*
				(groundHeight-masses[index]->getPos().data[1]);

			masses[index]->applyForce(force);							//the ground repulsion force is applied
		}
	}
}

void RopeSimulator::resetMassesForce()									//call the init() method of every mass
{
	for(int count=0;count<numOfMasses;++count)							//init() every mass
		masses[count]->init();											//call init() method of the mass
}