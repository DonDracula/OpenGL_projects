#include "ropesimulator.h"
RopeSimulator::RopeSimulator()
{
//ctor
}

RopeSimulator::~RopeSimulator()
{
//dtor
}

RopeSimulator::RopeSimulator(int numOfMasses,						//1. the number if masses
							 float m,								//2. weight of each mass
							 float springConstant,					//3. how stiff the springs are
							 float springLength,					//4.the length that a spring does not exert and force
							 float springFrictionCnstant,			//5. inner friction constant of spring 
							 Vec3 g,								//6. gravitational acceleration
							 float airFrictionConstant,				//7. air friction constant
							 float groundRepulsionConstant,			//8. ground repulsion constant
							 float groundFrictionConstant,			//9. ground friction constant
							 float groundAbsorptionConstant,		//10. ground absorption constant
							 float groundHeight						//11. height of the ground(y position)
							 )			
{
	this->numOfMasses = numOfMasses;
	this->gravitation =g;
	this->airFrictionConstant = airFrictionConstant;
	this->groundFrictionConstant =groundFrictionConstant;
	this->groundRepulsionConstant =groundRepulsionConstant;
	this->groundAbsorptonConstant = groundAbsorptionConstant;
	this->groundHeight = groundHeight;

	this->masses = new Mass*[numOfMasses];

	for(int count =0;count<numOfMasses;++count)						//we will step to every pointer in th array
	{
	masses[count]=new Mass(m);
	masses[count]->init();
	}

	for(int index =0; index<numOfMasses;++index)					//to set the initial positions of masses loop with for(;;)
	{
		masses[index]->setPos(Vec3(index*springLength,0,0));
	}

	springs =new Spring*[numOfMasses-1];
	for(int index =0;index<numOfMasses-1;++index)					//to create each spring , start a loop
	{
		//create the spring with index "a" by the mass with index "a" and another mass with index "a+1"
	springs[index] =new Spring(masses[index],masses[index+1],
		springConstant,springLength,springFrictionCnstant);
	}
}

void RopeSimulator::release()
{
for(int count =0; count <numOfMasses; ++count)						// we will delete all of them
{
delete(masses[count]);
masses[count] = NULL;
}
delete(masses);
masses=NULL;

for(int index =0; index<numOfMasses-1;++index)						//to delete all springs, start a loop
{
delete(springs[index]);
springs[index]=NULL;
}

delete(springs);
springs =NULL;
}