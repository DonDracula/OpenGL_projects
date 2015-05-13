#ifndef ROPESIMULATOR_H
#define ROPESIMULATION_H
#include "spring.h"
#include <iostream>

class RopeSimulator
{
public :
	RopeSimulator();
	~RopeSimulator();

	RopeSimulator(int numOfMasses, float m, float springConstant, float springLength,float springFrictionConstant,
		Vec3 g,float airFrictionConstant,float groundRepulsionConstant,float groundFrictionConstant,
		float groundAbsorptionConstant,float groundHeight);

	void release();
	void simulate(float dt);
	void setRopeConnectionPos(Vec3 p);
	void setRopeConnectionVel(Vec3 v);
	float getGroundHeight();
	int getNumOfMasses();

	Mass* getMass(int index);
	void operate(float dt);
	void solve();
	void resetMassesForce();
protected:
private:
	Vec3 gravitation;
	
	int numOfMasses;
	float m;
	float springConstant;
	float springLength;
	float springFrictionConstant;
	Vec3 g;

	Mass **masses;			//a mass pointer which point to the struction
	Spring **springs;		//Springs binding the masses(there shall be [numofmasses-1] of them)

	float airFrictionConstant;
	float groundRepulsionConstant;
	float groundFrictionConstant;
	float groundAbsorptonConstant;
	float groundHeight;
};

#endif