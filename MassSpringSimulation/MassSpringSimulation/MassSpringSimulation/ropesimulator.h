#ifndef ROPESIMULATIOR_H
#define ROPESIMULATOR_H
#include "spring.h"

class RopeSimulator
{
public:
	RopeSimulator();								//Constructor
	~RopeSimulator();								//destructor
	RopeSimulator(									//constructor
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
		);

	void release();

	void simulate(float dt);

	void setRopeConnectionPos(Vec3 p);
	void setRopeConnectionVel(Vec3 v);

	float getGroundHeight();
	float getNumOfMasses();
	Mass* getMass(int index);
	void operate(float dt);

	void solve();
	void resetMassesForce();
protected:
private:
	Spring** springs;								//Springs binding the masses
	Mass** masses;									//masses are held by pointer to pointer.(Mass** represents a 1 dimensional array)

	Vec3 gravitation;								//gravitational acceleration
	Vec3 ropeConnectionPos;							//a point in space that is used to set position 
	Vec3 ropeConnectionVel;							//a variable to move the ropeConnectionPos()

	int numOfMasses;								//number of masses in this container

	float airFrictionConstant;						//a constant of air friciton applied to masses

	float groundFrictionConstant;					//a constant of friction applied to masses by the ground
													//(used for the sliding of rope on the ground)

	float groundRepulsionConstant;					//a constant to represent how much the ground shall repel the masses
	float groundAbsorptionConstant;					//a constant of absorption frcition applied to masses by the ground
													//(used for vertical collisions of the rope with the ground)
	float groundHeight;								//y position value of the ground
};

#endif //ROPESIMULATOR_H