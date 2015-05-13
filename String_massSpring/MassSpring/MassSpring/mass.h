#ifndef MASS_H
#define MASS_H
//#include <glm/glm.cpp>
#include <iostream>
#include <vector>

#include "rx_utility.h"		// Vector classes
#include "rx_matrix.h"		// Matrix classes
#include "rx_quaternion.h"	// Quaternion classes

class Mass
{
public:
	Mass(float m);
	~Mass();
	void applyForce(Vec3 force);
	void init();
	void simulate(float dt);
	void setPos(Vec3 p);
	void setVel(Vec3 v);

	Vec3 getPos();
	Vec3 getVel();
	float getM();
protected:
private:
	//the mass value
	float m;
	//position in space
	Vec3 pos;
	//velocity
	Vec3 vel;
	//force applied on this mass at an instance
	Vec3 force;
};
#endif //MASS_H