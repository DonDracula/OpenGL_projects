/*
 Mass.h: define a mass class to represent a mass, which has mass,velocity,positon in it
*/
#ifndef MASS_H
#define MASS_H
#include <iostream>
#include <vector>
#include <string>

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"			//Vector classes
#include "rx_matrix.h"

#include "rx_nnsearch.h"

using namespace std;

//calss Mass					->an object to represent a mass
class Mass
{
public:
	Mass(float m);				//constructor
	~Mass();					//destructor
	void applyForce(Vec3 force);	//add external force to the mass
	void init();					//set the force vallues to zero
	void simulate(float dt);		//calculate the new velocity and new positon of the mass
									//according to change in time
	void setPos(Vec3 p);				//set position
	void setVel(Vec3 v);				//set velocity

	Vec3 getPos();						//get the positon
	Vec3 getVel();						//get the velocity
	float getM();						//get the mass
protected:
private:
	float m;					//the mass value
	Vec3 pos;					//positon in space
	Vec3 vel;					//velocity
	Vec3 force;					//force applied on this mass at an instance
};

#endif //MASS_H