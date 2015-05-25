//#ifndef MASS_H
//#define MASS_H
//#include <glm/glm.cpp>
#include <iostream>
#include <vector>
#include <string>

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "rx_utility.h"		// Vector classes
//#include "rx_matrix.h"		// Matrix classes
//#include "rx_quaternion.h"	// Quaternion classes
using namespace std;



//class Mass					//An object to represent a mass
class Mass
{
public:
	float m;					//the mass value
	Vec3 pos;					//positon in space
	Vec3 vel;					//velocity
	Vec3 force;					//force applied on this mass at an instance

	Mass(float m)				//constructor
	{
		this->m = m;
	}

	/*
		void applyForce(vec3 force) :  add extrnal force to the mass.
		At an instance in time, several sources of force might affect the mass. The vector sum
		or these forces make up the net force applied to the mass at the instance/
	*/
	void applyForce(Vec3 force)
	{
		this->force +=force;			//the external force is added to the force of the mass
	}
	/*
		void init() :set the force values to zero
	*/
	void init()
	{
		force.data[0] = 0;
		force.data[1]=0;
		force.data[2]=0;
	}

	/*
		void simulate(float dt): calculate the new velocity and new positon of 
		the mass according to change in time(dt). 
	*/
	void simulate(float dt)
	{
		vel +=(force/m)*dt;				//Change in velocity is added to the velocity.
										//The change is proportinal with the acceleration(force/m) and change in time
		pos +=vel*dt;					//Change in positon is added to the positon
										//Change in position is velocity times the change in time
	}

};

//class Simulation			A container object for simulationg masses
class Simulation
{
public:
	int numOfMasses;					//number of masses in this container
	Mass** masses;						//masses are held by pointer to pointer. ( Mass** represents a 1 dimensional array)

	Simulation(int numOfMasses,float m)			//Constructor creates some masses with mass values m
	{
		this->numOfMasses = numOfMasses;

		masses = new Mass*[numOfMasses];		//create an array of pointers

		for(int a =0; a<numOfMasses; ++a)		//we will step to every pointer in the array
			masses[a]= new Mass(m);				//create a mass as a pointer and put it in the array
	}

	virtual void release()
	{
		for(int a = 0; a<numOfMasses;++a)		//delete all the masses
		{
			delete(masses[a]);
			masses[a]=NULL;
		}

		delete(masses);
		masses = NULL;
	}

	Mass* getMass(int index)
	{
		if(index < 0|| index>=numOfMasses)		//if the index is not in the array
			return NULL;						//return null

		return masses[index];					//get the mass at the index
	}

	virtual void init()							//call the init() method of every mass
	{
		for(int a =0;a<numOfMasses;++a)			
			masses[a]->init();					//call init() method of the mass
	}

	virtual void solve()						//no implementation because no forces are wanted in this basic container
	{
	
	}
	
	virtual void simulate(float dt)				//Iterate the masses by the change in time
	{
		for(int a =0;a<numOfMasses;++a)			
			masses[a]->simulate(dt);			//Iterate the mass and obtain new position and new velocity
	}

	virtual void operate(float dt)
	{
		init();
		solve();
		simulate(dt);
	}
};

	/*
		class ConstantVelocity is derived from class Simulation
		It created 1 mass with mass value 1 kg and sets its velocity to (1.0f,0.0f,0.0f)
		so that the mass moves in the x direction with 1 m/s velocity.
	*/
class ConstantVelocity : public Simulation
{
	ConstantVelocity() : Simulation(1,1.0f)
	{
		masses[0]->pos = Vec3(0.0f,0.0f,0.0f);
		masses[0]->vel = Vec3(1.0f,0.0f,0.0f);
	}
};

/*
	class MotionUnderGravitation: creates 1 mass with mass value 1 kg and sets its velocity to (10.0f,15.0f,0.0f) 
	and its position to(-10.0f,0.0f,0.0f).The purpose is to apply a gravitational force to mass and
	observe the path it follows. 
*/
class MotionUnderGravitation : public Simulation
{
public:
	Vec3 gravitation;

	MotionUnderGravitation(Vec3 gravitation) : Simulation(1,1.0f)
	{
		this->gravitation = gravitation;
		masses[0]->pos = Vec3(-10.0f,0.0f,0.0f);
		masses[0]->vel = Vec3(10.0f,15.0f,0.0f);
	}

	virtual void solve()
	{
		for(int a = 0;a<numOfMasses;++a)
			masses[a]->applyForce(gravitation*masses[a]->m);
	}
};

/*
	class MassConnectedWithSpring:creates 1 mass with mass value 1 kg and binds the mass to an arbitrary constant point with a spring.
	This point is refered as the connectionPos and the spring has a springConstant value to represent its stiffness.
*/
class MassConnectedWithSpring: public Simulation
{
public :
	float springConstant;
	Vec3 connectionPos;

	MassConnectedWithSpring(float springConstant):Simulation(1,1.0f)
	{
		this->springConstant = springConstant;

		connectionPos = Vec3(0.0f,-5.0f,0.0f);

		masses[0]->pos = connectionPos + Vec3(10.0f,0.0f,0.0f);
		masses[0]->vel = Vec3(0.0f,0.0f,0.0f);
	}

	virtual void solve()
	{
		for(int a =0;a<numOfMasses;++a)
		{
			Vec3 springVector=masses[a]->pos-connectionPos;
			masses[a]->applyForce(-springVector*springConstant);
		}
	}
};
//#endif //MASS_H
