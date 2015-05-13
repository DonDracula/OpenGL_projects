#ifndef SPRING_H
#define SPRING_H
#include "mass.h"
#include <iostream>
class Spring 
{
public :
	Spring();
	Spring(Mass* mass1,Mass*mass2,float springConstant,float springLength,float frictionConstant);
	~Spring();

	void solve();

protected:
private:
	Mass* mass1;
	Mass* mass2;

	float springConstant;
	float resLength;
	float frictionConstant;
};

#endif //SPRING_H