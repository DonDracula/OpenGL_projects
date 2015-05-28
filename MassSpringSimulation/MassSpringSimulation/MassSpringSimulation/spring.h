#ifndef SPRING_H
#define SPRING_H
#include "mass.h"

class Spring
{
public:
	Spring();													//constructor
	Spring(Mass* mass1,Mass* mass2,float springConstant,
		float springLength,float frictionConstant);				//constructor
	~Spring();												//destructor

	void solve();										//the method where forces can be applied

protected:
private:
	Mass* mass1;										//the first mass at one tio of the spirng
	Mass* mass2;										//the second mass at the other tip of the spring

	float springConstant;								//a constant to reprsent the stiffness of the spring
	float restLength;									//the length that the spring does not exert any force
	float frictionConstant;								//a constant to be used for the inner friction of the spring
};

#endif //STRING_H