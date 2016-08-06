/*
 * Class definition for a Sphere fairly straightforward.
 */

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

#include <vector>
#include <string>

class Sphere {
public:
    //Default constructor
	Sphere() {
        pos = Vec3(0.0f,0.0f,0.0f);
        vel = Vec3(0.0f,0.0f,0.0f);
        radius = 1.0f;
		mass = 1.0f;
    }

    //Explicit constructor
	Sphere(Vec3 npos, Vec3 nvel, float nradius, float nmass) {
		pos = npos;
		vel = nvel;
		radius = nradius;
		mass = nmass;
    }

    //Destructor
	~Sphere(){}

//private:
	Vec3 pos;
    Vec3 vel;
    Vec3 accel;
	float radius;
	float mass;
};