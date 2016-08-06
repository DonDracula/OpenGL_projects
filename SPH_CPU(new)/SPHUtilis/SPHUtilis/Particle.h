/*
 * Class definition for a Particle
 *
 */

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

class Particle {
public:
    //Default constructor
	Particle() {
		pos = Vec3(0.0f,0.0f,0.0f);
        vel = Vec3(0.0f,0.0f,0.0f);
        accel = Vec3(0.0f,0.0f,0.0f);

        mass = 0.0f;
        density = 0.0f;
        pressure = 0.0f;
        thermal = 0.0f;
    }

    //Explicit constructor
	Particle(Vec3 npos, Vec3 nvel, Vec3 nforce, float nmass, float ndensity, float npressure, float nthermal) {
        pos = npos;
        vel = nvel;
        accel = nforce;
        mass = nmass;
        density = ndensity;
        pressure = npressure;
        thermal = nthermal;
    }

    //Destructor
	~Particle(){}
/*
    //Displays this particle
    //TODO use a real sphere making algorithm, not this crap
	void display(){
        glutSolidSphere(1, 10, 10);
    }
*/
//private:
    Vec3 pos;
    Vec3 vel;
    Vec3 accel;

    float mass;
    float density;
    float pressure;
    float thermal;
};
