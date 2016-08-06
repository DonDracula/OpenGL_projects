#ifndef _PARTICLE_H_
#define _PARTICLE_H_

#include <vector>
#include "vector.h"

class Particle
{
	public:
		Particle();
	
	public:
		vector3 pos; // position
		vector3 vel; // velocity
		float mass;  // mass
		std::vector< std::pair<unsigned, float> > nbs; // this particle's neighbors
		float density; // density
		float pressure; // pressure
		vector3 acc; // acceleration
};

#endif
