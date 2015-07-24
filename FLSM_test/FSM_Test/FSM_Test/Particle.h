#include "Summation.h"

//the information pertaining to a particle. the dynamic data is stored in a sumdata
class  Particle:public Summation

{
public:
	float mass;
	Vec3 x0;
	float perRegionMass;				//mass of each region = mass*(1.0/numParentRegions)

	//Dynamic values
	Vec3 x, v, f;
	Vec3 g;
	rxMatrix3 R;						//the rotation of the transformations of its parent regions

	//relations
	vector<LatticeLocation *> parentRegions;

	//collision stuff
	Particle *nextParticleInCollisionCell;		//mamintaining a linked list of particles in the collision cell
};
