/*
 * Class definition for a Particle
 *
 */

#include "Particle.h"
#include "Wall.h"
#include "Sphere.h"
#include <math.h>
#include <vector>
#include <iostream>

#define PI 3.1415926

class SphUtils {
public:
    //Default constructor
	SphUtils() {
    }

    //Explicit constructor
	SphUtils(float nsmooth_length, float nrho0, float nc, float ndt, float nepsilon) {
		smooth_length = nsmooth_length;
		rho0 = nrho0;
		c = nc;
		dt = ndt;
		epsilon = nepsilon;

		cell_dimensions = 2*nsmooth_length;
		maxx=maxy=maxz=minx=miny=minz=0.0f;
    }

    //Destructor
	~SphUtils(){}

	//Kernel/grad/laplace functions for the SPH
	float kernel_function(Vec3 i, Vec3 j);
	Vec3 grad_kernel(Vec3 i, Vec3 j);
	float SphUtils::lap_kernel(Vec3 i, Vec3 j);

	//Public callers
	void update_cells(std::vector<Particle> &particles);
	void update_density(std::vector<Particle> &particles);
	void update_forces(std::vector<Particle> &particles,std::vector<Sphere> &spheres);
	void update_posvel(std::vector<Particle> &particles,std::vector<Sphere> &spheres);
	void collision(std::vector<Particle> &particles, std::vector<Wall> &walls, std::vector<Sphere> &spheres);

private:
	float smooth_length;
	float rho0;
	float c;
	float dt;
	float epsilon;

	//My attempt at accelerating. This array containts a list of indices of praticles that are in a given cell. For example,
	//call [0] = {1,3,7,100} means that particles 1,3,7,100 are located in cell 0.
	float maxx, maxy, maxz, minx, miny, minz;
	float cell_dimensions;
	std::vector<std::vector<int>> cells;

	//Helper functions for the above
	void update_h_vel(std::vector<Particle> &particles);
	void gravity_forces(std::vector<Particle> &particles,std::vector<Sphere> &spheres);
	void pressure_forces(std::vector<Particle> &particles);
	void viscosity_forces(std::vector<Particle> &particles);
	Vec3 velocity_incompressible(std::vector<Particle>::size_type i, std::vector<Particle> &particles);
};