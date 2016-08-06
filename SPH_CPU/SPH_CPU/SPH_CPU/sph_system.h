/*
	Basic SPH fluid system

*/

#ifndef _SPHSYSTEM_H
#define _SPHSYSTEM_H

#include "sph_type.h"

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"
#include <GL\glew.h>
#include <GL\glut.h>
#include <queue>

class Particle
{
public:
	uint id;
	Vec3_f current_pos;			//position
	Vec3_f previous_pos[10];		//previous position
	//std::queue<Vec3_f> previous_pos;

	Vec3_f vel;			//velocity
	Vec3_f acc;			//acceleration
	Vec3_f ev;

	float dens;			//density
	float pres;			//pressure

	float surf_norm;

	Particle *next;
};

class SPHSystem
{
public:
	int count ;
	uint max_particle;
	uint num_particle;

	float kernel;
	float mass;

	Vec3_f world_size;
	float cell_size;		//size of a cell
	uint3 grid_size;
	uint tot_cell;			//total number of the cells

	std::vector<Particle> particleUnion;
	std::vector<std::vector<Particle>> particleTrack;

	Vec3_f gravity;
	float wall_damping;
	float rest_density;
	float gas_constant;
	float viscosity;
	float time_step;
	float surf_norm;
	float surf_coe;

	float poly6_value;
	float spiky_value;
	float visco_value;

	float grad_poly6;
	float lplc_poly6;

	float kernel_2;
	float self_dens;
	float self_lplc_color;

	Particle *mem;
	Particle **cell;

	uint sys_running;

public:
	SPHSystem();
	~SPHSystem();
	void update();
	void init_system();
	void add_particle(Vec3_f pos,Vec3_f vel);
	void draw();

private:
	void build_table();
	void comp_dens_pres();
	void comp_force_adv();
	void advection();

private:
	Vec3_i calc_cell_pos(Vec3_f p);
	uint calc_cell_hash(Vec3_i cell_pos);
};


#endif	//_SPHSYSTEM_H