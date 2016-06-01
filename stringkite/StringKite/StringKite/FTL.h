/*
	brief FTL method of Muller
	date: 2015.12
*/

#ifndef _FTL_H
#define _FTL_H

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>
#include "utils.h"

#include "rx_utility.h"				//Vector classes
#include <cmath>

namespace ftl{
	//粒子机构体
	struct Particle{
		Particle(Vec3 position, float m);
		Vec3 position;			//position
		Vec3 tmp_position;		//temporary position
		Vec3 forces;			//force
		Vec3 velocity;
		Vec3 d;					//distance between two particles
		float mass;				//mass of the particle
		float inv_mass;			//inverse of the mass
		bool enabled;			//the position state
	};

	class FTL{
	public:
		FTL();
		void setup(int num, float d);
		void addForce(Vec3 f);
		void moveforce(Vec3 f);
		Vec3 getLastPos() { return particles[particles.size()-1]->position; }
		void setLastPos(Vec3 pos) { particles[particles.size()-1]->position=pos; }
		void setLastVel(Vec3 vel) { particles[particles.size()-1]->velocity = vel;}
		void update();
		void draw();
		Vec3 calTension(Vec3 mywind);
	public:
		float len;			//constraint length
		std::vector<Particle*> particles;
		Vec3 color;
	};

}


#endif //_FTL_H