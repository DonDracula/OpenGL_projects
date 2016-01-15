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

	struct Particle{
		Particle(Vec3 position, float m);
		Vec3 position;			//position
		Vec3 tem_position;		//temporary position
		Vec3 forces;			//force
		Vec3 velocitiy;
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
		void update();
		void draw();
	public:
		float len;			//constraint length
		std::vector<Particle*> particles;
		Vec3 color;
	};

}


#endif //_FTL_H