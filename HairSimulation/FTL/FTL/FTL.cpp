/*
	FTL class
	date : 2015.12
*/


#include "FTL.h"

namespace ftl{
	
	Particle::Particle(Vec3 p, float m):position(p),tem_position(p),enabled(true)
	{
		if(m<0.001){
			m=0.001;
		}
		
		mass = m;
		inv_mass = 1.0 / mass;
	}

	//-----------------------------------
	//FTL function
	FTL::FTL():len(10)
	{
	}
	//init the position
	void FTL::setup(int num, float d){
		float dim = 50;
		len = d;
		Vec3 pos(0,0,0);
		float mass=2.0F;
		
		for(int i = 0; i<num; ++i){
			Particle* p = new Particle(pos, mass); 
			particles.push_back(p);
			pos.data[1] +=d;
		}
		//set the fixed particles
		particles[0]->enabled = false;
	}
	//set the force of each particles
	void FTL::addForce(Vec3 f){

		for(std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it){
			Particle* p = *it;
			if(p->enabled){
				p->forces +=Vec3(-0.5,0,0);
			}
		}
	}

	void FTL::update() {
	
		float dt = 1.0f/20.0f;

		//update velocities
		for(std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it){
			Particle* p = *it;
			//first particle
			if(!p->enabled){
				p->tem_position = p->position;
				continue;
			}
			//cal v=v+t*(f/m)
			p->velocitiy = p->velocitiy + dt * (p->forces * p->inv_mass);
			p->tem_position +=(p->velocitiy * dt);
			p->forces = 0;
			p->velocitiy *= 0.99;
		}

		//solve constrants
		Vec3 dir;		//moved position
		Vec3 curr_pos;	//current position 
		for(int i = 1; i<particles.size(); ++i){
			Particle* pa= particles[i -1];
			Particle* pb = particles[i];
			curr_pos = pb->tem_position;
			dir = pb->tem_position - pa->tem_position;
			normalize(dir);

			pb->tem_position = pa->tem_position + dir*len;
			pb->d = curr_pos - pb->tem_position;  //curr_pos
		}

		for(int i = 1;i<particles.size(); ++i){
			Particle* pa = particles[i-1];
			Particle* pb = particles[i];
			if(!pa->enabled){
				continue;
			}
			//cal the v and p
			pa->velocitiy = ((pa->tem_position - pa->position)/dt) + 0.9 * (pb->d/dt);
			pa->position = pa->tem_position;
		}
		//get the last particle position
		Particle* last = particles.back();
		last->position = last->tem_position;
	}
	//draw the hair
	void FTL::draw(){
		glLineWidth(5.0f);
		glBegin(GL_LINE_STRIP);
		for(std::vector<Particle*>::iterator it = particles.begin(); it !=particles.end(); ++it){
			Particle* p= *it;

			if(!p->enabled){
				glColor3f(217,41,41);
				
					
			}
			else{
				glColor3f(251,251,251);

			}

			glVertex3f(p->position.data[0],p->position.data[1],p->position.data[2]);
		}
		glEnd();
	}
}

