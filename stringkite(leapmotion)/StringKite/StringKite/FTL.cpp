/*
	FTL class
	date : 2015.12
*/


#include "FTL.h"
//ftl命名空间
namespace ftl{
	//建立粒子结构，包含，位置，前一帧位置，质量mass，以及活跃状态
	Particle::Particle(Vec3 p, float m):position(p),tmp_position(p),enabled(true)
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
		float mass=0.5F;
		Vec3 Over;
		//添加粒子信息到particles容器中
		for(int i = 0; i<num; ++i){
		//	Particle* p = new Particle(pos, mass); 
		//	particles.push_back(p);
		//	pos.data[0] +=d;

			Over=Vec3(1.0,0.0,0.2);	//斜め上方
			normalize(Over);
			Over*=d*num*((double)i)/((double)num);

		Particle* p = new Particle(Over, mass); 
			particles.push_back(p);
	//	particles.push_back(firstParticle+Over);
		}
		//set the fixed particles
		particles[0]->enabled = false;
	//	particles[particles.size()-1]->enabled=false;
	}
	//set the force of each particles
	void FTL::addForce(Vec3 f){

		for(std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it){
			Particle* p = *it;
			if(p->enabled){
				//p->forces +=Vec3(-0.5,0,0);
				p->forces +=f;
			}
		}

		
	}

	void FTL::moveforce(Vec3 f){
		particles[0]->forces +=f;
	}
	void FTL::update() {
	
		float dt = 1.0f/20.0f;

		//update velocities
		for(std::vector<Particle*>::iterator it = particles.begin(); it != particles.end(); ++it){
			Particle* p = *it;
			//first particle
			if(!p->enabled){
				p->tmp_position = p->position;
				continue;
			}
			//cal v=v+t*(f/m)
			p->velocity = p->velocity + dt * (p->forces * p->inv_mass);
			p->tmp_position +=(p->velocity * dt);
			p->forces = 0;
			p->velocity *= 1;
		}

		//solve constrants
		Vec3 dir;		//moved position
		Vec3 curr_pos;	//current position 
		for(int i = 1; i<particles.size(); ++i){
			Particle* pa= particles[i -1];
			Particle* pb = particles[i];
			curr_pos = pb->tmp_position;
			dir = pb->tmp_position - pa->tmp_position;
			normalize(dir);

			pb->tmp_position = pa->tmp_position + dir*len;
			pb->d = curr_pos - pb->tmp_position;  //curr_pos
		}

		for(int i = 1;i<particles.size(); ++i){
			Particle* pa = particles[i-1];
			Particle* pb = particles[i];
			if(!pa->enabled){
				continue;
			}
			//cal the v and p
			pa->velocity = ((pa->tmp_position - pa->position)/dt) + 0.8 * (pb->d/dt);
			pa->position = pa->tmp_position;
		}
		//get the last particle position
		Particle* last = particles.back();
		last->position = last->tmp_position;
	}
	//draw the hair
	void FTL::draw(){
		glLineWidth(1.0f);
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
	//
	Vec3 FTL::calTension(Vec3 mywind)
	{
	
		Vec3 g_vec=Vec3(0.0,0.0,1.0);
		g_vec*=(2*0.2)*(-9.8);

		Vec3 d=particles[particles.size()-1]->position-particles[particles.size()-2]->position;

		Vec3 vel=particles[particles.size()-1]->velocity-particles[particles.size()-2]->velocity;

		Vec3 f1=(100 * (norm(d) - 2) + 1 * ( dot(vel,d) / norm(d) )) * ( d / norm(d) );

		Vec3 stringForce = Vec3();
	//	Vec3 w_V = stringObj->m_vWindSpeed;

		stringForce = mywind+g_vec+f1;
		return stringForce;
	}
}

