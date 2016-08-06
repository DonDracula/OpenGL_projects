#include "application.h"

#include <iostream>
#include <cassert>
#include <cstdio>
#include <sstream>

using namespace std;

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include "matrix.h"
#include "transform.h"

enum { NONE, AMBIENT, DIFFUSE, SPECULAR, NUM_MODES };

void draw_grid(int dim);
int ScreenShot(int m);

void set_pixel(int x, int y, float col[3])
{
    // write a 1x1 block of pixels of color col to framebuffer
    // coordinates (x, y)
    //glRasterPos2i(x, y);
    //glDrawPixels(1, 1, GL_RGB, GL_FLOAT, col);
    
    // use glVertex instead of glDrawPixels (faster)
    glBegin(GL_POINTS);
    glColor3fv(col);
    glVertex2f(x, y);
    glEnd();
}

application::application() 
    : raytrace(false), rendmode(SPECULAR), paused(false), sim_t(0.0)
{	
}

application::~application()
{
}

// triggered once after the OpenGL context is initialized
void application::init_event()
{

    cout << "CAMERA CONTROLS: \n  LMB: Rotate \n  MMB: Pan \n  LMB: Dolly" << endl;
    cout << "KEYBOARD CONTROLS: \n  ' ': Pause simulation" << endl;

    const GLfloat ambient[] = { 0.0, 0.0, 0.0, 1.0 };
    const GLfloat diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
    const GLfloat specular[] = { 1.0, 1.0, 1.0, 1.0 };
    
    // enable a light
    glLightfv(GL_LIGHT1, GL_AMBIENT, ambient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
    glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
    glEnable(GL_LIGHT1);
    
    // set global ambient lighting
    GLfloat amb[] = { 0.4, 0.4, 0.4, 1.0 };
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, amb);
    
    // enable depth-testing, colored materials, and lighting
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_LIGHTING);
    
    // normalize normals so lighting calculations are correct
    // when using GLUT primitives
    glEnable(GL_RESCALE_NORMAL);
    
    // enable smooth shading
    glShadeModel(GL_SMOOTH);

    glClearColor(0,0,0,0);
    
    // set the cameras default coordinates
    camera.set_distance(10);
    camera.set_elevation(35);
    camera.set_twist(45);

    t.reset();

}

// set the parameter for the particles' 
const int N = 2000;
std::vector<Particle> particles(N);
bool isinitialized = false;

float delta_t = 0.05;
float bbox[6] = { -2.0, 2.0, 0, 3.0, -2.0, 2.0 };
float h = 0.6;
float radius = 0.1;
float K = 8.0;
float rho0 = 20.0;
float viscosity = 0.5;
float damp = 0.8;
const vector3 gravity = vector3(0, -9.8, 0);
float kernel_poly6 = 315.0 / (64.0 * M_PI * powf(h, 9.0) );
float kernel_poly6_diff = -945.0 / (32.0 * M_PI * powf(h, 9.0) );
float kernel_poly6_laplacian = 945.0 / ( 32.0 * M_PI * powf(h, 6.0) );
float kernel_spiky_diff = -45.0 / (M_PI * powf(h, 6.0) );
float kernel_laplacian = 454.0 / (M_PI * powf(h, 6.0) );    
Grid* p_grid;

int m = 0; // the number for the frame
const int M = 90; // the number of frames we need for the animation	

// triggered each time the application needs to redraw
void application::draw_event()
{
	//RUN ONCE
	if(not isinitialized)
	{
		float ratio = sqrt(4*M_PI/3.0*radius*radius*radius*N/(bbox[3]-bbox[2])/2);
		for(int p=0; p<N; p++)
		{
			particles[p] = Particle();
			// the particle' system consists of two parts, 
			// one half in the corner (x=bbox[0], z=bbox[5]), the other half in the diagonal corner
			if (p < N/2)
			{
				particles[p].pos[0] = rand()*1.0/RAND_MAX * ratio + bbox[0];
				particles[p].pos[2] = bbox[5] - rand()*1.0/RAND_MAX * ratio;				
			}
			else
			{
				particles[p].pos[0] = bbox[1] - rand()*1.0/RAND_MAX * ratio;
				particles[p].pos[2] = rand()*1.0/RAND_MAX * ratio + bbox[4];
			}
			particles[p].pos[1] = rand()*1.0/RAND_MAX * (bbox[3]-bbox[2]);
							
			particles[p].vel[0] = 0;
			particles[p].vel[1] = 0;
			particles[p].vel[2] = 0;
		}
		
		isinitialized = true;
	}

	camera.apply_gl_transform();

	const GLfloat light_pos1[] = { 0.0, 10.0, 0.0, 1 };   
	glLightfv(GL_LIGHT1, GL_POSITION, light_pos1);
  
	if (!paused) 
	{
		// if delta_t = 0.05, you'll simulate for 1/20th of a second;
		// if delta_t = 0.015, you'll simulate for 1/60th of a second
		// 	
		/////////////////////////////////////////////////////////////
		if(p_grid != NULL) delete p_grid;
		p_grid = new Grid (bbox, h, particles);
		
		// Update particles' neighbors
		for( int i = 0; i < N; i++ )
		{
			p_grid -> Search(i, particles[i].nbs);
		}
		
		//////////////////////////////////////////////////////////////
		// Update particles' pressure
		unsigned nb_id;
		float dist;
    	float c;
		for( int i = 0; i < N; i++ )
		{
			Particle& particle = particles[i];
			particle.density = 0.0;
			for( unsigned j = 0; j < particle.nbs.size(); j++)
			{
				nb_id = particle.nbs[j].first;
				dist = particle.nbs[j].second;
				c = h * h - dist * dist;
				particle.density += particles[nb_id].mass * kernel_poly6 * (c*c*c);
			}
			
			particle.density = max(particle.density, rho0);
			particle.pressure = K * ( particle.density-rho0 );
			
		}
		
		////////////////////////////////////////////////////////////
		// Update particles' acceleration
		float c1, c2, c3, c4;
		vector3 dist_direction;
		for( int i = 0; i < N; i++ )
		{
			Particle& particle = particles[i];

			particle.acc = vector3(0, 0, 0);
			for( unsigned j=0; j < particle.nbs.size(); j++ )
			{
				Particle& nb_particle = particles[particle.nbs[j].first];
				dist = particle.nbs[j].second;

				//acceleration from pressure
				c1 = -0.5 * nb_particle.mass * (particle.pressure+nb_particle.pressure);
				c2 = kernel_spiky_diff * (h-dist) * (h-dist);
				c3 = 1.0 / (particle.density * nb_particle.density); 
				dist_direction = particle.pos-nb_particle.pos;
				dist_direction.normalize();			
				particle.acc += dist_direction * (c1*c2*c3);

				//acceleration from viscosity
				c4 = viscosity * nb_particle.mass * kernel_laplacian * (h-dist) / (nb_particle.density * particle.density);
				particle.acc += (nb_particle.vel-particle.vel) * c4;
			}
			if(particle.acc.mag() > 300.0)
				cout << "huge acc!!\n";
			particle.acc += gravity;
		}
		

		/////////////////////////////////////////////////////////////////////
		// update velocity and position
		vector3 prev_vel, mean_vel;
		float t_diff;
		for( int i = 0; i < N; i++ )
		{
			Particle& particle = particles[i];

			prev_vel = particle.vel;
			particle.vel += particle.acc * delta_t;
			
			mean_vel = (particle.vel + prev_vel) * 0.5;
			particle.pos += mean_vel * delta_t;
			
			// consider the boundary of the container
			while (particle.pos[0]<bbox[0] || particle.pos[1]<bbox[2] || particle.pos[2]<bbox[4] || particle.pos[0]>bbox[1] || particle.pos[1]>bbox[3] || particle.pos[2]>bbox[5])
			{
				for (unsigned i = 0; i < 3; i++ )
				{
					unsigned ii = 2 * i;
					if (particle.pos[i] < bbox[ii])
					{
						t_diff = (bbox[ii] - particle.pos[i]) / mean_vel.mag();
						mean_vel[i] = - mean_vel[i];
						mean_vel = mean_vel * damp;
						particle.pos[i] = bbox[ii] + mean_vel[i] * t_diff;
						particle.vel = mean_vel;
					}
				}
				for (unsigned i = 0; i < 3; i++ )
				{
					unsigned ii = 2 * i + 1;
					if (particle.pos[i] > bbox[ii])
					{
						t_diff = (particle.pos[i] - bbox[ii]) / mean_vel.mag();
						mean_vel[i] = - mean_vel[i];
						mean_vel = mean_vel * damp;
						particle.pos[i] = bbox[ii] + mean_vel[i] * t_diff;
						particle.vel = mean_vel;
					}
				}
			}				
		}			
		   
    }
    

	glEnable(GL_COLOR_MATERIAL);
    
	for (int i=0; i<N; i++)
	{
		// draw the particle		
		Particle& particle = particles[i];
		glPushMatrix();    	
    	glTranslatef(particle.pos[0], particle.pos[1], particle.pos[2]);    	
    	glutSolidSphere(radius, 4, 4);
    	glPopMatrix();
	}
	
	// draw the container
    glLineWidth(4.0);
    glColor3f(0, 1, 1);
    glBegin(GL_LINES);
  
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[5]+radius);
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[5]+radius);
    
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[5]+radius);
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[5]+radius);
    
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[4]-radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[4]-radius);
    glVertex3f(bbox[0]-radius,bbox[2]-radius,bbox[5]+radius);
    glVertex3f(bbox[0]-radius,bbox[3]+radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[2]-radius,bbox[5]+radius);
    glVertex3f(bbox[1]+radius,bbox[3]+radius,bbox[5]+radius);
            
    glEnd();


    glColor3f(0.15, 0.15, 0.15);    
    draw_grid(40); // draw grid on the floor

	//
	// This makes sure that the frame rate is locked to close to 1/delta_t fps. 
	// For each call to draw_event you will want to run your integrate for delta_t s
	//
    float elap = t.elapsed();
    if (elap < delta_t) {
        usleep( (int)(1e6*(delta_t-elap)) );
    }     
    
    if(!paused) 
    {
    	//cout << " the simulation cost time: " << elap << "s" << endl;   	
    	ScreenShot(m); // take a screenshot for the current frame
    	m++;
    }
    t.reset();
    if(m > M)  // determine how many frames we need for the animation
	{
		paused = true;		
	}   
    
}

// triggered when mouse is clicked
void application::mouse_click_event(
    mouse_button button, mouse_button_state button_state, 
    int x, int y
    )
{
}
    
// triggered when mouse button is held down and the mouse is
// moved
void application::mouse_move_event(
    int x, int y
    )
{
}

// triggered when a key is pressed on the keyboard
void application::keyboard_event(unsigned char key, int x, int y)
{

    if (key == 'r') {
        sim_t = 0;
    } else if (key == ' ') {
        paused = !paused;
    } else if (key == 'q') {
        exit(0);
    }
}

void draw_grid(int dim)
{
    glLineWidth(2.0);

    
    //
    // Draws a grid along the x-z plane
    //
    glLineWidth(1.0);
    glBegin(GL_LINES);

    int ncells = dim;
    int ncells2 = ncells/2;

    for (int i= 0; i <= ncells; i++)
    {
        int k = -ncells2;
        k +=i;
        glVertex3f(ncells2,0,k);
        glVertex3f(-ncells2,0,k);
        glVertex3f(k,0,ncells2);
        glVertex3f(k,0,-ncells2);
    }
    glEnd();
    
    //
    // Draws the coordinate frame at origin
    //
    glPushMatrix();
    glScalef(1.0, 1.0, 1.0); 
    glBegin(GL_LINES);

    // x-axis
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0.0, 0.0);
    
    // y-axis
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 1.0, 0.0);
    
    // z-axis
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(0.0, 0.0, 1.0);
    glEnd();
    glPopMatrix();
}


string convertInt(int number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

//////////////////////////////////////////////////
// Grab the OpenGL screen and save it as a .tga //
// Copyright (C) Marius Andra 2001              //
// http://cone3d.gz.ee  EMAIL: cone3d@hot.ee    //
//////////////////////////////////////////////////
int ScreenShot(int m)
{
  // we will store the image data here
  unsigned char *pixels;
  // the thingy we use to write files
  FILE * shot;
  // we get the width/height of the screen into this array
  int screenStats[4];

  // get the width/height of the window
  glGetIntegerv(GL_VIEWPORT, screenStats);

  // generate an array large enough to hold the pixel data 
  // (width*height*bytesPerPixel)
  pixels = new unsigned char[screenStats[2]*screenStats[3]*3];
  // read in the pixel data, TGA's pixels are BGR aligned
  glReadPixels(0, 0, screenStats[2], screenStats[3], GL_BGR, 
                                   GL_UNSIGNED_BYTE, pixels);

  // open the file for writing. If unsucessful, return 1
  string buffer;
  if (m < 10)  
  	buffer = "00" + convertInt(m) + ".tga"; 
  else if (m < 100)
  	buffer = "0" + convertInt(m) + ".tga";
  else
  	buffer = convertInt(m) + ".tga";
  	
  if((shot=fopen(buffer.c_str(), "wb"))==NULL) return 1;

  // this is the tga header it must be in the beginning of 
  // every (uncompressed) .tga
  unsigned char TGAheader[12]={0,0,2,0,0,0,0,0,0,0,0,0};
  // the header that is used to get the dimensions of the .tga
  // header[1]*256+header[0] - width
  // header[3]*256+header[2] - height
  // header[4] - bits per pixel
  // header[5] - ?
  unsigned char header[6]={((int)(screenStats[2]%256)),
                   ((int)(screenStats[2]/256)),
                   ((int)(screenStats[3]%256)),
                   ((int)(screenStats[3]/256)),24,0};

  // write out the TGA header
  fwrite(TGAheader, sizeof(unsigned char), 12, shot);
  // write out the header
  fwrite(header, sizeof(unsigned char), 6, shot);
  // write the pixels
  fwrite(pixels, sizeof(unsigned char), 
                 screenStats[2]*screenStats[3]*3, shot);

  // close the file
  fclose(shot);
  // free the memory
  delete [] pixels;

  // return success
  return 0;
}
