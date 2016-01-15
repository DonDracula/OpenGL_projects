/*! 
/*
	a class to brief the fluid simulation
	date 2015
*/

#include <cmath>

#ifndef FLUIDSOLVER_H
#define FLUIDSOLVER_H

//-----------------------------------------------------------------------------------
//extern para
//-----------------------------------------------------------------------------------
extern double *g_u; //x
extern double *g_v;	//y
extern double *g_w; //z
extern double *g_u_prev;		//previous x
extern double *g_v_prev;		//previous y
extern double *g_w_prev;		//previous z

extern double *g_dens;	//density
extern double *g_dens_prev;	//previous density
extern double *g_p_prev;	//previous press
			   
			   
//-----------------------------------------------------------------------------------
//MACRO
//-----------------------------------------------------------------------------------

#define IX(i,j,k) ((i)+(N+2)*(j)+(N+2)*(N+2)*(k))	//storage
#define SWAP(x0,x) {double *tmp = x0; x0 = x; x = tmp;}	//reverse 

#define GRID 16			//1éüå≥ñàÇÃÉOÉäÉbÉhï™äÑêî
#define F_FORCE -1.0	//force
#define STEP 0.01		//time step 
#define VISC 0.0007		//viscosity coefficient
#define DIFF 0.0		//diffusion coefficient
using namespace std;

//---------------------------------------------------------------------------------------------------------------------
//fluid simulation
//---------------------------------------------------------------------------------------------------------------------
class fluidSimulation
{
public:
	static int X_wind;	//x direction of wind
	 static int Z_wind;	//z

	 static int V_field;	//visualization
	 static int D_tex;
	 static int frc_view;

public:
	fluidSimulation();
	~fluidSimulation();
	void free_data(void);
	//allocate data
	void allocate_data(void);
	//clear all the data
	void clear_data(void);

	//velocity 
	void vel_step ( int n, double * u, double * v, double * w, double * u0, double * v0, double * w0, double visc, double dt );

	//added force
	void add_source ( int n, double * x, double * s, double dt );
	//diffuse 
	void diffuse ( int n, int b, double * x, double * x0, double diff, double dt );
	//! à⁄ó¨çÄ
	void advect ( int n, int b, double * d, double * d0, double * u, double * v,double * w, double dt );
	//Gauss-Seidal iterative 
	void lin_solve ( int n, int b, double * x, double * x0, double a, double c );
	//boundary condition
	void set_bnd ( int n, int b, double * x );
	//! éøó ï€ë∂
	void project( int n, double * u, double * v, double * w, double * p, double * div );

	// user input
	void get_from_UI ( int N, double * d, double * u, double * v, double * w );

	//draw in OpenGL
	void draw_box(double mag);
	void draw_velocity(double mag, double scale);
};


#endif	//FLUIDSOLVER_H