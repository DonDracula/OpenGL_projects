/*
	kite solver.cpp
	to brief Fluid solver
	date 2015
*/

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

#include "fluid_solver.h"

double *g_u = 0;	//x
double *g_v = 0;	//y
double *g_w = 0;	//z
double *g_u_prev = 0;	//previous x
double *g_v_prev = 0;	//prevoius y
double *g_w_prev = 0;	//previous z

double *g_dens =0;		//density
double *g_dens_prev = 0;	//previous density

double *g_p_prev = 0;		//previous press

int fluidSimulation::X_wind=0;//!< x方向の外力源(demo)
int fluidSimulation::Z_wind=1;//!< z方向の外力源(demo)
int fluidSimulation::V_field=0;//!< 速度場の視覚化ON/OFF(demo)
int fluidSimulation::D_tex=0;
int fluidSimulation::frc_view=0;
//---------------------------------------------------------------------------------
// Fluid simulation solver
//---------------------------------------------------------------------------------
fluidSimulation::fluidSimulation()
{

}

fluidSimulation::~fluidSimulation()
{
}

//fluid simulation solver
void fluidSimulation::free_data(void)
{
	if ( g_u ) delete [] g_u;
	if ( g_v ) delete [] g_v;
	if ( g_w ) delete [] g_w;
	if ( g_u_prev ) delete [] g_u_prev;
	if ( g_v_prev ) delete [] g_v_prev;
	if ( g_w_prev ) delete [] g_w_prev;
	if ( g_p_prev ) delete [] g_p_prev;
	if ( g_dens ) delete [] g_dens;
	if ( g_dens_prev ) delete [] g_dens_prev;
}

//allocate data
void fluidSimulation::allocate_data(void )
{
	int N=GRID;
	int size = (N+2)*(N+2)*(N+2);
	g_u			= new double[size];
	g_v			= new double[size];
	g_w			= new double[size];
	g_u_prev	= new double[size];
	g_v_prev	= new double[size];
	g_w_prev	= new double[size];
	g_dens		= new double[size];	
	g_dens_prev	= new double[size];
	g_p_prev	= new double[size];
}

//init of vecocity
void fluidSimulation::clear_data(void)
{
	int size=(GRID+2)*(GRID+2)*(GRID+2);

	for ( int i=0 ; i<size ; i++ ) {
		g_u[i] = g_v[i] = g_w[i] 
		= g_u_prev[i] = g_v_prev[i]= g_w_prev[i] 
		= g_dens[i] = g_dens_prev[i] = g_p_prev[i] 
		= 0.0;
	}
}

//add force
void fluidSimulation::add_source(int N,double *x,double *s,double dt)
{
	int size=(N+2)*(N+2)*(N+2);
	for ( int i=0 ; i<size ; i++ )
		x[i] += dt*s[i];
}

//diffuse
void fluidSimulation::diffuse(int N,int b, double *x, double *x0,double diff,double dt)
{
	double a=dt*diff*N*N;
	fluidSimulation::lin_solve ( N, b, x, x0, a, 1+6*a );//iterative 
}

//移流項
void fluidSimulation::advect(int N,int b,double *d,double*d0,double *u,double *v,double *w,double dt)
{
	double dt0 = dt*N;

	for ( int i=1 ; i<=N ; i++ )
	{
		for ( int j=1 ; j<=N ; j++ )
		{
			for ( int k=1 ; k<=N ; k++ )
			{
				//backtrace
				double x = i-dt0*u[IX(i,j,k)];
				double y = j-dt0*v[IX(i,j,k)];
				double z = k-dt0*w[IX(i,j,k)];

				if (x<0.5) x=0.5; if (x>N+0.5) x=N+0.5;			//x boudary
				int i0=(int)x;							//x position of the grid before 1 timestep
				int i1=i0+1;

				if (y<0.5) y=0.5; if (y>N+0.5) y=N+0.5;		//y boudary
				int j0=(int)y;								//y position of the grid before 1 timestep
				int j1=j0+1;

				if (z<0.5) z=0.5; if (z>N+0.5) z=N+0.5;		//z boudary
				int k0=(int)z;								//z position of the grid before 1 timestep
				int k1=k0+1;

				double s1 = x-i0;
				double s0 = 1-s1;
				double t1 = y-j0;
				double t0 = 1-t1;
				double r1 = z-k0;
				double r0 = 1-r1;

				d[IX(i,j,k)] = r0*(s0*(t0*d0[IX(i0,j0,k0)]+t1*d0[IX(i0,j1,k0)])
								+s1*(t0*d0[IX(i1,j0,k0)]+t1*d0[IX(i1,j1,k0)]))
								+r1*(s0*(t0*d0[IX(i0,j0,k1)]+t1*d0[IX(i0,j1,k1)])
								+s1*(t0*d0[IX(i1,j0,k1)]+t1*d0[IX(i1,j1,k1)]));

			}
		}
	}
	fluidSimulation::set_bnd(N,b,d);	//boundary condition
}

//Gauss-Seidal iteration
void fluidSimulation::lin_solve(int N,int b,double *x, double *x0, double a, double c)
{
	int i, l;
	int size=(N+2)*(N+2)*(N+2);
	double bx[5832];
	double e = 1.0f / pow(10.0, 20.0);

	//init bbx
	for(i=0;i<size;i++)
	{
		bx[i]=0.0;
	}


	for ( l=0 ; l<20 ; l++ )
	{
		for ( i=1 ; i<=N ; i++ )
		{
			for ( int j=1 ; j<=N ; j++ )
			{
				for ( int k=1 ; k<=N ; k++ )
				{
					x[IX(i,j,k)] = (x0[IX(i,j,k)] + a*(x[IX(i-1,j,k)]+x[IX(i+1,j,k)]
													+x[IX(i,j-1,k)]+x[IX(i,j+1,k)]
													+x[IX(i,j,k-1)]+x[IX(i,j,k+1)]))/c;
				}
			}
		}
		fluidSimulation::set_bnd(N,b,x);		//boundary condition

		//compare
		l=0;
		for(i=0;i<size;i++)
		{
			if(fabs(bx[i]-x[i])<=e)
				l+=1;
		}		
		if(l==size)
			break;

		//value update 
		for(i=0;i<size;i++)
		{
			bx[i]=x[i];
		}
	}
}

//set boudary 
void fluidSimulation::set_bnd(int N,int b, double *x)
{
	for ( int j=1 ; j<=N ; j++ ) {
		for ( int i=1 ; i<=N ; i++ ) {

			x[IX(0  ,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(N+1,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(i,0  ,j)] = b==2 ? -x[IX(i,1,j)] : x[IX(i,1,j)];//below
			x[IX(i,N+1,j)] = b==2 ? 0.0 : 0.0;
			x[IX(i,j,  0)] = b==3 ? 0.0 : 0.0;
			x[IX(i,j,N+1)] = b==3 ? 0.0 : 0.0;//right

			
			if(j==1)
			{
				if(b==3)
				{
					x[IX(i,j,9)] = -0.5*x[IX(i,j,8)];
					g_v[IX(i,j,9)] += 0.5*x[IX(i,j,8)];
				}
				else
				{
					x[IX(i,j,9)] = x[IX(i,j,8)];
				}
			}
			else if(j==2)
			{
				if(b==3)
				{
					x[IX(i,j,10)] = -0.5*x[IX(i,j,9)];
					g_v[IX(i,j,10)] += 0.5*x[IX(i,j,9)];
				}
				else
				{
					x[IX(i,j,10)] = x[IX(i,j,9)];
				}
			}
			else if(j==3)
			{
				if(b==3)
				{
					x[IX(i,j,11)] = -0.5*x[IX(i,j,10)];
					g_v[IX(i,j,11)] += 0.5*x[IX(i,j,10)];
				}
				else
				{
					x[IX(i,j,11)] = x[IX(i,j,10)];
				}
			}
			else if(j==4)
			{
				if(b==3)
				{
					x[IX(i,j,12)] = -0.5*x[IX(i,j,11)];
					g_v[IX(i,j,12)] += 0.5*x[IX(i,j,11)];
				}
				else
				{
					x[IX(i,j,12)] = x[IX(i,j,11)];
				}
			}
			else if(j==5)
			{
				if(b==3)
				{
					x[IX(i,j,13)] = -0.5*x[IX(i,j,12)];
					g_v[IX(i,j,13)] += 0.5*x[IX(i,j,12)];
				}
				else
				{
					x[IX(i,j,13)] = x[IX(i,j,12)];
				}
			}
			else if(j==6)
			{
				if(b==3)
				{
					x[IX(i,j,14)] = -0.5*x[IX(i,j,13)];
					g_v[IX(i,j,14)] += 0.5*x[IX(i,j,13)];
				}
				else
				{
					x[IX(i,j,14)] = x[IX(i,j,13)];
				}
			}
			else if(j==7)
			{
				if(b==3)
				{
					x[IX(i,j,15)] = -0.5*x[IX(i,j,14)];
					g_v[IX(i,j,15)] += 0.5*x[IX(i,j,14)];
				}
				else
				{
					x[IX(i,j,15)] = x[IX(i,j,14)];
				}
			}
			else if(j==8)
			{
				if(b==3)
				{
					x[IX(i,j,16)] = -0.5*x[IX(i,j,15)];
					g_v[IX(i,j,16)] += 0.5*x[IX(i,j,15)];
				}
				else
				{
					x[IX(i,j,16)] = x[IX(i,j,15)];
				}
			}
		}
	}
}

//質量保存
void fluidSimulation::project(int N,double *u, double *v, double *w, double *p, double *div)
{
	double d=-0.5/(double)(N*N);

	//init
	for ( int i=1 ; i<=N ; i++ )
	{
		for ( int j=1 ; j<=N ; j++ )
		{
			for ( int k=1 ; k<=N ; k++ )
			{
				div[IX(i,j,k)] = d*u[IX(i+1,j,k)]-d*u[IX(i-1,j,k)]
								+d*v[IX(i,j+1,k)]-d*v[IX(i,j-1,k)]
								+d*w[IX(i,j,k+1)]-d*w[IX(i,j,k-1)];
				p[IX(i,j,k)] = 0.0; 
			}
		}
	}	

	fluidSimulation::set_bnd(N,0,div);
	fluidSimulation;;set_bnd(N,0,p);
	fluidSimulation::lin_solve(N,0,p,div,1,6);

	//勾配場を引くことで非圧縮場を得る
	for ( int i=1 ; i<=N ; i++ )
	{
		for ( int j=1 ; j<=N ; j++ )
		{
			for ( int k=1 ; k<=N ; k++ )
			{
				u[IX(i,j,k)] -= 0.5*N*p[IX(i+1,j,k)]-0.5f*N*p[IX(i-1,j,k)];
				v[IX(i,j,k)] -= 0.5*N*p[IX(i,j+1,k)]-0.5f*N*p[IX(i,j-1,k)];
				w[IX(i,j,k)] -= 0.5*N*p[IX(i,j,k+1)]-0.5f*N*p[IX(i,j,k-1)];
			}
		}
	}
	fluidSimulation::set_bnd(N,1,u); 
	fluidSimulation::set_bnd(N,2,v);
	fluidSimulation::set_bnd(N,3,w);
}

//velocity
void fluidSimulation::vel_step(int N, double *u, double *v, double *w, double *u0, double *v0, double *w0, double visc, double dt)
{
		//add force
	fluidSimulation::add_source(N,u,u0,dt);
	fluidSimulation::add_source(N,v,v0,dt);
	fluidSimulation::add_source(N,w,w0,dt);

	//diffuse
	SWAP(u0,u); SWAP(v0,v); SWAP(w0,w);
	fluidSimulation::diffuse(N,1,u,u0,visc,dt);
	fluidSimulation::diffuse(N,2,v,v0,visc,dt);
	fluidSimulation::diffuse(N,3,w,w0,visc,dt);
	fluidSimulation::project(N,u,v,w,u0,v0);

	//移流項
	SWAP ( u0, u );	SWAP ( v0, v );	SWAP ( w0, w );
	fluidSimulation::advect ( N, 1, u, u0, u0, v0, w0, dt );
	fluidSimulation::advect ( N, 2, v, v0, u0, v0, w0, dt );
	fluidSimulation::advect ( N, 3, w, w0, u0, v0, w0, dt );
	fluidSimulation::project ( N, u, v, w, u0, v0 );
}

//user interaction
void fluidSimulation::get_from_UI(int N,double *d, double *u, double *v, double *w)
{
	int i,j;

	int size = (N+2)*(N+2)*(N+2);
	//init vectory,density
	for(i=0;i<size;i++)
	{
		u[i] = v[i] = w[i] = d[i] = 0.0;
	}

	double a = 8.0f;
	int x = N-3;
	int z = 3;
	if(Z_wind == 1)
	{
		for(i=1;i<N;i++)
		{
			for(j=1;j<N;j++)
			{
				w[IX(i,j,z)] = -F_FORCE*a;
			}
		}
	}
	if(X_wind == 1)
	{
		for(i=1;i<N;i++)
		{
			for(j=1;j<N;j++)
			{
				u[IX(x,i,j)] = F_FORCE*a;
			}
		}
	}
}

//visualization
void fluidSimulation::draw_velocity(double mag, double scale)
{
	int i, j, k;
	double x, y, z, h;

	int N=GRID;

	h = 1.0/N;	//limitation 

	glDisable(GL_LIGHTING);
	glColor3f ( 0.5f, 0.5f, 0.5f );
	glLineWidth ( 1.0f );
	glBegin ( GL_LINES );

	for ( i=1 ; i<=N ; i++ ) {
		x = (i-0.5)*h;
		for ( j=1 ; j<=N ; j++ ) {
			y = (j-0.5)*h;
			for ( k=1 ; k<=N ; k++ ) {
				z = (k-0.5)*h;

				glVertex3d ( mag*z, mag*y, mag*x );
				glVertex3d ( mag*(z+g_w[IX(i,j,k)]*scale), mag*(y+g_v[IX(i,j,k)]*scale), mag*(x+g_u[IX(i,j,k)]*scale) );
			}
		}
	}

	glEnd ();
	glEnable(GL_LIGHTING);
  
}

//fluid box
void fluidSimulation::draw_box(double mag)
{
	glDisable( GL_LIGHTING );

	double a,b;
	a=0.0;
	b=mag;

	glColor3f ( 0.7f, 0.7f, 0.7f );
	glLineWidth ( 2.0f );

	//奥
	glBegin ( GL_LINE_STRIP );
		glVertex3d ( a,a,a );
		glVertex3d ( a,b,a );
		glVertex3d ( b,b,a );
		glVertex3d ( b,a,a );
	glEnd ();

	//上
	glBegin ( GL_LINE_STRIP );
		glVertex3d ( a,b,a );
		glVertex3d ( a,b,b );
		glVertex3d ( b,b,b );
		glVertex3d ( b,b,a );
	glEnd ();

	//手前
	glBegin ( GL_LINE_STRIP );
		glVertex3d ( a,b,b );
		glVertex3d ( a,a,b );
		glVertex3d ( b,a,b );
		glVertex3d ( b,b,b );
	glEnd ();

	//右
	glBegin ( GL_LINE_STRIP );
		glVertex3d ( b,a,a );
		glVertex3d ( b,a,b );
		glVertex3d ( b,b,b );
		glVertex3d ( b,b,a );
	glEnd ();

	//左
	glBegin ( GL_LINE_STRIP );
		glVertex3d ( a,a,a );
		glVertex3d ( a,a,b );
		glVertex3d ( a,b,b );
		glVertex3d ( a,b,a );
	glEnd ();

	//下
	glBegin ( GL_POLYGON );
		glVertex3d ( a,a,b );
		glVertex3d ( a,a,a );
		glVertex3d ( b,a,a );
		glVertex3d ( b,a,b );
	glEnd ();

	//斜め
	glColor3f ( 1.0f, 0.6f, 0.0f );
	glBegin ( GL_POLYGON );
		glVertex3d ( 0.6*b,a,b );
		glVertex3d ( 0.6*b,a,a );
		glVertex3d ( b,0.4*b,a );
		glVertex3d ( b,0.4*b,b );
	glEnd ();

	glEnable( GL_LIGHTING );
}
