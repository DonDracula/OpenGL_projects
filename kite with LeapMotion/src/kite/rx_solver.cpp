/*! 
 @file solver.cpp

 @brief Fluid solver
 
 @author Taichi Okamoto
 @date 2008
*/


// OpenGL
#include <GL/glew.h>
#include <GL/glut.h>


#include "rx_solver.h"

double *g_u = 0;		//!< 速度場のx方向成分
double *g_v = 0;		//!< 速度場のy方向成分
double *g_w = 0;		//!< 速度場のz方向成分
double *g_u_prev = 0;	//!< 1手順前の速度場のx方向成分
double *g_v_prev = 0;	//!< 1手順前の速度場のy方向成分
double *g_w_prev = 0;	//!< 1手順前の速度場のz方向成分

double *g_dens = 0;		//!< 密度場(現在未実装)
double *g_dens_prev = 0;//!< 1手順前の密度場(現在未実装)

double *g_p_prev = 0;	//!< 1手順前の圧力場


//---------------------------------------------------------------------------------
// Fluid simulation solver
//---------------------------------------------------------------------------------
void 
fluid::free_data ( void )
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

//メモリ確保
void 
fluid::allocate_data ( void )
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

//速度場の初期化
void 
fluid::clear_data ( void )
{
	int size=(GRID+2)*(GRID+2)*(GRID+2);

	for ( int i=0 ; i<size ; i++ ) {
		g_u[i] = g_v[i] = g_w[i] 
		= g_u_prev[i] = g_v_prev[i]= g_w_prev[i] 
		= g_dens[i] = g_dens_prev[i] = g_p_prev[i] 
		= 0.0;
	}
}

//外力項
void 
fluid::add_source ( int N, double * x, double * s, double dt )
{
	int size=(N+2)*(N+2)*(N+2);
	for ( int i=0 ; i<size ; i++ )
		x[i] += dt*s[i];
}

//拡散項
void 
fluid::diffuse ( int N, int b, double * x, double * x0, double diff, double dt )
{
	double a=dt*diff*N*N;
	fluid::lin_solve ( N, b, x, x0, a, 1+6*a );//反復法
}

//移流項
void 
fluid::advect ( int N, int b, double * d, double * d0, double * u, double * v,double * w, double dt )
{
	double dt0 = dt*N;

	for ( int i=1 ; i<=N ; i++ )
	{
		for ( int j=1 ; j<=N ; j++ )
		{
			for ( int k=1 ; k<=N ; k++ )
			{
				//バックトレース
				double x = i-dt0*u[IX(i,j,k)];
				double y = j-dt0*v[IX(i,j,k)];
				double z = k-dt0*w[IX(i,j,k)];
			
				if (x<0.5) x=0.5; if (x>N+0.5) x=N+0.5;		//xの位置が領域内にあるように調整
				int i0=(int)x;								//1timestep前のxの含まれるグリッドのx座標
				int i1=i0+1;

				if (y<0.5) y=0.5; if (y>N+0.5) y=N+0.5;		//yの位置が領域内にあるように調整
				int j0=(int)y;								//1timestep前のyの含まれるグリッドのy座標
				int j1=j0+1;

				if (z<0.5) z=0.5; if (z>N+0.5) z=N+0.5;		//zの位置が領域内にあるように調整
				int k0=(int)z;								//1timestep前のzの含まれるグリッドのz座標
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
	fluid::set_bnd ( N, b, d );
}

//Gauss-Seidal反復法
void 
fluid::lin_solve ( int N, int b, double * x, double * x0, double a, double c )
{
	int i, l;
	int size=(N+2)*(N+2)*(N+2);
	double bx[5832];
	double e = 1.0f / pow(10.0, 20.0);

	//bxの初期化
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
		fluid::set_bnd ( N, b, x );

		//比較
		l=0;
		for(i=0;i<size;i++)
		{
			if(fabs(bx[i]-x[i])<=e)
				l+=1;
		}		
		if(l==size)
			break;

		//値の更新
		for(i=0;i<size;i++)
		{
			bx[i]=x[i];
		}
	}
}//*/

//境界条件
void 
fluid::set_bnd ( int N, int b, double * x )
{
	for ( int j=1 ; j<=N ; j++ ) {
		for ( int i=1 ; i<=N ; i++ ) {
			/*/境界面で法線成分を0にする(箱の中に流体を閉じ込める)
			x[IX(0  ,i,j)] = b==1 ? -x[IX(1,i,j)] : x[IX(1,i,j)];
			x[IX(N+1,i,j)] = b==1 ? -x[IX(N,i,j)] : x[IX(N,i,j)];
			x[IX(i,0  ,j)] = b==2 ? -x[IX(i,1,j)] : x[IX(i,1,j)];
			x[IX(i,N+1,j)] = b==2 ? -x[IX(i,N,j)] : x[IX(i,N,j)];
			x[IX(i,j,  0)] = b==3 ? -x[IX(i,j,1)] : x[IX(i,j,1)];
			x[IX(i,j,N+1)] = b==3 ? -x[IX(i,j,N)] : x[IX(i,j,N)];//*/

			/*/箱開放
			x[IX(0  ,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(N+1,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(i,0  ,j)] = b==2 ? 0.0 : 0.0;
			x[IX(i,N+1,j)] = b==2 ? 0.0 : 0.0;
			x[IX(i,j,  0)] = b==3 ? 0.0 : 0.0;
			x[IX(i,j,N+1)] = b==3 ? 0.0 : 0.0;//*/

			//
			x[IX(0  ,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(N+1,i,j)] = b==1 ? 0.0 : 0.0;
			x[IX(i,0  ,j)] = b==2 ? -x[IX(i,1,j)] : x[IX(i,1,j)];//下
			x[IX(i,N+1,j)] = b==2 ? 0.0 : 0.0;
			x[IX(i,j,  0)] = b==3 ? 0.0 : 0.0;
			x[IX(i,j,N+1)] = b==3 ? 0.0 : 0.0;//右

			
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
void 
fluid::project( int N, double * u, double * v, double * w, double * p, double * div )
{
	double d=-0.5/(double)(N*N);

	//初期化
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

	fluid::set_bnd ( N, 0, div );
	fluid::set_bnd ( N, 0, p );

	fluid::lin_solve ( N, 0, p, div, 1, 6 );

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
	fluid::set_bnd ( N, 1, u ); fluid::set_bnd ( N, 2, v ); fluid::set_bnd ( N, 3, w );
}

//速度場
void 
fluid::vel_step ( int N, double * u, double * v, double * w, double * u0, double * v0, double * w0, double visc, double dt )
{
	//外力項
	fluid::add_source ( N, u, u0, dt );
	fluid::add_source ( N, v, v0, dt );
	fluid::add_source ( N, w, w0, dt );

	//拡散項
	SWAP ( u0, u );	SWAP ( v0, v );	SWAP ( w0, w );
	fluid::diffuse ( N, 1, u, u0, visc, dt );
	fluid::diffuse ( N, 2, v, v0, visc, dt );
	fluid::diffuse ( N, 3, w, w0, visc, dt );
	fluid::project ( N, u, v, w, u0, v0 );

	//移流項
	SWAP ( u0, u );	SWAP ( v0, v );	SWAP ( w0, w );
	fluid::advect ( N, 1, u, u0, u0, v0, w0, dt );
	fluid::advect ( N, 2, v, v0, u0, v0, w0, dt );
	fluid::advect ( N, 3, w, w0, u0, v0, w0, dt );
	fluid::project ( N, u, v, w, u0, v0 );
}

//ユーザーインタラクション(未完成)
void 
fluid::get_from_UI ( int N, double * d, double * u, double * v, double * w )
{
	int i,j;

	int size = (N+2)*(N+2)*(N+2);
	//速度・密度の初期化
	for (i=0 ; i<size ; i++ ) {
		u[i] = v[i] = w[i] = d[i] = 0.0;
	}

	double a=8.0;
	//double a=20.0;
	int x=N-3;
	int z=3;
	//力源
	if(Z_wind==1)
	//if(nsteps<400)
	{
		for(i=1;i<N;i++)
		{
			//for(j=1;j<N-7;j++)
			for(j=1;j<N;j++)
			{
				w[IX(i,j,z)] = -F_FORCE*a;
			}
		}
	}
	if(X_wind==1)
	//else if(nsteps>=400)
	{
		for(i=1;i<N;i++)
		{
			for(j=1;j<N;j++)
			{
				u[IX(x,i,j)] = F_FORCE*a;
			}
		}
	}
	//return;
}



//速度の視覚化
void 
fluid::draw_velocity(double mag, double scale)
{
	int i, j, k;
	double x, y, z, h;

	int N=GRID;

	h = 1.0/N;//描画の範囲を決定

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

//流体Box
void 
fluid::draw_box(double mag)
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
