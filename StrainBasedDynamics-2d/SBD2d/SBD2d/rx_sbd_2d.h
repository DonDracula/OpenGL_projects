/*

 brief Strain Based Dynamic 2d
 2015

*/

#ifndef RX_SBD2D_H
#define RX_SBD2D_H

//include files
#include<vector>
#include<string>

//OpenGL
#include<GL/glew.h>
#include<glut.h>

//utility file
#include "rx_utility.h"

//plygon mesh
#include "rx_mesh_e.h"

//define
using namespace std;

//SBD2d
// strain based dynamic funciton

class rxSBD2D
{
	vector<Vec2> m_vX,m_vP,m_vV;		//vertex positoin, predict postion ,velocity of the polygon
	vector<double> m_vM;				//vertex mass
	vector< vector<int> > m_vTri;		//index of the vertex
	vector<int> m_vFix;					//fix the positon
	vector<double> m_vQ;				//material positions of the polygon
	vector<double> m_vInvQ;				//inverse matrix of Q

	int m_iNt, m_iNv;					//acount of the Polygon and vertex

	int m_iNx, m_iNy;					//number of grid partitions
	double m_fGravity;					//gravity
	double m_fMass;	

	int m_iModifiedConstraint;			//constraint to modify the vertex postion

	double m_fTime;						//simulation time
	int m_iStep;						//step

public:
	rxSBD2D(int n);
	~rxSBD2D();

public:
	//initialization
	void Init(void);
	//updata
	int Update(double dt);
	//draw fucntion
	void Draw(int drw = 2+4, Vec3 line_col = Vec3(1.0),Vec3 vertex_col = Vec3(0.0,0.0,1.0));

	//search nearest vertex
	int Search(Vec2 pos, double h=0.05);
	int SearchNearest(Vec2 pos);

	//set fix vertex
	void SetFix(int idx, Vec2 pos);
	//unset the fixed vertex
	void UnsetFix(int idx);

protected:
	//calculate the strain tensor of the triangle
	void calStrainTensor(Vec2 *v, double *invq, Vec2 f[2], Vec2 S[2], Vec2 dS[3][2][2]);

	//calculate the correct postion of the vector 
	void calPositionCorrectionStrain(Vec2 *v, double *invq, double *invm, Vec2 dp[3]);

	//calculate the correct area
	void calPositionCorrectionArea(Vec2 *v, Vec2 *q,double *invm, Vec2 dp[3]);

	//create the mesh
	void generateMesh(Vec2 c1, Vec2 c2, int nx, int ny);
	//
	void generateRandomMesh(Vec2 c1, Vec2 c2, double min_dist,int n );

	//calculate the grid index
	inline int IDX(int i, int j, int n){ return i+j*n; };

};

#endif //RX_SBD2D_H