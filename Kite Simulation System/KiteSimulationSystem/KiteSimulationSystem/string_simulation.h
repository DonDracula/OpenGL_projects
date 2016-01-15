/*
	a class to brief the line simulation
	date 2015
*/
#ifndef _String?H
#define _STRING_H

#include <cmath>
#include "CSM.h"
#include "utils.h"

#define RX_GOUND_HEIGHT  -1.0		//ground height
//draw kiteline

class StringSimulation
{
public:
	Vec3 firstParticle;
	Vec3 lastParticle;
private:
	CSM *stringObj;
	vector<Vec3> particles;

	double m_weight;
	Vec3 g_v3EnvMin;
	Vec3 g_v3EnvMax;
	double g_fDt;
	//set the alpha and beta of the csm
	double alpha;
	double beta;
	// the cluster size
	int regionSize;

public:
	StringSimulation();
	~StringSimulation();

	//return the number of the vertice
	int getVerticeNum()	{	return stringObj->GetNumOfVertices();	}
	//set the speed of wind
	void setWindSpeed(Vec3 speed)	{	stringObj->m_vWindSpeed; 	}
	//initialize the string
	void setup();
	//draw the string
	void draw();
	//update  the string
	void update();
	//clear the data
	void free_data();
	void fixVertexPos(int index,Vec3 vPos){stringObj->FixVertex(index,vPos);}
	Vec3 calTension();
};

#endif	//_STRINGSIMULATION
