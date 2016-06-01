/*
	a class to brief the line simulation
	date 2015
*/
#ifndef _STRING_H
#define _STRING_H

#include <cmath>
#include "CSM.h"
#include "utils.h"

class StringSimulation:public CSM
{
public:
	Vec3 lastParticle;
	Vec3 midParticle;
	float Length;
	Vec3 firstParticle;		//
private:
	//CSM *stringObj;
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

	//void setFirstParticle(Vec3 pos) { firstParticle = pos;}
	//void setBeginPos(Vec3 pos) { FixVertex(0,pos); }
	//void setLastPOs(Vec3 pos) {  FixVertex(GetNumOfVertices()-1,pos);}
	//void setLastVel(Vec3 vel) {setLastVel(vel);}
	//Vec3 getLastPos()  {return GetVertexPos(getVerticeNum()-1);}
	//return the number of the vertice
	//int getVerticeNum()	{	return GetNumOfVertices();	}

	//set the speed of wind
	void setWindSpeed(Vec3 speed)	{	m_vWindSpeed =speed; 	}
	//initialize the string
	void setup();
	//draw the string
	void draw();
	//update  the string
	void update();
	//clear the data
	void free_data();
	void fixVertexPos(int index,Vec3 vPos){FixVertex(index,vPos);}
	Vec3 calTension(int posIndex);

	
	int getIndex(int index) { return (int)(10+(GetNumOfVertices()-11)*index/5); }
	Vec3 getParticlePos(int i) {  return GetVertexPos(getIndex(i)); }
	
};

#endif	//_STRINGSIMULATION
