//---------------------------------------------------------------------------

#ifndef deformableH
#define deformableH
//---------------------------------------------------------------------------
#include <iostream>
#include <vector>
#include "Math2d/math2d.h"
#include <string>

//OpenGL
#include <GL/glew.h>
#include <GL/glut.h>

using namespace std;

struct DeformableParameters
{
	DeformableParameters() { setDefaults(); }
	float timeStep;
	m2Vector gravity;
	m2Bounds bounds;

	float alpha;
    float beta;

    bool quadraticMatch;
	bool volumeConservation;

	bool allowFlip;

	void setDefaults();
};


//---------------------------------------------------------------------------
class Deformable
{
public:
    Deformable();
    ~Deformable();

	void reset();
	void addVertex(const m2Vector &pos, float mass);

	void externalForces();
	void projectPositions();
	void integrate();

	void timeStep();

	DeformableParameters params;

	int  getNumVertices() const { return mNumVertices; }
	const m2Vector & getVertexPos(int nr) { return mPos[nr]; }
	const m2Vector & getOriginalVertexPos(int nr) { return mOriginalPos[nr]; }
	const m2Vector & getGoalVertexPos(int nr) { return mGoalPos[nr]; }
	const m2Real getMass(int nr) { return mMasses[nr]; }

	void fixVertex(int nr, const m2Vector &pos);
	bool isFixed(int nr) { return mFixed[nr]; }
	void releaseVertex(int nr);

	void saveToFile(char *filename);
	void loadFromFile(char *filename);

private:
	void initState();

	int mNumVertices;
	std::vector<m2Vector> mOriginalPos;
	std::vector<m2Vector> mPos;
	std::vector<m2Vector> mNewPos;
	std::vector<m2Vector> mGoalPos;
	std::vector<float> mMasses;
	std::vector<m2Vector> mVelocities;
    std::vector<bool> mFixed;
};

#endif