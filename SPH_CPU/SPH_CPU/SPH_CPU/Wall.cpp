#include "Wall.h"


std::vector<Wall> walls;

//Init the values of the walls
void initWalls() {
	//The side walls
	Vec3 center1(-1.0f, 0.0f, 0.0f);
	Vec3 normal1(1.0f, 0.0f, 0.0f);
	float xlength = 1.0f;
	float ylength = 1.0f;

	Vec3 center2(1.0f, 0.0f, 0.0f);
	Vec3 normal2(-1.0f, 0.0f, 0.0f);
	Wall w1(center1,normal1,xlength,ylength);
	Wall w2(center2,normal2,xlength,ylength);

	//The bottom wall
	Vec3 center3(0.0f, -1.0f, 0.0f);
	Vec3 normal3(0.0f, 1.0f, 0.0f);
	Wall w3(center3,normal3,xlength,xlength);

	//The front wall
	Vec3 center4(0.0f, 0.0f,1.0);
	Vec3 normal4(0.0f, 0.0f,-1.0f);
	Wall w4(center4,normal4,xlength,xlength);

	//The back wall
	Vec3 center5(0.0f, 0.0f, -1.0f);
	Vec3 normal5(0.0f, 0.0f,1.0f);
	Wall w5(center5,normal5,xlength,xlength);

	walls.push_back(w1);
	walls.push_back(w2);
	walls.push_back(w4);
	walls.push_back(w5);
	walls.push_back(w3);
}