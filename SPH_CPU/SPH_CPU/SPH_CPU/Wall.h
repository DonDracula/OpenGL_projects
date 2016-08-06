
/*
 * Class definition for a Wall
 */

#include <GL/glut.h>
#include <vector>
#include <string>

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

class Wall {
public:
    //Default constructor
	Wall() {
		center = Vec3(0.0f,0.0f,0.0f);
        normal = Vec3(0.0f,0.0f,0.0f);
        xlength = 0.0f;
		ylength = 0.0f;
    }

    //Explicit constructor
	Wall(Vec3 ncenter, Vec3 nnormal, float nxlength, float nylength) {
		center = ncenter;
		normal = nnormal;
		xlength = nxlength;
		ylength = nylength;
    }

    //Destructor
	~Wall(){}

//private:
    Vec3 normal;
    Vec3 center;
	float xlength;
	float ylength;
};