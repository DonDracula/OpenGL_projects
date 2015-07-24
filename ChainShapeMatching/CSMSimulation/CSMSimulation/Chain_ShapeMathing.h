/*
	Chain_ShapeMatching.h file
	brief Chain Shape Mathing Method
	data 2015-07
*/
#ifndef CHAINSHAPEMATCHING_H
#define CHAINSHAPEMATCHING_H

#include <vector>
#include <string>

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

using namespace std;

typedef void (*CollisionFunc)(Vec3&,Vec3&,Vec3&,int);
class ChainShapeMatching
{
public:
	struct Cluster
	{
		vector<int> Node;				//the list number of the cluster node
		int NumNode;					//the count of the node
		vector<Vec3> Disp;				//the position of the node
	};
};

#endif //CHAINSHAPEMATCHING