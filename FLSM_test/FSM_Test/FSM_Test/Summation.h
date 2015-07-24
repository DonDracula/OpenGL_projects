#include <vector>
#include <string>

#include "rx_utility.h"				//Vector classes
#include "rx_matrix.h"

using namespace std;

class LatticeLocatino;
class Particle;

//represents a region of object that will be summed independently,usually as a
//building block in generating the region sums. Defined principally by teh list
//of particles it contains. will be responsible for generating its own children
class Summation
{
public:
//	LatticeLocation *lp;
	vector<Particle*> particles;
	vector<Summation*> children,parents;
	vector<Summation*> *connection[2];  //pointers to above
	int minDim, maxDim;				//range along the split dimension

	Summation();
	void FindParticleRange(int dimension, int *minDim, int *maxDim);
	vector<Summation*> GenerateChildSums(int childLevel);		//returns the child summatinos that were generated

	void SUmFromChildren();
	void SUmFromParents();
	void PerformSummation(int direction);						//generalization
};

Summation *FindIdenticalSummation(vector<Particle*> &particles, int myLevel);