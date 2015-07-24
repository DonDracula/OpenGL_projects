#include <algorithm>
#include "Summation.h"
#include "Particle.h"
#include "Body.h"

vector<Summation*> Summation::GenerateChildSums(int childLevel)
{
	int i;
	unsigned int m;
	Particle *p;

	int splitDimension = childLevel;
	FindParticleRange(splitDimension, &minDim, &maxDim);

	vector<Summation*> newSums;

	//Generate the array of children, one per value of the sorting dimension
	Summation **childArray = new Summation*[maxDim - minDim+1];
	for(i = 0; i<maxDim-minDim+1;i++)
	{
		childArray[i] = new Summation();
	}

	//sort the particles into their correct children
	for(m = 0;m<particles.size(); m++)
	{
		p = particles[m];
	//	childArray[p->lp]
	}

	//now process each child 
	for(i = minDim;i<=maxDim;i++)
	{
		Summation *child = childArray[i-minDim];

		if(child->particles.empty())
		{
			delete child;
		}
		else{
			sort(child->particles.begin(),child->particles.end());
		}
	}
}

	void Summation::PerformSummation(int direction)
	{
		vector<Summation*> *sources = connection[direction];
		if(sources->size() == 0)
		{
		return;
		}
		else if(sources->size() == 1)
		{
			sumData.v = (*sources)[0]->sumData.v
		}
	}
}