#include "Summation.h"

//holds the info - the dynamic properties are in a SumData
class Region : public Summation
{
public:
	Vec3 Ex0;
	Vec3 c0;
	float M;

	Vec3 c;
	rxMatrix3 A;
	rxMatrix3 eigenVectors;

	Vec3 t;
	rxMatrix3 R;
};