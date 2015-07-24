
#include "deformable.h"
#include <stdio.h>

//---------------------------------------------------------------------------
void DeformableParameters::setDefaults()
{
	timeStep = 0.01f;
	gravity.set(0.0f, -0.9f);

	bounds.min.zero();
	bounds.max.set(1.0f, 1.0f);

	alpha = 0.9f;
	beta = 0.98f;

	quadraticMatch = false;
	volumeConservation = false;

	allowFlip = true;
}

//---------------------------------------------------------------------------
Deformable::Deformable()
{
	reset();
}

//---------------------------------------------------------------------------
Deformable::~Deformable()
{
}


//---------------------------------------------------------------------------
void Deformable::reset()
{
	mNumVertices = 0;
	mOriginalPos.clear();
	mPos.clear();
	mNewPos.clear();
	mGoalPos.clear();
	mMasses.clear();
	mVelocities.clear();
    mFixed.clear();
}

//---------------------------------------------------------------------------
void Deformable::initState()
{
	for (int i = 0; i < mNumVertices; i++) {
		mPos[i] = mOriginalPos[i];
		mNewPos[i] = mOriginalPos[i];
		mGoalPos[i] = mOriginalPos[i];
		mVelocities[i].zero();
        mFixed[i] = false;
	}
}

//---------------------------------------------------------------------------
void Deformable::addVertex(const m2Vector &pos, float mass)
{
	mOriginalPos.push_back(pos);
	mPos.push_back(pos);
	mNewPos.push_back(pos);
	mGoalPos.push_back(pos);
	mMasses.push_back(mass);
	mVelocities.push_back(m2Vector(0,0));
    mFixed.push_back(false);
	mNumVertices++;

	initState();
}

//---------------------------------------------------------------------------
void Deformable::externalForces()
{
	int i;

	for (i = 0; i < mNumVertices; i++) {
    	if (mFixed[i]) continue;
		mVelocities[i] += params.gravity * params.timeStep;
		mNewPos[i] = mPos[i] + mVelocities[i] * params.timeStep;
		mGoalPos[i] = mOriginalPos[i];
	}

	// boundaries
	m2Real restitution = 0.9f;
	for (i = 0; i < mNumVertices; i++) {
    	if (mFixed[i]) continue;
		m2Vector &p = mPos[i];
		m2Vector &np = mNewPos[i];
		m2Vector &v = mVelocities[i];
		if (np.x < params.bounds.min.x || np.x > params.bounds.max.x) {
			np.x = p.x - v.x * params.timeStep * restitution;
			np.y = p.y;
		}
		if (np.y < params.bounds.min.y || np.y > params.bounds.max.y) {
			np.y = p.y - v.y * params.timeStep * restitution;
			np.x = p.x;
		}
        params.bounds.clamp(mNewPos[i]);
	}

}
//---------------------------------------------------------------------------
void Deformable::projectPositions()
{
	if (mNumVertices <= 1) return;
	int i,j,k;

	// center of mass
	m2Vector cm, originalCm;
	cm.zero(); originalCm.zero();
    float mass = 0.0f;

	for (i = 0; i < mNumVertices; i++) {
    	m2Real m = mMasses[i];
        if (mFixed[i]) m *= 100.0f;
    	mass += m;
		cm += mNewPos[i] * m;
		originalCm += mOriginalPos[i] * m;
	}

    cm /= mass;
    originalCm /= mass;

	m2Matrix Apq, Aqq;
	m2Vector p,q;
	Apq.zero();
	Aqq.zero();

	for (i = 0; i < mNumVertices; i++) {
		p = mNewPos[i] - cm;
		q = mOriginalPos[i] - originalCm;
		m2Real m = mMasses[i];
		Apq.r00 += m * p.x * q.x;
		Apq.r01 += m * p.x * q.y;
		Apq.r10 += m * p.y * q.x;
		Apq.r11 += m * p.y * q.y;

		Aqq.r00 += m * q.x * q.x;
		Aqq.r01 += m * q.x * q.y;
		Aqq.r10 += m * q.y * q.x;
		Aqq.r11 += m * q.y * q.y;
	}

	if (!params.allowFlip && Apq.determinant() < 0.0f) {  	// prevent from flipping
		Apq.r01 = -Apq.r01;
		Apq.r11 = -Apq.r11;
    }

	m2Matrix R,S;
	m2Matrix::polarDecomposition(Apq, R,S);

    if (!params.quadraticMatch) {	// --------- linear match
		m2Matrix A = Aqq;
		A.invert();
		A.multiply(Apq, A);

        if (params.volumeConservation) {
	        m2Real det = A.determinant();
	        if (det != 0.0f) {
            	det = 1.0f / sqrt(fabs(det));
                if (det > 2.0f) det = 2.0f;
		        A *= det;
            }
    	}

		m2Matrix T = R * (1.0f - params.beta) + A * params.beta;

        for (i = 0; i < mNumVertices; i++) {
            if (mFixed[i]) continue;
            q = mOriginalPos[i] - originalCm;
            mGoalPos[i] = T.multiply(q) + cm;
            mNewPos[i] += (mGoalPos[i] - mNewPos[i]) * params.alpha;
        }
    }
	else {	// -------------- quadratic match---------------------

        m2Real A5pq[2][5];
        A5pq[0][0] = 0.0f; A5pq[0][1] = 0.0f; A5pq[0][2] = 0.0f; A5pq[0][3] = 0.0f; A5pq[0][4] = 0.0f;
        A5pq[1][0] = 0.0f; A5pq[1][1] = 0.0f; A5pq[1][2] = 0.0f; A5pq[1][3] = 0.0f; A5pq[1][4] = 0.0f;
        m5Matrix A5qq;
        A5qq.zero();

        for (i = 0; i < mNumVertices; i++) {
            p = mNewPos[i] - cm;
            q = mOriginalPos[i] - originalCm;
            m2Real q5[5];
            q5[0] = q.x; q5[1] = q.y; q5[2] = q.x*q.x; q5[3] = q.y*q.y; q5[4] = q.x*q.y;
            m2Real m = mMasses[i];
            A5pq[0][0] += m * p.x * q5[0];
            A5pq[0][1] += m * p.x * q5[1];
            A5pq[0][2] += m * p.x * q5[2];
            A5pq[0][3] += m * p.x * q5[3];
            A5pq[0][4] += m * p.x * q5[4];
            A5pq[1][0] += m * p.y * q5[0];
            A5pq[1][1] += m * p.y * q5[1];
            A5pq[1][2] += m * p.y * q5[2];
            A5pq[1][3] += m * p.y * q5[3];
            A5pq[1][4] += m * p.y * q5[4];

            for (j = 0; j < 5; j++)
                for (k = 0; k < 5; k++)
                    A5qq(j,k) += m * q5[j]*q5[k];
        }

        A5qq.invert();

        m2Real A5[2][5];
        for (i = 0; i < 2; i++) {
            for (j = 0; j < 5; j++) {
                A5[i][j] = 0.0f;
                for (k = 0; k < 5; k++) {
                    A5[i][j] += A5pq[i][k] * A5qq(k,j);
                }
                A5[i][j] *= params.beta;
                if (j < 2)
                    A5[i][j] += (1.0f - params.beta) * R(i,j);
            }
        }

        m2Real det = A5[0][0]*A5[1][1] - A5[0][1]*A5[1][0];
        if (!params.allowFlip && det < 0.0f) {         		// prevent from flipping
           	A5[0][1] = -A5[0][1];
            A5[1][1] = -A5[1][1];
        }

        if (params.volumeConservation) {
	        if (det != 0.0f) {
            	det = 1.0f / sqrt(fabs(det));
                if (det > 2.0f) det = 2.0f;
		        A5[0][0] *= det; A5[0][1] *= det;
                A5[1][0] *= det; A5[1][1] *= det;
            }
    	}


        for (i = 0; i < mNumVertices; i++) {
            if (mFixed[i]) continue;
            q = mOriginalPos[i] - originalCm;
            mGoalPos[i].x = A5[0][0]*q.x + A5[0][1]*q.y + A5[0][2]*q.x*q.x + A5[0][3]*q.y*q.y + A5[0][4]*q.x*q.y;
            mGoalPos[i].y = A5[1][0]*q.x + A5[1][1]*q.y + A5[1][2]*q.x*q.x + A5[1][3]*q.y*q.y + A5[1][4]*q.x*q.y;
            mGoalPos[i] += cm;
            mNewPos[i] += (mGoalPos[i] - mNewPos[i]) * params.alpha;
        }
    }
}

//---------------------------------------------------------------------------
void Deformable::integrate()
{
	m2Real dt1 = 1.0f / params.timeStep;
	for (int i = 0; i < mNumVertices; i++) {
		mVelocities[i] = (mNewPos[i] - mPos[i]) * dt1;
		mPos[i] = mNewPos[i];
	}
}

//---------------------------------------------------------------------------
void Deformable::timeStep()
{
	externalForces();
	projectPositions();
	integrate();
}


//---------------------------------------------------------------------------
void Deformable::fixVertex(int nr, const m2Vector &pos)
{
    mNewPos[nr] = pos;
    mFixed[nr] = true;
}


//---------------------------------------------------------------------------
void Deformable::releaseVertex(int nr)
{
	mFixed[nr] = false;
}

//---------------------------------------------------------------------------
void Deformable::saveToFile(char *filename)
{
	FILE *f = fopen(filename, "w");
	if (!f) return;

	fprintf(f, "%i\n", mNumVertices);
	for (int i = 0; i < mNumVertices; i++) {
		fprintf(f, "%f %f %f\n", mOriginalPos[i].x, mOriginalPos[i].y, mMasses[i]);
	}
	fprintf(f, "%f\n", params.timeStep);
	fprintf(f, "%f %f\n", params.gravity.x, params.gravity.y);

	fprintf(f, "%f\n", params.alpha);
	fprintf(f, "%f\n", params.beta);

	fprintf(f, "%i\n", params.quadraticMatch);
	fprintf(f, "%i\n", params.volumeConservation);
	fprintf(f, "%i\n", params.allowFlip);

	fclose(f);
}


//---------------------------------------------------------------------------
void Deformable::loadFromFile(char *filename)
{
	FILE *f = fopen(filename, "r");
	if (!f) return;

	const int len = 100;
	char s[len+1];
	m2Vector pos;
	m2Real mass;
	int i;

	reset();
	int numVerts;
	fgets(s, len, f); sscanf(s, "%i", &numVerts);

	for (i = 0; i < numVerts; i++) {
		fgets(s, len, f); sscanf(s, "%f %f %f", &pos.x, &pos.y, &mass);
		addVertex(pos, mass);
	}

	fgets(s, len, f); sscanf(s, "%f", &params.timeStep);
	fgets(s, len, f); sscanf(s, "%f %f", &params.gravity.x, &params.gravity.y);

	fgets(s, len, f); sscanf(s, "%f", &params.alpha);
	fgets(s, len, f); sscanf(s, "%f", &params.beta);

	fgets(s, len, f); sscanf(s, "%i", &i); params.quadraticMatch = i;
	fgets(s, len, f); sscanf(s, "%i", &i); params.volumeConservation = i;
	fgets(s, len, f); sscanf(s, "%i", &i); params.allowFlip = i;

	fclose(f);
}

