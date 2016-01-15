/*
	shape_matching.cpp
	brief Chain shape matching method
	date 2015-06
*/
//include file
#include "CSM.h"

CSM ::CSM()
{
	m_fDt = 0.01f;

	m_vGraviity = Vec3(0.0f,-9.8f,0.0);
	segmentLength = 0.05f;

	m_vWindSpeed = Vec3();

	m_vMin = Vec3(-1.0);
	m_vMax = Vec3(1.0);

	m_fAlpha = 0.8;
	m_fBeta = 0.9;

	m_bLinearDeformation = true;
	m_bVolumeConservation = true;

	m_Collision = 0;

	Clear();
}

CSM::~CSM()
{
}

void CSM::initialize(void)
{
	for(int i = 0; i<m_iNumOfVertices; ++i)
	{
		m_vCurPos[i] = m_vOrgPos[i];
		m_vNewPos[i] = m_vOrgPos[i];
	//	m_vGoalPos[i] = m_vOrgPos[i];
		m_vVel[i] = Vec3(0.0);
		m_vFix[i] = false;
		m_vNumCluster[i] = 0;
	}
}
//clear all the vertices
void CSM :: Clear()
{
	m_iNumOfVertices = 0;
	m_vOrgPos.clear();
	m_vCurPos.clear();
	m_vNewPos.clear();
//	m_vGoalPos.clear();
	m_vMass.clear();
	m_vVel.clear();
	m_vFix.clear();
	m_vNumCluster.clear();
	m_vCluster.clear();
}

//add vertices
//void ShapeMatching::AddVertex(const Vec3 &pos, double mass)
void CSM::AddVertex(const Vec3 &pos, double mass)
{
	m_vOrgPos.push_back(pos);
	m_vCurPos.push_back(pos);
	m_vNewPos.push_back(pos);
//	m_vGoalPos.push_back(p.position);
	m_vMass.push_back(mass);
	m_vVel.push_back(Vec3(0.0));
	m_vFix.push_back(false);
	m_vNumCluster.push_back(0);
	m_iNumOfVertices++;

	initialize();
}

//add the cluster of the vector
void CSM::AddCluster(const vector<int> &list)
{
	Cluster cl;
	cl.Node = list;
	cl.NumNode = (int)list.size();
	cl.Disp.resize(cl.NumNode,Vec3(0.0));
	m_vCluster.push_back(cl);

	for(int l= 0; l<cl.NumNode; ++l)
	{
		int i= cl.Node[l];
		m_vNumCluster[i]++;
	}
}

//externall forces add to the vertex(gravity.etc)
void CSM :: calExternalForces(double dt)
{
	for(int i = 0; i< m_iNumOfVertices;++i)
	{
		if(m_vFix[i]) continue;
		m_vVel[i] += m_vGraviity*dt;
		m_vNewPos[i] = m_vCurPos[i]+m_vVel[i]*dt;
//		m_vGoalPos[i] = m_vOrgPos[i];
	}

	double res = 0.9f;
	for(int i= 0; i< m_iNumOfVertices;++i)
	{
		if(m_vFix[i]) continue;
		Vec3 &p = m_vCurPos[i];
		Vec3 &np = m_vNewPos[i];
		Vec3 &v = m_vVel[i];
		if(np[0] < m_vMin[0] || np[0] > m_vMax[0]){
			np[0] = p[0]-v[0]*dt*res;
			np[1] = p[1];
			np[2] = p[2];
		}
		if(np[1] < m_vMin[1] || np[1] > m_vMax[1]){
			np[1] = p[1]-v[1]*dt*res;
			np[0] = p[0];
			np[2] = p[2];
		}
		if(np[2] < m_vMin[2] || np[2] > m_vMax[2]){
			np[2] = p[2]-v[2]*dt*res;
			np[0] = p[0];
			np[1] = p[1];
		}
		clamp(m_vNewPos[i]);
	}
}

void CSM::calCollision(double dt)
{
	//collision with other objects
	if(m_Collision != 0){
		for(int i = 0; i < m_iNumOfVertices; ++i){
			if(m_vFix[i]) continue;
			Vec3 &p = m_vCurPos[i];
			Vec3 &np = m_vNewPos[i];
			Vec3 &v = m_vVel[i];
			m_Collision(p, np, v, 1);
		}
	}
}

//shape matching function calculate the goal position
void CSM::shapeMatchingFun(Cluster &cl, double dt)
{
	if(cl.NumNode <=1) return;

	Vec3 cm(0.0),cm_org(0.0);						//center of gravity
	double mass = 0.0f;

	//calculate the center of gravity
	for(int l= 0; l<cl.NumNode;++l)
	{
		int i = cl.Node[l];

		double m= m_vMass[i];
		if(m_vFix[i]) m*=30.0f;
		mass+=m;
		cm+=m_vNewPos[i]*m;
		cm_org +=m_vOrgPos[i]*m;
	}
	cm /=mass;
	cm_org /= mass;

	rxMatrix3 Apq(0.0),Aqq(0.0);
	Vec3 p,q;

	// Apq = É∞mpq^T
	// Aqq = É∞mqq^T
	for(int l = 0;l< cl.NumNode; l++)
	{
		int i = cl.Node[l];

		p = m_vNewPos[i] - cm;
		q = m_vOrgPos[i] - cm_org;
		double m = m_vMass[i];

		Apq(0,0) += m*p[0]*q[0];
		Apq(0,1) += m*p[0]*q[1];
		Apq(0,2) += m*p[0]*q[2];
		Apq(1,0) += m*p[1]*q[0];
		Apq(1,1) += m*p[1]*q[1];
		Apq(1,2) += m*p[1]*q[2];
		Apq(2,0) += m*p[2]*q[0];
		Apq(2,1) += m*p[2]*q[1];
		Apq(2,2) += m*p[2]*q[2];

		Aqq(0,0) += m*q[0]*q[0];
		Aqq(0,1) += m*q[0]*q[1];
		Aqq(0,2) += m*q[0]*q[2];
		Aqq(1,0) += m*q[1]*q[0];
		Aqq(1,1) += m*q[1]*q[1];
		Aqq(1,2) += m*q[1]*q[2];
		Aqq(2,0) += m*q[2]*q[0];
		Aqq(2,1) += m*q[2]*q[1];
		Aqq(2,2) += m*q[2]*q[2];
	}

	rxMatrix3 R,S;
	PolarDecomposition(Apq,R,S);
	//polarDecompositionStable(Apq, eps, R);
	//Linear Deformations
	if(m_bLinearDeformation)
	{
		rxMatrix3 A;
		A = Apq*Aqq.Inverse();					// A = Apq*Aqq^-1
		// to make sure the volume is conserved, we use the Å„(det(A))
		if(m_bVolumeConservation)
		{
			double det = fabs(A.Determinant());
			if(det > RX_FEQ_EPS)
			{
				det = 1.0/sqrt(det);
				if(det>2.0) det = 2.0f;
				A *=det;
			}
		}

		//we use RL=É¿A+(1-É¿)R to calculate instead of the rotation R
		rxMatrix3 RL = m_fBeta*A+(1.0-m_fBeta)*R;

		//calculate the goal positin, and move current position to the new position
		for(int l=0; l< cl.NumNode;++l){
			int i = cl.Node[l];

			if(m_vFix[i]) continue;
			q = m_vOrgPos[i] - cm_org;
			Vec3 goal_pos = RL*q + cm;
		//	m_vGoalPos[i] = RL*q+cm;
			cl.Disp[l] = (goal_pos-m_vNewPos[i])*m_fAlpha;
		}
	}
	//Quadratic Deformatinos
	else
	{
		double Atpq[3][9];
		for(int j= 0;j<9;++j)
		{
			Atpq[0][j] = 0.0;
			Atpq[1][j] = 0.0;
			Atpq[2][j] = 0.0;
		}
		rxMatrixN<double ,9> Atqq;
		Atqq.SetValue(0.0);

		for(int l = 0; l<cl.NumNode; ++l)
		{
			int i = cl.Node[l];

			p=m_vNewPos[i] - cm;
			q = m_vOrgPos[i] - cm_org;
			//calculation of q~
			double qt[9];
			qt[0] = q[0];      qt[1] = q[1];      qt[2] = q[2];
			qt[3] = q[0]*q[0]; qt[4] = q[1]*q[1]; qt[5] = q[2]*q[2];
			qt[6] = q[0]*q[1]; qt[7] = q[1]*q[2]; qt[8] = q[2]*q[0];
			//calculate A~pq = É∞mpq~ 
			double m = m_vMass[i];
			for(int j = 0; j < 9; ++j){
				Atpq[0][j] += m*p[0]*qt[j];
				Atpq[1][j] += m*p[1]*qt[j];
				Atpq[2][j] += m*p[2]*qt[j];
			}

			//calculate A~qq = É∞mq~q~ 
			for(int j = 0; j < 9; ++j){
				for(int k = 0; k < 9; ++k){
					Atqq(j,k) += m*qt[j]*qt[k];
				}
			}
		}

		// calculate A~qq invert
		Atqq.Invert();

		double At[3][9];
		for(int i = 0; i < 3; ++i){
			for(int j = 0; j < 9; j++){
				At[i][j] = 0.0f;
				for(int k = 0; k < 9; k++){
					At[i][j] += Atpq[i][k]*Atqq(k,j);
				}

				// É¿A~+(1-É¿)R~
				At[i][j] *= m_fBeta;
				if(j < 3){
					At[i][j] += (1.0f-m_fBeta)*R(i,j);
				}
			}
		}
		 // a00a11a22+a10a21a02+a20a01a12-a00a21a12-a20a11a02-a10a01a22
		double det = At[0][0]*At[1][1]*At[2][2]+At[1][0]*At[2][1]*At[0][2]+At[2][0]*At[0][1]*At[1][2]
					-At[0][0]*At[2][1]*At[1][2]-At[2][0]*At[1][1]*At[0][2]-At[1][0]*At[0][1]*At[2][2];
		//make sure the volume is conserved
		if(m_bVolumeConservation){
			if(det != 0.0f){
				det = 1.0f/sqrt(fabs(det));
				if(det > 2.0f) det = 2.0f;
				for(int i = 0; i < 3; ++i){
					for(int j = 0; j < 3; ++j){
						At[i][j] *= det;
					}
				}
			}
		}

		//calculate the goal position,and move
		for(int l = 0; l < m_iNumOfVertices; ++l){
			int i = cl.Node[l];

			if(m_vFix[i]) continue;
			q = m_vOrgPos[i]-cm_org;

			Vec3 goal_pos = m_vCurPos[i];
			for(int k = 0; k < 3; ++k){
				goal_pos[k] = At[k][0]*q[0]+At[k][1]*q[1]+At[k][2]*q[2]
								  +At[k][3]*q[0]*q[0]+At[k][4]*q[1]*q[1]+At[k][5]*q[2]*q[2]+
								  +At[k][6]*q[0]*q[1]+At[k][7]*q[1]*q[2]+At[k][8]*q[2]*q[0];
			}

			goal_pos += cm;
			cl.Disp[l]= (goal_pos - m_vNewPos[i])*m_fAlpha;
		}

	}
}

//integrate the position an speed
void CSM::integrate(double dt)
{
	double dt1 = 1.0f/dt;
	Vec3 dir;	//to calculate the distance between two points
	//solve constrants
	for(int i = 1; i < m_iNumOfVertices; ++i){
		dir = m_vNewPos[i]-m_vNewPos[i-1];
		normalize(dir);
		m_vNewPos[i] = m_vNewPos[i-1] + dir*segmentLength;
	}

	for(int i = 0; i < m_iNumOfVertices; ++i){
		m_vVel[i] = (m_vNewPos[i]-m_vCurPos[i])*dt1;
		m_vCurPos[i] = m_vNewPos[i];
	}
	
}
//update the simulation step
void CSM::Update()
{
	calExternalForces(m_fDt);
	calCollision(m_fDt);
	
	if(m_vCluster.empty())
	{
		Cluster cl;
		cl.Node.resize(m_iNumOfVertices);
		cl.Disp.resize(m_iNumOfVertices,Vec3(0.0));
		for(int i = 0; i<m_iNumOfVertices;++i)
		{
			cl.Node[i] = i;
		}
		cl.NumNode = m_iNumOfVertices;
		shapeMatchingFun(cl,m_fDt);
	}
	else
	{
		int max_iter = 1;
		double dmax = 1.0e-8;
		double d2 = 0.0;
		int k;
		for(k = 0;k<max_iter;++k)
		{
			vector<Cluster>::iterator itr = m_vCluster.begin();
			for(; itr != m_vCluster.end(); ++itr)
			{
				shapeMatchingFun(*itr,m_fDt);
			}
			itr = m_vCluster.begin();

			d2 = 0.0;
			for(;itr!=m_vCluster.end(); ++itr)
			{
				for(int l = 0;l < itr->NumNode; ++l){
					int i = itr->Node[l];
					if(m_vFix[i]) continue;

					Vec3 dp = itr->Disp[l]/(double)m_vNumCluster[i];
					m_vNewPos[i] += dp;

					d2 += norm2(dp);
				}
			}
			if(d2 < dmax) break;
		}
	}

	integrate(m_fDt);
}

//set the fixed vertex
void CSM::FixVertex(int i, const Vec3 &pos)
{
	m_vNewPos[i] = pos;
	m_vFix[i] = true;
}

//unlock the fixed vertex
void CSM::UnFixVertex(int i)
{
	m_vFix[i] = false;
}
