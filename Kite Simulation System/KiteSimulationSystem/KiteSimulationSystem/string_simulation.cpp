
/*
	to brief the line simulation function
	date 2015
*/
#include "string_simulation.h"



StringSimulation::StringSimulation()
{
	m_weight = 0.01;
	g_v3EnvMin = Vec3(-3.0, RX_GOUND_HEIGHT-1, -3.0);
	g_v3EnvMax= Vec3(3.0, RX_GOUND_HEIGHT+3.0, 3.0);
	g_fDt = 0.01;
	//set the alpha and beta of the csm
	alpha = 0.8;
	beta = 0.7;
	// the cluster size
	regionSize = 4;
}

StringSimulation::~StringSimulation()
{
}

//init the object
void StringSimulation::setup()
{
	stringObj = new CSM();

	vector<vector<int>> clusters;
	particles = vector<Vec3>();
	//get masses vertices
	Vec3 Over;
	//init masses' position
	for(int index=0;index<=LINE_SP;index++)
	{
		particles.push_back(Vec3(index*stringObj->segmentLength,0,0));
	}
	//create clusters
	for(int i = 0;i<particles.size()-regionSize+1;i++)
	{
		vector<int> c;
		for(int startPos= i; startPos < i+regionSize; startPos++)
		{
			c.push_back(startPos);
		}
		clusters.push_back(c);	
	}

	for(int i = 0;i<particles.size();i++)
	{
		stringObj->AddVertex(particles[i],m_weight);
		Vec3 mmmmm = stringObj->GetVertexPos(i);
		cout <<mmmmm;
	}

	//cluster
	for(int i = 0; i< (int)clusters.size(); ++i)
	{
		stringObj->AddCluster(clusters[i]);
	}
	//cout <<"cluster size::::::::::::::::::::::::::::::::"<<clusters.size()<<endl;

	// Shape Matchingの設定
	stringObj->SetSimulationSpace(g_v3EnvMin, g_v3EnvMax);
	stringObj->SetTimeStep(g_fDt);
	stringObj->SetCollisionFunc(0);
	stringObj->SetStiffness(alpha, beta);

	//get the first and last point of the line
	fixVertexPos(0,Vec3(0,0,0));

	firstParticle = stringObj->GetVertexPos(0); 
	stringObj->FixVertex(0,*firstParticle);

	lastParticle = stringObj->GetVertexPos(getVerticeNum()-1) ;

}

void StringSimulation::draw()
{
		glDisable(GL_LIGHTING);

		glPushMatrix();
			glColor3f ( 1.0f, 0.0f, 0.0f );//red
			//力の作用点
			glTranslated(firstParticle.data[0],firstParticle.data[1],firstParticle.data[2]);
				glutSolidSphere(0.05,15,15);
		glPopMatrix();

		glPushMatrix();
		//力の作用点
		glColor3f ( 0.0f, 1.0f, 0.0f );//red
			glTranslated(lastParticle.data[0],lastParticle.data[1],lastParticle.data[2]);
			glutSolidSphere(0.05,15,15);
		glPopMatrix();

		static const GLfloat difr[] = { 1.0f, 0.4f, 0.4f, 1.0f };	// 拡散色 : 赤
		glColor3f ( 0.0f, 0.0f, 0.0f );//black
		
		glLineWidth ( 1.0f );
		//draw line strip
		glBegin ( GL_LINE_STRIP );

		for(int index=0;index<getVerticeNum();index++)
			{
				glVertex3dv(stringObj->GetVertexPos(index).data);
			}
		glEnd();
		//drawing the rope ends here
		glEnable(GL_LIGHTING);
}
void StringSimulation::update()
{
	stringObj->Update();
	//lastParticle =stringObj->GetVertexPos(getVerticeNum()-1) ;
	lastParticle = stringObj->GetVertexPos(getVerticeNum()-1) ;
}

void StringSimulation::free_data()
{
	if(stringObj)
		delete (stringObj);
}

Vec3 StringSimulation::calTension()
{
	double L=4/((double)LINE_SP);
	Vec3 g_vec=Vec3(0.0,0.0,1.0);
	g_vec*=(L*LINE_RHO)*(-9.8);

	Vec3 d=stringObj->GetVertexPos(getVerticeNum()-1)-stringObj->GetVertexPos(getVerticeNum()-2);
	Vec3 vel=stringObj->GetVertexVel(getVerticeNum()-1)-stringObj->GetVertexVel(getVerticeNum()-2);

	Vec3 f1=(LINE_K * (norm(d) - L) + LINE_D * ( dot(vel,d) / norm(d) )) * ( d / norm(d) );

	Vec3 stringForce = Vec3();
//	Vec3 w_V = stringObj->m_vWindSpeed;

	stringForce = stringObj->m_vWindSpeed+g_vec+f1;
	return stringForce;
}