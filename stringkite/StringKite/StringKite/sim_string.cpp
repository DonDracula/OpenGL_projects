
/*
	to brief the line simulation function
	date 2015
*/
#include "sim_string.h"
#include "macros4cc.h"

//Vec3 firstParticle;

StringSimulation::StringSimulation()
{
	firstParticle = Vec3(-2,0,0);
	
	Length=SEG_LENGTH*LINE_SP;
	
	m_weight = 0.01;
	g_v3EnvMin = Vec3(-6.0, RX_GOUND_HEIGHT-6, -6.0);
	g_v3EnvMax= Vec3(6.0, RX_GOUND_HEIGHT+5.0, 6.0);
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
	//stringObj = new CSM();

	vector<vector<int>> clusters;
	particles = vector<Vec3>();
	//get masses vertices
	Vec3 Over,Side;
	//init masses' position
	for(int index=0;index<=LINE_SP;index++)
	{
		Over=Vec3(1.0,0.0,0.2);	//斜め上方
		normalize(Over);
	//	cout<<"ssssssssssssssssssssssssssssssssssssssssssssssssss"<<Over<<endl;
		Over*=SEG_LENGTH*LINE_SP*((double)index)/((double)LINE_SP);
		Side=Vec3(SEG_LENGTH*LINE_SP*((double)index)/((double)LINE_SP),0.0,0.0);	//真横

		particles.push_back(firstParticle+Over);
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
		AddVertex(particles[i],m_weight);
	//	Vec3 mmmmm = stringObj->GetVertexPos(i);
		//cout <<mmmmm;
	}

	//cluster
	for(int i = 0; i< (int)clusters.size(); ++i)
	{
		AddCluster(clusters[i]);
	}
	//cout <<"cluster size::::::::::::::::::::::::::::::::"<<clusters.size()<<endl;

	// Shape Matchingの設定
	SetSimulationSpace(g_v3EnvMin, g_v3EnvMax);
	SetTimeStep(g_fDt);
	SetCollisionFunc(0);
	SetStiffness(alpha, beta);

	//get the first and last point of the line
	//setBeginPos(Vec3(-1,0,0));

	lastParticle = GetVertexPos(GetNumOfVertices()-1);
	midParticle= GetVertexPos((int)(GetNumOfVertices()-1)/2);
}

//画图函数
void StringSimulation::draw()
{
		glDisable(GL_LIGHTING);

		glPushMatrix();
			glColor3f ( 1.0f, 0.0f, 0.0f );//red
			//力の作用点
			glTranslated(GetVertexPos(0).data[0],GetVertexPos(0).data[2],GetVertexPos(0).data[1]);
				glutSolidSphere(0.02,15,15);
		glPopMatrix();

		glPushMatrix();
		//力の作用点
		glColor3f ( 0.0f, 1.0f, 0.0f );//red
			glTranslated(lastParticle.data[0],lastParticle.data[2],lastParticle.data[1]);
			glutSolidSphere(0.02,15,15);
		glPopMatrix();

		static const GLfloat difr[] = { 1.0f, 0.4f, 0.4f, 1.0f };	// 拡散色 : 赤
		glColor3f ( 0.0f, 0.0f, 0.0f );//black
		
		glLineWidth ( 1.0f );
		//draw line strip
		glBegin ( GL_LINE_STRIP );

		for(int index=0;index<GetNumOfVertices();index++)
			{
				//glVertex3dv(GetVertexPos(index).data);
				glVertex3d ( GetVertexPos(index).data[0],GetVertexPos(index).data[2],-GetVertexPos(index).data[1] );
			}
		glEnd();
		//drawing the rope ends here
		glEnable(GL_LIGHTING);
}

//更新绳子坐标
void StringSimulation::update()
{
	Update();

	lastParticle = GetVertexPos(GetNumOfVertices()-1) ;
		midParticle= GetVertexPos((int)(GetNumOfVertices()-1)/2);
}

//释放内存
void StringSimulation::free_data()
{
	//if(stringObj)
	//	delete (stringObj);
}

//calculate tension
Vec3 StringSimulation::calTension(int posIndex)
{
	
	Vec3 g_vec=Vec3(0.0,0.0,1.0);
	g_vec*=(SEG_LENGTH*LINE_RHO)*(-9.8);

	Vec3 d=GetVertexPos(posIndex)-GetVertexPos(posIndex-1);
	Vec3 vel=GetVertexVel(posIndex)-GetVertexVel(posIndex-1);

	Vec3 f1=(LINE_K * (norm(d) - SEG_LENGTH) + LINE_D * ( dot(vel,d) / norm(d) )) * ( d / norm(d) );

	Vec3 stringForce = Vec3();
//	Vec3 w_V = stringObj->m_vWindSpeed;

	stringForce = m_vWindSpeed+g_vec+f1;
	return stringForce;
}
