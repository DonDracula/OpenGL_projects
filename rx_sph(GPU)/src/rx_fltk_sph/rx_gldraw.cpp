/*! @file rx_gldraw.cpp
	
	@brief OpenGL描画関数群
 
*/


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include "rx_gldraw.h"



//
// OpenGL描画関数 
//
/*! 
 * x-z平面の描画
 * @param[in] n 網目の数
 * @param[in] delta 網目の幅
 * @param[in] color 描画色
 */
void RxGLDraw::DrawChackerBoard(const int &n, const float &delta, const Vec3 &color)
{
	GLfloat v0[3],v1[3],v2[3],v3[3];

	glLineWidth(1.0);
	glColor3f(color[0], color[1], color[2]);

	v0[1]=v1[1]=v2[1]=v3[1]=0.0f;
	for(int x = -n/2; x <= n/2; ++x){
		for(int z = -n/2; z <= n/2; ++z){
			v0[0]=0.0f+delta*z;
			v0[2]=0.0f+delta*x;

			v1[0]=v0[0]+delta;
			v1[2]=v0[2];

			v2[0]=v0[0]+delta;
			v2[2]=v0[2]+delta;

			v3[0]=v0[0];
			v3[2]=v0[2]+delta;

			glBegin(GL_LINE_LOOP);
			glVertex3fv(v1);
			glVertex3fv(v2);
			glVertex3fv(v3);
			glVertex3fv(v0);
			glEnd();	
		}
	}
}

/*! 
 * 3D軸の描画
 * @param[in] length 軸の長さ
 * @param[in] color_x x軸の描画色
 * @param[in] color_y y軸の描画色
 * @param[in] color_z z軸の描画色
 */
void RxGLDraw::DrawAxis3D(const float &length, const Vec3 &color_x, const Vec3 &color_y, const Vec3 &color_z)
{
	glBegin(GL_LINES);
	// x軸
	glColor3f(color_x[0], color_x[1], color_x[2]);
	glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(length, 0.0, 0.0);
	// y軸
	glColor3f(color_y[0], color_y[1], color_y[2]);
	glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, length, 0.0);
	// z軸
	glColor3f(color_z[0], color_z[1], color_z[2]);
	glVertex3f(0.0, 0.0, 0.0);
	glVertex3f(0.0, 0.0, length);
	glEnd();
}



/*! 
 * 立方体のワイヤーフレーム描画
 * @param[in] center 中心座標
 * @param[in] s_length 一辺の長さ
 * @param[in] color 描画色
 */
void RxGLDraw::DrawWireCube(const Vec3 &center, const float &s_length, const Vec3 &color)
{
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glColor3f(color[0], color[1], color[2]);
	glutWireCube(s_length);
	glPopMatrix();
}


/*! 
 * 直方体のワイヤーフレーム描画
 * @param[in] min 最小座標値
 * @param[in] max 最大座標値
 * @param[in] color 描画色
 */
void RxGLDraw::DrawWireCuboid(const Vec3 &min, const Vec3 &max, const Vec3 &color)
{
	glPushMatrix();
	glColor3f(color[0], color[1], color[2]);

	glBegin(GL_LINES);
	
	// x軸平行
	glVertex3f(min[0], min[1], min[2]);	glVertex3f(max[0], min[1], min[2]);
	glVertex3f(min[0], min[1], max[2]); glVertex3f(max[0], min[1], max[2]);
	glVertex3f(min[0], max[1], min[2]);	glVertex3f(max[0], max[1], min[2]);
	glVertex3f(min[0], max[1], max[2]);	glVertex3f(max[0], max[1], max[2]);
	
	// z軸平行
	glVertex3f(min[0], min[1], min[2]);	glVertex3f(min[0], min[1], max[2]);
	glVertex3f(min[0], max[1], min[2]);	glVertex3f(min[0], max[1], max[2]);
	glVertex3f(max[0], min[1], min[2]);	glVertex3f(max[0], min[1], max[2]);
	glVertex3f(max[0], max[1], min[2]);	glVertex3f(max[0], max[1], max[2]);

	// z軸平行
	glVertex3f(min[0], min[1], min[2]);	glVertex3f(min[0], max[1], min[2]);
	glVertex3f(min[0], min[1], max[2]);	glVertex3f(min[0], max[1], max[2]);
	glVertex3f(max[0], min[1], min[2]);	glVertex3f(max[0], max[1], min[2]);
	glVertex3f(max[0], min[1], max[2]);	glVertex3f(max[0], max[1], max[2]);
	
	glEnd();

	glPopMatrix();
}


/*! 
 * 立方体の描画
 * @param[in] center 中心座標
 * @param[in] s_length 一辺の長さ
 * @param[in] color 描画色
 */
void RxGLDraw::DrawSolidCube(const Vec3 &center, const float &s_length, const Vec3 &color)
{
	float sl = s_length/2.0f;
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);
	glColor3f(color[0], color[1], color[2]);
	glutSolidCube(s_length);
	glPopMatrix();
}

/*! 
 * 立方体の描画(各面色分け)
 * @param[in] center 中心座標
 * @param[in] s_length 一辺の長さ
 * @param[in] color[6] 各面の描画色(x+,x-,y+,y-,z+,z-)
 */
void RxGLDraw::DrawSolidCubeColor(const Vec3 &center, const float &s_length, const Vec3 color[6])
{
	float sl = s_length/2.0f;
	glPushMatrix();
	glTranslatef(center[0], center[1], center[2]);

	glColor3f(color[0][0], color[0][1], color[0][2]);
	glNormal3f(1.0, 0.0, 0.0); // 右面
	glBegin(GL_QUADS);
	glVertex3f( sl, -sl, -sl);
	glVertex3f( sl,  sl, -sl);
	glVertex3f( sl,  sl,  sl);
	glVertex3f( sl, -sl,  sl);
	glEnd();

	glColor3f(color[1][0], color[1][1], color[1][2]);
	glNormal3f(-1.0, 0.0, 0.0); // 左面
	glBegin(GL_QUADS);
	glVertex3f(-sl, -sl, -sl);
	glVertex3f(-sl, -sl,  sl);
	glVertex3f(-sl,  sl,  sl);
	glVertex3f(-sl,  sl, -sl);
	glEnd();

	glColor3f(color[2][0], color[2][1], color[2][2]);
	glNormal3f(0.0, 1.0, 0.0); // 上面
	glBegin(GL_QUADS);
	glVertex3f(-sl,  sl, -sl);
	glVertex3f(-sl,  sl,  sl);
	glVertex3f( sl,  sl,  sl);
	glVertex3f( sl,  sl, -sl);
	glEnd();

	glColor3f(color[3][0], color[3][1], color[3][2]);
	glNormal3f(0.0, -1.0, 0.0); // 下面
	glBegin(GL_QUADS);
	glVertex3f(-sl, -sl, -sl);
	glVertex3f( sl, -sl, -sl);
	glVertex3f( sl, -sl,  sl);
	glVertex3f(-sl, -sl,  sl);
	glEnd();

	glColor3f(color[4][0], color[4][1], color[4][2]);
	glNormal3f(0.0, 0.0, -1.0); // 前面
	glBegin(GL_QUADS);
	glVertex3f(-sl, -sl,  sl);
	glVertex3f( sl, -sl,  sl);
	glVertex3f( sl,  sl,  sl);
	glVertex3f(-sl,  sl,  sl);
	glEnd();

	glColor3f(color[5][0], color[5][1], color[5][2]);
	glNormal3f(0.0, 0.0, 1.0); // 後面
	glBegin(GL_QUADS);
	glVertex3f(-sl, -sl, -sl);
	glVertex3f(-sl,  sl, -sl);
	glVertex3f( sl,  sl, -sl);
	glVertex3f( sl, -sl, -sl);
	glEnd();

	glPopMatrix();
}


/*! 
 * 直方体の描画(各面色分け)
 * @param[in] min 最小座標値
 * @param[in] max 最大座標値
 * @param[in] color[6] 各面の描画色(x+,x-,y+,y-,z+,z-)
 */
void RxGLDraw::DrawSolidCuboidColor(const Vec3 &min, const Vec3 &max, const Vec3 color[6], const int &sign)
{
	glPushMatrix();

	glColor3f(color[0][0], color[0][1], color[0][2]);
	glNormal3f(1.0*sign, 0.0, 0.0); // 右面
	glBegin(GL_POLYGON);
	glVertex3f(max[0], min[1], min[2]);
	glVertex3f(max[0], max[1], min[2]);
	glVertex3f(max[0], max[1], max[2]);
	glVertex3f(max[0], min[1], max[2]);
	glEnd();

	glColor3f(color[1][0], color[1][1], color[1][2]);
	glNormal3f(-1.0*sign, 0.0, 0.0); // 左面
	glBegin(GL_POLYGON);
	glVertex3f(min[0], min[1], min[2]);
	glVertex3f(min[0], min[1], max[2]);
	glVertex3f(min[0], max[1], max[2]);
	glVertex3f(min[0], max[1], min[2]);
	glEnd();

	glColor3f(color[2][0], color[2][1], color[2][2]);
	glNormal3f(0.0, 1.0*sign, 0.0); // 上面
	glBegin(GL_POLYGON);
	glVertex3f(min[0], max[1], min[2]);
	glVertex3f(min[0], max[1], max[2]);
	glVertex3f(max[0], max[1], max[2]);
	glVertex3f(max[0], max[1], min[2]);
	glEnd();

	glColor3f(color[3][0], color[3][1], color[3][2]);
	glNormal3f(0.0, -1.0*sign, 0.0); // 下面
	glBegin(GL_POLYGON);
	glVertex3f(min[0], min[1], min[2]);
	glVertex3f(max[0], min[1], min[2]);
	glVertex3f(max[0], min[1], max[2]);
	glVertex3f(min[0], min[1], max[2]);
	glEnd();

	glColor3f(color[4][0], color[4][1], color[4][2]);
	glNormal3f(0.0, 0.0, -1.0*sign); // 前面
	glBegin(GL_POLYGON);
	glVertex3f(min[0], min[1], max[2]);
	glVertex3f(max[0], min[1], max[2]);
	glVertex3f(max[0], max[1], max[2]);
	glVertex3f(min[0], max[1], max[2]);
	glEnd();

	glColor3f(color[5][0], color[5][1], color[5][2]);
	glNormal3f(0.0, 0.0, 1.0*sign); // 後面
	glBegin(GL_POLYGON);
	glVertex3f(min[0], min[1], min[2]);
	glVertex3f(min[0], max[1], min[2]);
	glVertex3f(max[0], max[1], min[2]);
	glVertex3f(max[0], min[1], min[2]);
	glEnd();

	glPopMatrix();
}



/*!
 * 矢印の描画
 * @param origin 矢印の原点
 * @param dir 矢印の方向
 * @param scale 矢印の大きさ
 * @return 
 */
void RxGLDraw::DrawArrow2D(const Vec3 &origin, const Vec3 &dir, const float &scale)
{
	double theta;
	double arrow_length, arrow_x, arrow_y;

	if(fabs(dir[0]) < RX_EPS){
		theta = 90.0;
	}
	else{
		theta = 180.0/RX_PI*atan(dir[1]/dir[0]);
	}
	
	// 矢印の傘部分の設定
	arrow_length = scale*norm2(dir);
	if(arrow_length < scale*0.4){	// 矢印の長さが小さいときは矢印の長さを基準に傘を決定
		arrow_x = 0.2*arrow_length;
	}
	else{						// 矢印の長さが大きいときは傘の大きさは固定
		arrow_x = 0.2*scale;
	}
	arrow_y = arrow_x*tan(0.174532925);

	glPushMatrix();

	glTranslatef(origin[0], origin[1], 0.0);	// 矢印原点に移動
	glRotatef(theta, 0.0, 0.0, 1.0);			// 矢印方向に回転(z軸中心)

//	glColor3f(0.0, 0.8, 0.2);
	glBegin(GL_LINES);
	glVertex2f(0.0f, 0.0f);
	glVertex2f(arrow_length, 0.0f);
	glVertex2f(arrow_length, 0.0f);
	glVertex2f(arrow_length-arrow_x, arrow_y);
	glVertex2f(arrow_length, 0.0f);
	glVertex2f(arrow_length-arrow_x, -arrow_y);
	glEnd();

	glPopMatrix();
}

/*!
 * 球の描画(glutSolidSphereを使用)
 * @param cen 球の中心
 * @param rad 球の半径
 * @param col 描画色
 */
void RxGLDraw::DrawSphere(const Vec3 &cen, const float &rad, const Vec3 &col)
{
	glPushMatrix();
	glTranslatef(cen[0], cen[1], cen[2]);
	glRotated(90, 1.0, 0.0, 0.0);
	glColor3f(col[0], col[1], col[2]);
	glutSolidSphere(rad, 20, 10);
	glPopMatrix();
}

/*!
 * 円の描画
 * @param cen 円の中心
 * @param rad 円の半径
 */
void RxGLDraw::DrawCircle(const Vec3 &cen, const float &rad)
{
	float t = 0.0f;
	float dt = (float)RX_PI/16.0f;

	glPushMatrix();

	glTranslatef(cen[0], cen[1], cen[2]);
	glBegin(GL_POLYGON);
	do{
		glVertex2f(rad*cos(t), rad*sin(t));
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();

	glPopMatrix();
}

/*!
 * 円のワイヤーフレーム描画
 * @param cen 円の中心
 * @param rad 円の半径
 * @param n 分割数
 */
void RxGLDraw::DrawWireCircle(const Vec3 &cen, const float &rad, const int &n)
{
	float t = 0.0f;
	float dt = 2.0*RX_PI/(float)n;

	glPushMatrix();

	glTranslatef(cen[0], cen[1], cen[2]);
	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), rad*sin(t), 0.0);
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();

	glPopMatrix();
}

/*!
 * 原点中心の円のワイヤーフレーム描画
 * @param rad 円の半径
 * @param n 分割数
 */
void RxGLDraw::DrawWireCircle(const float &rad, const int &n)
{
	float t = 0.0f;
	float dt = 2.0*RX_PI/(float)n;

	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), rad*sin(t), 0.0);
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();
}

/*!
 * 原点中心の円のワイヤーフレーム描画(XZ平面)
 * @param rad 円の半径
 * @param n 分割数
 */
void RxGLDraw::DrawWireCircleXZ(const float &rad, const int &n)
{
	float t = 0.0f;
	float dt = 2.0*RX_PI/(float)n;

	glBegin(GL_LINE_LOOP);
	do{
		glVertex3f(rad*cos(t), 0.0, rad*sin(t));
		t += dt;
	}while(t < 2.0*RX_PI);
	glEnd();
}

/*!
 * 球のワイヤーフレーム描画
 * @param cen 球の中心
 * @param rad 球の半径
 * @param col 描画色
 */
void RxGLDraw::DrawWireSphere(const Vec3 &cen, const float &rad, const Vec3 &col)
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glTranslatef(cen[0], cen[1], cen[2]);
	glRotatef(90, 1.0, 0.0, 0.0);
	glColor3f(col[0], col[1], col[2]);

	// 緯度(x-y平面に平行)
	float z, dz;
	dz = 2.0*rad/8.0f;
	z = -(rad-dz);
	do{
		glPushMatrix();
		glTranslatef(0.0, 0.0, z);
		DrawWireCircle(sqrt(rad*rad-z*z), 32);
		glPopMatrix();
		z += dz;
	}while(z < rad);

	// 経度(z軸まわりに回転)
	float t, dt;
	t = 0.0f;
	dt = 180.0/8.0;
	do{
		glPushMatrix();
		glRotatef(t,  0.0, 0.0, 1.0);
		DrawWireCircleXZ(rad, 32);
		glPopMatrix();

		t += dt;
	}while(t < 180);

	//glutWireSphere(rad, 10, 5);
	glPopMatrix();
}



/*! 
 * 単一ポリゴンの描画
 * @param[in] vrts 頂点座標格納コンテナ
 * @param[in] idxes 頂点位相
 * @param[in] nrm ポリゴン法線
 * @param[in] col 描画色
 */
void RxGLDraw::DrawPolygon(const vector<Vec3> &vrts, 
						   const vector<int> &idx,
						   const Vec3 &nrm, 
						   const Vec3 &col)
{
	glBegin(GL_POLYGON);
	glColor3dv(col.data);
	glNormal3dv(nrm.data); 

	for(int i = 0; i < (int)idx.size(); ++i){
		glVertex3dv(vrts[idx[i]-1].data);
	}
	glEnd();
}
/*! 
 * 単一ポリゴンの描画(ポリゴン法線計算含む)
 * @param[in] vrts 頂点座標格納コンテナ
 * @param[in] idxes 頂点位相
 * @param[in] nrm ポリゴン法線
 * @param[in] col 描画色
 */
void RxGLDraw::DrawPolygon(const vector<Vec3> &vrts, 
						   const vector<int> &idx,
						   const Vec3 &col)
{
	Vec3 nrm = Unit(cross(vrts[idx[1]-1]-vrts[idx[0]-1], vrts[idx.back()-1]-vrts[idx[0]-1]));
	DrawPolygon(vrts, idx, nrm, col);

}

/*! 
 * OpenGLによるグローシェーディングで単一ポリゴン描画
 * @param[in] vrts 頂点座標格納コンテナ
 * @param[in] idxes 頂点位相
 * @param[in] nrms 頂点法線コンテナ
 * @param[in] col 描画色
 */
void RxGLDraw::DrawPolygonGouraud(const vector<Vec3> &vrts, 
								  const vector<int> &idx,
								  const vector<Vec3> &nrms, 
								  const Vec3 &col)
{
	glColor3dv(col.data);

	glBegin(GL_POLYGON);
	for(int i = 0; i < (int)idx.size(); ++i){
		glNormal3dv(nrms[idx[i]-1].data);
		glVertex3dv(vrts[idx[i]-1].data);
	}
	glEnd();
}


/*! 
 * OpenGLによるグローシェーディングで複数ポリゴン描画(ポリゴン法線)
 * @param[in] polys ポリゴンを格納したコンテナ
 * @param[in] gmat 全体の材質
 */
void RxGLDraw::DrawPolygonsNormal(const vector<Vec3> &vrts, 
								  const vector< vector<int> > &idxes, 
								  const Vec3 &col, bool select)
{
	for(int i = 0; i < (int)idxes.size(); ++i){
		if(select) glLoadName(i);
		DrawPolygon(vrts, idxes[i], col);
	}
}

/*! 
 * OpenGLによるグローシェーディングで複数ポリゴン描画(頂点法線)
 * @param[in] polys ポリゴンを格納したコンテナ
 * @param[in] gmat 全体の材質
 */
void RxGLDraw::DrawPolygonsGouraud(const vector<Vec3> &vrts, 
								   const vector< vector<int> > &idxes, 
								   const vector<Vec3> &nrms, 
								   const Vec3 &col, bool select)
{
	for(int i = 0; i < (int)idxes.size(); ++i){
		if(select) glLoadName(i);
		DrawPolygonGouraud(vrts, idxes[i], nrms, col);
	}
}

/*! 
 * ポリゴンの描画(ワイヤーフレーム)
 * @param[in] verts 頂点
 * @param[in] index 位相情報
 * @param[in] width 線幅
 * @param[in] color 線色
 */
void RxGLDraw::DrawLineLoop(const vector<Vec3> &vrts, 
							const vector<int> &idx, 
							const double &width, 
							const Vec3 &col)
{
	glLineWidth((GLfloat)width);
	glColor3dv(col.data);
	glBegin(GL_LINE_LOOP);
	for(int i = 0; i < (int)idx.size(); ++i){
		glVertex3dv(vrts[idx[i]-1].data);
	}
	glEnd();
	glLineWidth(1.0);
}



/*!
 * ポリゴン描画
 * @param[in] vrts 頂点列
 * @param[in] idxs 位相列
 * @param[in] draw_nrm 法線描画フラグ
 */
void RxGLDraw::DrawPolygons(const vector<Vec3> &vrts, const vector< vector<int> > &idxs, bool draw_nrm)
{
	for(int i = 0; i < (int)idxs.size(); ++i){
		vector<int> idx = idxs[i];
		int nv = (int)idx.size();

		// 法線
		Vec3 nrm;
		nrm = cross(vrts[idx[1]]-vrts[idx[0]], vrts[idx[nv-1]]-vrts[idx[0]]);
		normalize(nrm);

		if(draw_nrm){
			// 重心を求める
			Vec3 mc(0.0);
			for(int j = 0; j < nv; ++j){
				mc += vrts[idx[j]];
			}
			mc /= (double)nv;

			GLboolean lighting = glIsEnabled(GL_LIGHTING);
			glDisable(GL_LIGHTING);
			glBegin(GL_LINES);
			glVertex3dv(mc.data);
			glVertex3dv((mc+0.05*nrm).data);
			glEnd();

			(lighting == GL_TRUE) ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
		}

		glNormal3dv(nrm.data);

		// 面
		glBegin(GL_POLYGON);
		for(int j = 0; j < nv; ++j){
			glVertex3dv(vrts[idx[j]].data);
		}
		glEnd();
	}
}


/*!
 * ポリゴン描画
 * @param[in] vrts 頂点列
 * @param[in] idxs 位相列
 * @param[in] draw_nrm 法線描画フラグ
 */
void RxGLDraw::DrawPolygons(const float *vrts, const float *nrms, int vnum, bool draw_nrm)
{
	int pnum = vnum/3;
	for(int i = 0; i < pnum; ++i){
		// 面
		glBegin(GL_POLYGON);
		for(int j = 0; j < 3; ++j){
			glNormal3fv(&nrms[4*(3*i+j)]);
			glVertex3fv(&vrts[4*(3*i+j)]);
		}
		glEnd();

		// 法線描画
		if(draw_nrm){
			// 重心を求める
			Vec3 mc(0.0);
			for(int j = 0; j < 3; ++j){
				mc[0] += vrts[4*(3*i+j)+0];
				mc[1] += vrts[4*(3*i+j)+1];
				mc[2] += vrts[4*(3*i+j)+2];
			}
			mc /= 3.0;

			Vec3 nr(0.0);
			for(int j = 0; j < 3; ++j){
				nr[0] += nrms[4*(3*i+j)+0];
				nr[1] += nrms[4*(3*i+j)+1];
				nr[2] += nrms[4*(3*i+j)+2];
			}
			nr /= 3.0;

			normalize(nr);

			GLboolean lighting = glIsEnabled(GL_LIGHTING);
			glDisable(GL_LIGHTING);
			glColor3f(1.0, 1.0, 0.0);
			glBegin(GL_LINES);
			glVertex3dv(mc.data);
			glVertex3dv((mc+0.03*nr).data);
			glEnd();

			(lighting == GL_TRUE) ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
		}
	}
}

/*!
 * 三角形メッシュ描画
 * @param[in] vrts 頂点列
 * @param[in] idxes メッシュを構成する頂点インデックス
 * @param[in] normal 法線描画ON/OFF
 */
void RxGLDraw::DrawMesh(const vector<Vec3> &vrts, vector<int> &idxes, bool normal)
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	int nv = 3;
	int ntris = (int)idxes.size()/nv;
	for(int i = 0; i < ntris; ++i){
		int *idx = &idxes[3*i];

		// 法線
		Vec3 nrm;
		nrm = cross(vrts[idx[1]]-vrts[idx[0]], vrts[idx[nv-1]]-vrts[idx[0]]);
		normalize(nrm);

		if(normal){
			// 重心を求める
			Vec3 mc(0.0);
			for(int j = 0; j < nv; ++j){
				mc += vrts[idx[j]];
			}
			mc /= (double)nv;

			int lighting = glIsEnabled(GL_LIGHTING);
			glDisable(GL_LIGHTING);
			glBegin(GL_LINES);
			glVertex3dv(mc.data);
			glVertex3dv((mc+0.05*nrm).data);
			glEnd();

			lighting ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
		}

		glNormal3dv(nrm.data);

		// 面
		glBegin(GL_POLYGON);
		for(int j = 0; j < nv; ++j){
			glVertex3dv(vrts[idx[j]].data);
		}
		glEnd();
	}
}

/*!
 * 三角形メッシュ描画
 * @param[in] vrts 頂点列
 * @param[in] idxes メッシュを構成する頂点インデックス
 * @param[in] normal 法線描画ON/OFF
 */
void RxGLDraw::DrawTriangles(const vector<Vec3> &vrts, const vector<int> &tris, const vector<Vec3> &vnrms, bool normal)
{
	int nvt = 3;
	int ntris = (int)tris.size()/nvt;
	cout << "ntris = " << ntris << endl;
	int k = 0;
	for(int i = 0; i < ntris; ++i){
		// 面
		glColor4d(0.0, 0.0, 1.0, 1.0);
		glBegin(GL_POLYGON);
		for(int j = 0; j < nvt; ++j){
			Vec3 pos = vrts[tris[k]];
			Vec3 nrm = vnrms[tris[k]];
			glNormal3dv(nrm.data);
			glVertex3dv(pos.data);

			k++;
		}
		glEnd();
	}

	// 法線
	//if(normal){
		//for(int i = 0; i < ntris; ++i){
		//	for(int j = 0; j < nvt; ++j){
		//		Vec3 pos = vrts[tris[3*i+j]];
		//		Vec3 nrm = vnrms[tris[3*i+j]];

		//		int lighting = glIsEnabled(GL_LIGHTING);
		//		glDisable(GL_LIGHTING);
		//		glBegin(GL_LINES);
		//		glColor3d(1.0, 1.0, 0.3);
		//		glVertex3dv(pos.data);
		//		glColor3d(0.5, 0.5, 0.0);
		//		glVertex3dv((pos+0.05*nrm).data);
		//		glEnd();
		//		lighting ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
		//	}
		//}
	//}
}
/*!
 * 三角形メッシュ描画
 * @param[in] vrts 頂点列
 * @param[in] idxes メッシュを構成する頂点インデックス
 * @param[in] normal 法線描画ON/OFF
 */
void RxGLDraw::DrawTriangles(const Vec3 *vrts, const int *tris, const Vec3 *vnms, int nvrts, int ntris, bool normal)
{
/*	glColor4d(0.0, 0.0, 1.0, 1.0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	glVertexPointer(3, GL_DOUBLE, 0, (GLdouble*)vrts);
	glNormalPointer(GL_DOUBLE, 0, (GLdouble*)vnms);

	glDrawElements(GL_TRIANGLES, ntris*3, GL_UNSIGNED_INT, tris);

	glDisableClientState(GL_VERTEX_ARRAY); 
	glDisableClientState(GL_NORMAL_ARRAY); 
*/
	int k = 0;
	for(int i = 0; i < ntris; ++i){
		// 面
		glBegin(GL_POLYGON);
		for(int j = 0; j < 3; ++j){
			Vec3 pos = vrts[tris[k]];
			Vec3 nrm = vnms[tris[k]];
			glNormal3dv(nrm.data);
			glVertex3dv(pos.data);

			k++;
		}
		glEnd();
	}

	// 法線
	if(normal){
		for(int i = 0; i < ntris; ++i){
			for(int j = 0; j < 3; ++j){
				Vec3 pos = vrts[tris[3*i+j]];
				Vec3 nrm = vnms[tris[3*i+j]];

				int lighting = glIsEnabled(GL_LIGHTING);
				glDisable(GL_LIGHTING);
				glBegin(GL_LINES);
				glColor3d(1.0, 1.0, 0.3);
				glVertex3dv(pos.data);
				glColor3d(0.5, 0.5, 0.0);
				glVertex3dv((pos+0.05*nrm).data);
				glEnd();
				lighting ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
			}
		}
	}
}


/*!
 * 三角形メッシュ描画
 * @param[in] vrts 頂点列
 * @param[in] idxes メッシュを構成する頂点インデックス
 * @param[in] normal 法線描画ON/OFF
 */
void RxGLDraw::DrawMesh(const vector<Vec3> &vrts, vector<int> &idxes, const vector<Vec3> &vnrms, 
						double col[4], double ncol0[4], double ncol1[4], bool normal)
{
	int nv = 3;
	int ntris = (int)idxes.size()/nv;
	for(int i = 0; i < ntris; ++i){
		int *idx = &idxes[3*i];

		// 面
		glColor4dv(col);
		glBegin(GL_POLYGON);
		for(int j = 0; j < nv; ++j){
			Vec3 pos = vrts[idx[j]];
			Vec3 nrm = vnrms[idx[j]];
			glNormal3dv(nrm.data);
			glVertex3dv(pos.data);
		}
		glEnd();

		// 法線
		if(normal){
			for(int j = 0; j < nv; ++j){
				Vec3 pos = vrts[idx[j]];
				Vec3 nrm = vnrms[idx[j]];

				int lighting = glIsEnabled(GL_LIGHTING);
				glDisable(GL_LIGHTING);
				glBegin(GL_LINES);
				glColor4dv(ncol0);
				glVertex3dv(pos.data);
				glColor4dv(ncol1);
				glVertex3dv((pos+0.05*nrm).data);
				glEnd();
				lighting ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
			}
		}
	}
}

/*!
 * 幾何情報を持たない頂点列から三角形メッシュ描画
 * @param[in] vrts 頂点座標
 * @param[in] vmns 頂点法線
 * @param[in] normal 法線描画ON/OFF
 */
void RxGLDraw::DrawMeshV(const vector<Vec3> &vrts, const vector<Vec3> &vnms, bool normal)
{
	int ntris = (int)vrts.size()/3;
	for(int i = 0; i < ntris; ++i){
		// 面
		glColor4d(0.3, 0.3, 1.0, 1.0);
		glBegin(GL_POLYGON);
		for(int j = 0; j < 3; ++j){
			Vec3 pos = vrts[3*i+j];
			Vec3 nrm = vnms[3*i+j];
			glNormal3dv(nrm.data);
			glVertex3dv(pos.data);
		}
		glEnd();

		// 法線
		if(normal){
			int lighting = glIsEnabled(GL_LIGHTING);
			glDisable(GL_LIGHTING);
			glBegin(GL_LINES);
			for(int j = 0; j < 3; ++j){
				Vec3 pos = vrts[3*i+j];
				Vec3 nrm = Unit(vnms[3*i+j]);

				glColor3d(1.0, 1.0, 0.3);
				glVertex3dv(pos.data);
				glColor3d(0.5, 0.5, 0.0);
				glVertex3dv((pos+0.05*nrm).data);
			}
			glEnd();
			lighting ? glEnable(GL_LIGHTING) : glDisable(GL_LIGHTING);
		}
	}
}

/*! 
 * 点群の描画
 * @param[in] vrts 頂点列(Vec3)
 * @param[in] size 点の大きさ
 * @param[in] col 点色
 */
void RxGLDraw::DrawPoints(const vector<Vec3> &vrts, const double &size, const Vec3 &col)
{
	glPointSize((GLfloat)size);
	glColor3dv(col.data);
	glBegin(GL_POINTS);
	for(int i = 0; i < (int)vrts.size(); ++i){
		glVertex3dv(vrts[i].data);
	}
	glEnd();
	glPointSize(1.0);
}


/*!
 * 文字列描画
 * @param[in] str 文字列
 * @param[in] w,h 描画領域の大きさ
 */
void RxGLDraw::DrawString(const string &str, int w, int h)
{
	glDisable(GL_LIGHTING);
	glColor3f(0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0, w, 0, h);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	
	float x0 = 5;
	float y0 = h-20;

	glRasterPos2f(x0, y0);

	int i, size = (int)str.size();
	for(i = 0; i < size; ++i){
		char ic = str[i];
		if(ic == '\n'){
			y0 -= 20;
			glRasterPos2f(x0, y0);
		}
		else{
			glutBitmapCharacter(GLUT_BITMAP_9_BY_15, ic);
		}
	}

	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}
//
///*! 
// * 環境マップ用のキューブマップテクスチャの読み込み
// * @param cube_map キューブマップデータ
// * @param base キューブマップ用画像のファイル名のベース部分
// * @param ext キューブマップ用画像のファイルの拡張子
// * @retval true  キューブマップ用画像の読み込み成功
// * @retval false キューブマップ用画像の読み込み失敗
// */
//bool RxGLDraw::LoadCubeMap(rxCubeMapData &cube_map, string base, string ext)
//{
//	// キューブマップ用画像の読み込み(x+,x-,y+,y-,z+,z-)(右,左,上,下,後,前)
//	string fn[6];
//	fn[0] = base+"posx"+ext;
//	fn[1] = base+"negx"+ext;
//	fn[2] = base+"posy"+ext;
//	fn[3] = base+"negy"+ext;
//	fn[4] = base+"posz"+ext;
//	fn[5] = base+"negz"+ext;
//
//	if(!RxGLDraw::LoadCubeMapTexture(fn, cube_map)){
//		return false;
//	}
//
//	return true;
//}
//
//
///*! 
// * 環境マップ用のキューブマップテクスチャの読み込み
// * @param[in] fn[6] テクスチャ画像(6枚)のパス(x+,x-,y+,y-,z+,z-)(右,左,上,下,後,前)
// * @param[out] cube_map rxCubeMapData型
// * @retval true  キューブマップ用画像の読み込み成功
// * @retval false キューブマップ用画像の読み込み失敗
// */
//bool RxGLDraw::LoadCubeMapTexture(const string fn[6], rxCubeMapData &cube_map)
//{
//	GLuint tex_name;
//	glGenTextures(1, &tex_name);
//	glBindTexture(GL_TEXTURE_CUBE_MAP, tex_name);
//
//	// キューブマップテクスチャパラメータの設定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);		// 画像境界の扱いの指定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	// 画像フィルタの指定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	
//	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);
//	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 6);
//
//	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//	
//	GLenum target[6] = { GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
//		                 GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
//						 GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };
//
//	for(int i = 0; i < 6; ++i){
//		int w, h, c;
//		unsigned char* pimg;
//		pimg = LoadPngFile(fn[i], w, h, c);
//		if(pimg == NULL){
//			return false;
//		}
//
//		GLuint format;
//		format = GL_RGBA;
//		if(c == 1){
//			format = GL_LUMINANCE;
//		}
//		else if(c == 3){
//			format = GL_RGB;
//		}
//
//		GLuint iformat;
//		iformat = GL_BGRA;
//		if(c == 1){
//			iformat = GL_LUMINANCE;
//		}
//		else if(c == 3){
//			iformat = GL_BGR;
//		}
//
//		gluBuild2DMipmaps(target[i], format, w, h, iformat, GL_UNSIGNED_BYTE, pimg); 
//
//
//		free(pimg);	
//	}
//
//	glBindTexture(GL_TEXTURE_2D, 0);	
//
///*
//	for(int i = 0; i < 6; ++i){
//		int w, h, c;
//		unsigned char* pimg = NULL;
//		pimg = LoadPngFile(fn[i], w, h, c);
//		if(pimg != NULL){
//			// 画像データをテクスチャとして登録
//			cube_map.tex[i].SetSize(w, h, c);
//		
//			int ic, jc;
//			for(jc = 0; jc < cube_map.tex[i].m_iH; ++jc){
//				for(ic = 0; ic < cube_map.tex[i].m_iW; ++ic){
//					cube_map.tex[i].SetColor(ic, h-jc-1, pimg[c*(jc*w+ic)], pimg[c*(jc*w+ic)+1], pimg[c*(jc*w+ic)+2], pimg[c*(jc*w+ic)+3]);
//				}
//			}
//
//			delete [] pimg;
//		}
//		else{
//			return false;
//		}
//	}
//
//	// キューブマップテクスチャ登録
//	GLuint tex_name;
//	glGenTextures(1, &tex_name);
//	glBindTexture(GL_TEXTURE_CUBE_MAP, tex_name);
//
//	// キューブマップテクスチャパラメータの設定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);		// 画像境界の扱いの指定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);	// 画像フィルタの指定
//	glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	
//	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//	
//	GLenum target[6] = { GL_TEXTURE_CUBE_MAP_POSITIVE_X, GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
//		                 GL_TEXTURE_CUBE_MAP_POSITIVE_Y, GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
//						 GL_TEXTURE_CUBE_MAP_POSITIVE_Z, GL_TEXTURE_CUBE_MAP_NEGATIVE_Z };
//
//	for(int i = 0; i < 6; ++i){
//		GLuint format;
//		format = GL_RGBA;
//		if(cube_map.tex[i].m_iC == 1){
//			format = GL_LUMINANCE;
//		}
//		else if(cube_map.tex[i].m_iC == 3){
//			format = GL_RGB;
//		}
//		
//		glTexImage2D(target[i], 0, GL_RGB8, cube_map.tex[i].m_iW, cube_map.tex[i].m_iH, 0, GL_RGB, GL_UNSIGNED_BYTE, cube_map.tex[i].m_pImage);
//	}
//	
//	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
//*/
//	cube_map.id = tex_name;
//
//	return true;
//}
//
//
///*! 
// * キューブマップテクスチャを内部に貼り付けた立方体の描画
// * @param[in] cube_map キューブマップデータ
// * @param[in] side 立方体の一辺の長さ
// */
//void RxGLDraw::DrawCubeMap(const rxCubeMapData &cube_map, double side)
//{
//	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
//
//	glDisable(GL_DEPTH_TEST);
//	glDisable(GL_LIGHTING);
//	glDisable(GL_CULL_FACE);
//
//	// bind textures
////	glActiveTextureARB(GL_TEXTURE0_ARB);
//	glEnable(GL_TEXTURE_CUBE_MAP);
//    glBindTexture(GL_TEXTURE_CUBE_MAP, cube_map.id);
//
//	// initialize object linear texgen
//	glPushMatrix();
//	GLfloat s_plane[] = { 1.0, 0.0, 0.0, 0.0 };
//	GLfloat t_plane[] = { 0.0, 1.0, 0.0, 0.0 };
//	GLfloat r_plane[] = { 0.0, 0.0, 1.0, 0.0 };
//	glTexGenfv(GL_S, GL_OBJECT_PLANE, s_plane);
//	glTexGenfv(GL_T, GL_OBJECT_PLANE, t_plane);
//	glTexGenfv(GL_R, GL_OBJECT_PLANE, r_plane);
//	glPopMatrix();
//
//	glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
//	glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
//	glTexGeni(GL_R, GL_TEXTURE_GEN_MODE, GL_OBJECT_LINEAR);
//
//	glEnable(GL_TEXTURE_GEN_S);
//	glEnable(GL_TEXTURE_GEN_T);
//	glEnable(GL_TEXTURE_GEN_R);
//
//	glPushMatrix();
//	glutSolidCube(side);
//	glPopMatrix();
//
//	glDisable(GL_TEXTURE_GEN_S);
//	glDisable(GL_TEXTURE_GEN_T);
//	glDisable(GL_TEXTURE_GEN_R);
//
//	glDisable(GL_TEXTURE_CUBE_MAP);
//
//	glEnable(GL_DEPTH_TEST);
//	glEnable(GL_CULL_FACE);
//}

//-----------------------------------------------------------------------------
// HACK:GLSLの実装
//-----------------------------------------------------------------------------
rxGLSL m_glslFresnel;
rxGLSL m_glslFresnelWithFoam;
rxGLSL m_glslPhong;
rxGLSL m_glslToon;

/*!
 * コンストラクタ
 */
void RxGLDraw::InitGLSL(void)
{
	// GLSLのコンパイル
	//printf("glsl compile : Fresnel shader\n");
	m_glslFresnel = CreateGLSL(fresnel_vs, fresnel_fs, "fresnel");
	//m_glslFresnel = CreateGLSLFromFile("shader/fresnel.vs", "shader/fresnel.fs", "fresnel");

	//printf("glsl compile : Phong shader\n");
	m_glslPhong   = CreateGLSL(phong_vs,   phong_fs,   "phong");
	//m_glslPhong   = CreateGLSLFromFile("shader/phong.vs",   "shader/phong.fs",   "phong");

	//printf("glsl compile : Toon shader\n");
	m_glslToon    = CreateGLSL(toon_vs,    toon_fs,    "toon");
	//m_glslToon    = CreateGLSLFromFile("shader/toon.vs",    "shader/toon.fs",    "toon");
}

/*! 
 * 鏡面反射・屈折描画(キューブマップ,GLSL)
 * @param[in] fpDraw 描画関数ポインタ
 * @param[in] eta 屈折率
 * @param[in] bias  Fresnelバイアス
 * @param[in] scale Fresnel倍率
 * @param[in] power Fresnel指数
 * @param[in] eye_pos 視点座標値
 * @param[in] cube_map キューブマップ
 */
void RxGLDraw::DrawPolygonsRefrac(RxFpDraw fpDraw, float eta, float bias, float power, float scale, 
								  const Vec3 &eye_pos, const rxCubeMapData &cube_map)
{
	// MARK:DrawPolygonsRefrac
	GLuint prog = m_glslFresnel.Prog;
	glUseProgram(prog);

	// パラメータ設定
	// バーテックスシェーダ用パラメータ
	glUniform1f(glGetUniformLocation(prog, "etaRatio"), eta);
	glUniform1f(glGetUniformLocation(prog, "fresnelBias"), bias);
	glUniform1f(glGetUniformLocation(prog, "fresnelPower"), power);
	glUniform1f(glGetUniformLocation(prog, "fresnelScale"), scale);
	glUniform3f(glGetUniformLocation(prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);

	// フラグメントシェーダ用パラメータ
	glUniform1i(glGetUniformLocation(prog, "envmap"), 0);

	glUniform1f(glGetUniformLocation(prog, "maxNoise"), 1e10);
	glUniform1f(glGetUniformLocation(prog, "minNoise"), 1e10);
	glUniform3f(glGetUniformLocation(prog, "FoamColor"), 0, 0, 0);

	fpDraw();

	glUseProgram(0);
}

/*! 
 * 鏡面反射・屈折描画(キューブマップ,GLSL)
 * @param[in] fpDraw 描画関数ポインタ
 * @param[in] eta 屈折率
 * @param[in] bias  Fresnelバイアス
 * @param[in] scale Fresnel倍率
 * @param[in] power Fresnel指数
 * @param[in] eye_pos 視点座標値
 * @param[in] cube_map キューブマップ
 */
void RxGLDraw::DrawPolygonsRefracWithFoam(RxFpDraw fpDraw, float eta, float bias, float power, float scale, 
										  const Vec3 &eye_pos, const rxCubeMapData &cube_map, 
										  const Vec3 &foam_color, float noise_max, float noise_min)
{
	// MARK:DrawPolygonsRefrac
	GLuint prog = m_glslFresnel.Prog;
	glUseProgram(prog);

	// パラメータ設定
	// バーテックスシェーダ用パラメータ
	glUniform1f(glGetUniformLocation(prog, "etaRatio"), eta);
	glUniform1f(glGetUniformLocation(prog, "fresnelBias"), bias);
	glUniform1f(glGetUniformLocation(prog, "fresnelPower"), power);
	glUniform1f(glGetUniformLocation(prog, "fresnelScale"), scale);
	glUniform3f(glGetUniformLocation(prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);

	// フラグメントシェーダ用パラメータ
	glUniform1i(glGetUniformLocation(prog, "envmap"), 0);

	glUniform1f(glGetUniformLocation(prog, "maxNoise"), noise_max);
	glUniform1f(glGetUniformLocation(prog, "minNoise"), noise_min);
	glUniform3f(glGetUniformLocation(prog, "FoamColor"), foam_color[0], foam_color[1], foam_color[2]);

	fpDraw();

	glUseProgram(0);
}
/*! 
 * 鏡面反射・屈折ポリゴン描画(キューブマップ,GLSL)
 * @param[in] vrts 頂点座標
 * @param[in] plys 頂点の接続情報
 * @param[in] vnms 頂点法線
 * @param[in] eta 屈折率
 * @param[in] bias  Fresnelバイアス
 * @param[in] scale Fresnel倍率
 * @param[in] power Fresnel指数
 * @param[in] eye_pos 視点座標値
 * @param[in] cube_map キューブマップ
 * @param[in] select ポリゴンピック有無
 */
void RxGLDraw::DrawPolygonsRefrac(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
								  float eta, float bias, float power, float scale, const Vec3 &eye_pos, 
								  const rxCubeMapData &cube_map, bool select)
{
	// MARK:DrawPolygonsRefrac
	GLuint prog = m_glslFresnel.Prog;
	glUseProgram(prog);


	// パラメータ設定
	// バーテックスシェーダ用パラメータ
	glUniform1f(glGetUniformLocation(prog, "etaRatio"), eta);
	glUniform1f(glGetUniformLocation(prog, "fresnelBias"), bias);
	glUniform1f(glGetUniformLocation(prog, "fresnelPower"), power);
	glUniform1f(glGetUniformLocation(prog, "fresnelScale"), scale);
	glUniform3f(glGetUniformLocation(prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);

	// フラグメントシェーダ用パラメータ
	glUniform1i(glGetUniformLocation(prog, "envmap"), 0);

	for(int i = 0; i < (int)plys.size(); ++i){
		if(select) glLoadName(i);

		glBegin(GL_POLYGON);
		//glColor3f(color[0], color[1], color[2]);

		Vec3 norm;
		Vec3 vert;
		for(int j = 0; j < (int)plys[i].size(); ++j){
			vert = vrts[plys[i][j]];
			norm = vnms[plys[i][j]];

			glNormal3d(norm[0], norm[1], norm[2]);
			glVertex3f(vert[0], vert[1], vert[2]);
		}
		glEnd();
	}

	glUseProgram(0);
}




/*! 
 * Phongシェーディングでポリゴン描画(GLSL)
 * @param[in] vrts 頂点座標
 * @param[in] plys 頂点の接続情報
 * @param[in] vnms 頂点法線
 * @param[in] mat 材質
 * @param[in] light 光源
 * @param[in] light_pos 光源位置
 * @param[in] eye_pos 視点座標値
 * @param[in] select ポリゴンピック有無
 */
void RxGLDraw::DrawPolygonsPhong(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
								 rxMaterial *mat, rxMaterial *light, Vec3 light_pos, const Vec3 &eye_pos, bool select)
{
	// MARK:DrawPolygonsPhong
	GLuint prog = m_glslPhong.Prog;
	glUseProgram(prog);


	// パラメータ設定
	// フラグメントシェーダ用パラメータ
	glUniform3f(glGetUniformLocation(prog, "Ke"), mat->GetEmit()[0], mat->GetEmit()[1], mat->GetEmit()[2]);	// emit
	glUniform3f(glGetUniformLocation(prog, "Kd"), mat->GetDiff()[0], mat->GetDiff()[1], mat->GetDiff()[2]);	// diffuse
	glUniform3f(glGetUniformLocation(prog, "Ks"), mat->GetSpec()[0], mat->GetSpec()[1], mat->GetSpec()[2]);	// specular
	glUniform3f(glGetUniformLocation(prog, "Ka"), mat->GetAmbi()[0], mat->GetAmbi()[1], mat->GetAmbi()[2]);	// ambient
	glUniform1f(glGetUniformLocation(prog, "shine"), mat->GetShin());

	glUniform3f(glGetUniformLocation(prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);
	glUniform3f(glGetUniformLocation(prog, "Lpos"), light_pos[0], light_pos[1], light_pos[2]);

	glUniform3f(glGetUniformLocation(prog, "Ld"), light->GetDiff()[0], light->GetDiff()[1], light->GetDiff()[2]);	// ライトの環境光
	glUniform3f(glGetUniformLocation(prog, "Ls"), light->GetSpec()[0], light->GetSpec()[1], light->GetSpec()[2]);	// ライトの拡散反射光
	glUniform3f(glGetUniformLocation(prog, "La"), light->GetAmbi()[0], light->GetAmbi()[1], light->GetAmbi()[2]);	// ライトの鏡面反射光

	for(int i = 0; i < (int)plys.size(); ++i){
		if(select) glLoadName(i);

		glBegin(GL_POLYGON);

		Vec3 norm;
		Vec3 vert;
		for(int j = 0; j < (int)plys[i].size(); ++j){
			vert = vrts[plys[i][j]];
			norm = vnms[plys[i][j]];

			glNormal3d(norm[0], norm[1], norm[2]);
			glVertex3f(vert[0], vert[1], vert[2]);
		}
		glEnd();
	}

	glUseProgram(0);
}


/*! 
 * Toonシェーディングでポリゴン描画(GLSL)
 * @param[in] vrts 頂点座標
 * @param[in] plys 頂点の接続情報
 * @param[in] vnms 頂点法線
 * @param[in] mat 材質
 * @param[in] light_pos 光源位置
 * @param[in] eye_pos 視点座標値
 * @param[in] select ポリゴンピック有無
 */
void RxGLDraw::DrawPolygonsToon(const vector<Vec3> &vrts, const vector< vector<int> > &plys, const vector<Vec3> &vnms, 
								rxMaterial *mat, Vec3 light_pos, const Vec3 &eye_pos, bool select)
{
	GLuint prog = m_glslToon.Prog;
	glUseProgram(prog);

	// パラメータ設定
	// バーテックスシェーダ用パラメータ
	glUniform3f(glGetUniformLocation(prog, "lightPosition"), light_pos[0], light_pos[1], light_pos[2]);
	glUniform3f(glGetUniformLocation(prog, "eyePosition"), eye_pos[0], eye_pos[1], eye_pos[2]);
	glUniform1f(glGetUniformLocation(prog, "shininess"), mat->GetShin());

	// フラグメントシェーダ用パラメータ
	glUniform3f(glGetUniformLocation(prog, "Kd"), mat->GetDiff()[0], mat->GetDiff()[1], mat->GetDiff()[2]);
	glUniform3f(glGetUniformLocation(prog, "Ks"), mat->GetSpec()[0], mat->GetSpec()[1], mat->GetSpec()[2]);

	for(int i = 0; i < (int)plys.size(); ++i){
		if(select) glLoadName(i);

		glBegin(GL_POLYGON);

		Vec3 norm;
		Vec3 vert;
		for(int j = 0; j < (int)plys[i].size(); ++j){
			vert = vrts[plys[i][j]];
			norm = vnms[plys[i][j]];

			glNormal3d(norm[0], norm[1], norm[2]);
			glVertex3f(vert[0], vert[1], vert[2]);
		}
		glEnd();
	}

	glUseProgram(0);
}


