/*!
  @file rx_mc_cpu.cpp
	
  @brief 陰関数表面からのポリゴン生成(MC法)
	
	http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
 
  @author Raghavendra Chandrashekara (basesd on source code
			provided by Paul Bourke and Cory Gene Bloyd)
  @date   2010-03
*/


//-----------------------------------------------------------------------------
// インクルードファイル
//-----------------------------------------------------------------------------
#include <math.h>
#include "rx_mc.h"
#include "rx_mc_tables.h"



//-----------------------------------------------------------------------------
// rxMCMeshCPUの実装
//-----------------------------------------------------------------------------
/*!
 * コンストラクタ
 */
rxMCMeshCPU::rxMCMeshCPU()
{
	// MARK:コンストラクタ
	m_Grid.fMin = Vec3(0.0);
	m_Grid.fWidth = Vec3(0.0);
	m_Grid.iNum[0] = 0;
	m_Grid.iNum[1] = 0;
	m_Grid.iNum[2] = 0;

	m_nTriangles = 0;
	m_nNormals = 0;
	m_nVertices = 0;

	m_ptScalarField = NULL;
	m_fpScalarFunc = 0;
	m_tIsoLevel = 0;
	m_bValidSurface = false;
}

/*!
 * デストラクタ
 */
rxMCMeshCPU::~rxMCMeshCPU()
{
	DeleteSurface();
}


/*!
 * 陰関数から三角形メッシュを生成
 * @param[in] func 陰関数値取得用関数ポインタ
 * @param[in] min_p グリッドの最小座標
 * @param[in] h グリッドの幅
 * @param[in] n[3] グリッド数(x,y,z)
 * @param[in] threshold しきい値(陰関数値がこの値のところをメッシュ化)
 * @param[in] method メッシュ生成方法("mc", "rmt", "bloomenthal")
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 * @retval true  メッシュ生成成功
 * @retval false メッシュ生成失敗
 */
bool rxMCMeshCPU::CreateMesh(RXREAL (*func)(double, double, double), Vec3 min_p, double h, int n[3], RXREAL threshold, 
							 vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face)
{
	if(func == NULL) return false;

	RxScalarField sf;
	for(int i = 0; i < 3; ++i){
		sf.iNum[i] = n[i];
		sf.fWidth[i] = h;
		sf.fMin[i] = min_p[i];
	}

	vector<int> tris;
	GenerateSurfaceV(sf, func, threshold, vrts, nrms, tris);

	if(IsSurfaceValid()){
		int nv = (int)GetNumVertices();
		int nm = (int)GetNumTriangles();
		int nn = (int)GetNumNormals();
		//cout << "mesh was created : " << nv << ", " << nm << ", " << nn << endl;

		// 法線反転
		for(int i = 0; i < nn; ++i){
			nrms[i] *= -1.0;
		}

		face.resize(nm);
		for(int i = 0; i < nm; ++i){
			face[i].vert_idx.resize(3);
			for(int j = 0; j < 3; ++j){
				face[i][j] = tris[3*i+(2-j)];
			}
		}

		return true;
	}

	return false;
}


/*!
 * 陰関数(rxImplicitFunc)から三角形メッシュを生成
 * @param[in] func 陰関数値取得用関数ポインタ
 * @param[in] method メッシュ生成方法("mc", "rmt", "bloomenthal")
 * @param[in] threshold しきい値(陰関数値がこの値のところをメッシュ化)
 * @param[in] n 分割グリッド数
 * @param[out] vrt 頂点座標
 * @param[out] idx 頂点位相
 * @retval true  メッシュ生成成功
 * @retval false メッシュ生成失敗

bool CalMeshImplicitCPU(RXREAL (*func)(double, double, double), Vec3 min_p, double h, int n[3], RXREAL threshold, string method, 
						unsigned int *uvrts, unsigned int *unrms, vector<rxTriangle> &tris)
{
	if(func == NULL) return false;
	//printf("*** create meshes of the zero-isosurface ***");

	RxScalarField sf;
	for(int i = 0; i < 3; ++i){
		sf.iNum[i] = n[i];
		sf.fWidth[i] = h;
		sf.fMin[i] = min_p[i];
	}

	vector<int> tris0;

	static bool init = true;
	if(init){
//		g_uMaxVerts = n[0]*n[1]*3;
//
//#ifndef RX_CUMC_USE_GEOMETRY
//		g_hNrms = new float[g_uMaxVerts*4];
//#else
//		// VBOの確保
//		if(!g_uTriVBO) glGenBuffers(1, &g_uTriVBO);
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_uTriVBO);
//		glBufferData(GL_ELEMENT_ARRAY_BUFFER, g_uMaxVerts*3*3*sizeof(uint), 0, GL_DYNAMIC_DRAW);
//
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//
//#endif
//		// VBOの確保
//		if(!g_uVrtVBO) glGenBuffers(1, &g_uVrtVBO);
//		glBindBuffer(GL_ARRAY_BUFFER, g_uVrtVBO);
//		glBufferData(GL_ARRAY_BUFFER, g_uMaxVerts*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
//		glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//		if(!g_uNrmVBO) glGenBuffers(1, &g_uNrmVBO);
//		glBindBuffer(GL_ARRAY_BUFFER, g_uNrmVBO);
//		glBufferData(GL_ARRAY_BUFFER, g_uMaxVerts*4*sizeof(float), 0, GL_DYNAMIC_DRAW);
//		glBindBuffer(GL_ARRAY_BUFFER, 0);

		init = false;
	}


	rxMCMeshCPU mc;
	mc.GenerateSurfaceV(sf, func, threshold, vrts, nrms, tris0);

	if(mc.IsSurfaceValid()){
		int nv = (int)mc.GetNumVertices();
		int nm = (int)mc.GetNumTriangles();
		int nn = (int)mc.GetNumNormals();
		RXCOUT << "mesh was created : " << nv << ", " << nm << ", " << nn << endl;

		g_uMCVertNum = nv;
		g_uMCFaceNum = nm;

		// 法線反転
		for(int i = 0; i < nn; ++i){
			nrms[i] *= -1.0;
		}

		tris.resize(nm);
		for(int i = 0; i < nm; ++i){
			for(int j = 0; j < 3; ++j){
				tris[i][j] = tris0[3*i+(2-j)];
			}
		}

//		// 頂点アレイに格納
//		glBindBuffer(GL_ARRAY_BUFFER, g_uVrtVBO);
//		float *vrt_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
//
//		for(int i = 0; i < nv; ++i){
//			for(int j = 0; j < 3; ++j){
//				vrt_ptr[3*i+j] = vrts[i][j];
//			}
//		}
//
//		glUnmapBuffer(GL_ARRAY_BUFFER);
//		glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//#ifdef RX_CUMC_USE_GEOMETRY
//		// 法線情報の取得
//		glBindBuffer(GL_ARRAY_BUFFER, g_uNrmVBO);
//		float *nrm_ptr = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
//
//		for(int i = 0; i < nn; ++i){
//			for(int j = 0; j < 3; ++j){
//				nrm_ptr[3*i+j] = nrms[i][j];
//			}
//		}
//
//		glUnmapBuffer(GL_ARRAY_BUFFER);
//		glBindBuffer(GL_ARRAY_BUFFER, 0);
//
//		// 接続情報の取得
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_uTriVBO);
//		uint *tri_ptr = (uint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);
//
//		for(int i = 0; i < nm; ++i){
//			for(int j = 0; j < 3; ++j){
//				tri_ptr[3*i+j] = tris[i][j];
//			}
//		}
//
//		glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
//		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
//
//#else
//
//#endif
		return true;
	}

	return false;
}
 */


/*!
 * 陰関数から三角形メッシュを生成
 * @param[in] field サンプルボリューム
 * @param[in] min_p グリッドの最小座標
 * @param[in] h グリッドの幅
 * @param[in] n[3] グリッド数(x,y,z)
 * @param[in] threshold しきい値(陰関数値がこの値のところをメッシュ化)
 * @param[in] method メッシュ生成方法("mc", "rmt", "bloomenthal")
 * @param[out] vrts 頂点座標
 * @param[out] nrms 頂点法線
 * @param[out] tris メッシュ
 * @retval true  メッシュ生成成功
 * @retval false メッシュ生成失敗
 */
bool rxMCMeshCPU::CreateMeshV(RXREAL *field, Vec3 min_p, double h, int n[3], RXREAL threshold, 
								  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<rxFace> &face)
{
	if(field == NULL) return false;

	RxScalarField sf;
	for(int i = 0; i < 3; ++i){
		sf.iNum[i] = n[i];
		sf.fWidth[i] = h;
		sf.fMin[i] = min_p[i];
	}

	vector<int> tris;
	GenerateSurface(sf, field, threshold, vrts, nrms, tris);

	if(IsSurfaceValid()){
		int nv = (int)GetNumVertices();
		int nm = (int)GetNumTriangles();
		int nn = (int)GetNumNormals();
		//cout << "mesh was created : " << nv << ", " << nm << ", " << nn << endl;

		// 法線反転
		for(int i = 0; i < nn; ++i){
			nrms[i] *= -1.0;
		}

		face.resize(nm);
		for(int i = 0; i < nm; ++i){
			face[i].vert_idx.resize(3);
			for(int j = 0; j < 3; ++j){
				face[i][j] = tris[3*i+(2-j)];
			}
		}

		return true;
	}

	return false;
}


/*!
 * メッシュ生成
 * @param[in] sf 分割グリッド情報
 * @param[in] field サンプルボリューム
 * @param[in] threshold 閾値
 * @param[out] vrts メッシュ頂点
 * @param[out] nrms メッシュ頂点法線
 * @param[out] tris メッシュ幾何情報(頂点接続情報)
 */
void rxMCMeshCPU::GenerateSurface(const RxScalarField sf, RXREAL *field, RXREAL threshold, 
								  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris)
{
	// MARK:GenerateSurface
	if(m_bValidSurface){
		DeleteSurface();
	}

	m_tIsoLevel = threshold;
	m_Grid.iNum[0] = sf.iNum[0];
	m_Grid.iNum[1] = sf.iNum[1];
	m_Grid.iNum[2] = sf.iNum[2];
	m_Grid.fWidth = sf.fWidth;
	m_Grid.fMin = sf.fMin;
	m_ptScalarField = field;

	uint slice0 = (m_Grid.iNum[0] + 1);
	uint slice1 = slice0*(m_Grid.iNum[1] + 1);

	// 等値面の生成
	for(uint z = 0; z < m_Grid.iNum[2]; ++z){
		for(uint y = 0; y < m_Grid.iNum[1]; ++y){
			for(uint x = 0; x < m_Grid.iNum[0]; ++x){
				// グリッド内の頂点配置情報テーブル参照用インデックスの計算
				uint tableIndex = 0;
				if(m_ptScalarField[z*slice1 + y*slice0 + x] < m_tIsoLevel)
					tableIndex |= 1;
				if(m_ptScalarField[z*slice1 + (y+1)*slice0 + x] < m_tIsoLevel)
					tableIndex |= 2;
				if(m_ptScalarField[z*slice1 + (y+1)*slice0 + (x+1)] < m_tIsoLevel)
					tableIndex |= 4;
				if(m_ptScalarField[z*slice1 + y*slice0 + (x+1)] < m_tIsoLevel)
					tableIndex |= 8;
				if(m_ptScalarField[(z+1)*slice1 + y*slice0 + x] < m_tIsoLevel)
					tableIndex |= 16;
				if(m_ptScalarField[(z+1)*slice1 + (y+1)*slice0 + x] < m_tIsoLevel)
					tableIndex |= 32;
				if(m_ptScalarField[(z+1)*slice1 + (y+1)*slice0 + (x+1)] < m_tIsoLevel)
					tableIndex |= 64;
				if(m_ptScalarField[(z+1)*slice1 + y*slice0 + (x+1)] < m_tIsoLevel)
					tableIndex |= 128;

				if(edgeTable[tableIndex] != 0){
					// エッジ上の頂点算出
					if(edgeTable[tableIndex] & 8){
						RxVertexID pt = CalculateIntersection(x, y, z, 3);
						uint id = GetEdgeID(x, y, z, 3);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					if(edgeTable[tableIndex] & 1){
						RxVertexID pt = CalculateIntersection(x, y, z, 0);
						uint id = GetEdgeID(x, y, z, 0);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					if(edgeTable[tableIndex] & 256){
						RxVertexID pt = CalculateIntersection(x, y, z, 8);
						uint id = GetEdgeID(x, y, z, 8);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					
					if(x == m_Grid.iNum[0] - 1){
						if(edgeTable[tableIndex] & 4){
							RxVertexID pt = CalculateIntersection(x, y, z, 2);
							uint id = GetEdgeID(x, y, z, 2);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 2048){
							RxVertexID pt = CalculateIntersection(x, y, z, 11);
							uint id = GetEdgeID(x, y, z, 11);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if(y == m_Grid.iNum[1] - 1){
						if(edgeTable[tableIndex] & 2){
							RxVertexID pt = CalculateIntersection(x, y, z, 1);
							uint id = GetEdgeID(x, y, z, 1);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 512){
							RxVertexID pt = CalculateIntersection(x, y, z, 9);
							uint id = GetEdgeID(x, y, z, 9);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if(z == m_Grid.iNum[2] - 1){
						if(edgeTable[tableIndex] & 16){
							RxVertexID pt = CalculateIntersection(x, y, z, 4);
							uint id = GetEdgeID(x, y, z, 4);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 128){
							RxVertexID pt = CalculateIntersection(x, y, z, 7);
							uint id = GetEdgeID(x, y, z, 7);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if((x==m_Grid.iNum[0] - 1) && (y==m_Grid.iNum[1] - 1))
						if(edgeTable[tableIndex] & 1024){
							RxVertexID pt = CalculateIntersection(x, y, z, 10);
							uint id = GetEdgeID(x, y, z, 10);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					if((x==m_Grid.iNum[0] - 1) && (z==m_Grid.iNum[2] - 1))
						if(edgeTable[tableIndex] & 64){
							RxVertexID pt = CalculateIntersection(x, y, z, 6);
							uint id = GetEdgeID(x, y, z, 6);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					if((y==m_Grid.iNum[1] - 1) && (z==m_Grid.iNum[2] - 1))
						if(edgeTable[tableIndex] & 32){
							RxVertexID pt = CalculateIntersection(x, y, z, 5);
							uint id = GetEdgeID(x, y, z, 5);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					
					// ポリゴン生成
					for(uint i = 0; triTable[tableIndex][i] != 255; i += 3){
						RxTriangle triangle;
						uint pointID0, pointID1, pointID2;
						pointID0 = GetEdgeID(x, y, z, triTable[tableIndex][i]);
						pointID1 = GetEdgeID(x, y, z, triTable[tableIndex][i+1]);
						pointID2 = GetEdgeID(x, y, z, triTable[tableIndex][i+2]);
						triangle.pointID[0] = pointID0;
						triangle.pointID[1] = pointID1;
						triangle.pointID[2] = pointID2;
						m_trivecTriangles.push_back(triangle);
					}
				}
			}
		}
	}


	RenameVerticesAndTriangles(vrts, m_nVertices, tris, m_nTriangles);
	CalculateNormals(vrts, m_nVertices, tris, m_nTriangles, nrms, m_nNormals);

	m_bValidSurface = true;
}

/*!
 * メッシュ生成(サンプルボリューム作成)
 * @param[in] sf 分割グリッド情報
 * @param[in] func 陰関数値取得関数ポインタ
 * @param[in] threshold 閾値
 * @param[out] vrts メッシュ頂点
 * @param[out] nrms メッシュ頂点法線
 * @param[out] tris メッシュ幾何情報(頂点接続情報)
 */
void rxMCMeshCPU::GenerateSurfaceV(const RxScalarField sf, RXREAL (*func)(double, double, double), RXREAL threshold, 
										  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris)
{
	int nx, ny, nz;
	nx = sf.iNum[0]+1;
	ny = sf.iNum[1]+1;
	nz = sf.iNum[2]+1;

	Vec3 minp = sf.fMin;
	Vec3 d = sf.fWidth;

	RXREAL *field = new RXREAL[nx*ny*nz];
	for(int k = 0; k < nz; ++k){
		for(int j = 0; j < ny; ++j){
			for(int i = 0; i < nx; ++i){
				int idx = k*nx*ny+j*nx+i;
				Vec3 pos = minp+Vec3(i, j, k)*d;

				RXREAL val = func(pos[0], pos[1], pos[2]);
				field[idx] = val;
			}
		}
	}

	GenerateSurface(sf, field, threshold, vrts, nrms, tris);

	delete [] field;
}

/*!
 * メッシュ生成(関数から)
 * @param[in] sf 分割グリッド情報
 * @param[in] func 陰関数値取得関数ポインタ
 * @param[in] threshold 閾値
 * @param[out] vrts メッシュ頂点
 * @param[out] nrms メッシュ頂点法線
 * @param[out] tris メッシュ幾何情報(頂点接続情報)
 */
void rxMCMeshCPU::GenerateSurfaceF(const RxScalarField sf, RXREAL (*func)(double, double, double), RXREAL threshold, 
										  vector<Vec3> &vrts, vector<Vec3> &nrms, vector<int> &tris)
{
	// MARK:GenerateSurfaceF
	if(m_bValidSurface){
		DeleteSurface();
	}

	m_tIsoLevel = threshold;
	m_Grid.iNum[0] = sf.iNum[0];
	m_Grid.iNum[1] = sf.iNum[1];
	m_Grid.iNum[2] = sf.iNum[2];
	m_Grid.fWidth = sf.fWidth;
	m_Grid.fMin = sf.fMin;
	m_fpScalarFunc = func;

	uint slice0 = (m_Grid.iNum[0] + 1);
	uint slice1 = slice0*(m_Grid.iNum[1] + 1);

	double dx = m_Grid.fWidth[0];
	double dy = m_Grid.fWidth[1];
	double dz = m_Grid.fWidth[2];
	
	for(uint k = 0; k < m_Grid.iNum[2]; ++k){
		for(uint j = 0; j < m_Grid.iNum[1]; ++j){
			for(uint i = 0; i < m_Grid.iNum[0]; ++i){
				double x, y, z;
				x = m_Grid.fMin[0]+i*m_Grid.fWidth[0];
				y = m_Grid.fMin[1]+j*m_Grid.fWidth[1];
				z = m_Grid.fMin[2]+k*m_Grid.fWidth[2];

				// グリッド内の頂点配置情報テーブル参照用インデックスの計算
				uint tableIndex = 0;
				if(m_fpScalarFunc(x,    y,    z) < m_tIsoLevel) tableIndex |= 1;
				if(m_fpScalarFunc(x,    y+dy, z) < m_tIsoLevel) tableIndex |= 2;
				if(m_fpScalarFunc(x+dx, y+dy, z) < m_tIsoLevel) tableIndex |= 4;
				if(m_fpScalarFunc(x+dx, y,    z) < m_tIsoLevel) tableIndex |= 8;
				if(m_fpScalarFunc(x,    y,    z+dz) < m_tIsoLevel) tableIndex |= 16;
				if(m_fpScalarFunc(x,    y+dy, z+dz) < m_tIsoLevel) tableIndex |= 32;
				if(m_fpScalarFunc(x+dx, y+dy, z+dz) < m_tIsoLevel) tableIndex |= 64;
				if(m_fpScalarFunc(x+dx, y,    z+dz) < m_tIsoLevel) tableIndex |= 128;

				// 頂点情報，幾何情報生成
				if(edgeTable[tableIndex] != 0){
					if(edgeTable[tableIndex] & 8){
						RxVertexID pt = CalculateIntersectionF(i, j, k, 3);
						uint id = GetEdgeID(i, j, k, 3);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					if(edgeTable[tableIndex] & 1){
						RxVertexID pt = CalculateIntersectionF(i, j, k, 0);
						uint id = GetEdgeID(i, j, k, 0);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					if(edgeTable[tableIndex] & 256){
						RxVertexID pt = CalculateIntersectionF(i, j, k, 8);
						uint id = GetEdgeID(i, j, k, 8);
						m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
					}
					
					if(i == m_Grid.iNum[0] - 1){
						if(edgeTable[tableIndex] & 4){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 2);
							uint id = GetEdgeID(i, j, k, 2);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 2048){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 11);
							uint id = GetEdgeID(i, j, k, 11);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if(j == m_Grid.iNum[1] - 1){
						if(edgeTable[tableIndex] & 2){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 1);
							uint id = GetEdgeID(i, j, k, 1);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 512){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 9);
							uint id = GetEdgeID(i, j, k, 9);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if(k == m_Grid.iNum[2] - 1){
						if(edgeTable[tableIndex] & 16){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 4);
							uint id = GetEdgeID(i, j, k, 4);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
						if(edgeTable[tableIndex] & 128){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 7);
							uint id = GetEdgeID(i, j, k, 7);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					}
					if((i == m_Grid.iNum[0]-1) && (j == m_Grid.iNum[1]-1))
						if(edgeTable[tableIndex] & 1024){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 10);
							uint id = GetEdgeID(i, j, k, 10);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					if((i == m_Grid.iNum[0]-1) && (k == m_Grid.iNum[2]-1))
						if(edgeTable[tableIndex] & 64){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 6);
							uint id = GetEdgeID(i, j, k, 6);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					if((j == m_Grid.iNum[1]-1) && (k == m_Grid.iNum[2]-1))
						if(edgeTable[tableIndex] & 32){
							RxVertexID pt = CalculateIntersectionF(i, j, k, 5);
							uint id = GetEdgeID(i, j, k, 5);
							m_i2pt3idVertices.insert(ID2VertexID::value_type(id, pt));
						}
					
					for(uint t = 0; triTable[tableIndex][t] != 255; t += 3){
						RxTriangle triangle;
						uint pointID0, pointID1, pointID2;
						pointID0 = GetEdgeID(i, j, k, triTable[tableIndex][t]);
						pointID1 = GetEdgeID(i, j, k, triTable[tableIndex][t+1]);
						pointID2 = GetEdgeID(i, j, k, triTable[tableIndex][t+2]);
						triangle.pointID[0] = pointID0;
						triangle.pointID[1] = pointID1;
						triangle.pointID[2] = pointID2;
						m_trivecTriangles.push_back(triangle);
					}
				}
			}
		}
	}
	
	RenameVerticesAndTriangles(vrts, m_nVertices, tris, m_nTriangles);
	CalculateNormals(vrts, m_nVertices, tris, m_nTriangles, nrms, m_nNormals);
	m_bValidSurface = true;
}

/*!
 * 等値面メッシュの破棄
 */
void rxMCMeshCPU::DeleteSurface()
{
	m_Grid.fWidth[0] = 0;
	m_Grid.fWidth[1] = 0;
	m_Grid.fWidth[2] = 0;
	m_Grid.iNum[0] = 0;
	m_Grid.iNum[1] = 0;
	m_Grid.iNum[2] = 0;

	m_nTriangles = 0;
	m_nNormals = 0;
	m_nVertices = 0;
	
	//m_vVertices.clear();
	//m_vNormals.clear();
	//m_vTriangles.clear();

	m_ptScalarField = NULL;
	m_tIsoLevel = 0;
	m_bValidSurface = false;
}

/*!
 * メッシュ化に用いたグリッドの大きさ
 * @param[out] fVolLength* グリッドの大きさ
 * @return メッシュ生成されていれば1, そうでなければ-1
 */
int rxMCMeshCPU::GetVolumeLengths(double& fVolLengthX, double& fVolLengthY, double& fVolLengthZ)
{
	if(IsSurfaceValid()){
		fVolLengthX = m_Grid.fWidth[0]*m_Grid.iNum[0];
		fVolLengthY = m_Grid.fWidth[1]*m_Grid.iNum[1];
		fVolLengthZ = m_Grid.fWidth[2]*m_Grid.iNum[2];
		return 1;
	}
	else
		return -1;
}

/*!
 * エッジIDの取得
 * @param[in] nX,nY,nZ グリッド位置
 * @param[in] nEdgeNo エッジ番号
 * @return エッジID
 */
uint rxMCMeshCPU::GetEdgeID(uint nX, uint nY, uint nZ, uint nEdgeNo)
{
	switch(nEdgeNo){
	case 0:
		return GetVertexID(nX, nY, nZ) + 1;
	case 1:
		return GetVertexID(nX, nY + 1, nZ);
	case 2:
		return GetVertexID(nX + 1, nY, nZ) + 1;
	case 3:
		return GetVertexID(nX, nY, nZ);
	case 4:
		return GetVertexID(nX, nY, nZ + 1) + 1;
	case 5:
		return GetVertexID(nX, nY + 1, nZ + 1);
	case 6:
		return GetVertexID(nX + 1, nY, nZ + 1) + 1;
	case 7:
		return GetVertexID(nX, nY, nZ + 1);
	case 8:
		return GetVertexID(nX, nY, nZ) + 2;
	case 9:
		return GetVertexID(nX, nY + 1, nZ) + 2;
	case 10:
		return GetVertexID(nX + 1, nY + 1, nZ) + 2;
	case 11:
		return GetVertexID(nX + 1, nY, nZ) + 2;
	default:
		// Invalid edge no.
		return -1;
	}
}

/*!
 * 頂点IDの取得
 * @param[in] nX,nY,nZ グリッド位置
 * @return 頂点ID
 */
uint rxMCMeshCPU::GetVertexID(uint nX, uint nY, uint nZ)
{
	return 3*(nZ*(m_Grid.iNum[1] + 1)*(m_Grid.iNum[0] + 1) + nY*(m_Grid.iNum[0] + 1) + nX);
}


/*!
 * 補間によりエッジ上の等値点を計算(サンプルボリュームより)
 * @param[in] nX,nY,nZ グリッド位置
 * @param[in] nEdgeNo エッジ番号
 * @return メッシュ頂点情報
 */
RxVertexID rxMCMeshCPU::CalculateIntersection(uint nX, uint nY, uint nZ, uint nEdgeNo)
{
	double x1, y1, z1, x2, y2, z2;
	uint v1x = nX, v1y = nY, v1z = nZ;
	uint v2x = nX, v2y = nY, v2z = nZ;
	
	switch(nEdgeNo){
	case 0:
		v2y += 1;
		break;
	case 1:
		v1y += 1;
		v2x += 1;
		v2y += 1;
		break;
	case 2:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		break;
	case 3:
		v1x += 1;
		break;
	case 4:
		v1z += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 5:
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 6:
		v1x += 1;
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2z += 1;
		break;
	case 7:
		v1x += 1;
		v1z += 1;
		v2z += 1;
		break;
	case 8:
		v2z += 1;
		break;
	case 9:
		v1y += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 10:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 11:
		v1x += 1;
		v2x += 1;
		v2z += 1;
		break;
	}

	x1 = m_Grid.fMin[0]+v1x*m_Grid.fWidth[0];
	y1 = m_Grid.fMin[1]+v1y*m_Grid.fWidth[1];
	z1 = m_Grid.fMin[2]+v1z*m_Grid.fWidth[2];
	x2 = m_Grid.fMin[0]+v2x*m_Grid.fWidth[0];
	y2 = m_Grid.fMin[1]+v2y*m_Grid.fWidth[1];
	z2 = m_Grid.fMin[2]+v2z*m_Grid.fWidth[2];

	uint slice0 = (m_Grid.iNum[0] + 1);
	uint slice1 = slice0*(m_Grid.iNum[1] + 1);
	RXREAL val1 = m_ptScalarField[v1z*slice1 + v1y*slice0 + v1x];
	RXREAL val2 = m_ptScalarField[v2z*slice1 + v2y*slice0 + v2x];
	RxVertexID intersection = Interpolate(x1, y1, z1, x2, y2, z2, val1, val2);
	
	return intersection;
}

/*!
 * 補間によりエッジ上の等値点を計算(関数より)
 * @param[in] nX,nY,nZ グリッド位置
 * @param[in] nEdgeNo エッジ番号
 * @return メッシュ頂点情報
 */
RxVertexID rxMCMeshCPU::CalculateIntersectionF(uint nX, uint nY, uint nZ, uint nEdgeNo)
{
	double x1, y1, z1, x2, y2, z2;
	uint v1x = nX, v1y = nY, v1z = nZ;
	uint v2x = nX, v2y = nY, v2z = nZ;
	
	switch(nEdgeNo){
	case 0:
		v2y += 1;
		break;
	case 1:
		v1y += 1;
		v2x += 1;
		v2y += 1;
		break;
	case 2:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		break;
	case 3:
		v1x += 1;
		break;
	case 4:
		v1z += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 5:
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 6:
		v1x += 1;
		v1y += 1;
		v1z += 1;
		v2x += 1;
		v2z += 1;
		break;
	case 7:
		v1x += 1;
		v1z += 1;
		v2z += 1;
		break;
	case 8:
		v2z += 1;
		break;
	case 9:
		v1y += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 10:
		v1x += 1;
		v1y += 1;
		v2x += 1;
		v2y += 1;
		v2z += 1;
		break;
	case 11:
		v1x += 1;
		v2x += 1;
		v2z += 1;
		break;
	}

	x1 = m_Grid.fMin[0]+v1x*m_Grid.fWidth[0];
	y1 = m_Grid.fMin[1]+v1y*m_Grid.fWidth[1];
	z1 = m_Grid.fMin[2]+v1z*m_Grid.fWidth[2];
	x2 = m_Grid.fMin[0]+v2x*m_Grid.fWidth[0];
	y2 = m_Grid.fMin[1]+v2y*m_Grid.fWidth[1];
	z2 = m_Grid.fMin[2]+v2z*m_Grid.fWidth[2];

	RXREAL val1 = m_fpScalarFunc(x1, y1, z1);
	RXREAL val2 = m_fpScalarFunc(x2, y2, z2);
	RxVertexID intersection = Interpolate(x1, y1, z1, x2, y2, z2, val1, val2);
	
	return intersection;
}

/*!
 * グリッドエッジ両端の陰関数値から線型補間で等値点を計算
 * @param[in] fX1,fY1,fZ1 端点座標1
 * @param[in] fX2,fY2,fZ2 端点座標2
 * @param[in] tVal1 端点座標1でのスカラー値
 * @param[in] tVal2 端点座標2でのスカラー値
 * @return 頂点情報
 */
RxVertexID rxMCMeshCPU::Interpolate(double fX1, double fY1, double fZ1, double fX2, double fY2, double fZ2, RXREAL tVal1, RXREAL tVal2)
{
	RxVertexID interpolation;
	RXREAL mu;

	mu = RXREAL((m_tIsoLevel - tVal1))/(tVal2 - tVal1);
	interpolation.x = fX1 + mu*(fX2 - fX1);
	interpolation.y = fY1 + mu*(fY2 - fY1);
	interpolation.z = fZ1 + mu*(fZ2 - fZ1);

	return interpolation;
}


/*!
 * メッシュ頂点，幾何情報を出力形式で格納
 * @param[out] vrts 頂点座標
 * @param[out] nvrts 頂点数
 * @param[out] tris 三角形ポリゴン幾何情報
 * @param[out] ntris 三角形ポリゴン数
 */
void rxMCMeshCPU::RenameVerticesAndTriangles(vector<Vec3> &vrts, uint &nvrts, vector<int> &tris, uint &ntris)
{
	uint nextID = 0;
	ID2VertexID::iterator mapIterator = m_i2pt3idVertices.begin();
	RxTriangleVector::iterator vecIterator = m_trivecTriangles.begin();

	// Rename vertices.
	while(mapIterator != m_i2pt3idVertices.end()){
		(*mapIterator).second.newID = nextID;
		nextID++;
		mapIterator++;
	}

	// Now rename triangles.
	while(vecIterator != m_trivecTriangles.end()){
		for(uint i = 0; i < 3; i++){
			uint newID = m_i2pt3idVertices[(*vecIterator).pointID[i]].newID;
			(*vecIterator).pointID[i] = newID;
		}
		vecIterator++;
	}

	// Copy all the vertices and triangles into two arrays so that they
	// can be efficiently accessed.
	// Copy vertices.
	mapIterator = m_i2pt3idVertices.begin();
	nvrts = (int)m_i2pt3idVertices.size();
	vrts.resize(nvrts);
	for(uint i = 0; i < nvrts; i++, mapIterator++){
		vrts[i][0] = (*mapIterator).second.x;
		vrts[i][1] = (*mapIterator).second.y;
		vrts[i][2] = (*mapIterator).second.z;
	}
	// Copy vertex indices which make triangles.
	vecIterator = m_trivecTriangles.begin();
	ntris = (int)m_trivecTriangles.size();
	tris.resize(ntris*3);
	for(uint i = 0; i < ntris; i++, vecIterator++){
		tris[3*i+0] = (*vecIterator).pointID[0];
		tris[3*i+1] = (*vecIterator).pointID[1];
		tris[3*i+2] = (*vecIterator).pointID[2];
	}

	m_i2pt3idVertices.clear();
	m_trivecTriangles.clear();
}

/*!
 * 頂点法線計算
 * @param[in] vrts 頂点座標
 * @param[in] nvrts 頂点数
 * @param[in] tris 三角形ポリゴン幾何情報
 * @param[in] ntris 三角形ポリゴン数
 * @param[out] nrms 法線
 * @param[out] nnrms 法線数(=頂点数)
 */
void rxMCMeshCPU::CalculateNormals(const vector<Vec3> &vrts, uint nvrts, const vector<int> &tris, uint ntris, 
									   vector<Vec3> &nrms, uint &nnrms)
{
	nnrms = nvrts;
	nrms.resize(nnrms);
	
	// Set all normals to 0.
	for(uint i = 0; i < nnrms; i++){
		nrms[i][0] = 0;
		nrms[i][1] = 0;
		nrms[i][2] = 0;
	}

	// Calculate normals.
	for(uint i = 0; i < ntris; i++){
		Vec3 vec1, vec2, normal;
		uint id0, id1, id2;
		id0 = tris[3*i+0];
		id1 = tris[3*i+1];
		id2 = tris[3*i+2];

		vec1 = vrts[id1]-vrts[id0];
		vec2 = vrts[id2]-vrts[id0];
		normal = cross(vec1, vec2);

		nrms[id0] += normal;
		nrms[id1] += normal;
		nrms[id2] += normal;
	}

	// Normalize normals.
	for(uint i = 0; i < nnrms; i++){
		normalize(nrms[i]);
	}
}

