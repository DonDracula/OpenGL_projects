#ifndef _RX_PCUBE_
#define _RX_PCUBE_

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *\
 *                                                                           *
 *                       POLYGON-CUBE INTERSECTION                           *
 *                       by Don Hatch & Daniel Green                         *
 *                       January 1994                                        *
 *                                                                           *
 *    CONTENTS:                                                              *
 *        polygon_intersects_cube()                                          *
 *        fast_polygon_intersects_cube()                                     *
 *        trivial_vertex_tests()                                             *
 *        segment_intersects_cube()                                          *
 *        polygon_contains_point_3d()                                        *
 *                                                                           *
 *                                                                           *
 *  This module contains  routines that test points,  segments and polygons  *
 *  for intersections  with the  unit cube defined  as the  axially aligned  *
 *  cube of edge length 1 centered  at the origin.  Polygons may be convex,  *
 *  concave or  self-intersecting.  Also contained is a routine  that tests  *
 *  whether a point  is within a polygon.  All routines  are intended to be  *
 *  fast and robust. Note that the cube and polygons are defined to include  *
 *  their boundaries.                                                        *
 *                                                                           *
 *  The  fast_polygon_intersects_cube  routine  is  meant  to  replace  the  *
 *  triangle-cube  intersection routine  in  Graphics Gems  III by  Douglas  *
 *  Voorhies.   While  that  original  algorithm  is  still sound,   it  is  *
 *  specialized  for triangles  and  the  implementation contained  several  *
 *  bugs and inefficiencies.  The trivial_vertex_tests routine defined here  *
 *  is  almost an  exact copy  of the  trivial point-plane  tests from  the  *
 *  beginning of Voorhies' algorithm but broken out into a separate routine  *
 *  which is called by  fast_polygon_intersects_cube.  The segment-cube and  *
 *  polygon-cube intersection algorithms have been completely rewritten.     *
 *                                                                           *
 *  Notice that trivial_vertex_tests can be  used to quickly test an entire  *
 *  set of vertices  for trivial reject or accept.  This  can be useful for  *
 *  testing  polyhedra  or  entire  polygon  meshes.   When  used  to  test  *
 *  polyhedra, remember  that these  routines only  test points,  edges and  *
 *  surfaces, not volumes.  If no such intersection is reported, the caller  *
 *  should be  aware that the volume  of the polyhedra could  still contain  *
 *  the entire  unit box which  would then need to  be checked for  with an  *
 *  additional point-within-polyhedron test.  The origin would be a natural  *
 *  point to check in such a test.                                           *
 *                                                                           *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "rx_utility.h"

#include <vector>
using namespace std;

/*
 *                   POLYGON INTERSECTS CUBE
 *
 * Tells how the given polygon intersects the cube of edge length 1 centered
 * at the origin.
 * If any vertex or edge of the polygon intersects the cube,
 * a value of 1 will be returned.
 * Otherwise the value returned will be the multiplicity of containment
 * of the cross-section of the cube in the polygon; this may
 * be interpreted as a boolean value in any of the standard
 * ways; e.g. the even-odd rule (it's inside the polygon iff the
 * result is odd) or the winding rule (it's inside the polygon iff
 * the result is nonzero).
 *
 * The "p_nrm" argument is a vector perpendicular to the polygon.  It
 * need not be of unit length.  It is suggested that Newell's method be used
 * to calculate polygon normals (See Graphics Gems III).  Zero-lengthed normals
 * are quite acceptable for degenerate polygons but are not acceptable
 * otherwise.  In particular, beware of zero-length normals which Newell's
 * method can return for certain self-intersecting polygons (for example
 * a bow-tie quadrilateral).
 *
 * The already_know_verts_are_outside_cube flag is unused by this routine
 * but may be useful for alternate implementations.
 *
 * The already_know_edges_are_outside_cube flag is useful when testing polygon
 * meshes with shared edges in order to not test the same edge more than once.
 *
 * Note: usually users of this module would not want to call this routine
 * directly unless they have previously tested the vertices with the trivial
 * vertex test below.  Normally one would call the fast_polygon_intersects_cube
 * utility instead which combines both of these tests.
 */

namespace RXFunc
{
	#ifndef real
	#define real double
	#endif

	#ifndef __cplusplus
	#define inline
	#endif


	#define PCUBE_TEST_AGAINST_PARALLEL_PLANES(posbit, negbit, value, limit)	\
		if (mask & (posbit|negbit)) {					\
			register real temp = value;				\
			if ((mask & posbit) && temp > limit)			\
				outcode |= posbit;				\
			else if ((mask & negbit) && temp < -limit)		\
				outcode |= negbit;				\
		}								\


	/*
	 * Tells which of the six face-planes the given point is outside of.
	 * Only tests faces not represented in "mask".
	 */

	static inline unsigned long face_plane(const real p[3], unsigned long mask)
	{
		register unsigned long outcode = 0L;

		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x001, 0x002, p[0], 0.5)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x004, 0x008, p[1], 0.5)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x010, 0x020, p[2], 0.5)

		return(outcode);
	}



	/*
	 * Tells which of the twelve edge planes the given point is outside of.
	 * Only tests faces not represented in "mask".
	 */

	static inline unsigned long bevel_2d(const real p[3], unsigned long mask)
	{
		register unsigned long outcode = 0L;

		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x001, 0x002, p[0] + p[1], 1.0)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x004, 0x008, p[0] - p[1], 1.0)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x010, 0x020, p[0] + p[2], 1.0)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x040, 0x080, p[0] - p[2], 1.0)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x100, 0x200, p[1] + p[2], 1.0)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x400, 0x800, p[1] - p[2], 1.0)

		return(outcode);
	}



	/*
	 * Tells which of the eight corner planes the given point is outside of.
	 * Only tests faces not represented in "mask".
	 */

	static inline unsigned long bevel_3d(const real p[3], unsigned long mask)
	{
		register unsigned long outcode = 0L;

		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x001, 0x002, p[0] + p[1] + p[2], 1.5)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x004, 0x008, p[0] + p[1] - p[2], 1.5)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x010, 0x020, p[0] - p[1] + p[2], 1.5)
		PCUBE_TEST_AGAINST_PARALLEL_PLANES(0x040, 0x080, p[0] - p[1] - p[2], 1.5)

		return(outcode);
	}



	/*
	 *                   TRIVIAL VERTEX TESTS
	 *
	 * Returns 1 if any of the vertices are inside the cube of edge length 1
	 * centered at the origin (trivial accept), 0 if all vertices are outside
	 * of any testing plane (trivial reject), -1 otherwise (couldn't help).
	 */

	static int trivial_vertex_tests(int nverts, const real verts[][3],
				int already_know_verts_are_outside_cube)
	{
		register unsigned long cum_and;  /* cumulative logical ANDs */
		register int i;

		/*
		 * Compare the vertices with all six face-planes.
		 * If it's known that no vertices are inside the unit cube
		 * we can exit the loop early if we run out of bounding
		 * planes that all vertices might be outside of.  That simply means
		 * that this test failed and we can go on to the next one.
		 * If we must test for vertices within the cube, the loop is slightly
		 * different in that we can trivially accept if we ever do find a
		 * vertex within the cube, but we can't break the loop early if we run
		 * out of planes to reject against because subsequent vertices might
		 * still be within the cube.
		 */
		cum_and = ~0L;  /* Set to all "1" bits */
		if(already_know_verts_are_outside_cube) {
			for(i=0; i<nverts; i++)
				if(0L == (cum_and = face_plane(verts[i], cum_and)))
					break; /* No planes left to trivially reject */
		}
		else {
			for(i=0; i<nverts; i++) {
				/* Note the ~0L mask below to always test all planes */
				unsigned long face_bits = face_plane(verts[i], ~0L);
				if(0L == face_bits)  /* vertex is inside the cube */
					return 1; /* trivial accept */
				cum_and &= face_bits;
			}
		}
		if(cum_and != 0L)  /* All vertices outside some face plane. */
			return 0;  /* Trivial reject */

		/*
		 * Now do the just the trivial reject test against the 12 edge planes.
		 */
		cum_and = ~0L;  /* Set to all "1" bits */
		for(i=0; i<nverts; i++)
			if(0L == (cum_and = bevel_2d(verts[i], cum_and)))
				break; /* No planes left that might trivially reject */
		if(cum_and != 0L)  /* All vertices outside some edge plane. */
			return 0;  /* Trivial reject */

		/*
		 * Now do the trivial reject test against the 8 corner planes.
		 */
		cum_and = ~0L;  /* Set to all "1" bits */
		for(i=0; i<nverts; i++)
			if(0L == (cum_and = bevel_3d(verts[i], cum_and)))
				break; /* No planes left that might trivially reject */
		if(cum_and != 0L)  /* All vertices outside some corner plane. */
			return 0;  /* Trivial reject */

		/*
		 * By now we know that the polygon is not to the outside of any of the
		 * test planes and can't be trivially accepted *or* rejected.
		 */
		return -1;
	}




	#define PCUBE_FOR(i,n) for ((i) = 0; (i) < (n); ++(i))
	#define PCUBE_MAXDIM2(v) ((v)[0] > (v)[1] ? 0 : 1)
	#define PCUBE_MAXDIM3(v) ((v)[0] > (v)[2] ? PCUBE_MAXDIM2(v) : PCUBE_MAXDIM2((v)+1)+1)
	#define PCUBE_ABS(x) ((x)<0 ? -(x) : (x))
	#define PCUBE_SQR(x) ((x)*(x))
	#define PCUBE_SIGN_NONZERO(x) ((x) < 0 ? -1 : 1)
	/* note a and b can be in the reverse order and it still works! */
	#define PCUBE_IN_CLOSED_INTERVAL(a,x,b) (((x)-(a)) * ((x)-(b)) <= 0)
	#define PCUBE_IN_OPEN_INTERVAL(a,x,b) (((x)-(a)) * ((x)-(b)) < 0)


	#define seg_contains_point(a,b,x) (((b)>(x)) - ((a)>(x)))
	/*
	 *                   POLYGON CONTAINS POINT 3D
	 *
	 *  Tells whether a given polygon with nonzero area
	 *  contains a point which is assumed to lie in the plane of the polygon.
	 *  Actually returns the multiplicity of containment.
	 *  This will always be 1 or 0 for non-self-intersecting planar
	 *  polygons with the normal in the standard direction
	 *  (towards the eye when looking at the polygon so that it's CCW).
	 */
	static int polygon_contains_point_3d(int nverts, const real verts[/* nverts */][3],
				const real p_nrm[3],
				real point[3])
	{
		real absp_nrm[3];
		int zaxis, xaxis, yaxis, i, count;
		int xdirection;
		const real *v, *w;

		/*
		 * Determine which axis to ignore
		 * (the one in which the polygon normal is largest)
		 */
		PCUBE_FOR(i,3)
		absp_nrm[i] = PCUBE_ABS(p_nrm[i]);
		zaxis = PCUBE_MAXDIM3(absp_nrm);

		if (p_nrm[zaxis] < 0) {
		xaxis = (zaxis+2)%3;
		yaxis = (zaxis+1)%3;
		} else {
		xaxis = (zaxis+1)%3;
		yaxis = (zaxis+2)%3;
		}

		count = 0;
		PCUBE_FOR(i,nverts) {
		v = verts[i];
		w = verts[(i+1)%nverts];
		if (xdirection = seg_contains_point(v[xaxis], w[xaxis], point[xaxis])) {
			if (seg_contains_point(v[yaxis], w[yaxis], point[yaxis])) {
			if (xdirection * (point[xaxis]-v[xaxis])*(w[yaxis]-v[yaxis]) <= 
				xdirection * (point[yaxis]-v[yaxis])*(w[xaxis]-v[xaxis]))
				count += xdirection;
			} else {
			if (v[yaxis] <= point[yaxis])
				count += xdirection;
			}
		}
		}
		return count;
	}




	/*
	 *  A segment intersects the unit cube centered at the origin
	 *  iff the origin is contained in the solid obtained
	 *  by dragging a unit cube from one segment endpoint to the other.
	 *  (This solid is a warped rhombic dodecahedron.)
	 *  This amounts to 12 sidedness tests.
	 *  Also, this test works even if one or both of the segment endpoints is
	 *  inside the cube.
	 */
	static int segment_intersects_cube(const real v0[3], const real v1[3])
	{
		int i, iplus1, iplus2, edgevec_signs[3];
		real edgevec[3];

		edgevec[0] = v1[0]-v0[0];
		edgevec[1] = v1[1]-v0[1];
		edgevec[2] = v1[2]-v0[2];

		PCUBE_FOR(i,3)
		edgevec_signs[i] = PCUBE_SIGN_NONZERO(edgevec[i]);

		/*
		 * Test the three cube faces on the v1-ward side of the cube--
		 * if v0 is outside any of their planes then there is no intersection.
		 * Also test the three cube faces on the v0-ward side of the cube--
		 * if v1 is outside any of their planes then there is no intersection.
		 */

		PCUBE_FOR(i,3) {
		if (v0[i] * edgevec_signs[i] >  .5) return 0;
		if (v1[i] * edgevec_signs[i] < -.5) return 0;
		}

		/*
		 * Okay, that's the six easy faces of the rhombic dodecahedron
		 * out of the way.  Six more to go.
		 * The remaining six planes bound an infinite hexagonal prism
		 * joining the petrie polygons (skew hexagons) of the two cubes
		 * centered at the endpoints.
		 */

		PCUBE_FOR(i,3) {
		real rhomb_normal_dot_v0, rhomb_normal_dot_cubedge;

		iplus1 = (i+1)%3;
		iplus2 = (i+2)%3;

	#ifdef THE_EASY_TO_UNDERSTAND_WAY

		{
		real rhomb_normal[3], cubedge_midpoint[3];

		/*
		 * rhomb_normal = VXV3(edgevec, unit vector in direction i),
		 * being cavalier about which direction it's facing
		 */
		rhomb_normal[i] = 0;
		rhomb_normal[iplus1] = edgevec[iplus2];
		rhomb_normal[iplus2] = -edgevec[iplus1];

		/*
		 *  We now are describing a plane parallel to
		 *  both segment and the cube edge in question.
		 *  if |DOT3(rhomb_normal, an arbitrary point on the segment)| >
		 *  |DOT3(rhomb_normal, an arbitrary point on the cube edge in question|
		 *  then the origin is outside this pair of opposite faces.
		 *  (This is equivalent to saying that the line
		 *  containing the segment is "outside" (i.e. further away from the
		 *  origin than) the line containing the cube edge.
		 */

		cubedge_midpoint[i] = 0;
		cubedge_midpoint[iplus1] = edgevec_signs[iplus1]*.5;
		cubedge_midpoint[iplus2] = -edgevec_signs[iplus2]*.5;

		rhomb_normal_dot_v0 = DOT3(rhomb_normal, v0);
		rhomb_normal_dot_cubedge = DOT3(rhomb_normal,cubedge_midpoint);
		}

	#else /* the efficient way */

		rhomb_normal_dot_v0 = edgevec[iplus2] * v0[iplus1]
					- edgevec[iplus1] * v0[iplus2];

		rhomb_normal_dot_cubedge = .5 *
					(edgevec[iplus2] * edgevec_signs[iplus1] +
					 edgevec[iplus1] * edgevec_signs[iplus2]);

	#endif /* the efficient way */

		if (PCUBE_SQR(rhomb_normal_dot_v0) > PCUBE_SQR(rhomb_normal_dot_cubedge))
			return 0;	/* origin is outside this pair of opposite planes */
		}
		return 1;
	}





	/*
	 *                   POLYGON INTERSECTS CUBE
	 *
	 * Tells whether a given polygon intersects the cube of edge length 1
	 * centered at the origin.
	 * Always returns 1 if a polygon edge intersects the cube;
	 * returns the multiplicity of containment otherwise.
	 * (See explanation of polygon_contains_point_3d() above).
	 */
	static int polygon_intersects_cube(int nverts, const real verts[/* nverts */][3],
				const real p_nrm[3],
				int already_know_vertices_are_outside_cube, /*unused*/
				int already_know_edges_are_outside_cube)
	{
		int i, best_diag[3];
		real p[3], t;

		/*
		 * If any edge intersects the cube, return 1.
		 */
		if (!already_know_edges_are_outside_cube)
		PCUBE_FOR(i,nverts)
			if (segment_intersects_cube(verts[i], verts[(i+1)%nverts]))
			return 1;

		/*
		 * If the polygon normal is zero and none of its edges intersect the
		 * cube, then it doesn't intersect the cube
		 */
		if(p_nrm[0] == 0 && p_nrm[1] == 0 && p_nrm[2] == 0)
			return 0;

		/*
		 * Now that we know that none of the polygon's edges intersects the cube,
		 * deciding whether the polygon intersects the cube amounts
		 * to testing whether any of the four cube diagonals intersects
		 * the interior of the polygon.
		 *
		 * Notice that we only need to consider the cube diagonal that comes
		 * closest to being perpendicular to the plane of the polygon.
		 * If the polygon intersects any of the cube diagonals,
		 * it will intersect that one.
		 */

		PCUBE_FOR(i,3)
		best_diag[i] = PCUBE_SIGN_NONZERO(p_nrm[i]);

		/*
		 * Okay, we have the diagonal of interest.
		 * The plane containing the polygon is the set of all points p satisfying
		 *      DOT3(p_nrm, p) == DOT3(p_nrm, verts[0])
		 * So find the point p on the cube diagonal of interest
		 * that satisfies this equation.
		 * The line containing the cube diagonal is described parametrically by
		 *      t * best_diag
		 * so plug this into the previous equation and solve for t.
		 *      DOT3(p_nrm, t * best_diag) == DOT3(p_nrm, verts[0])
		 * i.e.
		 *      t = DOT3(p_nrm, verts[0]) / DOT3(p_nrm, best_diag)
		 *
		 * (Note that the denominator is guaranteed to be nonzero, since
		 * p_nrm is nonzero and best_diag was chosen to have the largest
		 * magnitude dot-product with p_nrm)
		 */
		t = (p_nrm[0]*verts[0][0]+p_nrm[1]*verts[0][1]+p_nrm[2]*verts[0][2])
		  / (p_nrm[0]*best_diag[0]+p_nrm[1]*best_diag[1]+p_nrm[2]*best_diag[2]);

		if (!PCUBE_IN_CLOSED_INTERVAL(-.5, t, .5))
		return 0;  /* intersection point is not in cube */

		/* p = t * best_diag */
		p[0] = t*best_diag[0];
		p[1] = t*best_diag[1];
		p[2] = t*best_diag[2];

		return polygon_contains_point_3d(nverts, verts, p_nrm, p);
	}

	
	/*
	 *                   FAST POLYGON INTERSECTS CUBE
	 *
	 * This is a version of the same polygon-cube intersection that first calls
	 * trivial_vertex_tests() to hopefully skip the more expensive definitive test.
	 * It simply calls polygon_intersects_cube() when that fails.
	 * Note that after the trivial tests we at least know that all vertices are
	 * outside the cube and can therefore pass a true flag to
	 * polygon_intersects_cube().
	 */
	static int fast_polygon_intersects_cube(int nverts, const real verts[][3],
							const real p_nrm[3],
				int already_know_verts_are_outside_cube,
				int already_know_edges_are_outside_cube)
	{
		int quick_test = trivial_vertex_tests(nverts, verts,
					already_know_verts_are_outside_cube);
		if(-1 == quick_test)
			return polygon_intersects_cube(nverts, verts, p_nrm, 1,
					already_know_edges_are_outside_cube);
		else
			return quick_test;
	}

	static int polygon_contains_point_3d(int nverts, const vector<Vec3> &verts, const Vec3 &p_nrm, real point[3])
	{
		int zaxis, xaxis, yaxis, i, count;
		int xdirection;
		const double *v, *w;

		Vec3 absp_nrm = RXFunc::Fabs(p_nrm);
		zaxis = RXFunc::Max3(absp_nrm);

		if(p_nrm[zaxis] < 0){
			xaxis = (zaxis+2)%3;
			yaxis = (zaxis+1)%3;
		}
		else{
			xaxis = (zaxis+1)%3;
			yaxis = (zaxis+2)%3;
		}

		count = 0;
		for(i = 0; i < nverts; ++i) {
			v = verts[i];
			w = verts[(i+1)%nverts];
			if(xdirection = seg_contains_point(v[xaxis], w[xaxis], point[xaxis])){
				if(seg_contains_point(v[yaxis], w[yaxis], point[yaxis])){
					if(xdirection * (point[xaxis]-v[xaxis])*(w[yaxis]-v[yaxis]) <= xdirection * (point[yaxis]-v[yaxis])*(w[xaxis]-v[xaxis]))
						count += xdirection;
				}
				else{
					if(v[yaxis] <= point[yaxis])
						count += xdirection;
				}
			}
		}
		return count;
	}

	static int polygon_intersects_cube(const vector<Vec3> &verts, const Vec3 &p_nrm, int already_know_edges_are_outside_cube)
	{
		int nverts = verts.size();
		int i, best_diag[3];
		double p[3], t;

		if(!already_know_edges_are_outside_cube){
			for(i = 0; i < nverts; ++i){
				if(segment_intersects_cube(verts[i], verts[(i+1)%nverts]))
					return 1;
			}
		}

		if(p_nrm[0] == 0 && p_nrm[1] == 0 && p_nrm[2] == 0)
			return 0;

		for(i = 0; i < 3; ++i)
			best_diag[i] = PCUBE_SIGN_NONZERO(p_nrm[i]);

		t = (p_nrm[0]*verts[0][0]+p_nrm[1]*verts[0][1]+p_nrm[2]*verts[0][2])/(p_nrm[0]*best_diag[0]+p_nrm[1]*best_diag[1]+p_nrm[2]*best_diag[2]);

		if(!PCUBE_IN_CLOSED_INTERVAL(-.5, t, .5))
			return 0;

		p[0] = t*best_diag[0];
		p[1] = t*best_diag[1];
		p[2] = t*best_diag[2];

		return polygon_contains_point_3d(nverts, verts, p_nrm, p);
	}

}

#endif

