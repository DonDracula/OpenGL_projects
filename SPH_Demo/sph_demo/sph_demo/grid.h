#ifndef _GRID_H_
#define _GRID_H_

#include <vector>
#include "particle.h"

class Grid
{
	public:
		
		Grid(float* boundingbox, float h, const std::vector<Particle>& particles);
		~Grid();
		
		void Search(unsigned index, std::vector<std::pair<unsigned, float> >& nbs);
	
	private:	
		void Locate(const vector3& pos, unsigned cell_index[3]);
		unsigned IndexToId(unsigned x, unsigned y, unsigned z);
		void GetNeighborCells(unsigned cell_index[3], std::vector<unsigned>& cells);
			
		std::vector<unsigned>* p_indices;
		vector3* p_points;
		unsigned points_num;
		float bbox[6];
		float cell_size;		
		unsigned Xdim, Ydim, Zdim, YZdim;	
	
};

#endif
