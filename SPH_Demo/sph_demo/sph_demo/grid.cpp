#include <cmath>
#include <cassert>

#include "grid.h"

Grid::Grid(float* boundingbox, float h, const std::vector<Particle>& particles)
{
	for(unsigned i = 0; i < 6; i++ )
	{
		if(i%2==0) bbox[i]=boundingbox[i]-h;
		else bbox[i]=boundingbox[i]+h;
	}
	cell_size = h;	

	Xdim = (unsigned)ceilf( (bbox[1]-bbox[0]) / cell_size );
	Ydim = (unsigned)ceilf( (bbox[3]-bbox[2]) / cell_size );
	Zdim = (unsigned)ceilf( (bbox[5]-bbox[4]) / cell_size );
	YZdim = Ydim * Zdim;

	p_indices = new std::vector<unsigned>[Xdim * Ydim * Zdim];
	p_points = new vector3[ particles.size() ];
	unsigned cell_index[3];
	
	points_num = particles.size();
	for( unsigned i=0; i < points_num; i++ )
	{
		p_points[i] = particles[i].pos;
		Locate( p_points[i], cell_index );
		unsigned id = IndexToId( cell_index[0], cell_index[1], cell_index[2] );
		p_indices[id].push_back(i);
	}	
}

Grid::~Grid()
{ 
	delete [] p_indices;
	delete [] p_points;
}

void Grid::Search(unsigned index, std::vector<std::pair<unsigned, float> >& nbs)
{
	nbs.clear();

	unsigned cell_index[3];
	const vector3& point =  p_points[index];
	
	Locate(point, cell_index);
	
	std::vector<unsigned> nb_cells;
	GetNeighborCells(cell_index, nb_cells);
	
	for( unsigned i=0; i < nb_cells.size(); i++ )
	{
		std::vector<unsigned>& nb_cell = p_indices[ nb_cells[i] ];
		for( unsigned j=0; j < nb_cell.size(); j++ )
		{
			unsigned nb_point_id = nb_cell[j];
			if( nb_point_id == index ) 
				continue;
			float dist = ( point - p_points[nb_point_id] ).mag();

			if( dist < cell_size)
			{
				nbs.push_back( std::make_pair<unsigned, float> (nb_point_id, dist) );
			}
		}
	}
}

/////////////////// private functions ///////////////////////
void Grid::Locate(const vector3& pos, unsigned cell_index[3])
{
	cell_index[0] = unsigned( floorf((pos[0]-bbox[0]) / cell_size) );
	cell_index[0] = std::min( cell_index[0], Xdim-1 );
	
	cell_index[1] = unsigned( floorf((pos[1]-bbox[2]) / cell_size) );
	cell_index[1] = std::min( cell_index[1], Ydim-1 );
	
	cell_index[2] = unsigned( floorf((pos[2]-bbox[4]) / cell_size) );
	cell_index[2] = std::min( cell_index[2], Zdim-1 );
}

unsigned Grid::IndexToId( unsigned x, unsigned y, unsigned z )
{
	assert( x < Xdim && y < Ydim && z < Zdim );
	return x * YZdim + y * Zdim + z;
}

void Grid::GetNeighborCells(unsigned cell_index[3], std::vector<unsigned>& cells)
{
	cells.clear();
	
	unsigned x = cell_index[0];
	unsigned y = cell_index[1];
	unsigned z = cell_index[2];
	
	assert( x < Xdim && y < Ydim && z < Zdim );
	
	for( int i = -1; i < 2; i++ )
	{
		int xi = x+i;
		for( int j = -1; j < 2; j++ )
		{
			int yj = y+j;
			for( int k = -1; k < 2; k++ )
			{
				int zk = z+k;
				if( xi >= 0 && xi < (int)Xdim && yj >= 0 && yj < (int)Ydim && zk >= 0 && zk < (int)Zdim )
					cells.push_back( IndexToId(xi, yj, zk) );
			}
		}
	}
}
