/*!	\file random.h

	This file is part of the renderBitch distribution.
	Copyright (C) 2002 Wojciech Jarosz

	original code by Makoto Matsumoto and Takuji Nishimura. converted to
	C++ from C code by Wojciech Jarosz.

	This program is free software; you can redistribute it and/or
	modify it under the terms of the GNU General Public License
	as published by the Free Software Foundation; either version 2
	of the License, or (at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

	Contact:
		Wojciech Jarosz - renderBitch@renderedRealities.net
		See http://renderedRealities.net/ for more information.
*/

#ifndef RANDOM_H
#define RANDOM_H

// Period parameters  
#define Nq 624
#define Mq 397
#define MATRIX_A 0x9908b0df		// constant vector a
#define UPPER_MASK 0x80000000	// most significant w-r bits
#define LOWER_MASK 0x7fffffff	// least significant r bits

// Tempering parameters
#define TEMPERING_MASK_B 0x9d2c5680
#define TEMPERING_MASK_C 0xefc60000
#define TEMPERING_SHIFT_U(y)  (y >> 11)
#define TEMPERING_SHIFT_S(y)  (y << 7)
#define TEMPERING_SHIFT_T(y)  (y << 15)
#define TEMPERING_SHIFT_L(y)  (y >> 18)

static unsigned long mt[Nq];	// the array for the state vector
static int mti=Nq+1;			// mti==N+1 means mt[N] is not initialized

/* Initializing the array with a seed */
inline void sgenrand(unsigned long seed)
{
	int i;
	
	for (i = 0; i < Nq; i++)
	{
		mt[i] = seed & 0xffff0000;
		seed = 69069 * seed + 1;
		mt[i] |= (seed & 0xffff0000) >> 16;
		seed = 69069 * seed + 1;
	}
	mti = Nq;
}

/* Initialization by "sgenrand()" is an example. Theoretically, 	 */
/* there are 2^19937-1 possible states as an intial state.			 */
/* This function allows to choose any of 2^19937-1 ones.			 */
/* Essential bits in "seed_array[]" is following 19937 bits:		 */
/*	(seed_array[0]&UPPER_MASK), seed_array[1], ..., seed_array[N-1]. */
/* (seed_array[0]&LOWER_MASK) is discarded. 						 */ 
/* Theoretically,													 */
/*	(seed_array[0]&UPPER_MASK), seed_array[1], ..., seed_array[N-1]  */
/* can take any values except all zeros.							 */
inline void lsgenrand(unsigned long seed_array[])
/* the length of seed_array[] must be at least N */
{
	int i;
	
	for (i = 0; i < Nq; i++) 
		mt[i] = seed_array[i];
	mti=Nq;
}

inline unsigned long genrand()
{
	unsigned long y;
	static unsigned long mag01[2]={0x0, MATRIX_A};
	/* mag01[x] = x * MATRIX_A	for x=0,1 */
	
	if (mti >= Nq)
	{	/* generate N words at one time */

		int kk;
		
		if (mti == Nq+1)		/* if sgenrand() has not been called, */
			sgenrand(4357);		/* a default initial seed is used	*/
		
		for (kk=0;kk<Nq-Mq;kk++)
		{
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+Mq] ^ (y >> 1) ^ mag01[y & 0x1];
		}
		for (;kk<Nq-1;kk++)
		{
			y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
			mt[kk] = mt[kk+(Mq-Nq)] ^ (y >> 1) ^ mag01[y & 0x1];
		}
		y = (mt[Nq-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
		mt[Nq-1] = mt[Mq-1] ^ (y >> 1) ^ mag01[y & 0x1];
		
		mti = 0;
	}
	
	y = mt[mti++];
	y ^= TEMPERING_SHIFT_U(y);
	y ^= TEMPERING_SHIFT_S(y) & TEMPERING_MASK_B;
	y ^= TEMPERING_SHIFT_T(y) & TEMPERING_MASK_C;
	y ^= TEMPERING_SHIFT_L(y);
	
	return y; 
}


// RANDOM_H
#endif
