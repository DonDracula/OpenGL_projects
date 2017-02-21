/*! @file rx_ply.h
	
	@brief PLY File Input/Output
 
	@author Makoto Fujisawa
	@date  
*/


#ifndef _RX_PLY_H_
#define _RX_PLY_H_


//-----------------------------------------------------------------------------
// Include Files
//-----------------------------------------------------------------------------

#include "rx_mesh.h"


//-----------------------------------------------------------------------------
// Name Space
//-----------------------------------------------------------------------------
using namespace std;

#ifndef RX_DEBUG_OUT
#define RX_DEBUG_OUT 0
#endif


//-----------------------------------------------------------------------------
// rxPLYÉNÉâÉXÇÃêÈåæ - OBJå`éÆÇÃì«Ç›çûÇ›
//-----------------------------------------------------------------------------
class rxPLY
{
public:
	rxPLY();
	~rxPLY();

	bool Read(string file_name, vector<Vec3> &vrts, vector<Vec3> &vnms, vector<rxFace> &plys, rxMTL &mats, bool triangle = true);
	bool Save(string file_name, const vector<Vec3> &vrts, const vector<Vec3> &vnms, const vector<rxFace> &plys, const rxMTL &mats);

private:
};




#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stddef.h>

#define PLY_ASCII      1        /* ascii PLY file */
#define PLY_BINARY_BE  2        /* binary PLY file, big endian */
#define PLY_BINARY_LE  3        /* binary PLY file, little endian */

#define PLY_OKAY    0           /* ply routine worked okay */
#define PLY_ERROR  -1           /* error in ply routine */

	/* scalar data types supported by PLY format */

#define PLY_START_TYPE 0
#define PLY_CHAR       1
#define PLY_SHORT      2
#define PLY_INT        3
#define PLY_UCHAR      4
#define PLY_USHORT     5
#define PLY_UINT       6
#define PLY_FLOAT      7
#define PLY_DOUBLE     8
#define PLY_END_TYPE   9

#define  PLY_SCALAR  0
#define  PLY_LIST    1


typedef struct PlyProperty {    /* description of a property */

	char *name;                           /* property name */
	int external_type;                    /* file's data type */
	int internal_type;                    /* program's data type */
	int offset;                           /* offset bytes of prop in a struct */

	int is_list;                          /* 1 = list, 0 = scalar */
	int count_external;                   /* file's count type */
	int count_internal;                   /* program's count type */
	int count_offset;                     /* offset byte for list count */

} PlyProperty;

typedef struct PlyElement {     /* description of an element */
	char *name;                   /* element name */
	int num;                      /* number of elements in this object */
	int size;                     /* size of element (bytes) or -1 if variable */
	int nprops;                   /* number of properties for this element */
	PlyProperty **props;          /* list of properties in the file */
	char *store_prop;             /* flags: property wanted by user? */
	int other_offset;             /* offset to un-asked-for props, or -1 if none*/
	int other_size;               /* size of other_props structure */
} PlyElement;

typedef struct PlyOtherProp {   /* describes other properties in an element */
	char *name;                   /* element name */
	int size;                     /* size of other_props */
	int nprops;                   /* number of properties in other_props */
	PlyProperty **props;          /* list of properties in other_props */
} PlyOtherProp;

typedef struct OtherData { /* for storing other_props for an other element */
	void *other_props;
} OtherData;

typedef struct OtherElem {     /* data for one "other" element */
	char *elem_name;             /* names of other elements */
	int elem_count;              /* count of instances of each element */
	OtherData **other_data;      /* actual property data for the elements */
	PlyOtherProp *other_props;   /* description of the property data */
} OtherElem;

typedef struct PlyOtherElems {  /* "other" elements, not interpreted by user */
	int num_elems;                /* number of other elements */
	OtherElem *other_list;        /* list of data for other elements */
} PlyOtherElems;

typedef struct PlyFile {        /* description of PLY file */
	FILE *fp;                     /* file pointer */
	int file_type;                /* ascii or binary */
	float version;                /* version number of file */
	int nelems;                   /* number of elements of object */
	PlyElement **elems;           /* list of elements */
	int num_comments;             /* number of comments */
	char **comments;              /* list of comments */
	int num_obj_info;             /* number of items of object information */
	char **obj_info;              /* list of object info items */
	PlyElement *which_elem;       /* which element we're currently writing */
	PlyOtherElems *other_elems;   /* "other" elements from a PLY file */
} PlyFile;

/* memory allocation */
extern char *my_alloc();
#define myalloc(mem_size) my_alloc((mem_size), __LINE__, __FILE__)


/*** delcaration of routines ***/

extern PlyFile *ply_write(FILE *, int, char **, int);
extern PlyFile *ply_open_for_writing(const char *, int, char **, int, float *);
extern void ply_describe_element(PlyFile *, char *, int, int, PlyProperty *);
extern void ply_describe_property(PlyFile *, char *, PlyProperty *);
extern void ply_element_count(PlyFile *, char *, int);
extern void ply_header_complete(PlyFile *);
extern void ply_put_element_setup(PlyFile *, char *);
extern void ply_put_element(PlyFile *, void *);
extern void ply_put_comment(PlyFile *, char *);
extern void ply_put_obj_info(PlyFile *, char *);
extern PlyFile *ply_read(FILE *, int *, char ***);
extern PlyFile *ply_open_for_reading(const char *, int *, char ***, int *, float *);
extern PlyProperty **ply_get_element_description(PlyFile *, char *, int*, int*);
extern void ply_get_element_setup( PlyFile *, char *, int, PlyProperty *);
extern void ply_get_property(PlyFile *, char *, PlyProperty *);
extern PlyOtherProp *ply_get_other_properties(PlyFile *, char *, int);
extern void ply_get_element(PlyFile *, void *);
extern char **ply_get_comments(PlyFile *, int *);
extern char **ply_get_obj_info(PlyFile *, int *);
extern void ply_close(PlyFile *);
extern void ply_get_info(PlyFile *, float *, int *);
extern PlyOtherElems *ply_get_other_element (PlyFile *, char *, int);
extern void ply_describe_other_elements ( PlyFile *, PlyOtherElems *);
extern void ply_put_other_elements (PlyFile *);
extern void ply_free_other_elements (PlyOtherElems *);

extern int equal_strings(char *, char *);


#ifdef __cplusplus
}
#endif


#endif // _RX_PLY_H_
