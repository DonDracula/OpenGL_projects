#ifndef APPLICATION_H
#define APPLICATION_H

#include <vector>
#include <map>

//#include "gl_texture.h"
#include "gl_viewer.h"
#include "timer.h"
#include "grid.h"
//#include "particle.h"

//typedef std::map<std::string, gl_image_texture*>
   // gl_image_texture_map;

class application : public gl_viewer
{
public:
    application();
    ~application();
    void init_event();
    void draw_event();
    void mouse_click_event(
        mouse_button button, mouse_button_state button_state, 
        int x, int y
    );
    void mouse_move_event(
        int x, int y
    );
    void keyboard_event(unsigned char key, int x, int y);

private:
    bool raytrace;
    int rendmode;
    int npart;
    timer t;
   
    bool paused;
    float sim_t;
    
}; 

#endif
