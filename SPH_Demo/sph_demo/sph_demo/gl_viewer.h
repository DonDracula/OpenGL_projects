#ifndef GL_VIEWER_H
#define GL_VIEWER_H

#include <GL/glut.h>
#include "gl_camera.h"

class gl_viewer
{
public:
    gl_viewer();
    virtual ~gl_viewer();

    enum mouse_button
    {
        LEFT_BUTTON=GLUT_LEFT_BUTTON,
        MIDDLE_BUTTON=GLUT_MIDDLE_BUTTON,
        RIGHT_BUTTON=GLUT_RIGHT_BUTTON
    };

    enum mouse_button_state
    {
        BUTTON_UP = GLUT_UP, // button is being released 
        BUTTON_DOWN=GLUT_DOWN // button is being pressed
    };

    
    // Polymorphic callback events. Implement in inheriting subclass. 
    //
    // draw_event is called by glut when the screen needs to be
    // redrawn. In this implementation this occurs during mouse 
    // and key inputs, as well as when the screen is resized,
    // overlapped by other windows, or minimized and maximized.
    //
    // keyboard_event is called when a key is pressed. The
    // character of the key as well as the x and y location of the
    // mouse cursors current location are passed into this callback.
    // The x and y value are in window coordinates.
    //
    // mouse_event is called when a button on the mouse
    // is first pressed and then once again when it is released. The
    // first parameter is the button id, the second parameter is
    // whether the button is being pushed down or released, and
    // finally x and y are once again the position of the mouse cursor
    // during the event.

    virtual void init_event() {}
    virtual void draw_event() {}
    virtual void keyboard_event(unsigned char key, int x, int y) {}
    virtual void mouse_click_event(
        mouse_button button, mouse_button_state button_state, 
        int x, int y
    ) {}
    virtual void mouse_move_event(
        int x, int y
    ) {}
    
    
    // Call before run. Initializes Glut. Glut is a OpenGL helper
    // library designed to allow easy GL window creation on all
    // of the various operating system.
    
    void init(int argc, char *argv[], int width, int height);

    
    // Begins the main loop. At this point execution is controlled by
    // glut.
    void run();

protected:
    // window width & height
    int width, height; 
    
    bool first_click;
    int mouse_last_x, mouse_last_y; 
    int mode; 
    int delta_x_total, delta_y_total;
    int num_motion_calls_thus_far;
    gl_camera camera;
    
    static void glut_display_event_wrapper();
    static void glut_mouse_click_event_wrapper(
        int button, int state, int x, int y
        );
    static void glut_mouse_move_event_wrapper(
        int x, int y
        );
    static void glut_keyboard_event_wrapper(unsigned char key, int x, int y);
    static void glut_reshape_event_wrapper(int width, int height); 


private:
    enum CAMERA_STATE { CAM_DOLLY, CAM_ROTATE, CAM_PAN, CAM_TWIST, CAM_ELEVATE, CAM_PAN_VERT, CAM_PAN_HORIZ };
    static gl_viewer* singleton;
}; 

#endif
