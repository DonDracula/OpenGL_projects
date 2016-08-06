#include "gl_viewer.h"

#include <cassert>
#include <iostream>
#include <cstdio>

using namespace std;

gl_viewer* gl_viewer::singleton = NULL; 

gl_viewer::gl_viewer()
{
    assert(!singleton); // ensure only one instance is created
    singleton = this;
}

gl_viewer::~gl_viewer()
{
    singleton = NULL;
}

void gl_viewer::init(int argc, char *argv[], int width, int height)
{
    // set up glut
    glutInit(&argc, argv);

    // create a 24-bit double-buffered window
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH); 

    glutInitWindowSize(width, height);
    //glutCreateWindow("OGL 3D Viewer");
    glutCreateWindow("Dam break!");

    // called when Glut needs to display
    glutDisplayFunc(&gl_viewer::glut_display_event_wrapper);

    // called when Glut has detected a mouse click
    glutMouseFunc(&gl_viewer::glut_mouse_click_event_wrapper);
    
    // called when Glut has detected mouse motion
    glutMotionFunc(&gl_viewer::glut_mouse_move_event_wrapper);

    // called when Glut has key input
    glutKeyboardFunc(&gl_viewer::glut_keyboard_event_wrapper);
    
    // called when GLUT has nothing to do
    glutIdleFunc(&gl_viewer::glut_display_event_wrapper);

    // called when the window is resized
    glutReshapeFunc(&gl_viewer::glut_reshape_event_wrapper);

    // clear our background to black when glClear is called
    glClearColor(0.9, 0.9, 1.0, 0); 
    
    this->width = width;
    this->height = height;
    first_click = true;

    singleton->init_event();
}

void gl_viewer::run()
{
    // pass execution to Glut. Now Glut is in control of the main loop.
    glutMainLoop();
}

void gl_viewer::glut_display_event_wrapper()
{
    // clear our color buffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    
    // call our gl_viewers draw event and render to back buffer
    singleton->draw_event();

    // swap the back buffer with the front (user always sees front
    // buffer on display)
    glutSwapBuffers();
}

void gl_viewer::glut_mouse_click_event_wrapper(
    int button, int state, int x, int y
    )
{
    if (singleton->first_click)
    {
        singleton->mouse_last_x = x;
        singleton->mouse_last_y = y;

        singleton->first_click = false;
    }

    if (state == gl_viewer::BUTTON_DOWN)
    {
        singleton->delta_x_total = 0;
        singleton->delta_y_total = 0; 
        singleton->num_motion_calls_thus_far = 0;
        if (button == gl_viewer::LEFT_BUTTON) 
        {
            singleton->mode = CAM_ROTATE;
        } 
        else if (button == gl_viewer::RIGHT_BUTTON) 
        {
            singleton->mode = CAM_DOLLY;
        } 
        else if (button == gl_viewer::MIDDLE_BUTTON) 
        {
            singleton->mode = CAM_PAN;
        }
    } else if (state == gl_viewer::BUTTON_UP) {
        singleton->first_click = true;
    }

    // normally (0,0) is at the top left of screen. Since this is
    // somewhat unintuitive, it has been changed to be the bottom
    // left
    singleton->mouse_click_event(
        (mouse_button)button, (mouse_button_state)state, 
        x, singleton->height - y
    );
}

void gl_viewer::glut_mouse_move_event_wrapper(
    int x, int y
    )
{
    // normally (0,0) is at the top left of screen. Since this is
    // somewhat unintuitive, it has been changed to be the bottom
    // left
    singleton->mouse_move_event(
        x, singleton->height - y
    );
    
    int delta_x = x - singleton->mouse_last_x;
    int delta_y = y - singleton->mouse_last_y;

    gl_camera &camera = singleton->camera;


    //
    // the following code is a finite state machine which controls
    // the camera
    //
    if (singleton->mode == CAM_DOLLY)
    {
        camera.set_distance(camera.get_distance() + delta_y / 100.0); 
    }
    else if (singleton->mode == CAM_ROTATE)
    {
        singleton->delta_x_total += abs(delta_x);
        singleton->delta_y_total += abs(delta_y); 
        singleton->num_motion_calls_thus_far += 1;

        if (singleton->num_motion_calls_thus_far > 1)
        {
            if (singleton->delta_x_total > singleton->delta_y_total)
                singleton->mode = CAM_TWIST;
            else
                singleton->mode = CAM_ELEVATE;
        }
    } 
    else if (singleton->mode == CAM_PAN) 
    {
        singleton->delta_x_total += abs(delta_x);
        singleton->delta_y_total += abs(delta_y); 
        singleton->num_motion_calls_thus_far += 1;

        if (singleton->num_motion_calls_thus_far > 1)
        {
            if (singleton->delta_x_total > singleton->delta_y_total)
                singleton->mode = CAM_PAN_HORIZ;
            else
                singleton->mode = CAM_PAN_VERT;
        }
    } 
    else if (singleton->mode == CAM_TWIST)
    {
        camera.set_twist(camera.get_twist() + delta_x / 3.0);
    }
    else if (singleton->mode == CAM_ELEVATE) 
    {
        camera.set_elevation(camera.get_elevation() + delta_y / 3.0);
    }
    else if (singleton->mode == CAM_PAN_HORIZ) 
    {
        vector3 focal = camera.get_focal_point();
        focal.add(camera.get_right() * delta_x/100);
        camera.set_focal_point(focal);
    } 
    else if (singleton->mode == CAM_PAN_VERT) {
        vector3 focal = camera.get_focal_point();
        focal.add(camera.get_up() * delta_y/100);
        camera.set_focal_point(focal);
    }

    singleton->mouse_last_x = x;
    singleton->mouse_last_y = y;
    
    singleton->mouse_move_event(x, y);
}

void gl_viewer::glut_keyboard_event_wrapper(unsigned char key, int x, int y)
{
    singleton->keyboard_event(key, x, y);
}

void gl_viewer::glut_reshape_event_wrapper(int width, int height)
{
    singleton->width = width;
    singleton->height = height;

    // adjust the view frustrum and viewport 
    // to reflect the new window dimensions
    glViewport(0, 0, width, height);
    
    if (height == 0) height = 1;

    // set up perspective projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, width / (float)height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
}

