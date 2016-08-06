CXX = g++
CXXFLAGS = -O2
CPPFLAGS = -Wall -g
LIBS  = -lGL -lGLU -lglut
OBJS  = gl_viewer.o gl_camera.o main.o application.o timer.o matrix.o particle.o grid.o
TARGET = SPH

a1:	$(OBJS)
	$(CXX) $(CXXLFAGS) $(CPPFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

clean:
	rm -f *.o $(TARGET) *.tga

gl_camera.o: gl_camera.h transform.h
gl_viewer.o: gl_viewer.h gl_camera.h 
main.o: application.h gl_viewer.h vector.h matrix.h 
application.o: application.h matrix.h transform.h
timer.o: timer.h
particle.o: particle.h
grid.o: grid.h
application.h : gl_viewer.h timer.h grid.h
vector.h : matrix.h 
gl_camera.h : vector.h 
grid.h: particle.h
particle.h: vector.h
