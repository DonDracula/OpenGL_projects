#ifndef _SPHTIMER_H
#define _SPHTIMER_H

#include <windows.h>		//获得系统函数

class Timer
{
private:
	int frames;
	int update_time;
	int last_time;
	double FPS;

public:
	Timer();
	void update();
	double get_fps();
};

#endif	//_SPHTIMER_H