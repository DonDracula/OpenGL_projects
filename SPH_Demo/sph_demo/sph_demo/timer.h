#ifndef TIMER_H
#define TIMER_H

//#include <sys/time.h>
#include <windows.h>

// a realtime timer
class timer
{
public:
    timer();

    // returns the amount of wall time that has elapsed
    // since last call to reset
    float elapsed() const;

    // resets the time to 0
    void reset();

private:
    timeval treset;
};

#endif
