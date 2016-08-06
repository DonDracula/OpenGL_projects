#include "timer.h"

timer::timer()
{
    reset();
}

float timer::elapsed() const
{
    timeval tnow;
    gettimeofday(&tnow, 0);
    
    long long ts, tus;
    ts = tnow.tv_sec - treset.tv_sec;
    tus = tnow.tv_usec - treset.tv_usec;

    return (float)ts + (float)(tus*1e-6);
}
    
void timer::reset()
{
    gettimeofday(&treset, 0);
}
