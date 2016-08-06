#include "application.h"

#define WIDTH 640
#define HEIGHT 480

int main(int argc, char* argv[])
{
    application app;
    app.init(argc, argv, WIDTH, HEIGHT);
    app.run();
}
