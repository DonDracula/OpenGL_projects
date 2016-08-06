all of the details about the fluid system is in the "application.cpp".
So if you want to change any parameters, please go to "application.cpp" to modify it.

************************** how to execute *************************************
compile and get the executable object "SPH", please type in:
>make

Now run:
>./SPH
it will show the animation and also save all of the frames as "###.tga" image files.
After we get all of the frame images, we can make video through the method described in the following.

************************** the method how to make the video ***********************************
A directory must be full of the images files (any format) other than .PPM, if your directory contains already PPM image files then skip this step.
Type in:
>mogrify -format ppm * // This will convert your files into PPM files

Then, we need to use the paramFile.
In line 31 you will specify the output, meaning the name of your mpg file
In line 34 you will give in the directory where the ppm files are (path)
In line 48 you will specify the range of the images 'image*.pp' if you want to use all the image files for the creation of the .mpg file

Now run:
>ppmtompeg paramfile

Play with:
>mplayer - loop 0

There are included samples in the parameter file to help guide.

Another easier way to view your images is with the command 'aminate'
>animate image*.ppm

If want to animate every 5 images for example:
>animate image*[05].ppm
