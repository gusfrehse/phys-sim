# phys-sim

A physics simulator in vulkan. Base engine based mostly from [Vulkan
Tutorial](https://vulkan-tutorial.com).

## Controls
 - `WASD`: move the camera.
 - `Shift`/`Space`: Zoom out/in.
 - `t`: Time step (once per frame, has the effect of speeding up time)
 - `[`/`]`: Deaccelerate/accelerate time. If you deaccelerate enough, time will start going backwards.

## Building
### Linux
Install the dependencies: Premake5, GLM, SDL2 and Vulkan SDK.


Then in the shell, do

    $ premake5 gmake
    $ ./make_shaders.sh
    $ make
    
To run:
    
    $ ./build/bin/Release/project
