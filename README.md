# phys-sim

A physics simulator in vulkan. Base engine based mostly from [Vulkan
Tutorial](https://vulkan-tutorial.com).

## Building
### Linux
Install the dependencies: Premake5, GLM, SDL2 and Vulkan SDK.


Then in the shell, do

    $ premake5 gmake
    $ ./make_shaders.sh
    $ make
    
To run:
    
    $ ./build/bin/Release/project
