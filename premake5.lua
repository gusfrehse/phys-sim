workspace "phys-sim"
configurations { "Release", "Debug" }

project "project"
architecture "x86_64"
location "build"
kind "ConsoleApp"
files { "*.hpp", "*.cpp" }
links { "vulkan", "SDL2" }
libdirs { os.findlib("SDL2", "vulkan") }

filter "configurations:Debug"
    defines { "DEBUG" }
    symbols "On"
    optimize "Off"

filter "configurations:Release"
    symbols "Off"
    optimize "Speed"

-- have to change this for windows...
-- prebuildcommands {
-- 	"glslc -fshader-stage=vertex -o ../basic_vert.spv ../basic_vert.glsl",
-- 	"glslc -fshader-stage=fragment -o ../basic_frag.spv ../basic_frag.glsl",
-- 	"echo -e '#ifndef SHADERS_H\n#define SHADERS_H' > ../shaders.h",
-- 	"xxd -i basic_vert.spv >> ../shaders.h ; done",
-- 	"xxd -i basic_vert.spv >> ../shaders.h ; done",
-- 	"echo -e '\n#endif' >> ../shaders.h"
-- }

