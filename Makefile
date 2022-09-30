#CC := g++
#CXX := g++

CC := x86_64-w64-mingw32-g++
CXX := x86_64-w64-mingw32-g++

CXXFLAGS := -g 

SDL_STATIC_LIBS := -L/usr/x86_64-w64-mingw32/lib -lmingw32 -lSDL2main /usr/x86_64-w64-mingw32/lib/libSDL2.a -mwindows -Wl,--dynamicbase -Wl,--nxcompat -Wl,--high-entropy-va -lm -ldinput8 -ldxguid -ldxerr8 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lshell32 -lsetupapi -lversion -luuid

LDLIBS := -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -lssp $(SDL_STATIC_LIBS) ./vulkan-1.dll
LDFLAGS :=

EXE := main
OBJS := main.o

SPVS := basic_frag.spv basic_vert.spv

$(EXE) : $(OBJS)

main.o : shaders.h

%_frag.spv : %_frag.glsl
	glslc -fshader-stage=fragment $< -o $@

%_vert.spv : %_vert.glsl
	glslc -fshader-stage=vertex $< -o $@

shaders.h : $(SPVS)
	rm -f $@
	echo -e '#ifndef SHADERS_H\n#define SHADERS_H' > $@
	for spv in $^ ; do \
		xxd -i $$spv >> $@ ; \
	done
	echo -e '\n#endif' >> $@

.PHONY : clean run zip

clean :
	rm -f $(EXE)
	rm -f $(OBJS)
	rm -f $(SPVS)
	rm -f shaders.h

run : $(EXE)
	./$(EXE)

zip :
	-rm $(EXE).zip
	zip $(EXE).zip $(EXE).exe textures/texture.jpg libvulkan-1.dll vulkan-1.dll SDL2.dll libwinpthread-1.dll
