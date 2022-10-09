CC := g++
CXX := g++

CXXFLAGS := -g -std=c++20
LDFLAGS := -g -std=c++20
LDLIBS := -lSDL2 -lvulkan

EXE := main
OBJS := main.o renderer.o vertex.o object.o

SPVS := basic_frag.spv basic_vert.spv

$(EXE) : $(OBJS)

main.o : renderer.hpp object.hpp
renderer.o : renderer.hpp vertex.hpp shaders.h
vertex.o : vertex.hpp
object.o : object.hpp

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

.PHONY : clean run

clean :
	rm -f $(EXE)
	rm -f $(OBJS)
	rm -f $(SPVS)
	rm -f shaders.h

run : $(EXE)
	./$(EXE)

