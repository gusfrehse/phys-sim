CC := g++
CXX := g++

CFLAGS := -g
LDLIBS := -lSDL2 -lvulkan

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

.PHONY : clean run

clean :
	rm -f $(EXE)
	rm -f $(OBJS)
	rm -f $(SPVS)
	rm -f shaders.h

run : $(EXE)
	./$(EXE)

