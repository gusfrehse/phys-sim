CC := g++
CXX := g++

CFLAGS := -g
LDLIBS := -lSDL2 -lvulkan

EXE := main
OBJS := main.o

$(EXE) : $(OBJS)

main.o : 

.PHONY : clean run

clean :
	rm -f $(EXE)
	rm -f $(OBJS)

run : $(EXE)
	./$(EXE)

