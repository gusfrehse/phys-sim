EXE := game

.PHONY : clean run zip all windows $(EXE)

$(EXE) : 
	g++ -lSDL2 -lvulkan main.cpp renderer.cpp -o $(EXE)

windows :
	x86_64-w64-mingw32-g++ -static-libgcc -static-libstdc++ -L/usr/x86_64-w64-mingw32/lib -lmingw32 -lSDL2main /usr/x86_64-w64-mingw32/lib/libSDL2.a -mconsole -Wl,--dynamicbase -Wl,--nxcompat -Wl,--high-entropy-va -lm -ldinput8 -ldxguid -ldxerr8 -luser32 -lgdi32 -lwinmm -limm32 -lole32 -loleaut32 -lshell32 -lsetupapi -lversion -luuid -lpthread vulkan-1.dll main.cpp renderer.cpp -o $(EXE)


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
