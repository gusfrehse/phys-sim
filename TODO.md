# TODOS
Probably won't fix 🫠.

 - [ ] Improve API for adding objects: currently we can only add a compile time defined number of objects because of `NUM_OBJECTS`, which is used in `renderer.cpp`. This makes the main application code really coupled with the renderer code.

 - [ ] Make it easy to switch from orthographic camera to perspective. I think this one is simple (famous last words).

 - [ ] Make an acceleration structure (taking the nomenclature from raytracing, not sure if it's correct usage here) for making collision detection faster. Currently it's n * n which is really bad. I think this will be complicated hehe.

 - [ ] Render everything instanced, as we use the same mesh for everything and I don't plan on changing that.

 - [ ] If objects are spawned near each other, they get 'glued' together, which shouldn't happen.
