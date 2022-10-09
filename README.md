# phys-sim

A physics simulator in vulkan. Base engine based mostly from (Vulkan
Tutorial)[https://vulkan-tutorial.com].

## TODO
 - [ ] render multiple objects.
     - [ ] create big uniform buffer.
         - big array, each element with model matrix.
         - changes every draw call (at least once per object).
     - [ ] create small uniform buffer.
         - with view / proj matrix.
         - updates when player / camera updates (every frame).

 - [ ] all the physics stuff heheh.
