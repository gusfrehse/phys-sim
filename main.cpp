#include <SDL2/SDL.h>

#include "renderer.hpp"

#include "object.hpp"

const uint32_t NUM_OBJECTS = 2; 

auto main() -> int {
  renderer render;
  render.init();
  render.set_num_objects(NUM_OBJECTS);

  std::vector<object> objects(NUM_OBJECTS, { .model = glm::mat4(1.0f)});

  // SDL_Delay(1000);
  bool running = true;
  SDL_Event event;
  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }
    }

    // update physics
    // TODO: :P

    // update uniforms
    //render.record_command_buffer(render.command_buffers, int image_index)
    render.draw_frame();
  }

  render.cleanup();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
