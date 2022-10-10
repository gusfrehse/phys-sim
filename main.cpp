#include <SDL2/SDL_keycode.h>
#include <cstdio>
#include <unordered_map>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>

#include "renderer.hpp"

#include "object.hpp"

const uint32_t NUM_OBJECTS = 20;

std::unordered_map<SDL_Keycode, bool> keys;

auto handle_event(SDL_Event e) {
  switch (e.type) {
    case SDL_KEYDOWN:
      keys[e.key.keysym.sym] = 1;
      break;

    case SDL_KEYUP:
      keys[e.key.keysym.sym] = 0;
      break;
  }
}

auto main() -> int {
  renderer render;
  render.set_num_objects(NUM_OBJECTS);
  render.init();

  //std::vector<object> objects(NUM_OBJECTS, { .model = glm::mat4(1.0f)});

  // SDL_Delay(1000);
  bool running = true;

  while (running) {
    SDL_Event event;

    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }

      handle_event(event);
    }

    if (keys[SDLK_SPACE]) {
      printf("space pressed\n");
    }

    // update physics
    // TODO: :P

    // update uniforms
    render.draw_frame();
  }

  render.cleanup();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
