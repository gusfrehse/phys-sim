#include <chrono>
#include <cstdio>
#include <iomanip>

#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_video.h>
#include <glm/ext/matrix_clip_space.hpp>
#include <unordered_map>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>

#include <glm/ext/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "renderer.hpp"

#include "object.hpp"

const uint32_t NUM_OBJECTS = 100;

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

void update_objects(glm::mat4 *data) {
  float gap = 2.0f;

  for (int i = 0; i < NUM_OBJECTS; i++) {
    data[i] = glm::mat4(1.0f);
    data[i] = glm::translate(data[i], (i * gap - NUM_OBJECTS * gap / 2.0f) *
                            glm::vec3(1.5f, 0.0f, 0.0f));
  }

}

void update_camera(uniform_buffer_object *cam, renderer &render) {
  cam->view = glm::lookAt(glm::vec3(0.0f,
                                    1.2f * (float)NUM_OBJECTS,
                                    1.2f * (float)NUM_OBJECTS),
                          glm::vec3(0.0f),
                          glm::vec3(0.0f, 0.0f, 1.0f));

  cam->proj = glm::perspective(glm::radians(45.0f),
                               render.get_width() /
                               (float) render.get_height(),
                               0.001f,
                               10000.0f);

  cam->proj[1][1] *= -1;
}

void show_frame_time(SDL_Window *w,
                     float dt_acc,
                     unsigned long long num_frames) {
  std::stringstream ss;
  auto averaged_dt = dt_acc / (float) num_frames;
  ss <<
    "average frame time: " <<
    std::setprecision(3) <<
    averaged_dt <<
    "ms fps: " << 
    std::setprecision(1) <<
    std::fixed <<
    1000.0f / averaged_dt;

  SDL_SetWindowTitle(w, ss.str().c_str());
}

auto main() -> int {
  renderer render;
  render.set_num_objects(NUM_OBJECTS);
  render.init();

  //std::vector<object> objects(NUM_OBJECTS, { .model = glm::mat4(1.0f)});

  auto prev_t = std::chrono::steady_clock::now();
  auto curr_t = std::chrono::steady_clock::now();

  unsigned long long frame = 0;
  float dt_acc = 0.0f;
  const unsigned long long num_frames_for_average_time = 100;

  // SDL_Delay(1000);
  bool running = true;

  while (running) {
    curr_t = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<float, std::chrono::milliseconds::period>(curr_t - prev_t).count();
    prev_t = curr_t;

    dt_acc += dt;

    if (frame % num_frames_for_average_time == 0) {
      show_frame_time(render.window, dt_acc, num_frames_for_average_time);
      dt_acc = 0;
    }


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
    auto object_uniform = render.map_object_uniform();
    update_objects(object_uniform);
    render.unmap_object_uniform(object_uniform);

    auto camera_uniform = render.map_camera_uniform();
    update_camera(camera_uniform, render);
    render.unmap_camera_uniform(camera_uniform);

    render.draw_frame();

    frame++;
  }

  render.cleanup();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
