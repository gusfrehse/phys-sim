#include <chrono>
#include <cstdio>
#include <glm/geometric.hpp>
#include <iomanip>

#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_video.h>
#include <unordered_map>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>

#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/trigonometric.hpp>

#include "renderer.hpp"
#include "camera.hpp" 
#include "physics.hpp"
#include "profiler.hpp"

const uint32_t NUM_OBJECTS = 3;

std::unordered_map<SDL_Keycode, bool> keys;

auto handle_event(SDL_Event e, orthographic_camera& cam,
                  const renderer& render) {
  PROFILE_FUNC();
  switch (e.type) {
    case SDL_KEYDOWN:
      keys[e.key.keysym.sym] = 1;
      break;

    case SDL_KEYUP:
      keys[e.key.keysym.sym] = 0;
      break;

    case SDL_WINDOWEVENT:
      switch (e.window.event) {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
          cam.set_aspect_ratio(render.get_width() / (float)render.get_height());
        break;
      }
      break;
  }
}

void update_objects(const renderer& render, const physics& phys, uint8_t *data) {
  PROFILE_FUNC();
  size_t alignment = render.get_uniform_alignment();
  for (int i = 0; i < NUM_OBJECTS; i++) {
    glm::mat4* curr = reinterpret_cast<glm::mat4*>(data + i * alignment);

    *curr = glm::mat4(1.0f);
    *curr = glm::translate(*curr, phys.get_position(i));
  }
}

void update_camera(orthographic_camera& cam,
                   uint8_t *uniform,
                   const renderer &render) {
  PROFILE_FUNC();
  camera_uniform *cam_uniform = reinterpret_cast<camera_uniform*>(uniform);

  cam_uniform->view = cam.get_view_matrix();
  cam_uniform->proj = cam.get_projection_matrix();

  //cam_uniform->proj[1][1] *= -1;
}

void show_frame_time(SDL_Window *w,
                     float dt_acc,
                     unsigned long long num_frames) {
  PROFILE_FUNC();
  
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

int main(int argc, char **argv) {
  PROFILE_FUNC();

  renderer render;
  render.set_num_objects(NUM_OBJECTS);
  render.init();

  physics phys({
    object {
      glm::vec3(0.0f, 1.0f, -1.0f),
      glm::vec3(0.0f, 0.0f, 0.0f),
    },
    object {
      glm::vec3(-10.0f, 1.0f, -1.0f),
      glm::vec3(1e-3f, 0.0f, 0.0f),
    },
    object {
       glm::vec3(20.0f, 1.0f, -1.0f),
       glm::vec3(-1e-3f, 0.0f, 0.0f),
    },
  });

  float aspect_ratio = render.get_width() / (float) render.get_height();

  orthographic_camera cam(aspect_ratio);
  cam.set_zoom(4.00);

  auto prev_t = std::chrono::steady_clock::now();
  auto curr_t = std::chrono::steady_clock::now();

  unsigned long long frame = 0;
  float dt_acc = 0.0f;
  const unsigned long long num_frames_for_average_time = 100;

  bool running = true;

  while (running) {
    // dynamic delta time calculation
    curr_t = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<float, std::milli>(curr_t - prev_t).count();
    prev_t = curr_t;

    dt_acc += dt;

    if (frame % num_frames_for_average_time == 0) {
      show_frame_time(render.get_window(), dt_acc, num_frames_for_average_time);
      dt_acc = 0;
    }

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }

      handle_event(event, cam, render);
    }

    // update physics
    phys.time_step(dt);

    phys.check_collisions();

    // update camera
    glm::vec3 pos = cam.get_position();

    float speed = 0.005f;
    glm::vec3 velocity(0.0f);

    float zoom = 1.0f;
    float zoom_speed = 1.005f;

    if (keys[SDLK_w]) {
      velocity += glm::vec3(0.0f, 1.0f,  0.0f);
    }

    if (keys[SDLK_s])  {
      velocity += glm::vec3(0.0f, -1.0f,  0.0f);
    }

    if (keys[SDLK_a])  {
      velocity += glm::vec3(-1.0f, 0.0f,  0.0f);
    }

    if (keys[SDLK_d]) {
      velocity += glm::vec3(1.0f,  0.0f,  0.0f);
    }

    if (keys[SDLK_LSHIFT]) {
      zoom *= zoom_speed;
    }

    if (keys[SDLK_SPACE]) {
      zoom /= zoom_speed;
    }

    cam.set_zoom(cam.get_zoom() * pow(zoom, dt));

    if (velocity != glm::vec3(0.0f)) {
      cam.set_position(pos +
                       dt * speed * cam.get_zoom() * glm::normalize(velocity));
    }


    // update uniforms
    auto object_uniform = render.map_object_uniform();
    update_objects(render, phys, object_uniform);
    render.unmap_object_uniform();

    auto camera_uniform = render.map_camera_uniform();
    update_camera(cam, camera_uniform, render);
    render.unmap_camera_uniform();

    render.draw_frame();

    frame++;
  }

  render.cleanup();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
