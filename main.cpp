#include <chrono>
#include <cmath>
#include <cstdio>
#include <glm/geometric.hpp>
#include <iomanip>
#include <random>

#include <SDL2/SDL_keycode.h>
#include <SDL2/SDL_video.h>
#include <unordered_map>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>

#include "glm.hpp"

#include "renderer.hpp"
#include "camera.hpp" 
#include "physics.hpp"
#include "profiler.hpp"

const uint32_t NUM_OBJECTS = 1000;

const float physics_refresh_rate = 1.0f / 60.0f;

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

void apply_collisions(physics& phys,
                      const std::vector<collision_response>& collisions,
                      int num_collisions) {
  PROFILE_FUNC();

  for (int i = 0; i < num_collisions; i++) {
    auto col = collisions[i];

    auto& a = phys.get_object(col.a_id);
    auto& b = phys.get_object(col.b_id);

    auto rel_velocity = a.velocity - b.velocity;

    auto normal = glm::normalize(a.position - b.position);

    auto impulse = glm::dot(rel_velocity, normal) * normal;

    a.velocity += -impulse;
    b.velocity +=  impulse;
  }
}

void update_objects(const renderer& render, const physics& phys, uint8_t *data) {
  PROFILE_FUNC();
  size_t alignment = render.get_uniform_alignment();
  for (int i = 0; i < NUM_OBJECTS; i++) {
    glm::mat4* curr = reinterpret_cast<glm::mat4*>(data + i * alignment);

    *curr = glm::mat4(1.0f);
    *curr = glm::translate(*curr, phys.get_position(i));
    *curr = glm::scale(*curr, glm::vec3(10.0f));
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

void gen_objects(physics& phys) {
  std::random_device rd;

  std::mt19937 e2(rd());

  std::uniform_real_distribution<> dist(50.0f, 100.0f);

  float phi = (1.0 + sqrt(5.0)) / 2.0;

  float r = 6.0 * dist(e2) / 1.0f;

  for (int i = 0; i < NUM_OBJECTS; i++) {
    auto pos = glm::vec3((r + i * 0.4) * cos(phi * i * 0.4), (r + i * 0.4) * sin(phi * i * 0.4), -1.0f);
    auto vel = -pos / dist(e2);
    vel.z = 0.0f;

    std::printf("Adding object at %g, %g, %g with velocity %g, %g, %g\n",
                pos.x, pos.y, pos.z,
                vel.x, vel.y, vel.z);

    phys.add_object(object {
                      pos,
                      vel,
                      1.0f,
                      10.0f
                    });
  }
}

int main(int argc, char **argv) {
  PROFILE_FUNC();

  renderer render;
  render.set_num_objects(NUM_OBJECTS);
  render.init();

  physics phys;

  gen_objects(phys);

  float aspect_ratio = render.get_width() / (float) render.get_height();

  orthographic_camera cam(aspect_ratio);
  cam.set_zoom(400.00f);

  auto prev_t = std::chrono::steady_clock::now();
  auto curr_t = std::chrono::steady_clock::now();

  /// Time left for integration, generated by rendering
  float integration_time_left = 0.0f;

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

    integration_time_left += dt;

    // prevent spiral of death if physics take too much time
    integration_time_left = std::min(integration_time_left, 1.0f / 10.0f);

    // update physics
    while (integration_time_left >= physics_refresh_rate) {
      //std::printf("stepping again: %g\n", integration_time_left);

      phys.time_step(physics_refresh_rate);
      //phys.time_step(dt);

      int num_collisions = phys.check_collisions();
      auto collisions = phys.get_collisions();

      apply_collisions(phys, collisions, num_collisions);

      integration_time_left -= physics_refresh_rate;
    }

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

    if (keys[SDLK_t]) {
      phys.time_step(dt * 0.1);
      int num_collisions = phys.check_collisions();
      auto collisions = phys.get_collisions();

      apply_collisions(phys, collisions, num_collisions);
    }

    if (keys[SDLK_LEFTBRACKET]) {
      auto& time_scale = phys.get_time_scale();
      time_scale -= 0.001f * dt;
    }

    if (keys[SDLK_RIGHTBRACKET]) {
      auto& time_scale = phys.get_time_scale();
      time_scale += 0.001f * dt;
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

    glm::vec3 last_pos = phys.get_position(NUM_OBJECTS - 1);
    //cam.set_position(last_pos + glm::vec3(0.0f, 0.0f, 1.0f));

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
