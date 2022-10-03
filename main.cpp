#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <chrono>
#include <random>
#include <vector>

#include "renderer.hpp"

#define TICK_PERIOD (5.0f)

bool running = true;

enum move {
  NORMAL,
  FLIP,
  JUMP,
  NUM_MOVES
};

struct game_state {
  std::chrono::steady_clock::time_point tick_time = std::chrono::steady_clock::now();
  float elapsed_since_tick = 0.0f;
  int level = 1;
  bool new_tick = true;

  std::vector<move> current_moves;

  static long long moves_in_level(int level) {
    return 1 << level;
  }

  void gen_moves() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, NUM_MOVES);

    long long num_moves = moves_in_level(level);

    current_moves.resize(num_moves);

    for (int i = 0; i < num_moves; i++) {
      move generated = static_cast<move>(distrib(gen)); 
      printf("%d ", generated);
      current_moves[i] = generated;
    }

    printf("\n");
  }

  void update_clock() {
    auto now = std::chrono::steady_clock::now();

    if (elapsed_since_tick > TICK_PERIOD) {
      tick_time = std::chrono::steady_clock::now();
      new_tick = true;
    }

    elapsed_since_tick = std::chrono::duration<float, std::chrono::seconds::period>(now - tick_time).count();
  }

} state;


void process_events() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    if (event.type == SDL_QUIT) {
      running = false;
    }
  }
}

int main(int argc, char** argv) {
  auto dt_prev_time = std::chrono::steady_clock::now();

  float time_acc = 0;

  renderer render;
  render.init();

  // SDL_Delay(1000);
  while (running) {
    // delta time calculation
    auto dt_curr_time = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float, std::chrono::seconds::period> (dt_curr_time - dt_prev_time).count();
    dt_prev_time = dt_curr_time;


    process_events();

    state.update_clock();

    if (state.new_tick) {
      printf("tickked \n");
      state.new_tick = false;
      state.level *= 2;
      state.gen_moves();
    }


    time_acc += dt;

    uniform_buffer_object ubo;
    ubo.model = glm::rotate(glm::mat4(1.0f),
                            time_acc * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                           glm::vec3(0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.proj = glm::perspective(glm::radians(45.0f),
                                render.swapchain_image_extent.width /
                                (float) render.swapchain_image_extent.height,
                                0.1f,
                                10.0f);


    ubo.proj[1][1] *= -1;


    render.send_ubo(ubo);
    
    render.draw_frame();
  }

  render.cleanup();

  return 0;
}

/* vim: set sts=2 ts=2 sw=2 et: */
