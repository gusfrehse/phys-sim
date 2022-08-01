#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#include <cstdio>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#define CHECK_SDL(x, pred)                                                     \
  if (x pred) {                                                                \
    SDL_Log("SDL_ERROR: " #x ": %s", SDL_GetError());                          \
    exit(1);                                                                   \
  }

struct window_and_vulkan_state {
  SDL_Window *window;
  struct {
    int width = 800;
    int height = 450;
  } window_dimensions;

  vk::DebugUtilsMessengerEXT debug_messenger;
  vk::Instance instance;
  vk::Device device;
  vk::PhysicalDevice physical_device;
  vk::SurfaceKHR surface;

  vk::SwapchainKHR swapchain;
  vk::Format swapchain_image_format;

  std::vector<vk::Image> swapchain_images;
  std::vector<vk::ImageView> swapchain_image_views;

  auto init_vulkan() -> auto{
    CHECK_SDL(SDL_Init(SDL_INIT_VIDEO), != 0);

    auto window = SDL_CreateWindow(
        "aim trainer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        window_dimensions.width, window_dimensions.height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    CHECK_SDL(window, == 0);

    // instance creation
    vk::ApplicationInfo application_info("Aim Trainer");

    // validation layer work
    std::vector<const char *> validation_layers = {
        "VK_LAYER_KHRONOS_validation"};

    vk::InstanceCreateInfo instance_create_info({}, &application_info,
                                                validation_layers.size(),
                                                validation_layers.data());

    instance = vk::createInstance(instance_create_info);

    SDL_Vulkan_CreateSurface(window, instance,
                             reinterpret_cast<VkSurfaceKHR *>(&surface));

    physical_device = nullptr;
    int max_mem = 0;

    printf("analyzing physical devices\n");
    // find device with most memory
    for (auto& p_dev : instance.enumeratePhysicalDevices()) {
      vk::PhysicalDeviceMemoryProperties prop = p_dev.getMemoryProperties();

      int mem = 0;
      for (auto& heap : prop.memoryHeaps) {
        mem += heap.size;
      }

      printf("analyzing physical device with %g giga bytes of total memory\n", (double) mem / 1e9);

      if (mem > max_mem) {
        max_mem = mem;
        physical_device = p_dev;
      }
    }

    // TODO create logical device
  }

  auto create_swapchain() -> auto{}

  auto cleanup() -> auto{
    device.destroySwapchainKHR(swapchain);

    for (auto &e : swapchain_image_views) {
      device.destroyImageView(e);
    }

    device.destroy();

    instance.destroySurfaceKHR(surface);
    instance.destroy();
    SDL_DestroyWindow(window);
  }

  auto init() -> auto{
    init_vulkan();
    create_swapchain();
  }
};

auto main() -> int {
  window_and_vulkan_state state;
  state.init();

  bool running = true;
  SDL_Event event;
  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }
    }
  }

  state.cleanup();
}
