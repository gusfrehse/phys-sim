#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#include <cstdio>
#include <optional>

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

  std::optional<uint32_t> queue_family_index = std::nullopt;

  vk::Queue queue;

  auto init_vulkan() -> auto{
    CHECK_SDL(SDL_Init(SDL_INIT_VIDEO), != 0);

    // create window
    window = SDL_CreateWindow(
        "aim trainer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        window_dimensions.width, window_dimensions.height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    CHECK_SDL(window, == 0);

    // instance creation
    vk::ApplicationInfo application_info("Aim Trainer");

    // validation layer work
    std::vector<const char *> validation_layers = {
        "VK_LAYER_KHRONOS_validation"};

    // SDL extensions
    unsigned int extensions_count = 0;

    CHECK_SDL(
        SDL_Vulkan_GetInstanceExtensions(window, &extensions_count, nullptr),
        != SDL_TRUE);

    std::vector<char *> extensions(extensions_count);

    CHECK_SDL(SDL_Vulkan_GetInstanceExtensions(
                  window, &extensions_count,
                  const_cast<const char **>(extensions.data())),
              != SDL_TRUE);

    vk::InstanceCreateInfo instance_create_info(
        {}, &application_info, validation_layers.size(),
        validation_layers.data(), extensions_count, extensions.data());

    instance = vk::createInstance(instance_create_info);

    CHECK_SDL(SDL_Vulkan_CreateSurface(
                  window, instance, reinterpret_cast<VkSurfaceKHR *>(&surface)),
              != SDL_TRUE);

    physical_device = nullptr;
    int max_mem = 0;

    printf("Analyzing physical devices\n");
    // find device with most memory
    for (auto &p_dev : instance.enumeratePhysicalDevices()) {
      vk::PhysicalDeviceMemoryProperties mem_prop = p_dev.getMemoryProperties();

      printf("Analyzing physical device named '%s'\n",
             p_dev.getProperties().deviceName.data());

      double mem = 0;
      for (int i = 0; i < mem_prop.memoryHeapCount; i++) {
        auto &heap = mem_prop.memoryHeaps[i];
        printf("\tHeap with %g GB of memory\n", (double)heap.size / 1e9);
        mem += heap.size;
      }

      printf("\tTotal memory %g GB\n", mem / 1e9);

      if (mem > max_mem) {
        max_mem = mem;
        physical_device = p_dev;
      }
    }

    printf("Chose device named '%s'\n",
           physical_device.getProperties().deviceName.data());

    // Choose a cool queue family with at least graphics
    auto queue_family_properties = physical_device.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
      if (queue_family_properties[i].queueFlags &
              vk::QueueFlagBits::eGraphics &&
          physical_device.getSurfaceSupportKHR(i, surface)) {

        // get first family that has present and graphics suppoort
        queue_family_index = i;
        break;
      }
    }

    float queue_priority = 1.0f;

    vk::DeviceQueueCreateInfo device_queue_create_info(
        {}, queue_family_index.value(), 1, &queue_priority);

    vk::PhysicalDeviceFeatures physical_device_features{};

    vk::DeviceCreateInfo device_create_info({}, 1, &device_queue_create_info, 0,
                                            nullptr, 0, nullptr,
                                            &physical_device_features);

    device = physical_device.createDevice(device_create_info);

    queue = device.getQueue(queue_family_index.value(), 0);
  }

  auto create_swapchain() -> auto{}

  auto cleanup() -> auto{
    // device.destroySwapchainKHR(swapchain);

    // for (auto &e : swapchain_image_views) {
    //   device.destroyImageView(e);
    // }

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

  SDL_Delay(1000);
  // bool running = true;
  // SDL_Event event;
  // while (running) {
  //   while (SDL_PollEvent(&event)) {
  //     if (event.type == SDL_QUIT) {
  //       running = false;
  //     }
  //   }
  // }

  state.cleanup();
}
