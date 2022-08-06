#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include "shaders.h"

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

  vk::Instance instance;
  vk::Device device;
  vk::PhysicalDevice physical_device;
  vk::SurfaceKHR surface;

  vk::SwapchainKHR swapchain;
  vk::Format swapchain_image_format;
  vk::Extent2D swapchain_image_extent;

  std::vector<vk::Image> swapchain_images;
  std::vector<vk::ImageView> swapchain_image_views;

  std::optional<uint32_t> queue_family_index = std::nullopt;

  vk::Queue queue;

  auto create_swapchain() -> auto{
    fprintf(stderr, "Creating swapchain\n");
    auto surface_capabilities =
        physical_device.getSurfaceCapabilitiesKHR(surface);
    auto surface_formats = physical_device.getSurfaceFormatsKHR(surface);
    auto surface_present_modes =
        physical_device.getSurfacePresentModesKHR(surface);

    // choose format, try to find SRGB, if failed choose the first available
    vk::SurfaceFormatKHR chosen_format = surface_formats[0];
    for (auto &format : surface_formats) {
      if (format.format == vk::Format::eB8G8R8A8Srgb &&
          format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
        chosen_format = format;
      }
    }

    swapchain_image_format = chosen_format.format;

    // choose present mode, try to find immediate, if failed choose FIFO
    vk::PresentModeKHR chosen_present_mode = vk::PresentModeKHR::eFifo;
    for (auto &present_mode : surface_present_modes) {
      if (present_mode == vk::PresentModeKHR::eImmediate) {
        chosen_present_mode = vk::PresentModeKHR::eImmediate;
      }
    }

    // find swapchain extent, for some reason it seems to need to check if the
    // current is equal to max_int and only then we actually change it???
    vk::Extent2D chosen_extent = surface_capabilities.currentExtent;
    if (surface_capabilities.currentExtent.width ==
        std::numeric_limits<uint32_t>::max()) {
      SDL_GL_GetDrawableSize(window, (int *)&chosen_extent.width,
                             (int *)&chosen_extent.height);

      chosen_extent.width = std::clamp(
          chosen_extent.width, surface_capabilities.minImageExtent.width,
          surface_capabilities.maxImageExtent.width);

      chosen_extent.height = std::clamp(
          chosen_extent.height, surface_capabilities.minImageExtent.height,
          surface_capabilities.maxImageExtent.height);
    }

    swapchain_image_extent = chosen_extent;

    // find the minimum image count + 1 and smaller than maximum image count
    uint32_t swapchain_image_count = surface_capabilities.minImageCount + 1;
    if (surface_capabilities.maxImageCount > 0 &&
        swapchain_image_count > surface_capabilities.maxImageCount) {
      swapchain_image_count = surface_capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchain_create_info{};
    swapchain_create_info.sType = vk::StructureType::eSwapchainCreateInfoKHR;
    swapchain_create_info.surface = surface;

    swapchain_create_info.minImageCount = swapchain_image_count;
    swapchain_create_info.imageFormat = chosen_format.format;
    swapchain_create_info.imageColorSpace = chosen_format.colorSpace;
    swapchain_create_info.imageExtent = chosen_extent;
    swapchain_create_info.imageArrayLayers = 1;

    swapchain_create_info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapchain_create_info.imageSharingMode =
        vk::SharingMode::eExclusive; // change this for more queue families
    swapchain_create_info.queueFamilyIndexCount = 0;
    swapchain_create_info.pQueueFamilyIndices = nullptr;
    swapchain_create_info.preTransform = surface_capabilities.currentTransform;
    swapchain_create_info.compositeAlpha =
        vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapchain_create_info.presentMode = chosen_present_mode;
    swapchain_create_info.clipped = true;

    // create the swapchain
    swapchain = device.createSwapchainKHR(swapchain_create_info);

    // save the images
    swapchain_images = device.getSwapchainImagesKHR(swapchain);

    // create the image views
    for (auto &image : swapchain_images) {
      vk::ImageViewCreateInfo image_view_create_info{};
      image_view_create_info.setImage(image);
      image_view_create_info.setViewType(vk::ImageViewType::e2D);
      image_view_create_info.setFormat(swapchain_image_format);
      image_view_create_info.setComponents(
          {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
           vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity});
      image_view_create_info.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
      image_view_create_info.subresourceRange.setBaseMipLevel(0);
      image_view_create_info.subresourceRange.setLevelCount(1);
      image_view_create_info.subresourceRange.setBaseArrayLayer(0);
      image_view_create_info.subresourceRange.setLayerCount(1);

      swapchain_image_views.push_back(device.createImageView(image_view_create_info));
    }

    fprintf(stderr, "Created swapchain\n");
  }

  auto create_instance() {
    vk::ApplicationInfo application_info("phys sim");

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
  }

  auto create_physical_device() {

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
  }

  auto create_device_and_queues() {
    CHECK_SDL(SDL_Vulkan_CreateSurface(
                  window, instance, reinterpret_cast<VkSurfaceKHR *>(&surface)),
              != SDL_TRUE);


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

    std::vector<const char *> required_device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::DeviceCreateInfo device_create_info(
        {}, 1, &device_queue_create_info, 0, nullptr,
        required_device_extensions.size(), required_device_extensions.data(),
        &physical_device_features);

    device = physical_device.createDevice(device_create_info);

    queue = device.getQueue(queue_family_index.value(), 0);
  }

  auto init_vulkan() {
    CHECK_SDL(SDL_Init(SDL_INIT_VIDEO), != 0);

    // create window
    window = SDL_CreateWindow(
        "aim trainer", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        window_dimensions.width, window_dimensions.height,
        SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

    CHECK_SDL(window, == 0);

    // instance creation
    create_instance();

    create_physical_device();

    create_device_and_queues();

    // swapchain
    create_swapchain();
  }

  auto cleanup() -> auto{
    for (auto &e : swapchain_image_views) {
      device.destroyImageView(e);
    }

    device.destroy(swapchain);

    device.destroy();

    instance.destroySurfaceKHR(surface);
    instance.destroy();
    SDL_DestroyWindow(window);
  }

  auto init() -> auto{ init_vulkan(); }
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
