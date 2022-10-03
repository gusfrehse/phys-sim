#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <optional>

#define CHECK_SDL(x, pred)                                                     \
  do {                                                                         \
    if (x pred) {                                                              \
      fprintf(stderr, "%s %s:%d, sdl check failed:\n",                         \
              __PRETTY_FUNCTION__,                                             \
              __FILE__,                                                        \
              __LINE__);                                                       \
      SDL_Log("SDL_ERROR: " #x ": %s", SDL_GetError());                        \
      exit(1);                                                                 \
    }                                                                          \
  } while(0);

const int MAX_FRAMES_IN_FLIGHT = 2;

struct uniform_buffer_object {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

struct vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 tex_coord;

  static std::array<vk::VertexInputAttributeDescription, 3>
  get_attribute_descriptions() {
    std::array<vk::VertexInputAttributeDescription, 3> attrib_description{};

    attrib_description[0].binding = 0;
    attrib_description[0].location = 0;
    attrib_description[0].format = vk::Format::eR32G32B32Sfloat;
    attrib_description[0].offset = offsetof(vertex, pos);

    attrib_description[1].binding = 0;
    attrib_description[1].location = 1;
    attrib_description[1].format = vk::Format::eR32G32B32Sfloat;
    attrib_description[1].offset = offsetof(vertex, color);

    attrib_description[2].binding = 0;
    attrib_description[2].location = 2;
    attrib_description[2].format = vk::Format::eR32G32Sfloat;
    attrib_description[2].offset = offsetof(vertex, tex_coord);

    return attrib_description;
  }


  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription binding_description{};
    binding_description.binding = 0;
    binding_description.stride = sizeof(vertex);
    binding_description.inputRate = vk::VertexInputRate::eVertex;

    return binding_description;
  }
};

const std::vector<vertex> vertices = {
  {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
  {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
  {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
  {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

  {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
  {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
  {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
  {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}}  
};

const std::vector<uint16_t> indices = {
  0, 1, 2, 2, 3, 0,
  4, 5, 6, 6, 7, 4
};

struct renderer {
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

    std::vector<vk::Framebuffer> swapchain_framebuffers;

    vk::Image depth_image;
    vk::DeviceMemory depth_image_memory;
    vk::ImageView depth_image_view;

    std::optional<uint32_t> queue_family_index = std::nullopt;

    vk::Queue queue;

    vk::RenderPass clear_renderpass;
    vk::RenderPass renderpass;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::PipelineLayout pipeline_layout;
    vk::Pipeline graphics_pipeline;

    vk::CommandPool command_pool;
    std::vector<vk::CommandBuffer> command_buffers;

    vk::CommandPool alloc_command_pool;

    std::vector<vk::Semaphore> image_available_semaphores;
    std::vector<vk::Semaphore> render_finished_semaphores;
    std::vector<vk::Fence> in_flight_fences;

    vk::Buffer vertex_buffer;
    vk::DeviceMemory vertex_buffer_memory;

    vk::Buffer index_buffer;
    vk::DeviceMemory index_buffer_memory;

    std::vector<vk::Buffer> uniform_buffers;
    std::vector<vk::DeviceMemory> uniform_buffers_memory;

    vk::DescriptorPool descriptor_pool;

    std::vector<vk::DescriptorSet> descriptor_sets;

    vk::Image texture_image;
    vk::DeviceMemory texture_image_memory;
    vk::ImageView texture_image_view;
    vk::Sampler texture_sampler;

    uniform_buffer_object next_ubo;

    uint32_t current_frame = 0;

    void record_command_buffer(vk::CommandBuffer command_buffer,
                               int image_index);

    void create_sync_objects();

    uint32_t find_memory_type(uint32_t type_filter,
                          vk::MemoryPropertyFlags properties);

    std::pair<vk::Buffer, vk::DeviceMemory> create_buffer(vk::DeviceSize size,
                                                          vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties);

    vk::CommandBuffer begin_single_time_commands();

    void end_single_time_commands(vk::CommandBuffer command_buffer);

    void copy_buffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size);

    void transition_image_layout(vk::Image image,
                                 vk::Format format,
                                 vk::ImageLayout old_layout,
                                 vk::ImageLayout new_layout);

    void copy_buffer_to_image(vk::Buffer buffer, vk::Image image, uint32_t width,
                              uint32_t height);

    void create_vertex_buffers();

    void create_index_buffers();

    void create_uniform_buffers();

    void create_descriptor_set_layout();

    void create_descriptor_pool();

    void create_descriptor_sets();

    void create_command_buffers();

    void create_command_pools();

    void create_renderpass();

    void create_clear_renderpass();

    void create_graphics_pipeline();

    void create_swapchain();

    void create_framebuffers();

    void cleanup_swapchain();

    void create_depth_resources();

    void recreate_swapchain();

    void create_instance();

    void create_physical_device();

    void create_device_and_queues();

    std::pair<vk::Image, vk::DeviceMemory>
    create_image(uint32_t width,
                 uint32_t height,
                 vk::Format format,
                 vk::ImageTiling tiling,
                 vk::ImageUsageFlags usage,
                 vk::MemoryPropertyFlags properties);

    void create_texture_image();

    vk::ImageView create_image_view(vk::Image image,
                                    vk::Format format,
                                    vk::ImageAspectFlags aspect_flags);

    void create_texture_image_view();

    void create_texture_sampler();

    vk::Format find_supported_format(const std::vector<vk::Format>& candidates,
                                     vk::ImageTiling tiling,
                                     vk::FormatFeatureFlags features);

    vk::Format find_depth_format();

    bool has_stencil_component(vk::Format format);

    void init_vulkan();

    void cleanup();

    void send_ubo(uniform_buffer_object ubo);

    void update_uniform_buffer(uint32_t current_image);

    void init();

    void draw_frame();
};
