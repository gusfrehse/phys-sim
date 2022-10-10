#include <optional>

#include <SDL2/SDL.h>
#include <glm/matrix.hpp>

#include "vertex.hpp"

const int MAX_FRAMES_IN_FLIGHT = 2;

struct uniform_buffer_object {
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

const std::string MODEL_PATH = "models/viking_room.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

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

  std::vector<vertex> vertices;
  std::vector<uint32_t> indices;

  vk::Buffer vertex_buffer;
  vk::DeviceMemory vertex_buffer_memory;

  vk::Buffer index_buffer;
  vk::DeviceMemory index_buffer_memory;

  std::vector<vk::Buffer> camera_uniform_buffers;
  std::vector<vk::DeviceMemory> camera_uniform_buffers_memory;

  std::vector<vk::Buffer> object_uniform_buffers;
  std::vector<vk::DeviceMemory> object_uniform_buffers_memory;

  vk::DescriptorPool descriptor_pool;

  std::vector<vk::DescriptorSet> descriptor_sets;

  uint32_t mip_levels;
  vk::Image texture_image;
  vk::DeviceMemory texture_image_memory;
  vk::ImageView texture_image_view;
  vk::Sampler texture_sampler;

  vk::SampleCountFlagBits msaa_samples = vk::SampleCountFlagBits::e1;

  vk::Image color_image;
  vk::DeviceMemory color_image_memory;
  vk::ImageView color_image_view;

  uint32_t num_objects = 0;

  uint32_t current_frame = 0;

  void set_num_objects(uint32_t num);

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
                               vk::ImageLayout new_layout,
                               uint32_t mip_levels);

  void copy_buffer_to_image(vk::Buffer buffer, vk::Image image, uint32_t width,
                            uint32_t height);

  void create_vertex_buffers();

  void create_index_buffers();

  void create_objects_uniform_buffer();

  void create_uniform_buffers();

  void create_objects_descriptor_set_layout();

  void create_descriptor_set_layout();

  void create_descriptor_pool();

  void create_descriptor_sets();

  void create_command_buffers();

  void create_command_pools();

  void create_renderpass();

  void create_graphics_pipeline();

  void create_swapchain();

  void create_framebuffers();

  void cleanup_swapchain();

  void create_depth_resources();

  void create_color_resources();

  void recreate_swapchain();

  void create_instance();

  vk::SampleCountFlagBits get_max_usable_sample_count();

  void create_physical_device();

  void create_device_and_queues();

  std::pair<vk::Image, vk::DeviceMemory>
  create_image(uint32_t width,
               uint32_t height,
               uint32_t mip_levels,
               vk::SampleCountFlagBits num_samples,
               vk::Format format,
               vk::ImageTiling tiling,
               vk::ImageUsageFlags usage,
               vk::MemoryPropertyFlags properties);

  void generate_mipmaps(vk::Image image, vk::Format image_format,
                        int32_t tex_width, int32_t tex_height,
                        uint32_t mip_levels );

  void create_texture_image();

  vk::ImageView create_image_view(vk::Image image,
                                  vk::Format format,
                                  vk::ImageAspectFlags aspect_flags,
                                  uint32_t mip_levels);

  void create_texture_image_view();

  void create_texture_sampler();

  vk::Format find_supported_format(const std::vector<vk::Format>& candidates,
                                   vk::ImageTiling tiling,
                                   vk::FormatFeatureFlags features);

  vk::Format find_depth_format();

  bool has_stencil_component(vk::Format format);

  void load_model();

  void init_vulkan();

  void cleanup();

  void update_uniform_buffer(uint32_t current_image);

  void update_objects_uniform_buffer(uint32_t current_image);

  uniform_buffer_object* map_camera_uniform();

  void unmap_camera_uniform(uniform_buffer_object* data);

  glm::mat4* map_object_uniform();

  void unmap_object_uniform(glm::mat4* data);

  uint32_t get_width();

  uint32_t get_height();

  void init();

  void draw_frame();
};


/* vim: set sts=2 ts=2 sw=2 et cc=81: */
