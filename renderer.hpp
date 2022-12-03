#include <optional>

#include <SDL2/SDL.h>

#include "glm.hpp"

#include <vulkan/vulkan.hpp>

#include "vertex.hpp"

const int MAX_FRAMES_IN_FLIGHT = 3;

struct camera_uniform {
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

struct object_uniform {
  alignas(16) glm::mat4 model;
};

const std::string MODEL_PATH = "models/sphere.obj";
const std::string TEXTURE_PATH = "textures/viking_room.png";

struct renderer {
private:
  SDL_Window *m_window;
  struct {
    int width = 800;
    int height = 450;
  } m_window_dimensions;

  vk::Instance m_instance;
  vk::Device m_device;
  vk::PhysicalDevice m_physical_device;
  vk::SurfaceKHR m_surface;

  vk::SwapchainKHR m_swapchain;
  vk::Format m_swapchain_image_format;
  vk::Extent2D m_swapchain_image_extent;

  std::vector<vk::Image> m_swapchain_images;
  std::vector<vk::ImageView> m_swapchain_image_views;

  std::vector<vk::Framebuffer> m_swapchain_framebuffers;

  vk::Image m_depth_image;
  vk::DeviceMemory m_depth_image_memory;
  vk::ImageView m_depth_image_view;

  std::optional<uint32_t> m_queue_family_index = std::nullopt;

  vk::Queue m_queue;

  vk::RenderPass m_renderpass;
  vk::DescriptorSetLayout m_descriptor_set_layout;
  vk::PipelineLayout m_pipeline_layout;
  vk::Pipeline m_graphics_pipeline;

  vk::CommandPool m_command_pool;
  std::vector<vk::CommandBuffer> m_command_buffers;

  vk::CommandPool m_alloc_command_pool;

  std::vector<vk::Semaphore> m_image_available_semaphores;
  std::vector<vk::Semaphore> m_render_finished_semaphores;
  std::vector<vk::Fence> m_in_flight_fences;

  std::vector<vertex> m_vertices;
  std::vector<uint32_t> m_indices;

  vk::Buffer m_vertex_buffer;
  vk::DeviceMemory m_vertex_buffer_memory;

  vk::Buffer m_index_buffer;
  vk::DeviceMemory m_index_buffer_memory;

  std::vector<vk::Buffer> m_camera_uniform_buffers;
  std::vector<vk::DeviceMemory> m_camera_uniform_buffers_memory;

  std::vector<vk::Buffer> m_object_uniform_buffers;
  std::vector<vk::DeviceMemory> m_object_uniform_buffers_memory;

  size_t m_uniform_offset_alignment;

  vk::DescriptorPool m_descriptor_pool;

  std::vector<vk::DescriptorSet> m_descriptor_sets;

  uint32_t m_mip_levels;
  vk::Image m_texture_image;
  vk::DeviceMemory m_texture_image_memory;
  vk::ImageView m_texture_image_view;
  vk::Sampler m_texture_sampler;

  vk::SampleCountFlagBits m_msaa_samples = vk::SampleCountFlagBits::e1;

  vk::Image m_color_image;
  vk::DeviceMemory m_color_image_memory;
  vk::ImageView m_color_image_view;

  uint32_t m_num_objects = 0;

  uint32_t m_current_frame = 0;

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

public:
  uint8_t* map_camera_uniform();

  void unmap_camera_uniform();

  uint8_t* map_object_uniform();

  void unmap_object_uniform();

  uint32_t get_width() const;

  uint32_t get_height() const;

  SDL_Window* get_window() const;

  size_t get_uniform_alignment() const;

  void init();

  void draw_frame();

  void set_num_objects(uint32_t num);

  void cleanup();
};


/* vim: set sts=2 ts=2 sw=2 et cc=81: */
