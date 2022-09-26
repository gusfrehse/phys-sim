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
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <limits>
#include <optional>
#include <chrono>

#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shaders.h"

#define CHECK_SDL(x, pred)                                                       \
  do {                                                                           \
    if (x pred) {                                                                \
      fprintf(stderr, "%s %s:%d, sdl check failed:\n",                           \
              __PRETTY_FUNCTION__,                                               \
              __FILE__,                                                          \
              __LINE__);                                                         \
      SDL_Log("SDL_ERROR: " #x ": %s", SDL_GetError());                          \
      exit(1);                                                                   \
    }                                                                            \
  } while(0);

const int MAX_FRAMES_IN_FLIGHT = 2;

struct uniform_buffer_object {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};

struct vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static std::array<vk::VertexInputAttributeDescription, 2>
  get_attribute_descriptions() {
    std::array<vk::VertexInputAttributeDescription, 2> attrib_description{};

    attrib_description[0].binding = 0;
    attrib_description[0].location = 0;
    attrib_description[0].format = vk::Format::eR32G32Sfloat;
    attrib_description[0].offset = offsetof(vertex, pos);

    attrib_description[1].binding = 0;
    attrib_description[1].location = 1;
    attrib_description[1].format = vk::Format::eR32G32B32Sfloat;
    attrib_description[1].offset = offsetof(vertex, color);

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
  {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
  {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
  {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
  {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};


const std::vector<uint16_t> indices = {
  0, 1, 2, 2, 3, 0
};

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

  std::vector<vk::Framebuffer> swapchain_framebuffers;

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

  vk::Buffer vertex_buffer;
  vk::DeviceMemory vertex_buffer_memory;

  vk::Buffer index_buffer;
  vk::DeviceMemory index_buffer_memory;

  std::vector<vk::Buffer> uniform_buffers;
  std::vector<vk::DeviceMemory> uniform_buffers_memory;

  vk::DescriptorPool descriptor_pool;

  std::vector<vk::DescriptorSet> descriptor_sets;

  uint32_t current_frame = 0;

  auto record_command_buffer(vk::CommandBuffer command_buffer,
                             int image_index) {
    vk::CommandBufferBeginInfo begin_info{};

    command_buffer.begin(begin_info);

    vk::RenderPassBeginInfo renderpass_begin_info{};
    renderpass_begin_info.renderPass = renderpass;
    renderpass_begin_info.framebuffer = swapchain_framebuffers[image_index];
    renderpass_begin_info.renderArea.offset = vk::Offset2D(0, 0);
    renderpass_begin_info.renderArea.extent = swapchain_image_extent;

    vk::ClearValue clear_color(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    renderpass_begin_info.clearValueCount = 1;
    renderpass_begin_info.pClearValues = &clear_color;

    command_buffer.beginRenderPass(renderpass_begin_info,
                                   vk::SubpassContents::eInline);
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                                graphics_pipeline);

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapchain_image_extent.width);
    viewport.height = static_cast<float>(swapchain_image_extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    command_buffer.setViewport(0, {viewport});

    vk::Rect2D scissor{{0, 0}, swapchain_image_extent};

    command_buffer.setScissor(0, {scissor});
    
    command_buffer.bindVertexBuffers(0, {vertex_buffer}, {0});

    command_buffer.bindIndexBuffer(index_buffer, 0, vk::IndexType::eUint16);

    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      pipeline_layout,
                                      0,
                                      {descriptor_sets[current_frame]},
                                      {});

    command_buffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

    command_buffer.endRenderPass();

    command_buffer.end();
  }

  auto create_sync_objects() {
    vk::SemaphoreCreateInfo semaphore_info{};

    vk::FenceCreateInfo fence_info{vk::FenceCreateFlagBits::eSignaled};

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      image_available_semaphores.push_back(
          device.createSemaphore(semaphore_info));
      render_finished_semaphores.push_back(
          device.createSemaphore(semaphore_info));

      in_flight_fences.push_back(device.createFence(fence_info));
    }
  }

  std::pair<vk::Buffer, vk::DeviceMemory> create_buffer(vk::DeviceSize size,
          vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {

      vk::Buffer buffer;
      vk::DeviceMemory buffer_memory;

      vk::BufferCreateInfo buffer_info{};
      buffer_info.size = size;
      buffer_info.usage = usage;
      buffer_info.sharingMode = vk::SharingMode::eExclusive;

      buffer = device.createBuffer(buffer_info);

      vk::MemoryRequirements mem_requirements =
          device.getBufferMemoryRequirements(buffer);

    auto find_memory_type = [this](uint32_t type_filter,
                                   vk::MemoryPropertyFlags properties){
        vk::PhysicalDeviceMemoryProperties mem_properties =
          physical_device.getMemoryProperties();

        for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
          if (type_filter & (1 << i)) {
            if ((mem_properties.memoryTypes[i].propertyFlags & properties)
              == properties) {
              return i;
            }
          }
        }

        fprintf(stderr, "ERROR: failed to find memory type\n");
        exit(1);
    };

    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
      find_memory_type(mem_requirements.memoryTypeBits, properties);

    buffer_memory = device.allocateMemory(alloc_info);

    device.bindBufferMemory(buffer, buffer_memory, 0);

    return {buffer, buffer_memory};
  }

  auto copy_buffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
    // create a command buffer that will be used only once to copy memory from
    // (in this specific case) from the src buffer to dst buffer


    vk::CommandBufferAllocateInfo alloc_info{};
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandPool = alloc_command_pool;
    alloc_info.commandBufferCount = 1;

    vk::CommandBuffer command_buffer =
      device.allocateCommandBuffers(alloc_info)[0];

    // record cmdbuffer
    vk::CommandBufferBeginInfo begin_info(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    command_buffer.begin(begin_info);

    vk::BufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;

    command_buffer.copyBuffer(src, dst, {copy_region});

    command_buffer.end();

    // submit command buffer
    vk::SubmitInfo submit_info{};
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    queue.submit({submit_info});

    // wait
    queue.waitIdle();

    device.freeCommandBuffers(alloc_command_pool, {command_buffer});
  }

  auto create_vertex_buffers() {
    vk::DeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

    // create staging buffer and memory
    auto [staging_buffer, staging_buffer_memory] =
      create_buffer(buffer_size,
                    vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);


    // copy data to staging buffer
    void* data = device.mapMemory(staging_buffer_memory, 0, buffer_size);

    memcpy(data, vertices.data(), (size_t) buffer_size);

    device.unmapMemory(staging_buffer_memory);


    // create vertex buffer
    std::tie(vertex_buffer, vertex_buffer_memory) =
      create_buffer(
        buffer_size,
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(staging_buffer, vertex_buffer, buffer_size);

    device.destroy(staging_buffer);
    device.free(staging_buffer_memory);
  }

  auto create_index_buffers() {
    vk::DeviceSize buffer_size = sizeof(indices[0]) * indices.size();

    // create staging buffer and memory
    auto [staging_buffer, staging_buffer_memory] =
      create_buffer(buffer_size,
                    vk::BufferUsageFlagBits::eTransferSrc,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);


    // copy data to staging buffer
    void* data = device.mapMemory(staging_buffer_memory, 0, buffer_size);

    memcpy(data, indices.data(), (size_t) buffer_size);

    device.unmapMemory(staging_buffer_memory);


    // create vertex buffer
    std::tie(index_buffer, index_buffer_memory) =
      create_buffer(
        buffer_size,
        vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    copy_buffer(staging_buffer, index_buffer, buffer_size);

    device.destroy(staging_buffer);
    device.free(staging_buffer_memory);
  }

  auto create_uniform_buffers() {
    vk::DeviceSize buffer_size = sizeof(uniform_buffer_object);

    uniform_buffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniform_buffers_memory.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      std::tie(uniform_buffers[i], uniform_buffers_memory[i]) = create_buffer(
          buffer_size,
          vk::BufferUsageFlagBits::eUniformBuffer,
          vk::MemoryPropertyFlagBits::eHostVisible |
          vk::MemoryPropertyFlagBits::eHostCoherent);
    }
  }

  auto create_descriptor_set_layout() {
    vk::DescriptorSetLayoutBinding ubo_layout_binding{};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;
    ubo_layout_binding.pImmutableSamplers = nullptr;

    vk::DescriptorSetLayoutCreateInfo layout_info{};
    layout_info.bindingCount = 1;
    layout_info.pBindings = &ubo_layout_binding;

    descriptor_set_layout = device.createDescriptorSetLayout(layout_info);
  }


  auto create_descriptor_pool() {
    vk::DescriptorPoolSize pool_size{};
    pool_size.type = vk::DescriptorType::eUniformBuffer;
    pool_size.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo pool_info{};
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    descriptor_pool = device.createDescriptorPool(pool_info);
  }

  auto create_descriptor_sets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 descriptor_set_layout);

    vk::DescriptorSetAllocateInfo alloc_info{};
    alloc_info.descriptorPool = descriptor_pool;
    alloc_info.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
    alloc_info.pSetLayouts = layouts.data();

    descriptor_sets = device.allocateDescriptorSets(alloc_info);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo buffer_info{};
      buffer_info.buffer = uniform_buffers[i];
      buffer_info.offset = 0;
      buffer_info.range = sizeof(uniform_buffer_object);

      vk::WriteDescriptorSet descriptor_write{};
      descriptor_write.dstSet = descriptor_sets[i];
      descriptor_write.dstBinding = 0;
      descriptor_write.dstArrayElement = 0;
      descriptor_write.descriptorType = vk::DescriptorType::eUniformBuffer;
      descriptor_write.descriptorCount = 1;
      descriptor_write.pBufferInfo = &buffer_info;
      descriptor_write.pImageInfo = nullptr;
      descriptor_write.pTexelBufferView = nullptr;

      device.updateDescriptorSets({descriptor_write}, {});

    }
  }

  auto create_command_buffers() {
    vk::CommandBufferAllocateInfo alloc_info{};
    alloc_info.commandPool = command_pool;
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    command_buffers = device.allocateCommandBuffers(alloc_info);
  }

  auto create_command_pools() {
    vk::CommandPoolCreateInfo pool_info{};
    pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    pool_info.queueFamilyIndex = queue_family_index.value();

    command_pool = device.createCommandPool(pool_info);

    vk::CommandPoolCreateInfo alloc_pool_info{};
    alloc_pool_info.flags = vk::CommandPoolCreateFlagBits::eTransient;
    alloc_pool_info.queueFamilyIndex = queue_family_index.value();

    alloc_command_pool = device.createCommandPool(alloc_pool_info);
  }

  auto create_renderpass() {
    vk::AttachmentDescription color_attachment_desc{};
    color_attachment_desc.format = swapchain_image_format;
    color_attachment_desc.samples = vk::SampleCountFlagBits::e1;
    color_attachment_desc.loadOp = vk::AttachmentLoadOp::eClear;
    color_attachment_desc.storeOp = vk::AttachmentStoreOp::eStore;

    color_attachment_desc.initialLayout = vk::ImageLayout::eUndefined;
    color_attachment_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.srcAccessMask = vk::AccessFlagBits::eNone;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

    vk::RenderPassCreateInfo renderpass_info{};
    renderpass_info.attachmentCount = 1;
    renderpass_info.pAttachments = &color_attachment_desc;
    renderpass_info.subpassCount = 1;
    renderpass_info.pSubpasses = &subpass;
    renderpass_info.dependencyCount = 1;
    renderpass_info.pDependencies = &dependency;

    renderpass = device.createRenderPass(renderpass_info);
  }

  auto create_graphics_pipeline() {
    // create fragment shader module
    vk::ShaderModuleCreateInfo frag_module_info{};
    frag_module_info.codeSize = basic_frag_spv_len;
    frag_module_info.pCode = (const uint32_t *)basic_frag_spv;

    vk::ShaderModule frag_module = device.createShaderModule(frag_module_info);

    // create vertex shader module
    vk::ShaderModuleCreateInfo vert_module_info{};
    vert_module_info.codeSize = basic_vert_spv_len;
    vert_module_info.pCode = (const uint32_t *)basic_vert_spv;

    vk::ShaderModule vert_module = device.createShaderModule(vert_module_info);

    // create fragment stage info
    vk::PipelineShaderStageCreateInfo frag_stage_info{};
    frag_stage_info.stage = vk::ShaderStageFlagBits::eFragment;
    frag_stage_info.module = frag_module;
    frag_stage_info.pName = "main";

    // create fragment stage info
    vk::PipelineShaderStageCreateInfo vert_stage_info{};
    vert_stage_info.stage = vk::ShaderStageFlagBits::eVertex;
    vert_stage_info.module = vert_module;
    vert_stage_info.pName = "main";

    vk::PipelineShaderStageCreateInfo shader_stages[] = {vert_stage_info,
                                                         frag_stage_info};

    // vertex input
    auto binding_desc = vertex::getBindingDescription();
    auto attrib_desc = vertex::get_attribute_descriptions();

    vk::PipelineVertexInputStateCreateInfo vert_input_info{};
    vert_input_info.vertexBindingDescriptionCount = 1;
    vert_input_info.pVertexBindingDescriptions = &binding_desc;
    vert_input_info.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(attrib_desc.size());
    vert_input_info.pVertexAttributeDescriptions = attrib_desc.data();

    // topology
    vk::PipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.topology = vk::PrimitiveTopology::eTriangleList;
    input_assembly.primitiveRestartEnable = false;

    std::vector<vk::DynamicState> dynamic_states = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamic_state_info{};
    dynamic_state_info.dynamicStateCount = dynamic_states.size();
    dynamic_state_info.pDynamicStates = dynamic_states.data();

    vk::PipelineViewportStateCreateInfo viewport_state_info{};
    viewport_state_info.viewportCount = 1;
    viewport_state_info.scissorCount = 1;
    // viewport would go here if it was static state

    vk::PipelineRasterizationStateCreateInfo rasterizer_info{};
    rasterizer_info.depthClampEnable = false;
    rasterizer_info.rasterizerDiscardEnable = false;
    rasterizer_info.polygonMode = vk::PolygonMode::eFill;
    rasterizer_info.lineWidth = 1.0f;
    rasterizer_info.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer_info.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer_info.depthBiasEnable = false;

    vk::PipelineMultisampleStateCreateInfo multisample_info{};
    multisample_info.sampleShadingEnable = false;
    multisample_info.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisample_info.minSampleShading = 1.0f;
    multisample_info.pSampleMask = nullptr;
    multisample_info.alphaToCoverageEnable = false;
    multisample_info.alphaToOneEnable = false;

    vk::PipelineColorBlendAttachmentState color_blend_attachment_state{};
    color_blend_attachment_state.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    color_blend_attachment_state.blendEnable = false;

    vk::PipelineColorBlendStateCreateInfo color_blend_info{};
    color_blend_info.logicOpEnable = false;
    color_blend_info.attachmentCount = 1;
    color_blend_info.pAttachments = &color_blend_attachment_state;

    vk::PipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;

    pipeline_layout = device.createPipelineLayout(pipeline_layout_info);

    vk::GraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;

    pipeline_info.pVertexInputState = &vert_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state_info;
    pipeline_info.pRasterizationState = &rasterizer_info;
    pipeline_info.pMultisampleState = &multisample_info;
    pipeline_info.pDepthStencilState = nullptr;
    pipeline_info.pColorBlendState = &color_blend_info;
    pipeline_info.pDynamicState = &dynamic_state_info;

    pipeline_info.layout = pipeline_layout;
    pipeline_info.renderPass = renderpass;
    pipeline_info.subpass = 0;

    pipeline_info.basePipelineHandle = nullptr;
    pipeline_info.basePipelineIndex = -1;

    graphics_pipeline =
        device.createGraphicsPipeline(nullptr, pipeline_info).value;

    device.destroy(frag_module);
    device.destroy(vert_module);
  }

  auto create_swapchain() {
    //fprintf(stderr, "Creating swapchain\n");
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
    swapchain_create_info.imageSharingMode = vk::SharingMode::eExclusive; // change this for more queue families
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
    swapchain_image_views.clear();
    for (auto &image : swapchain_images) {
      vk::ImageViewCreateInfo image_view_create_info{};
      image_view_create_info.setImage(image);
      image_view_create_info.setViewType(vk::ImageViewType::e2D);
      image_view_create_info.setFormat(swapchain_image_format);
      image_view_create_info.setComponents(
          {vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity,
           vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity});
      image_view_create_info.subresourceRange.setAspectMask(
          vk::ImageAspectFlagBits::eColor);
      image_view_create_info.subresourceRange.setBaseMipLevel(0);
      image_view_create_info.subresourceRange.setLevelCount(1);
      image_view_create_info.subresourceRange.setBaseArrayLayer(0);
      image_view_create_info.subresourceRange.setLayerCount(1);

      swapchain_image_views.push_back(
          device.createImageView(image_view_create_info));
    }

    //fprintf(stderr, "Created swapchain\n");
  }

  auto create_framebuffers() {
    swapchain_framebuffers.clear();

    for (int i = 0; i < swapchain_image_views.size(); i++) {
      vk::ImageView attachments[] = {swapchain_image_views[i]};

      vk::FramebufferCreateInfo framebuffer_info{};
      framebuffer_info.renderPass = renderpass;
      framebuffer_info.attachmentCount = 1;
      framebuffer_info.pAttachments = attachments;
      framebuffer_info.width = swapchain_image_extent.width;
      framebuffer_info.height = swapchain_image_extent.height;
      framebuffer_info.layers = 1;

      swapchain_framebuffers.push_back(
          device.createFramebuffer(framebuffer_info));
    }
  }

  auto cleanup_swapchain() {
    for (auto &fb : swapchain_framebuffers) {
      device.destroy(fb);
    }

    for (auto &e : swapchain_image_views) {
      device.destroy(e);
    }

    device.destroy(swapchain);
  }

  auto recreate_swapchain() {
    device.waitIdle();

    cleanup_swapchain();

    create_swapchain();

    create_framebuffers();
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

    create_renderpass();

    create_descriptor_set_layout();

    create_graphics_pipeline();

    create_framebuffers();

    create_command_pools();

    create_vertex_buffers();

    create_index_buffers();

    create_uniform_buffers();

    create_descriptor_pool();

    create_descriptor_sets();

    create_command_buffers();

    create_sync_objects();
  }

  auto cleanup() {
    device.waitIdle();

    cleanup_swapchain();

    device.destroy(vertex_buffer);
    device.free(vertex_buffer_memory);

    device.destroy(index_buffer);
    device.free(index_buffer_memory);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      device.destroy(uniform_buffers[i]);
      device.free(uniform_buffers_memory[i]);
    }

    device.destroy(descriptor_pool);

    device.destroy(descriptor_set_layout);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      device.destroy(image_available_semaphores[i]);
      device.destroy(render_finished_semaphores[i]);
      device.destroy(in_flight_fences[i]);
    }

    device.destroy(command_pool);
    device.destroy(alloc_command_pool);

    device.destroy(renderpass);
    device.destroy(graphics_pipeline);
    device.destroy(pipeline_layout);

    device.destroy();

    instance.destroy(surface);
    instance.destroy();
    SDL_DestroyWindow(window);
  }

  auto update_uniform_buffer(uint32_t current_image) {
    static auto start_time = std::chrono::high_resolution_clock::now();

    auto current_time = std::chrono::high_resolution_clock::now();

    float time = std::chrono::duration<float, std::chrono::seconds::period>(
        current_time - start_time).count();

    uniform_buffer_object ubo;
    ubo.model = glm::rotate(glm::mat4(1.0f),
                            time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f),
                           glm::vec3(0.0f),
                           glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.proj = glm::perspective(glm::radians(45.0f), swapchain_image_extent.width / (float) swapchain_image_extent.height, 0.1f, 10.0f);

    ubo.proj[1][1] *= -1;

    void *data;
    data = device.mapMemory(uniform_buffers_memory[current_image], 0, sizeof(ubo));

    memcpy(data, &ubo, sizeof(ubo));

    device.unmapMemory(uniform_buffers_memory[current_image]);
  }

  auto init() -> auto{ init_vulkan(); }

  auto draw_frame() {
    // wait for in flight fences (whatever that means...)
    if (device.waitForFences({in_flight_fences[current_frame]}, true,
                             UINT64_MAX) != vk::Result::eSuccess) {
      printf("wait for fence failed ... wtf\n");
      exit(123);
    }

    device.resetFences({in_flight_fences[current_frame]});

    uint32_t next_image;
    try {
      auto next_image_result = device.acquireNextImageKHR(
          swapchain, UINT64_MAX,
          image_available_semaphores[current_frame], nullptr);

      next_image = next_image_result.value;
    } catch (vk::OutOfDateKHRError const &e) {
      // probably resized the window, need to recreate the swapchain
      //fprintf(stderr, "A recreating swapchain\n");
      recreate_swapchain();
      return;
    }

    update_uniform_buffer(current_frame);

    command_buffers[current_frame].reset();
    record_command_buffer(command_buffers[current_frame], next_image);

    vk::SubmitInfo submit_info{};

    vk::Semaphore wait_semaphores[] = {
        image_available_semaphores[current_frame]};
    vk::PipelineStageFlags wait_stages[] = {
        vk::PipelineStageFlagBits::eColorAttachmentOutput};

    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffers[current_frame];

    vk::Semaphore signal_semaphores[] = {
        render_finished_semaphores[current_frame]};
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    queue.submit({submit_info}, in_flight_fences[current_frame]);

    vk::PresentInfoKHR present_info{};
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;

    vk::SwapchainKHR swapchains[] = {swapchain};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &next_image;
    present_info.pResults = nullptr;

    try {
      auto present_result = queue.presentKHR(present_info);

      if (present_result != vk::Result::eSuccess) {
        printf("present failed ... what\n");
        exit(231);
      }

    } catch (vk::OutOfDateKHRError const &e) {
      //fprintf(stderr, "B recreating swapchain\n");
      // needs to recreate the swapchain
      recreate_swapchain();

      return;
    }

    current_frame =
        (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
  }
};

auto main() -> int {
  window_and_vulkan_state state;
  state.init();

  // SDL_Delay(1000);
  bool running = true;
  SDL_Event event;
  while (running) {
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        running = false;
      }
    }

    state.draw_frame();
  }

  state.cleanup();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
