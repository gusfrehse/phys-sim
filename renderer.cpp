#include "renderer.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>
#include <unordered_map>

#include <SDL2/SDL.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_timer.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_vulkan.h>

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_structs.hpp>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "shaders.h"

#define CHECK_SDL(x, pred)                                                     \
  do {                                                                         \
    if (x pred) {                                                              \
      fprintf(stderr, "%s %s:%d, sdl check failed:\n",                         \
                __PRETTY_FUNCTION__,                                           \
                __FILE__,                                                      \
                __LINE__);                                                     \
      SDL_Log("SDL_ERROR: " #x ": %s", SDL_GetError());                        \
      exit(1);                                                                 \
    }                                                                          \
  } while(0);

template<> struct std::hash<vertex> {
  size_t operator()(vertex const& vertex) const {
    return ((std::hash<glm::vec3>()(vertex.pos) ^
    (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
    (hash<glm::vec2>()(vertex.tex_coord) << 1);
  }
};

void renderer::set_num_objects(uint32_t num) {
  if (num > m_num_objects) {
    // create num - num_objects 
  } else {
    // delete num_objects - num
  }

  m_num_objects = num;
}

void renderer::record_command_buffer(vk::CommandBuffer command_buffer,
                                     int image_index) {
  vk::CommandBufferBeginInfo begin_info{};

  command_buffer.begin(begin_info);

  vk::RenderPassBeginInfo renderpass_begin_info{};
  renderpass_begin_info.renderPass = m_renderpass;
  renderpass_begin_info.framebuffer = m_swapchain_framebuffers[image_index];
  renderpass_begin_info.renderArea.offset = vk::Offset2D(0, 0);
  renderpass_begin_info.renderArea.extent = m_swapchain_image_extent;

  std::array<vk::ClearValue, 2> clear_colors{};
  clear_colors[0].color = std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f};
  clear_colors[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

  renderpass_begin_info.clearValueCount = clear_colors.size();
  renderpass_begin_info.pClearValues = clear_colors.data();

  command_buffer.beginRenderPass(renderpass_begin_info,
                                 vk::SubpassContents::eInline);
  command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                              m_graphics_pipeline);

  vk::Viewport viewport{};
  viewport.x = 0.0f;
  viewport.y = 0.0f;
  viewport.width = static_cast<float>(m_swapchain_image_extent.width);
  viewport.height = static_cast<float>(m_swapchain_image_extent.height);
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  command_buffer.setViewport(0, {viewport});

  vk::Rect2D scissor{{0, 0}, m_swapchain_image_extent};

  command_buffer.setScissor(0, {scissor});

  command_buffer.bindVertexBuffers(0, {m_vertex_buffer}, {0});

  command_buffer.bindIndexBuffer(m_index_buffer, 0, vk::IndexType::eUint32);

  for (int i = 0; i < m_num_objects; i++) {
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                      m_pipeline_layout,
                                      0,
                                      {m_descriptor_sets[m_current_frame]},
                                      {static_cast<unsigned int>
                                        (m_dynamic_alignment * i)});

    command_buffer.drawIndexed(static_cast<uint32_t>(m_indices.size()),
                               1, 0, 0, 0);
  }

  command_buffer.endRenderPass();

  command_buffer.end();
}

void renderer::create_sync_objects() {
  vk::SemaphoreCreateInfo semaphore_info{};

  vk::FenceCreateInfo fence_info{vk::FenceCreateFlagBits::eSignaled};

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    m_image_available_semaphores.push_back(
        m_device.createSemaphore(semaphore_info));
    m_render_finished_semaphores.push_back(
        m_device.createSemaphore(semaphore_info));

    m_in_flight_fences.push_back(m_device.createFence(fence_info));
  }
}

uint32_t renderer::find_memory_type(uint32_t type_filter,
                                    vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties mem_properties =
    m_physical_device.getMemoryProperties();

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

std::pair<vk::Buffer, vk::DeviceMemory>
renderer::create_buffer(vk::DeviceSize size,
                        vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags properties) {

  vk::Buffer buffer;
  vk::DeviceMemory buffer_memory;

  vk::BufferCreateInfo buffer_info{};
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = vk::SharingMode::eExclusive;

  buffer = m_device.createBuffer(buffer_info);

  vk::MemoryRequirements mem_requirements =
    m_device.getBufferMemoryRequirements(buffer);


  vk::MemoryAllocateInfo alloc_info{};
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
    find_memory_type(mem_requirements.memoryTypeBits, properties);

  buffer_memory = m_device.allocateMemory(alloc_info);

  m_device.bindBufferMemory(buffer, buffer_memory, 0);

  return {buffer, buffer_memory};
}

vk::CommandBuffer renderer::begin_single_time_commands() {
  vk::CommandBufferAllocateInfo alloc_info{};
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandPool = m_alloc_command_pool;
  alloc_info.commandBufferCount = 1;

  vk::CommandBuffer command_buffer =
    m_device.allocateCommandBuffers(alloc_info)[0];

  // record cmdbuffer
  vk::CommandBufferBeginInfo begin_info(
    vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  command_buffer.begin(begin_info);

  return command_buffer;
}

void renderer::end_single_time_commands(vk::CommandBuffer command_buffer) {
  command_buffer.end();

  // submit command buffer
  vk::SubmitInfo submit_info{};
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  m_queue.submit({submit_info});

  // wait
  m_queue.waitIdle();

  m_device.freeCommandBuffers(m_alloc_command_pool, {command_buffer});
}

void renderer::copy_buffer(vk::Buffer src,
                           vk::Buffer dst,
                           vk::DeviceSize size) {
  vk::CommandBuffer command_buffer = begin_single_time_commands();

  vk::BufferCopy copy_region{};
  copy_region.srcOffset = 0;
  copy_region.dstOffset = 0;
  copy_region.size = size;

  command_buffer.copyBuffer(src, dst, {copy_region});

  end_single_time_commands(command_buffer);
}

void renderer::transition_image_layout(vk::Image image,
                                       vk::Format format,
                                       vk::ImageLayout old_layout,
                                       vk::ImageLayout new_layout,
                                       uint32_t mip_levels) {

  auto command_buffer = begin_single_time_commands();

  vk::ImageMemoryBarrier barrier{};
  barrier.oldLayout = old_layout;
  barrier.newLayout = new_layout;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = image;
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseMipLevel = 0;
  barrier.subresourceRange.levelCount = mip_levels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  //barrier.srcAccessMask = vk::AccessFlagBits::eNone; // TODO
  //barrier.dstAccessMask = vk::AccessFlagBits::eNone; // TODO

  vk::PipelineStageFlags src_stage, dst_stage;

  if (old_layout == vk::ImageLayout::eUndefined &&
    new_layout == vk::ImageLayout::eTransferDstOptimal) {

    barrier.srcAccessMask = vk::AccessFlagBits::eNone;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

    src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
    dst_stage = vk::PipelineStageFlagBits::eTransfer;

  } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
    new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {

    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    src_stage = vk::PipelineStageFlagBits::eTransfer;
    dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
  } else {
    fprintf(stderr, "ERROR: unsupported layout transition!\n");
    exit(1);
  }

  command_buffer.pipelineBarrier(src_stage,
                                 dst_stage,
                                 static_cast<vk::DependencyFlags>(0),
                                 {},
                                 {},
                                 {barrier});

  end_single_time_commands(command_buffer);

}

void renderer::copy_buffer_to_image(vk::Buffer buffer,
                                    vk::Image image,
                                    uint32_t width,
                                    uint32_t height) {
  vk::CommandBuffer command_buffer = begin_single_time_commands();

  vk::BufferImageCopy region{};
  region.bufferOffset = 0;
  region.bufferRowLength = 0;
  region.bufferImageHeight = 0;

  region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
  region.imageSubresource.mipLevel = 0;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount = 1;

  region.imageOffset = vk::Offset3D {0, 0, 0};
  region.imageExtent = vk::Extent3D {width, height, 1};

  command_buffer.copyBufferToImage(buffer,
                                   image,
                                   vk::ImageLayout::eTransferDstOptimal,
                                   {region});

  end_single_time_commands(command_buffer);
}

void renderer::create_vertex_buffers() {
  vk::DeviceSize buffer_size = sizeof(m_vertices[0]) * m_vertices.size();

  // create staging buffer and memory
  auto [staging_buffer, staging_buffer_memory] =
    create_buffer(buffer_size,
                  vk::BufferUsageFlagBits::eTransferSrc,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                  vk::MemoryPropertyFlagBits::eHostCoherent);


  // copy data to staging buffer
  void* data = m_device.mapMemory(staging_buffer_memory, 0, buffer_size);

  memcpy(data, m_vertices.data(), (size_t) buffer_size);

  m_device.unmapMemory(staging_buffer_memory);


  // create vertex buffer
  std::tie(m_vertex_buffer, m_vertex_buffer_memory) =
    create_buffer(
      buffer_size,
      vk::BufferUsageFlagBits::eVertexBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
      vk::MemoryPropertyFlagBits::eDeviceLocal);

  copy_buffer(staging_buffer, m_vertex_buffer, buffer_size);

  m_device.destroy(staging_buffer);
  m_device.free(staging_buffer_memory);
}

void renderer::create_index_buffers() {
  vk::DeviceSize buffer_size = sizeof(m_indices[0]) * m_indices.size();

  // create staging buffer and memory
  auto [staging_buffer, staging_buffer_memory] =
    create_buffer(buffer_size,
                  vk::BufferUsageFlagBits::eTransferSrc,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                  vk::MemoryPropertyFlagBits::eHostCoherent);


  // copy data to staging buffer
  void* data = m_device.mapMemory(staging_buffer_memory, 0, buffer_size);

  memcpy(data, m_indices.data(), (size_t) buffer_size);

  m_device.unmapMemory(staging_buffer_memory);


  // create vertex buffer
  std::tie(m_index_buffer, m_index_buffer_memory) =
    create_buffer(
      buffer_size,
      vk::BufferUsageFlagBits::eIndexBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
      vk::MemoryPropertyFlagBits::eDeviceLocal);

  copy_buffer(staging_buffer, m_index_buffer, buffer_size);

  m_device.destroy(staging_buffer);
  m_device.free(staging_buffer_memory);
}

void renderer::create_objects_uniform_buffer() {
  // for model matrices and object dependent information.
  
  // Uniform buffer alignment
  size_t min_ubo_alignment = m_physical_device.getProperties()
    .limits.minUniformBufferOffsetAlignment;

  m_dynamic_alignment = sizeof(glm::mat4); // TODO: rm mat4

  if (min_ubo_alignment > 0) {
    m_dynamic_alignment = (m_dynamic_alignment + min_ubo_alignment - 1) &
      ~(min_ubo_alignment - 1);
  }

  vk::DeviceSize buffer_size = m_num_objects * min_ubo_alignment;

  // TODO: change the glm::mat4 to something not hardcoded
  //vk::DeviceSize buffer_size = sizeof(glm::mat4) * m_num_objects;

  m_object_uniform_buffers.resize(MAX_FRAMES_IN_FLIGHT);
  m_object_uniform_buffers_memory.resize(MAX_FRAMES_IN_FLIGHT);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    std::tie(m_object_uniform_buffers[i], m_object_uniform_buffers_memory[i]) =
      create_buffer(buffer_size,  // TODO:change: sizeof(model) * num_objs prob
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);
  }
}

void renderer::create_uniform_buffers() {
  vk::DeviceSize buffer_size = sizeof(camera_uniform);

  m_camera_uniform_buffers.resize(MAX_FRAMES_IN_FLIGHT);
  m_camera_uniform_buffers_memory.resize(MAX_FRAMES_IN_FLIGHT);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    std::tie(m_camera_uniform_buffers[i], m_camera_uniform_buffers_memory[i]) =
      create_buffer(buffer_size,
                    vk::BufferUsageFlagBits::eUniformBuffer,
                    vk::MemoryPropertyFlagBits::eHostVisible |
                    vk::MemoryPropertyFlagBits::eHostCoherent);

  }
}

void renderer::create_descriptor_set_layout() {
  vk::DescriptorSetLayoutBinding ubo_layout_binding{};
  ubo_layout_binding.binding = 0;
  ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBuffer;
  ubo_layout_binding.descriptorCount = 1;
  ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;
  ubo_layout_binding.pImmutableSamplers = nullptr;

  vk::DescriptorSetLayoutBinding sampler_layout_binding{};
  sampler_layout_binding.binding = 1;
  sampler_layout_binding.descriptorType =
    vk::DescriptorType::eCombinedImageSampler;
  sampler_layout_binding.descriptorCount = 1;
  sampler_layout_binding.stageFlags = vk::ShaderStageFlagBits::eFragment;
  sampler_layout_binding.pImmutableSamplers = nullptr;

  vk::DescriptorSetLayoutBinding object_ubo_layout_binding{};
  object_ubo_layout_binding.binding = 2;
  object_ubo_layout_binding.descriptorType = vk::DescriptorType::eUniformBufferDynamic;
  object_ubo_layout_binding.descriptorCount = 1;
  object_ubo_layout_binding.stageFlags = vk::ShaderStageFlagBits::eVertex;
  object_ubo_layout_binding.pImmutableSamplers = nullptr;

  std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
    ubo_layout_binding,
    sampler_layout_binding,
    object_ubo_layout_binding
  };

  vk::DescriptorSetLayoutCreateInfo layout_info{};
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  m_descriptor_set_layout = m_device.createDescriptorSetLayout(layout_info);
}

void renderer::create_descriptor_pool() {
  std::array<vk::DescriptorPoolSize, 3> pool_sizes{};

  // camera proj view
  pool_sizes[0].type = vk::DescriptorType::eUniformBuffer;
  pool_sizes[0].descriptorCount =
    static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  pool_sizes[1].type = vk::DescriptorType::eCombinedImageSampler;
  pool_sizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  // per object model mat
  pool_sizes[2].type = vk::DescriptorType::eUniformBufferDynamic;
  pool_sizes[2].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  vk::DescriptorPoolCreateInfo pool_info{};
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

  m_descriptor_pool = m_device.createDescriptorPool(pool_info);
}

void renderer::create_descriptor_sets() {
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                               m_descriptor_set_layout);

  vk::DescriptorSetAllocateInfo alloc_info{};
  alloc_info.descriptorPool = m_descriptor_pool;
  alloc_info.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
  alloc_info.pSetLayouts = layouts.data();

  m_descriptor_sets = m_device.allocateDescriptorSets(alloc_info);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo buffer_info{};
    buffer_info.buffer = m_camera_uniform_buffers[i];
    buffer_info.offset = 0;
    buffer_info.range = sizeof(camera_uniform);

    vk::DescriptorImageInfo image_info{};
    image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    image_info.imageView = m_texture_image_view;
    image_info.sampler = m_texture_sampler;

    vk::DescriptorBufferInfo objects_buffer_info{};
    objects_buffer_info.buffer = m_object_uniform_buffers[i];
    objects_buffer_info.offset = 0;
    objects_buffer_info.range = sizeof(glm::mat4);

    std::array<vk::WriteDescriptorSet, 3> descriptor_writes{};
    descriptor_writes[0].dstSet = m_descriptor_sets[i];
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].dstArrayElement = 0;
    descriptor_writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].pBufferInfo = &buffer_info;

    descriptor_writes[1].dstSet = m_descriptor_sets[i];
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].dstArrayElement = 0;
    descriptor_writes[1].descriptorType =
      vk::DescriptorType::eCombinedImageSampler;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].pImageInfo = &image_info;

    descriptor_writes[2].dstSet = m_descriptor_sets[i];
    descriptor_writes[2].dstBinding = 2;
    descriptor_writes[2].dstArrayElement = 0;
    descriptor_writes[2].descriptorType =
      vk::DescriptorType::eUniformBufferDynamic;
    descriptor_writes[2].descriptorCount = 1;
    descriptor_writes[2].pBufferInfo = &objects_buffer_info;

    m_device.updateDescriptorSets(descriptor_writes, {});
  }
}

void renderer::create_command_buffers() {
  vk::CommandBufferAllocateInfo alloc_info{};
  alloc_info.commandPool = m_command_pool;
  alloc_info.level = vk::CommandBufferLevel::ePrimary;
  alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

  m_command_buffers = m_device.allocateCommandBuffers(alloc_info);
}

void renderer::create_command_pools() {
  vk::CommandPoolCreateInfo pool_info{};
  pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  pool_info.queueFamilyIndex = m_queue_family_index.value();

  m_command_pool = m_device.createCommandPool(pool_info);

  vk::CommandPoolCreateInfo alloc_pool_info{};
  alloc_pool_info.flags = vk::CommandPoolCreateFlagBits::eTransient;
  alloc_pool_info.queueFamilyIndex = m_queue_family_index.value();

  m_alloc_command_pool = m_device.createCommandPool(alloc_pool_info);
}

void renderer::create_renderpass() {
  vk::AttachmentDescription color_attachment_desc{};
  color_attachment_desc.format = m_swapchain_image_format;
  color_attachment_desc.samples = m_msaa_samples;
  color_attachment_desc.loadOp = vk::AttachmentLoadOp::eClear;
  color_attachment_desc.storeOp = vk::AttachmentStoreOp::eStore;
  color_attachment_desc.initialLayout = vk::ImageLayout::eUndefined;
  color_attachment_desc.finalLayout =
    vk::ImageLayout::eColorAttachmentOptimal;

  vk::AttachmentReference color_attachment_ref{};
  color_attachment_ref.attachment = 0;
  color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;


  vk::AttachmentDescription depth_attachment_desc{};
  depth_attachment_desc.format = find_depth_format();
  depth_attachment_desc.samples = m_msaa_samples;
  depth_attachment_desc.loadOp = vk::AttachmentLoadOp::eClear;
  depth_attachment_desc.storeOp = vk::AttachmentStoreOp::eDontCare;
  depth_attachment_desc.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
  depth_attachment_desc.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
  depth_attachment_desc.initialLayout = vk::ImageLayout::eUndefined;
  depth_attachment_desc.finalLayout =
    vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::AttachmentReference depth_attachment_ref{};
  depth_attachment_ref.attachment = 1;
  depth_attachment_ref.layout =
    vk::ImageLayout::eDepthStencilAttachmentOptimal;

  vk::AttachmentDescription color_attachment_resolve_desc{};
  color_attachment_resolve_desc.format = m_swapchain_image_format;
  color_attachment_resolve_desc.samples = vk::SampleCountFlagBits::e1;
  color_attachment_resolve_desc.loadOp = vk::AttachmentLoadOp::eDontCare;
  color_attachment_resolve_desc.storeOp = vk::AttachmentStoreOp::eStore;
  color_attachment_resolve_desc.stencilLoadOp =
    vk::AttachmentLoadOp::eDontCare;
  color_attachment_resolve_desc.stencilLoadOp =
    vk::AttachmentLoadOp::eDontCare;
  color_attachment_resolve_desc.initialLayout = vk::ImageLayout::eUndefined;
  color_attachment_resolve_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;

  vk::AttachmentReference color_attachment_resolve_ref{};
  color_attachment_resolve_ref.attachment = 2;
  color_attachment_resolve_ref.layout =
    vk::ImageLayout::eColorAttachmentOptimal;

  vk::SubpassDescription subpass{};
  subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
  subpass.colorAttachmentCount = 1;
  subpass.pColorAttachments = &color_attachment_ref;
  subpass.pDepthStencilAttachment = &depth_attachment_ref;
  subpass.pResolveAttachments = &color_attachment_resolve_ref;

  vk::SubpassDependency dependency{};
  dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
  dependency.dstSubpass = 0;
  dependency.srcStageMask = // FIGURE: are these masks what we will use in this specific subpass?
    vk::PipelineStageFlagBits::eColorAttachmentOutput |
    vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.srcAccessMask = vk::AccessFlagBits::eNone;
  dependency.dstStageMask =
    vk::PipelineStageFlagBits::eColorAttachmentOutput |
    vk::PipelineStageFlagBits::eEarlyFragmentTests;
  dependency.dstAccessMask =
    vk::AccessFlagBits::eColorAttachmentWrite |
    vk::AccessFlagBits::eDepthStencilAttachmentWrite;

  std::array<vk::AttachmentDescription, 3> attachments = {
    color_attachment_desc,
    depth_attachment_desc,
    color_attachment_resolve_desc,
  };

  vk::RenderPassCreateInfo renderpass_info{};
  renderpass_info.attachmentCount = attachments.size();
  renderpass_info.pAttachments = attachments.data();
  renderpass_info.subpassCount = 1;
  renderpass_info.pSubpasses = &subpass;
  renderpass_info.dependencyCount = 1;
  renderpass_info.pDependencies = &dependency;

  m_renderpass = m_device.createRenderPass(renderpass_info);
}

void renderer::create_graphics_pipeline() {
  // create fragment shader module
  vk::ShaderModuleCreateInfo frag_module_info{};
  frag_module_info.codeSize = basic_frag_spv_len;
  frag_module_info.pCode = (const uint32_t *)basic_frag_spv;

  vk::ShaderModule frag_module = m_device.createShaderModule(frag_module_info);

  // create vertex shader module
  vk::ShaderModuleCreateInfo vert_module_info{};
  vert_module_info.codeSize = basic_vert_spv_len;
  vert_module_info.pCode = (const uint32_t *)basic_vert_spv;

  vk::ShaderModule vert_module = m_device.createShaderModule(vert_module_info);

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
  auto binding_desc = vertex::get_binding_description();
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
  multisample_info.sampleShadingEnable = true;
  multisample_info.rasterizationSamples = m_msaa_samples;
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
  pipeline_layout_info.pSetLayouts = &m_descriptor_set_layout;

  m_pipeline_layout = m_device.createPipelineLayout(pipeline_layout_info);

  vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{};
  depth_stencil_info.depthTestEnable = VK_TRUE;
  depth_stencil_info.depthWriteEnable = VK_TRUE;
  depth_stencil_info.depthCompareOp = vk::CompareOp::eLess;
  depth_stencil_info.depthBoundsTestEnable = VK_FALSE;
  depth_stencil_info.stencilTestEnable = VK_FALSE;

  vk::GraphicsPipelineCreateInfo pipeline_info{};
  pipeline_info.stageCount = 2;
  pipeline_info.pStages = shader_stages;

  pipeline_info.pVertexInputState = &vert_input_info;
  pipeline_info.pInputAssemblyState = &input_assembly;
  pipeline_info.pViewportState = &viewport_state_info;
  pipeline_info.pRasterizationState = &rasterizer_info;
  pipeline_info.pMultisampleState = &multisample_info;
  pipeline_info.pDepthStencilState = &depth_stencil_info;
  pipeline_info.pColorBlendState = &color_blend_info;
  pipeline_info.pDynamicState = &dynamic_state_info;

  pipeline_info.layout = m_pipeline_layout;
  pipeline_info.renderPass = m_renderpass;
  pipeline_info.subpass = 0;

  pipeline_info.basePipelineHandle = nullptr;
  pipeline_info.basePipelineIndex = -1;

  m_graphics_pipeline =
    m_device.createGraphicsPipeline(nullptr, pipeline_info).value;

  m_device.destroy(frag_module);
  m_device.destroy(vert_module);
}

void renderer::create_swapchain() {
  //fprintf(stderr, "Creating swapchain\n");
  auto surface_capabilities =
    m_physical_device.getSurfaceCapabilitiesKHR(m_surface);
  auto surface_formats = m_physical_device.getSurfaceFormatsKHR(m_surface);
  auto surface_present_modes =
    m_physical_device.getSurfacePresentModesKHR(m_surface);

  // choose format, try to find SRGB, if failed choose the first available
  vk::SurfaceFormatKHR chosen_format = surface_formats[0];
  for (auto &format : surface_formats) {
    if (format.format == vk::Format::eB8G8R8A8Srgb &&
      format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
      chosen_format = format;
    }
  }

  m_swapchain_image_format = chosen_format.format;

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
    SDL_GL_GetDrawableSize(m_window, (int *)&chosen_extent.width,
                           (int *)&chosen_extent.height);

    chosen_extent.width = std::clamp(
        chosen_extent.width, surface_capabilities.minImageExtent.width,
        surface_capabilities.maxImageExtent.width);

    chosen_extent.height = std::clamp(
        chosen_extent.height, surface_capabilities.minImageExtent.height,
        surface_capabilities.maxImageExtent.height);
  }

  m_swapchain_image_extent = chosen_extent;

  // find the minimum image count + 1 and smaller than maximum image count
  uint32_t swapchain_image_count = surface_capabilities.minImageCount + 1;
  if (surface_capabilities.maxImageCount > 0 &&
    swapchain_image_count > surface_capabilities.maxImageCount) {
    swapchain_image_count = surface_capabilities.maxImageCount;
  }

  vk::SwapchainCreateInfoKHR swapchain_create_info{};
  swapchain_create_info.sType = vk::StructureType::eSwapchainCreateInfoKHR;
  swapchain_create_info.surface = m_surface;

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
  m_swapchain = m_device.createSwapchainKHR(swapchain_create_info);

  // save the images
  m_swapchain_images = m_device.getSwapchainImagesKHR(m_swapchain);

  // create the image views
  m_swapchain_image_views.clear();
  for (auto &image : m_swapchain_images) {
    m_swapchain_image_views
      .push_back(create_image_view(image,
                                   m_swapchain_image_format,
                                   vk::ImageAspectFlagBits::eColor,
                                   1));
  }

  //fprintf(stderr, "Created swapchain\n");
}

void renderer::create_framebuffers() {
  m_swapchain_framebuffers.clear();

  for (int i = 0; i < m_swapchain_image_views.size(); i++) {
    std::array<vk::ImageView, 3> attachments = {
      m_color_image_view,
      m_depth_image_view,
      m_swapchain_image_views[i],
    };

    vk::FramebufferCreateInfo framebuffer_info{};
    framebuffer_info.renderPass = m_renderpass;
    framebuffer_info.attachmentCount = attachments.size();
    framebuffer_info.pAttachments = attachments.data();
    framebuffer_info.width = m_swapchain_image_extent.width;
    framebuffer_info.height = m_swapchain_image_extent.height;
    framebuffer_info.layers = 1;

    m_swapchain_framebuffers.push_back(
        m_device.createFramebuffer(framebuffer_info));
  }
}

void renderer::cleanup_swapchain() {
  m_device.destroy(m_color_image_view);
  m_device.destroy(m_color_image);
  m_device.free(m_color_image_memory);

  m_device.destroy(m_depth_image_view);
  m_device.destroy(m_depth_image);
  m_device.free(m_depth_image_memory);

  for (auto &fb : m_swapchain_framebuffers) {
    m_device.destroy(fb);
  }

  for (auto &e : m_swapchain_image_views) {
    m_device.destroy(e);
  }

  m_device.destroy(m_swapchain);
}

void renderer::create_depth_resources() {
  vk::Format depth_format = find_depth_format();

  std::tie(m_depth_image, m_depth_image_memory) =
    create_image(m_swapchain_image_extent.width,
                 m_swapchain_image_extent.height,
                 1,
                 m_msaa_samples,
                 depth_format,
                 vk::ImageTiling::eOptimal,
                 vk::ImageUsageFlagBits::eDepthStencilAttachment,
                 vk::MemoryPropertyFlagBits::eDeviceLocal);

  m_depth_image_view = create_image_view(m_depth_image,
                                       depth_format,
                                       vk::ImageAspectFlagBits::eDepth,
                                       1);

  // here we don't need to transition the layout from undefined to depth
  // optimal because the renderpass will do that for us
}

void renderer::create_color_resources() {
  vk::Format color_format = m_swapchain_image_format;

  std::tie(m_color_image, m_color_image_memory) =
    create_image(m_swapchain_image_extent.width,
                 m_swapchain_image_extent.height, 
                 1, 
                 m_msaa_samples, 
                 color_format, 
                 vk::ImageTiling::eOptimal, 
                 vk::ImageUsageFlagBits::eTransientAttachment |
                 vk::ImageUsageFlagBits::eColorAttachment, 
                 vk::MemoryPropertyFlagBits::eDeviceLocal);

  m_color_image_view = create_image_view(m_color_image,
                                       color_format,
                                       vk::ImageAspectFlagBits::eColor,
                                       1);
}

void renderer::recreate_swapchain() {
  m_device.waitIdle();

  cleanup_swapchain();

  create_swapchain();

  create_color_resources();
  create_depth_resources();

  create_framebuffers();
}

void renderer::create_instance() {
  vk::ApplicationInfo application_info("phys sim");

  // validation layer work
  std::vector<const char *> validation_layers = {
    "VK_LAYER_KHRONOS_validation",
    "VK_LAYER_KHRONOS_validation"
  };

  // SDL extensions
  unsigned int extensions_count = 0;

  CHECK_SDL(
      SDL_Vulkan_GetInstanceExtensions(m_window, &extensions_count, nullptr),
      != SDL_TRUE);

  std::vector<char *> extensions(extensions_count);

  CHECK_SDL(SDL_Vulkan_GetInstanceExtensions(
              m_window, &extensions_count,
              const_cast<const char **>(extensions.data())),
            != SDL_TRUE);

  vk::InstanceCreateInfo instance_create_info(
      {}, &application_info, validation_layers.size(),
      validation_layers.data(), extensions_count, extensions.data());

  m_instance = vk::createInstance(instance_create_info);
}

vk::SampleCountFlagBits renderer::get_max_usable_sample_count() {
  auto physical_device_properties = m_physical_device.getProperties();

  auto count =
    physical_device_properties.limits.framebufferColorSampleCounts &
    physical_device_properties.limits.framebufferDepthSampleCounts;

  if (count & vk::SampleCountFlagBits::e64)
    return vk::SampleCountFlagBits::e64;

  if (count & vk::SampleCountFlagBits::e32)
    return vk::SampleCountFlagBits::e32;

  if (count & vk::SampleCountFlagBits::e16)
    return vk::SampleCountFlagBits::e16;

  if (count & vk::SampleCountFlagBits::e8)
    return vk::SampleCountFlagBits::e8;

  if (count & vk::SampleCountFlagBits::e4)
    return vk::SampleCountFlagBits::e4;

  if (count & vk::SampleCountFlagBits::e2)
    return vk::SampleCountFlagBits::e2;

  return vk::SampleCountFlagBits::e1;
}

void renderer::create_physical_device() {

  m_physical_device = nullptr;
  int max_mem = 0;

  printf("Analyzing physical devices\n");
  // find device with most memory
  for (auto &p_dev : m_instance.enumeratePhysicalDevices()) {
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
      m_physical_device = p_dev;
    }
  }

  m_msaa_samples = get_max_usable_sample_count();

  printf("Chose device named '%s'\n",
         m_physical_device.getProperties().deviceName.data());
}

void renderer::create_device_and_queues() {
  CHECK_SDL(SDL_Vulkan_CreateSurface(
              m_window, m_instance, reinterpret_cast<VkSurfaceKHR *>(&m_surface)),
            != SDL_TRUE);

  // Choose a cool queue family with at least graphics
  auto queue_family_properties = m_physical_device.getQueueFamilyProperties();

  for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
    if (queue_family_properties[i].queueFlags &
      vk::QueueFlagBits::eGraphics &&
      m_physical_device.getSurfaceSupportKHR(i, m_surface)) {

      // get first family that has present and graphics suppoort
      m_queue_family_index = i;
      break;
    }
  }

  float queue_priority = 1.0f;

  vk::DeviceQueueCreateInfo device_queue_create_info(
      {}, m_queue_family_index.value(), 1, &queue_priority);

  vk::PhysicalDeviceFeatures physical_device_features{};
  physical_device_features.samplerAnisotropy = VK_TRUE;
  physical_device_features.sampleRateShading = VK_TRUE;

  std::vector<const char *> required_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  vk::DeviceCreateInfo device_create_info(
      {}, 1, &device_queue_create_info, 0, nullptr,
      required_device_extensions.size(), required_device_extensions.data(),
      &physical_device_features);

  m_device = m_physical_device.createDevice(device_create_info);

  m_queue = m_device.getQueue(m_queue_family_index.value(), 0);
}

std::pair<vk::Image, vk::DeviceMemory>
renderer::create_image(uint32_t width,
                       uint32_t height,
                       uint32_t mip_levels,
                       vk::SampleCountFlagBits num_samples,
                       vk::Format format,
                       vk::ImageTiling tiling,
                       vk::ImageUsageFlags usage,
                       vk::MemoryPropertyFlags properties) {
  vk::Image image;
  vk::DeviceMemory image_memory;

  vk::ImageCreateInfo image_info{};
  image_info.imageType = vk::ImageType::e2D;
  image_info.extent.width = width;
  image_info.extent.height = height;
  image_info.extent.depth = 1;
  image_info.mipLevels = mip_levels;
  image_info.arrayLayers = 1;
  image_info.format = format;
  image_info.tiling = tiling;
  image_info.initialLayout = vk::ImageLayout::eUndefined;
  image_info.usage = usage;
  image_info.sharingMode = vk::SharingMode::eExclusive;
  image_info.samples = num_samples;

  image = m_device.createImage(image_info);

  vk::MemoryRequirements mem_requirements =
    m_device.getImageMemoryRequirements(image);

  vk::MemoryAllocateInfo alloc_info{};
  alloc_info.allocationSize = mem_requirements.size;
  alloc_info.memoryTypeIndex =
    find_memory_type(mem_requirements.memoryTypeBits,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

  image_memory = m_device.allocateMemory(alloc_info);

  m_device.bindImageMemory(image, image_memory, 0);

  return {image, image_memory};
}

void renderer::generate_mipmaps(vk::Image image, vk::Format image_format,
                                int32_t tex_width, int32_t tex_height,
                                uint32_t mip_levels ) {

  // check properties for linear bliting support 
  auto format_properties = m_physical_device.getFormatProperties(image_format);

  if (!(format_properties.optimalTilingFeatures &
    vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {

    fprintf(stderr,
            "ERROR Physical device don't support blit filter linear\n");
  }

  auto command_buffer = begin_single_time_commands();

  vk::ImageMemoryBarrier barrier{};
  barrier.image = image;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount = 1;
  barrier.subresourceRange.levelCount = 1;

  int32_t mip_width = tex_width;
  int32_t mip_height = tex_height;

  for (int i = 1; i < mip_levels; i++) {
    barrier.subresourceRange.baseMipLevel = i - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eTransfer,
                                   static_cast<vk::DependencyFlags>(0),
                                   {},
                                   {},
                                   {barrier});

    vk::ImageBlit blit{};
    blit.srcOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.srcOffsets[1] = vk::Offset3D{mip_width, mip_height, 1};
    blit.srcSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blit.srcSubresource.mipLevel = i - 1;
    blit.srcSubresource.baseArrayLayer = 0;
    blit.srcSubresource.layerCount = 1;

    blit.dstOffsets[0] = vk::Offset3D{0, 0, 0};
    blit.dstOffsets[1] = vk::Offset3D{
      mip_width > 1 ? mip_width / 2 : 1,
      mip_height > 1 ? mip_height / 2: 1,
      1
    };
    blit.dstSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    blit.dstSubresource.mipLevel = i;
    blit.dstSubresource.baseArrayLayer = 0;
    blit.dstSubresource.layerCount = 1;

    command_buffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                             image, vk::ImageLayout::eTransferDstOptimal,
                             {blit}, vk::Filter::eLinear);

    barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                   vk::PipelineStageFlagBits::eFragmentShader,
                                   static_cast<vk::DependencyFlags>(0),
                                   {}, {}, {barrier});

    if (mip_width > 1) mip_width /= 2;
    if (mip_height > 1) mip_height /= 2;

  }

  barrier.subresourceRange.baseMipLevel = mip_levels - 1;
  barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
  barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
  barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

  command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eFragmentShader,
                                 static_cast<vk::DependencyFlags>(0),
                                 {}, {}, {barrier});


  end_single_time_commands(command_buffer);

  fprintf(stderr, "LOG Created mipmaps\n");
}

void renderer::create_texture_image() {
  int tex_width, tex_height, tex_channels;

  stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &tex_width, &tex_height,
                              &tex_channels, STBI_rgb_alpha);

  if (!pixels) {
    fprintf(stderr, "ERROR loading texture '%s': %s\n",
            TEXTURE_PATH.c_str(),
            stbi_failure_reason());
    exit(1);
  }

  vk::DeviceSize image_size = tex_width * tex_height * 4;
  m_mip_levels = static_cast<uint32_t>(
      std::floor(std::log2(std::max(tex_width, tex_height)))) + 1;

  auto [staging_buffer, staging_buffer_memory] =
    create_buffer(image_size,
                  vk::BufferUsageFlagBits::eTransferSrc,
                  vk::MemoryPropertyFlagBits::eHostVisible |
                  vk::MemoryPropertyFlagBits::eHostCoherent);

  void *data = m_device.mapMemory(staging_buffer_memory, 0, image_size);

  memcpy(data, pixels, static_cast<size_t>(image_size));

  m_device.unmapMemory(staging_buffer_memory);

  stbi_image_free(pixels);

  std::tie(m_texture_image, m_texture_image_memory)  =
    create_image(tex_width, tex_height, m_mip_levels,
                 vk::SampleCountFlagBits::e1,
                 vk::Format::eR8G8B8A8Srgb,
                 vk::ImageTiling::eOptimal,
                 vk::ImageUsageFlagBits::eTransferDst |
                 vk::ImageUsageFlagBits::eTransferSrc |
                 vk::ImageUsageFlagBits::eSampled,
                 vk::MemoryPropertyFlagBits::eDeviceLocal);

  transition_image_layout(m_texture_image,
                          vk::Format::eR8G8B8A8Srgb,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal,
                          m_mip_levels);

  copy_buffer_to_image(staging_buffer,
                       m_texture_image,
                       static_cast<uint32_t>(tex_width),
                       static_cast<uint32_t>(tex_height));

  // Will transition to eShaderReadOnlyOptimal while generation mipmaps.
  generate_mipmaps(m_texture_image,
                   vk::Format::eR8G8B8A8Srgb,
                   tex_width,
                   tex_height,
                   m_mip_levels);

  m_device.destroy(staging_buffer);

  m_device.free(staging_buffer_memory);
}

vk::ImageView renderer::create_image_view(vk::Image image,
                                          vk::Format format,
                                          vk::ImageAspectFlags aspect_flags,
                                          uint32_t mip_levels) {
  vk::ImageViewCreateInfo view_info{};
  view_info.image = image;
  view_info.viewType = vk::ImageViewType::e2D;
  view_info.format = format;
  view_info.subresourceRange.aspectMask = aspect_flags;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.levelCount = mip_levels;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount = 1;
  view_info.components = vk::ComponentSwizzle::eIdentity;

  return m_device.createImageView(view_info);
}

void renderer::create_texture_image_view() {
  m_texture_image_view = create_image_view(m_texture_image,
                                         vk::Format::eR8G8B8A8Srgb,
                                         vk::ImageAspectFlagBits::eColor,
                                         m_mip_levels);
}

void renderer::create_texture_sampler() {
  vk::SamplerCreateInfo sampler_info{};
  sampler_info.magFilter = vk::Filter::eLinear;
  sampler_info.minFilter = vk::Filter::eLinear;
  sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
  sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
  sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;

  auto properties = m_physical_device.getProperties();

  sampler_info.anisotropyEnable = VK_TRUE;
  sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;  
  sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = vk::CompareOp::eAlways;
  sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
  sampler_info.mipLodBias = 0.0f;
  sampler_info.minLod = 0.0f;
  sampler_info.maxLod = static_cast<float>(m_mip_levels);

  m_texture_sampler = m_device.createSampler(sampler_info);
}

vk::Format renderer::find_supported_format(const std::vector<vk::Format>& candidates,
                                           vk::ImageTiling tiling,
                                           vk::FormatFeatureFlags features) {

  for (auto format : candidates) {
    auto props = m_physical_device.getFormatProperties(format);

    if (tiling == vk::ImageTiling::eLinear &&
      (props.linearTilingFeatures & features) == features) {
      return format;
    }

    if (tiling == vk::ImageTiling::eOptimal &&
      (props.optimalTilingFeatures & features) == features) {
      return format;
    }
  }

  fprintf(stderr, "ERROR: could not find supported format!\n");
  exit(1);
}

vk::Format renderer::find_depth_format() {
  return find_supported_format(
    { vk::Format::eD32Sfloat,
      vk::Format::eD32SfloatS8Uint,
      vk::Format::eD24UnormS8Uint }, 
    vk::ImageTiling::eOptimal, 
    vk::FormatFeatureFlagBits::eDepthStencilAttachment);
}

bool renderer::has_stencil_component(vk::Format format) {
  return format == vk::Format::eD32SfloatS8Uint ||
  format == vk::Format::eD24UnormS8Uint;
}

void renderer::load_model() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, error;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &error,
                        MODEL_PATH.c_str())) {

    fprintf(stderr, "ERROR loading obj: %s\n", error.c_str());
    exit(1);

  }

  fprintf(stderr, "WARNING from loading obj: %s\n", warn.c_str());

  std::unordered_map<vertex, uint32_t> unique_vertices{};

  for (const auto& shape : shapes) {
    for (const auto& index : shape.mesh.indices) {
      vertex v{};

      v.pos = {
        attrib.vertices[3 * index.vertex_index + 0],
        attrib.vertices[3 * index.vertex_index + 1],
        attrib.vertices[3 * index.vertex_index + 2],
      };

      v.tex_coord = {
        attrib.texcoords[2 * index.texcoord_index + 0],
        1.0f - attrib.texcoords[2 * index.texcoord_index + 1],
      };

      v.color = { 1.0f, 1.0f, 1.0f };

      if (unique_vertices.count(v) == 0) {
        unique_vertices[v] = static_cast<uint32_t>(m_vertices.size());
        m_vertices.push_back(v);
      }

      m_indices.push_back(unique_vertices[v]);
    }
  }

  fprintf(stderr,
          "LOG vertices size: %zu bytes, indices size: %zu bytes\n",
          m_vertices.size() * sizeof(m_vertices[0]),
          m_indices.size() * sizeof(m_indices[0]));

}

void renderer::init_vulkan() {
  CHECK_SDL(SDL_Init(SDL_INIT_VIDEO), != 0);

  // create window
  m_window = SDL_CreateWindow(
      "phys-sim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      m_window_dimensions.width, m_window_dimensions.height,
      SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

  CHECK_SDL(m_window, == 0);

  // instance creation
  create_instance();

  create_physical_device();

  create_device_and_queues();

  // swapchain
  create_swapchain();

  create_renderpass();

  create_descriptor_set_layout();

  create_graphics_pipeline();

  create_color_resources();

  create_depth_resources();

  create_framebuffers();

  create_command_pools();

  create_texture_image();

  create_texture_image_view();

  create_texture_sampler();

  load_model();

  create_vertex_buffers();

  create_index_buffers();

  create_objects_uniform_buffer();

  create_uniform_buffers();

  create_descriptor_pool();

  create_descriptor_sets();

  create_command_buffers();

  create_sync_objects();

  auto properties = m_physical_device.getProperties();
  printf("min alignment uniform buffer: %lu\n",
         properties.limits.minUniformBufferOffsetAlignment);
}

void renderer::cleanup() {
  m_device.waitIdle();

  cleanup_swapchain();

  m_device.destroy(m_texture_sampler);
  m_device.destroy(m_texture_image_view);

  m_device.destroy(m_texture_image);
  m_device.free(m_texture_image_memory);

  m_device.destroy(m_vertex_buffer);
  m_device.free(m_vertex_buffer_memory);

  m_device.destroy(m_index_buffer);
  m_device.free(m_index_buffer_memory);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    m_device.destroy(m_camera_uniform_buffers[i]);
    m_device.free(m_camera_uniform_buffers_memory[i]);

    m_device.destroy(m_object_uniform_buffers[i]);
    m_device.free(m_object_uniform_buffers_memory[i]);
  }

  m_device.destroy(m_descriptor_pool);

  m_device.destroy(m_descriptor_set_layout);

  for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    m_device.destroy(m_image_available_semaphores[i]);
    m_device.destroy(m_render_finished_semaphores[i]);
    m_device.destroy(m_in_flight_fences[i]);
  }

  m_device.destroy(m_command_pool);
  m_device.destroy(m_alloc_command_pool);

  m_device.destroy(m_renderpass);
  m_device.destroy(m_graphics_pipeline);
  m_device.destroy(m_pipeline_layout);

  m_device.destroy();

  m_instance.destroy(m_surface);
  m_instance.destroy();
  SDL_DestroyWindow(m_window);
}

size_t renderer::get_uniform_alignment() const {
  return m_dynamic_alignment;
}

uint8_t* renderer::map_object_uniform() {
  void *data = m_device.mapMemory(m_object_uniform_buffers_memory[m_current_frame],
                          0,
                          sizeof(glm::mat4) * m_num_objects);

  return static_cast<uint8_t*>(data);
}

void renderer::unmap_object_uniform() {
  m_device.unmapMemory(m_object_uniform_buffers_memory[m_current_frame]);
}

uint8_t* renderer::map_camera_uniform() {
  void *data = m_device.mapMemory(m_camera_uniform_buffers_memory[m_current_frame],
                          0,
                          sizeof(camera_uniform));

  return static_cast<uint8_t*>(data);
}

void renderer::unmap_camera_uniform() {
  m_device.unmapMemory(m_camera_uniform_buffers_memory[m_current_frame]);
}

void renderer::init() { init_vulkan(); }

void renderer::draw_frame() {
  // wait for in flight fences (whatever that means...)
  if (m_device.waitForFences({m_in_flight_fences[m_current_frame]}, true,
                           UINT64_MAX) != vk::Result::eSuccess) {
    printf("wait for fence failed ... wtf\n");
    exit(123);
  }

  m_device.resetFences({m_in_flight_fences[m_current_frame]});

  uint32_t next_image;
  try {
    auto next_image_result = m_device.acquireNextImageKHR(
        m_swapchain, UINT64_MAX,
        m_image_available_semaphores[m_current_frame], nullptr);

    next_image = next_image_result.value;
  } catch (vk::OutOfDateKHRError const &e) {
    // probably resized the window, need to recreate the swapchain
    recreate_swapchain();

    // draw the frame asked. Maybe not needed, but garantees at least one frame
    // is drawn.
    draw_frame();
    return;
  }

  m_command_buffers[m_current_frame].reset();
  record_command_buffer(m_command_buffers[m_current_frame], next_image);

  vk::SubmitInfo submit_info{};

  vk::Semaphore wait_semaphores[] = {
    m_image_available_semaphores[m_current_frame]};
  vk::PipelineStageFlags wait_stages[] = {
    vk::PipelineStageFlagBits::eColorAttachmentOutput};

  submit_info.waitSemaphoreCount = 1;
  submit_info.pWaitSemaphores = wait_semaphores;
  submit_info.pWaitDstStageMask = wait_stages;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &m_command_buffers[m_current_frame];

  vk::Semaphore signal_semaphores[] = {
    m_render_finished_semaphores[m_current_frame]
  };

  submit_info.signalSemaphoreCount = 1;
  submit_info.pSignalSemaphores = signal_semaphores;

  m_queue.submit({submit_info}, m_in_flight_fences[m_current_frame]);

  vk::PresentInfoKHR present_info{};
  present_info.waitSemaphoreCount = 1;
  present_info.pWaitSemaphores = signal_semaphores;

  vk::SwapchainKHR swapchains[] = {m_swapchain};
  present_info.swapchainCount = 1;
  present_info.pSwapchains = swapchains;
  present_info.pImageIndices = &next_image;
  present_info.pResults = nullptr;

  try {
    auto present_result = m_queue.presentKHR(present_info);

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

  m_current_frame =
    (m_current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

uint32_t renderer::get_width() const {
  return m_swapchain_image_extent.width;
}

uint32_t renderer::get_height() const {
  return m_swapchain_image_extent.height;
}

SDL_Window* renderer::get_window() const {
  return m_window;
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
