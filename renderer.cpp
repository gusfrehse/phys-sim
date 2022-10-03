#include "renderer.hpp"

#include <chrono>

#include "shaders.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void renderer::record_command_buffer(vk::CommandBuffer command_buffer,
                                     int image_index) {
    vk::CommandBufferBeginInfo begin_info{};

    command_buffer.begin(begin_info);

    vk::RenderPassBeginInfo renderpass_begin_info{};
    renderpass_begin_info.renderPass = renderpass;
    renderpass_begin_info.framebuffer = swapchain_framebuffers[image_index];
    renderpass_begin_info.renderArea.offset = vk::Offset2D(0, 0);
    renderpass_begin_info.renderArea.extent = swapchain_image_extent;

    std::array<vk::ClearValue, 2> clear_colors{};
    clear_colors[0].color = std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f};
    clear_colors[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

    renderpass_begin_info.clearValueCount = clear_colors.size();
    renderpass_begin_info.pClearValues = clear_colors.data();

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

    command_buffer.drawIndexed(static_cast<uint32_t>(indices.size()),
                               1, 0, 0, 0);

    command_buffer.endRenderPass();

    command_buffer.end();
}

void renderer::create_sync_objects() {
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

uint32_t renderer::find_memory_type(uint32_t type_filter,
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

std::pair<vk::Buffer, vk::DeviceMemory> renderer::create_buffer(vk::DeviceSize size,
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


    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(mem_requirements.memoryTypeBits, properties);

    buffer_memory = device.allocateMemory(alloc_info);

    device.bindBufferMemory(buffer, buffer_memory, 0);

    return {buffer, buffer_memory};
}

vk::CommandBuffer renderer::begin_single_time_commands() {
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

    return command_buffer;
}

void renderer::end_single_time_commands(vk::CommandBuffer command_buffer) {
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

void renderer::copy_buffer(vk::Buffer src, vk::Buffer dst, vk::DeviceSize size) {
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
                                       vk::ImageLayout new_layout) {

    auto command_buffer = begin_single_time_commands();

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
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
                                   static_cast<vk::DependencyFlags>(0), // this is weird hopefully doesn't break anything
                                   {},
                                   {},
                                   {barrier});

    end_single_time_commands(command_buffer);

}

void renderer::copy_buffer_to_image(vk::Buffer buffer, vk::Image image, uint32_t width,
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

void renderer::create_index_buffers() {
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

void renderer::create_uniform_buffers() {
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

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
        ubo_layout_binding,
        sampler_layout_binding,
    };

    vk::DescriptorSetLayoutCreateInfo layout_info{};
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    descriptor_set_layout = device.createDescriptorSetLayout(layout_info);
}


void renderer::create_descriptor_pool() {
    std::array<vk::DescriptorPoolSize, 2> pool_sizes{};

    pool_sizes[0].type = vk::DescriptorType::eUniformBuffer;
    pool_sizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    pool_sizes[1].type = vk::DescriptorType::eCombinedImageSampler;
    pool_sizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    vk::DescriptorPoolCreateInfo pool_info{};
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    pool_info.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

    descriptor_pool = device.createDescriptorPool(pool_info);
}

void renderer::create_descriptor_sets() {
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

        vk::DescriptorImageInfo image_info{};
        image_info.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
        image_info.imageView = texture_image_view;
        image_info.sampler = texture_sampler;

        std::array<vk::WriteDescriptorSet, 2> descriptor_writes{};
        descriptor_writes[0].dstSet = descriptor_sets[i];
        descriptor_writes[0].dstBinding = 0;
        descriptor_writes[0].dstArrayElement = 0;
        descriptor_writes[0].descriptorType = vk::DescriptorType::eUniformBuffer;
        descriptor_writes[0].descriptorCount = 1;
        descriptor_writes[0].pBufferInfo = &buffer_info;

        descriptor_writes[1].dstSet = descriptor_sets[i];
        descriptor_writes[1].dstBinding = 1;
        descriptor_writes[1].dstArrayElement = 0;
        descriptor_writes[1].descriptorType =
            vk::DescriptorType::eCombinedImageSampler;
        descriptor_writes[1].descriptorCount = 1;
        descriptor_writes[1].pImageInfo = &image_info;

        device.updateDescriptorSets(descriptor_writes, {});
    }
}

void renderer::create_command_buffers() {
    vk::CommandBufferAllocateInfo alloc_info{};
    alloc_info.commandPool = command_pool;
    alloc_info.level = vk::CommandBufferLevel::ePrimary;
    alloc_info.commandBufferCount = MAX_FRAMES_IN_FLIGHT;

    command_buffers = device.allocateCommandBuffers(alloc_info);
}

void renderer::create_command_pools() {
    vk::CommandPoolCreateInfo pool_info{};
    pool_info.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    pool_info.queueFamilyIndex = queue_family_index.value();

    command_pool = device.createCommandPool(pool_info);

    vk::CommandPoolCreateInfo alloc_pool_info{};
    alloc_pool_info.flags = vk::CommandPoolCreateFlagBits::eTransient;
    alloc_pool_info.queueFamilyIndex = queue_family_index.value();

    alloc_command_pool = device.createCommandPool(alloc_pool_info);
}

void renderer::create_clear_renderpass() {
    vk::AttachmentDescription color_attachment_desc{};
    color_attachment_desc.format = swapchain_image_format;
    color_attachment_desc.samples = vk::SampleCountFlagBits::e1;
    color_attachment_desc.loadOp = vk::AttachmentLoadOp::eClear; // change this to clear screen
    color_attachment_desc.storeOp = vk::AttachmentStoreOp::eStore;
    color_attachment_desc.initialLayout = vk::ImageLayout::eUndefined;
    color_attachment_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentDescription depth_attachment_desc{};
    depth_attachment_desc.format = find_depth_format();
    depth_attachment_desc.samples = vk::SampleCountFlagBits::e1;
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

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

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

    std::array<vk::AttachmentDescription, 2> attachments = {
        color_attachment_desc,
        depth_attachment_desc,
    };

    vk::RenderPassCreateInfo renderpass_info{};
    renderpass_info.attachmentCount = attachments.size();
    renderpass_info.pAttachments = attachments.data();
    renderpass_info.subpassCount = 1;
    renderpass_info.pSubpasses = &subpass;
    renderpass_info.dependencyCount = 1;
    renderpass_info.pDependencies = &dependency;

    clear_renderpass = device.createRenderPass(renderpass_info);
}

void renderer::create_renderpass() {
    vk::AttachmentDescription color_attachment_desc{};
    color_attachment_desc.format = swapchain_image_format;
    color_attachment_desc.samples = vk::SampleCountFlagBits::e1;
    color_attachment_desc.loadOp = vk::AttachmentLoadOp::eClear; // change this to clear screen
    color_attachment_desc.storeOp = vk::AttachmentStoreOp::eStore;
    color_attachment_desc.initialLayout = vk::ImageLayout::eUndefined;
    color_attachment_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentDescription depth_attachment_desc{};
    depth_attachment_desc.format = find_depth_format();
    depth_attachment_desc.samples = vk::SampleCountFlagBits::e1;
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

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;

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

    std::array<vk::AttachmentDescription, 2> attachments = {
        color_attachment_desc,
        depth_attachment_desc,
    };

    vk::RenderPassCreateInfo renderpass_info{};
    renderpass_info.attachmentCount = attachments.size();
    renderpass_info.pAttachments = attachments.data();
    renderpass_info.subpassCount = 1;
    renderpass_info.pSubpasses = &subpass;
    renderpass_info.dependencyCount = 1;
    renderpass_info.pDependencies = &dependency;

    renderpass = device.createRenderPass(renderpass_info);
}

void renderer::create_graphics_pipeline() {
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

void renderer::create_swapchain() {
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
        swapchain_image_views
            .push_back(create_image_view(image,
                                         swapchain_image_format,
                                         vk::ImageAspectFlagBits::eColor));
    }

    //fprintf(stderr, "Created swapchain\n");
}

void renderer::create_framebuffers() {
    swapchain_framebuffers.clear();

    for (int i = 0; i < swapchain_image_views.size(); i++) {
        std::array<vk::ImageView, 2> attachments = {
            swapchain_image_views[i],
            depth_image_view,
        };

        vk::FramebufferCreateInfo framebuffer_info{};
        framebuffer_info.renderPass = renderpass;
        framebuffer_info.attachmentCount = attachments.size();
        framebuffer_info.pAttachments = attachments.data();
        framebuffer_info.width = swapchain_image_extent.width;
        framebuffer_info.height = swapchain_image_extent.height;
        framebuffer_info.layers = 1;

        swapchain_framebuffers.push_back(
                device.createFramebuffer(framebuffer_info));
    }
}

void renderer::cleanup_swapchain() {
    device.destroy(depth_image_view);
    device.destroy(depth_image);
    device.free(depth_image_memory);

    for (auto &fb : swapchain_framebuffers) {
        device.destroy(fb);
    }

    for (auto &e : swapchain_image_views) {
        device.destroy(e);
    }

    device.destroy(swapchain);
}

void renderer::create_depth_resources() {
    vk::Format depth_format = find_depth_format();

    std::tie(depth_image, depth_image_memory) =
        create_image(swapchain_image_extent.width,
                     swapchain_image_extent.height,
                     depth_format,
                     vk::ImageTiling::eOptimal,
                     vk::ImageUsageFlagBits::eDepthStencilAttachment,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    depth_image_view = create_image_view(depth_image,
                                         depth_format,
                                         vk::ImageAspectFlagBits::eDepth);

    // here we don't need to transition the layout from undefined to depth
    // optimal because the renderpass will do that for us
}

void renderer::recreate_swapchain() {
    device.waitIdle();

    cleanup_swapchain();

    create_swapchain();

    create_depth_resources();

    create_framebuffers();
}

void renderer::create_instance() {
    vk::ApplicationInfo application_info("phys sim");

    // validation layer work
#ifdef DISABLE_LAYERS
    std::vector<const char *> validation_layers = {};
#else
    std::vector<const char *> validation_layers = {
        "VK_LAYER_KHRONOS_validation"
    };
#endif

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

void renderer::create_physical_device() {

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

void renderer::create_device_and_queues() {
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
    physical_device_features.samplerAnisotropy = VK_TRUE;

    std::vector<const char *> required_device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    vk::DeviceCreateInfo device_create_info(
            {}, 1, &device_queue_create_info, 0, nullptr,
            required_device_extensions.size(), required_device_extensions.data(),
            &physical_device_features);

    device = physical_device.createDevice(device_create_info);

    queue = device.getQueue(queue_family_index.value(), 0);
}

std::pair<vk::Image, vk::DeviceMemory>
renderer::create_image(uint32_t width,
                       uint32_t height,
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
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.format = format;
    image_info.tiling = tiling;
    image_info.initialLayout = vk::ImageLayout::eUndefined;
    image_info.usage = usage;
    image_info.sharingMode = vk::SharingMode::eExclusive;
    image_info.samples = vk::SampleCountFlagBits::e1;

    image = device.createImage(image_info);

    vk::MemoryRequirements mem_requirements =
        device.getImageMemoryRequirements(image);

    vk::MemoryAllocateInfo alloc_info{};
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(mem_requirements.memoryTypeBits,
                         vk::MemoryPropertyFlagBits::eDeviceLocal);

    image_memory = device.allocateMemory(alloc_info);

    device.bindImageMemory(image, image_memory, 0);

    return {image, image_memory};
}

void renderer::create_texture_image() {
    int tex_width, tex_height, tex_channels;
    stbi_uc* pixels = stbi_load("textures/texture.jpg", &tex_width, &tex_height,
                                &tex_channels, STBI_rgb_alpha);
    vk::DeviceSize image_size = tex_width * tex_height * 4;

    if (!pixels) {
        fprintf(stderr, "ERROR: loading texture\n");
        exit(1);
    }

    auto [staging_buffer, staging_buffer_memory] =
        create_buffer(image_size,
                      vk::BufferUsageFlagBits::eTransferSrc,
                      vk::MemoryPropertyFlagBits::eHostVisible |
                      vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = device.mapMemory(staging_buffer_memory, 0, image_size);

    memcpy(data, pixels, static_cast<size_t>(image_size));

    device.unmapMemory(staging_buffer_memory);

    stbi_image_free(pixels);

    std::tie(texture_image, texture_image_memory)  =
        create_image(tex_width, tex_height,
                     vk::Format::eR8G8B8A8Srgb,
                     vk::ImageTiling::eOptimal,
                     vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eSampled,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    transition_image_layout(texture_image,
                            vk::Format::eR8G8B8A8Srgb,
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eTransferDstOptimal);

    copy_buffer_to_image(staging_buffer,
                         texture_image,
                         static_cast<uint32_t>(tex_width),
                         static_cast<uint32_t>(tex_height));

    transition_image_layout(texture_image, 
                            vk::Format::eR8G8B8A8Srgb, 
                            vk::ImageLayout::eTransferDstOptimal, 
                            vk::ImageLayout::eShaderReadOnlyOptimal);

    device.destroy(staging_buffer);
    device.free(staging_buffer_memory);

}

vk::ImageView renderer::create_image_view(vk::Image image,
                                          vk::Format format,
                                          vk::ImageAspectFlags aspect_flags) {
    vk::ImageViewCreateInfo view_info{};
    view_info.image = image;
    view_info.viewType = vk::ImageViewType::e2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    view_info.components = vk::ComponentSwizzle::eIdentity;

    return device.createImageView(view_info);
}

void renderer::create_texture_image_view() {
    texture_image_view = create_image_view(texture_image,
                                           vk::Format::eR8G8B8A8Srgb,
                                           vk::ImageAspectFlagBits::eColor);
}

void renderer::create_texture_sampler() {
    vk::SamplerCreateInfo sampler_info{};
    sampler_info.magFilter = vk::Filter::eLinear;
    sampler_info.minFilter = vk::Filter::eLinear;
    sampler_info.addressModeU = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeV = vk::SamplerAddressMode::eRepeat;
    sampler_info.addressModeW = vk::SamplerAddressMode::eRepeat;

    auto properties = physical_device.getProperties();

    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;  
    sampler_info.borderColor = vk::BorderColor::eIntOpaqueBlack;
    sampler_info.unnormalizedCoordinates = VK_FALSE;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = vk::CompareOp::eAlways;
    sampler_info.mipmapMode = vk::SamplerMipmapMode::eLinear;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = 0.0f;

    texture_sampler = device.createSampler(sampler_info);
}

vk::Format renderer::find_supported_format(const std::vector<vk::Format>& candidates,
                                           vk::ImageTiling tiling,
                                           vk::FormatFeatureFlags features) {

    for (auto format : candidates) {
        auto props = physical_device.getFormatProperties(format);

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

bool has_stencil_component(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint ||
    format == vk::Format::eD24UnormS8Uint;
}

void renderer::init_vulkan() {
    CHECK_SDL(SDL_Init(SDL_INIT_VIDEO), != 0);

    // create window
    window = SDL_CreateWindow(
            "phys-sim", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
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

    create_depth_resources();

    create_framebuffers();

    create_command_pools();

    create_texture_image();

    create_texture_image_view();

    create_texture_sampler();

    create_vertex_buffers();

    create_index_buffers();

    create_uniform_buffers();

    create_descriptor_pool();

    create_descriptor_sets();

    create_command_buffers();

    create_sync_objects();
}

void renderer::cleanup() {
    device.waitIdle();

    cleanup_swapchain();

    device.destroy(texture_sampler);
    device.destroy(texture_image_view);

    device.destroy(texture_image);
    device.free(texture_image_memory);

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

void renderer::send_ubo(uniform_buffer_object ubo) {
    next_ubo = ubo;
}

void renderer::update_uniform_buffer(uint32_t current_image) {
    void *data;

    data = device.mapMemory(uniform_buffers_memory[current_image], 0,
                            sizeof(next_ubo));

    memcpy(data, &next_ubo, sizeof(next_ubo));

    device.unmapMemory(uniform_buffers_memory[current_image]);
}

void renderer::init() { init_vulkan(); }

void renderer::draw_frame() {
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