#pragma once

#include <vulkan/vulkan.hpp>

#include "glm.hpp"

struct vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 tex_coord;

  static std::array<vk::VertexInputAttributeDescription, 3>
  get_attribute_descriptions();

  static vk::VertexInputBindingDescription get_binding_description();

  bool operator==(const vertex& other) const;
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
