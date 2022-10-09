#include "vertex.hpp"

#include <vulkan/vulkan.hpp>
#include <glm/vec3.hpp>

std::array<vk::VertexInputAttributeDescription, 3>
vertex::get_attribute_descriptions() {
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


vk::VertexInputBindingDescription vertex::get_binding_description() {
  vk::VertexInputBindingDescription binding_description{};
  binding_description.binding = 0;
  binding_description.stride = sizeof(vertex);
  binding_description.inputRate = vk::VertexInputRate::eVertex;

  return binding_description;
}

bool vertex::operator==(const vertex& other) const {
  return pos == other.pos && color == other.color && tex_coord == tex_coord;
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
