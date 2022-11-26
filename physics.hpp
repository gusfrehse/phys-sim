#pragma once

#include <glm/vec3.hpp>
#include <initializer_list>
#include <vector>

#include "object.hpp"

class physics {
  const glm::vec3 m_gravity = glm::vec3(0.0f, -98e-9f, 0.0f);
  std::vector<object> m_objects;

public:

  physics(std::initializer_list<object> const &list) : m_objects(list) {};

  void time_step(float dt);

  void check_collisions();

  glm::vec3 get_position(uint32_t i) const;
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
