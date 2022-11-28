#pragma once

#include <glm/vec3.hpp>
#include <initializer_list>
#include <vector>

#include "object.hpp"

struct collision_response {
  float val;
  int a_id;
  int b_id;
};

class physics {
  const glm::vec3 m_gravity = glm::vec3(0.0f, -98e-8f, 0.0f);
  std::vector<object> m_objects;
  std::vector<collision_response> m_collisions;

public:

  physics(std::initializer_list<object> const &list);

  void time_step(float dt);

  int check_collisions();

  std::vector<collision_response>& get_collisions();

  glm::vec3 get_position(uint32_t i) const;
  object& get_object(uint32_t i);
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
