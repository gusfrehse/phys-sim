#pragma once

#include <vector>

#include "glm.hpp"

#include "object.hpp"

struct collision_response {
  float val;
  int a_id;
  int b_id;
};

class physics {
  //const glm::vec3 m_gravity = glm::vec3(0.0f, -98e-8f, 0.0f);
  const glm::vec3 m_gravity = glm::vec3(0.0f);
  float m_time_scale = 1.0f;
  std::vector<object> m_objects;
  std::vector<collision_response> m_collisions;

public:

  physics() = default;

  physics(std::initializer_list<object> const &list);

  void add_object(object o);

  void time_step(float dt);

  float& get_time_scale();

  int check_collisions();

  std::vector<collision_response>& get_collisions();

  glm::vec3 get_position(uint32_t i) const;
  object& get_object(uint32_t i);
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
