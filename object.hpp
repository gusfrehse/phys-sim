#pragma once
#include <glm/vec3.hpp>

enum class obj_type { SPHERE };

class object {
public:
  static int next_id;

  int id;
  glm::vec3 position;
  glm::vec3 velocity;
  glm::vec3 force;
  obj_type type;
  float mass;
  float radius;

  object(glm::vec3 position = glm::vec3(0.0f),
         glm::vec3 velocity = glm::vec3(0.0f),
         float mass = 1.0f,
         float radius = 1.0f,
         obj_type type = obj_type::SPHERE);
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
