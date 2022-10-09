#pragma once

#include <glm/matrix.hpp>

struct object {
  glm::mat4 model;

  void set_pos(glm::vec3 pos);
  glm::vec3 get_pos();
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
