#include "object.hpp"

#include <glm/vec3.hpp>
#include <glm/matrix.hpp>

void object::set_pos(glm::vec3 pos) {
  // TODO: check last component is always 1.0f. If it is, can change get_pos()
  // also.
  model[3] = glm::vec4(pos, 1.0f);
}

glm::vec3 object::get_pos() {
  return glm::vec3(model[3].x, model[3].y, model[3].z ) / model[3].w;
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
