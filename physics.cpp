#include "physics.hpp"
#include <cstdio>

void physics::time_step(float dt) {
    for (auto& obj : m_objects) {
        obj.force    += m_gravity * obj.mass;
        obj.velocity += obj.force / obj.mass * dt;
        obj.position += obj.velocity * dt;

        obj.force = glm::vec3(0.0f);
    }
}

glm::vec3 physics::get_position(uint32_t i) const {
  return m_objects[i].position;
}
/* vim: set sts=2 ts=2 sw=2 et cc=81: */
