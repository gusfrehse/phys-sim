#include "physics.hpp"

#include <cstdio>

#include <glm/gtx/norm.hpp>

#include "profiler.hpp"
#include "object.hpp"

void physics::time_step(float dt) {
  PROFILE_FUNC();
  for (auto& obj : m_objects) {
    obj.force    += m_gravity * obj.mass;
    obj.velocity += obj.force / obj.mass * dt;
    obj.position += obj.velocity * dt;

    obj.force = glm::vec3(0.0f);
  }
}

struct collision_response {
  float val;
  int a_id;
  int b_id;
};

static collision_response collides(const object& a, const object& b) {
  // collision for now are only between spheres. Maybe for ever.
  if (a.type == obj_type::SPHERE && b.type == obj_type::SPHERE) {
    auto dif = glm::length2(a.position - b.position);
    auto radius_sum = a.radius + b.radius;

    auto val = dif - radius_sum * radius_sum; 

    //std::printf("Testing A { position: %g, velocity: %g, radius: %g }\n",
    //            a.position.x, a.velocity.x, a.radius);
    //std::printf("Testing B { position: %g, velocity: %g, radius: %g }\n",
    //            b.position.x, b.velocity.x, a.radius);

    //std::printf("dif: %g radius_sum: %g val: %g\n", dif, radius_sum, val);

    return collision_response {
      .val = val,
      .a_id = a.id,
      .b_id = b.id,
    };

  } else {
    std::printf("collision with non spheres is not implemented!\n");
  }

  return collision_response {
    .val = -1.0f,
    .a_id = a.id,
    .b_id = b.id
  };
}

void physics::check_collisions() {
  PROFILE_FUNC();
  for (auto object_a = m_objects.begin(); object_a != m_objects.end(); object_a++) {
    for (auto object_b = object_a + 1; object_b != m_objects.end(); object_b++) {
      auto collision = collides(*object_a, *object_b);

      if (collision.val <= -0.01f) {
        std::printf("Collision! a: %d b: %d\n", collision.a_id, collision.b_id);
      }
    }
  }
}

glm::vec3 physics::get_position(uint32_t i) const {
  PROFILE_FUNC();
  return m_objects[i].position;
}
/* vim: set sts=2 ts=2 sw=2 et cc=81: */
