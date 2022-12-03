#include "physics.hpp"

#include <cstdio>

#include "glm.hpp"

#include "profiler.hpp"
#include "object.hpp"

physics::physics(std::initializer_list<object> const &list) : m_objects(list) {
  m_collisions.resize(list.size() * list.size());
};

void physics::add_object(object o) {
  m_objects.push_back(o);
  
  auto size = m_objects.size();

  m_collisions.resize(size * size);
}

void physics::time_step(float dt) {
  PROFILE_FUNC();

  float total_kinetic_energy = 0.0f;

  for (auto& obj : m_objects) {
    obj.force    += m_gravity * obj.mass;
    obj.velocity += obj.force / obj.mass * dt;
    obj.position += obj.velocity * dt;

    obj.force = glm::vec3(0.0f);

    total_kinetic_energy += (obj.mass * glm::length2(obj.velocity));
  }

  total_kinetic_energy /= 2.0f;

  //std::printf("total energy is %g joules \n", total_kinetic_energy);
}

static collision_response collides(const object& a, const object& b) {
  // collision for now are only between spheres. Maybe for ever.
  if (a.type == obj_type::SPHERE && b.type == obj_type::SPHERE) {
    auto dif = glm::length2(a.position - b.position);
    auto radius_sum = a.radius + b.radius;

    auto val = radius_sum * radius_sum - dif; 

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

int physics::check_collisions() {
  PROFILE_FUNC();
  int index = 0;
  for (auto object_a = m_objects.begin(); object_a != m_objects.end(); object_a++) {
    for (auto object_b = object_a + 1; object_b != m_objects.end(); object_b++) {
      auto collision = collides(*object_a, *object_b);

      if (collision.val >= 0.01f) {
        // std::printf("collision!\n");
        m_collisions[index++] = collision;
      }
    }
  }

  return index;
}

std::vector<collision_response>& physics::get_collisions() {
  return m_collisions;
}

glm::vec3 physics::get_position(uint32_t i) const {
  PROFILE_FUNC();
  return m_objects[i].position;
}

object& physics::get_object(uint32_t i) {
  return m_objects[i];
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
