#pragma once

#include <glm/vec3.hpp>
#include <initializer_list>
#include <vector>

struct object {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 force;
    float mass;
};

class physics {
    const glm::vec3 m_gravity = glm::vec3(0.0f, -98e-9f, 0.0f);
    std::vector<object> m_objects;

public:
    physics(std::initializer_list<object> const &list) : m_objects(list) {};
    void time_step(float dt);
    glm::vec3 get_position(uint32_t i) const;
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
