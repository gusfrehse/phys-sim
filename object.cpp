#include "object.hpp"

int object::next_id = 0;

object::object(glm::vec3 position, glm::vec3 velocity, float mass, float radius,
               obj_type type) :
    id(next_id++),
    position(position),
    velocity(velocity),
    force(0.0f),
    type(type),
    mass(mass),
    radius(radius)
{}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
