#include "object.hpp"

int object::next_id = 0;

object::object(glm::vec3 position, glm::vec3 velocity, obj_type type,
               float mass, float radius) :
    id(next_id++),
    position(position),
    velocity(velocity),
    force(0.0f),
    type(type),
    mass(mass),
    radius(radius)
{}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
