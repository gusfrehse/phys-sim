#include "camera.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>


camera::camera() :
  m_position(0.0f),
  m_projection_matrix(1.0f),
  m_view_matrix(1.0f),
  m_projection_view_matrix(1.0f) {}

void camera::set_position(const glm::vec3 &pos) {
  m_position = pos;
  recalculate_view_matrix();
}

orthographic_camera::orthographic_camera() :
  orthographic_camera(1.0f) {}

orthographic_camera::orthographic_camera(float aspect_ratio) : m_zoom(1.0f) {
  m_projection_matrix = glm::ortho(-aspect_ratio, aspect_ratio, 1.0f, -1.0f);
  m_projection_view_matrix = m_projection_matrix * m_view_matrix;
}

void orthographic_camera::recalculate_view_matrix() {
  glm::mat4 transform = glm::translate(glm::mat4(1.0f), m_position);
  transform = glm::scale(transform, glm::vec3(m_zoom, m_zoom, 1.0f));

  m_view_matrix = glm::inverse(transform);
  m_projection_view_matrix = m_projection_matrix * m_view_matrix;
}

void orthographic_camera::set_zoom(float zoom) {
  m_zoom = zoom;
  recalculate_view_matrix();
}

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
