#pragma once

#include <memory>
#include <glm/matrix.hpp>

class camera {
protected:
  glm::vec3 m_position;
  glm::mat4 m_projection_matrix;
  glm::mat4 m_view_matrix;
  glm::mat4 m_projection_view_matrix;

  virtual void recalculate_view_matrix() = 0;

  camera();

public:
  void set_position(const glm::vec3& pos);

  const glm::vec3& get_position() const {
    return m_position;
  }

  const glm::mat4& get_projection_matrix() const {
    return m_projection_matrix;
  }

  const glm::mat4& get_view_matrix() const {
    return m_view_matrix;
  }

  const glm::mat4& get_projection_view_matrix() const {
    return m_projection_view_matrix;
  }
};

class orthographic_camera : public camera {
private:
  float m_zoom;

  void recalculate_view_matrix() override;

public:
  float get_zoom() const { return m_zoom; }
  void set_zoom(float zoom);
  orthographic_camera();
  orthographic_camera(float aspect_ratio);
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */
