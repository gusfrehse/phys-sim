#pragma once

#include <memory>

#include "glm.hpp"

class camera {
protected:
  float m_aspect_ratio;
  glm::vec3 m_position;
  glm::mat4 m_projection_matrix;
  glm::mat4 m_view_matrix;
  glm::mat4 m_projection_view_matrix;

  virtual void recalculate_view_matrix() = 0;
  virtual void recalculate_proj_matrix() = 0;

  camera();

public:
  void set_position(const glm::vec3& pos);
  void set_aspect_ratio(const float aspect_ratio);

  const float get_aspect_ratio() const {
    return m_aspect_ratio;
  }

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
  void recalculate_proj_matrix() override;

public:
  float get_zoom() const { return m_zoom; }
  void set_zoom(float zoom);
  orthographic_camera();
  orthographic_camera(float aspect_ratio);
};

class perspective_camera : public camera {
private:
  float m_fov = 90.0f;
  float m_yaw = 0.0f;
  float m_pitch = 0.0f;

  void recalculate_view_matrix() override;
  void recalculate_proj_matrix() override;

public:
  float get_fov() const { return m_fov; }
  void set_fov(float fov);
  perspective_camera();
  perspective_camera(float aspect_ratio);
};
/* vim: set sts=2 ts=2 sw=2 et cc=81: */
