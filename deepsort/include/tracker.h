#ifndef TRACKER_H
#define TRACKER_H

#include <vector>

#include "kalmanfilter.h"
#include "track.h"
#include "model.hpp"

using namespace std;

class NearNeighborDisMetric;

constexpr float MAX_IOU_DISTANCE = 0.7;
constexpr int MAX_AGE = 200;
constexpr int N_INIT = 3; // 20

class tracker {
  public:
  NearNeighborDisMetric *metric;
  float max_iou_distance;
  int max_age;
  int n_init;

  KalmanFilter *kf;

  int _next_idx;

  public:
  std::vector<Track> tracks;
  tracker(/*NearNeighborDisMetric* metric,*/
          float max_cosine_distance, int nn_budget,
          float max_iou_distance = MAX_IOU_DISTANCE,
          int max_age = MAX_AGE,
          int n_init = N_INIT);
  void predict();
  void update(const DETECTIONS &detections);
  void update(const DETECTIONSV2 &detectionsv2);
  typedef DYNAMICM (tracker::*GATED_METRIC_FUNC)(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices);

  private:
  void _match(const DETECTIONS &detections, TRACHER_MATCHD &res);
  void _initiate_track(const DETECTION_ROW &detection);
  void _initiate_track(const DETECTION_ROW &detection, CLSCONF clsConf);

  public:
  DYNAMICM gated_matric(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices);
  DYNAMICM iou_cost(
    std::vector<Track> &tracks,
    const DETECTIONS &dets,
    const std::vector<int> &track_indices,
    const std::vector<int> &detection_indices);
  Eigen::VectorXf iou(DETECTBOX &bbox,
                      DETECTBOXSS &candidates);
};

#endif // TRACKER_H
