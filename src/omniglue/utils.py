# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utility functions for OmniGlue."""

import math
from typing import Optional
import cv2
import numpy as np
import tensorflow as tf


def lookup_descriptor_bilinear(
    keypoint: np.ndarray, descriptor_map: np.ndarray
) -> np.ndarray:
  """Looks up descriptor value for keypoint from a dense descriptor map.

  Uses bilinear interpolation to find descriptor value at non-integer
  positions.

  Args:
    keypoint: 2-dim numpy array containing (x, y) keypoint image coordinates.
    descriptor_map: (H, W, D) numpy array representing a dense descriptor map.

  Returns:
    D-dim descriptor value at the input 'keypoint' location.

  Raises:
    ValueError, if kepoint position is out of bounds.
  """
  height, width = np.shape(descriptor_map)[:2]
  if (
      keypoint[0] < 0
      or keypoint[0] > width
      or keypoint[1] < 0
      or keypoint[1] > height
  ):
    raise ValueError(
        'Keypoint position (%f, %f) is out of descriptor map bounds (%i w x'
        ' %i h).' % (keypoint[0], keypoint[1], width, height)
    )

  x_range = [math.floor(keypoint[0])]
  if not keypoint[0].is_integer() and keypoint[0] < width-1:
    x_range.append(x_range[0] + 1)
  y_range = [math.floor(keypoint[1])]
  if not keypoint[1].is_integer() and keypoint[1] < height-1:
    y_range.append(y_range[0] + 1)

  bilinear_descriptor = np.zeros(np.shape(descriptor_map)[2])
  for curr_x in x_range:
    for curr_y in y_range:
      curr_descriptor = descriptor_map[curr_y, curr_x, :]
      bilinear_scalar = (1.0 - abs(keypoint[0] - curr_x)) * (
          1.0 - abs(keypoint[1] - curr_y)
      )
      bilinear_descriptor += bilinear_scalar * curr_descriptor
  return bilinear_descriptor


def soft_assignment_to_match_matrix(
    soft_assignment: tf.Tensor, match_threshold: float
) -> tf.Tensor:
  """Converts a matrix of soft assignment values to binary yes/no match matrix.

  Searches soft_assignment for row- and column-maximum values, which indicate
  mutual nearest neighbor matches between two unique sets of keypoints. Also,
  ensures that score values for matches are above the specified threshold.

  Args:
    soft_assignment: (B, N, M) tensor, contains matching likelihood value
      between features of different sets. N is number of features in image0, and
      M is number of features in image1. Higher value indicates more likely to
      match.
    match_threshold: float, thresholding value to consider a match valid.

  Returns:
    (B, N, M) tensor of binary values. A value of 1 at index (x, y) indicates
    a match between index 'x' (out of N) in image0 and index 'y' (out of M) in
    image 1.
  """

  def _range_like(x, dim):
    """Returns tensor with values (0, 1, 2, ..., N) for dimension in input x."""
    return tf.range(tf.shape(x)[dim], dtype=x.dtype)

  # TODO(omniglue): batch loop & SparseTensor are slow. Optimize with tf ops.
  matches = tf.TensorArray(tf.float32, size=tf.shape(soft_assignment)[0])
  for i in range(tf.shape(soft_assignment)[0]):
    # Iterate through batch and process one example at a time.
    scores = tf.expand_dims(soft_assignment[i, :], 0)  # Shape: (1, N, M).

    # Find indices for max values per row and per column.
    max0 = tf.math.reduce_max(scores, axis=2)  # Shape: (1, N).
    indices0 = tf.math.argmax(scores, axis=2)  # Shape: (1, N).
    indices1 = tf.math.argmax(scores, axis=1)  # Shape: (1, M).

    # Find matches from mutual argmax indices of each set of keypoints.
    mutual = tf.expand_dims(_range_like(indices0, 1), 0) == tf.gather(
        indices1, indices0, axis=1
    )

    # Create match matrix from sets of index pairs and values.
    kp_ind_pairs = tf.stack(
        [_range_like(indices0, 1), tf.squeeze(indices0)], axis=1
    )
    mutual_max0 = tf.squeeze(tf.squeeze(tf.where(mutual, max0, 0), 0))
    sparse = tf.sparse.SparseTensor(
        kp_ind_pairs, mutual_max0, tf.shape(scores, out_type=tf.int64)[1:]
    )
    match_matrix = tf.sparse.to_dense(sparse)
    matches = matches.write(i, match_matrix)

  # Threshold on match_threshold value and convert to binary (0, 1) values.
  match_matrix = matches.stack()
  match_matrix = match_matrix > match_threshold
  return match_matrix


def visualize_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    match_matrix: np.ndarray,
    match_labels: Optional[np.ndarray] = None,
    show_keypoints: bool = False,
    highlight_unmatched: bool = False,
    title: Optional[str] = None,  # ここは受け取るが使わない
    line_width: int = 1,
    circle_radius: int = 4,
    circle_thickness: int = 2,
    rng: Optional['np.random.Generator'] = None,
    gap_width: int = 50  # 画像間の余白ピクセル数を追加
):
    """Generates visualization of keypoints and matches without a legend."""
    if rng is None:
        rng = np.random.default_rng()

    kp1 = np.copy(kp1)

    # 画像1の高さに合わせる
    height0 = image0.shape[0]
    height1 = image1.shape[0]
    if height0 != height1:
        scale_factor = height0 / height1
        interp_method = cv2.INTER_AREA if scale_factor <= 1.0 else cv2.INTER_LINEAR
        new_dim1 = (int(image1.shape[1] * scale_factor), height0)
        image1 = cv2.resize(image1, new_dim1, interpolation=interp_method)
        kp1 *= scale_factor

    # 画像の間に白い余白を作成
    gap = 255 * np.ones((height0, gap_width, 3), dtype=np.uint8)

    # 画像を横に並べる
    viz = np.hstack((image0, gap, image1))
    w0 = image0.shape[1] + gap_width  # Keypoint座標のオフセットを変更

    # マッチング線を描画
    matches = np.argwhere(match_matrix)
    for match in matches:
        pt0 = (int(kp0[match[0], 0]), int(kp0[match[0], 1]))
        pt1 = (int(kp1[match[1], 0] + w0), int(kp1[match[1], 1]))
        color = tuple(rng.integers(0, 255, size=3).tolist())  # ランダムな色で線を描画
        cv2.line(viz, pt0, pt1, color, line_width)

    # Keypointを描画
    if show_keypoints:
        for kp in kp0:
            cv2.circle(viz, tuple(kp.astype(np.int32).tolist()), circle_radius, (0, 0, 255), circle_thickness)
        for kp in kp1:
            kp[0] += w0
            cv2.circle(viz, tuple(kp.astype(np.int32).tolist()), circle_radius, (0, 0, 255), circle_thickness)

    return viz