#!/usr/bin/env python3
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

"""Demo script for performing OmniGlue inference."""

import os
import sys
import time
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
import omniglue
from omniglue import utils
from PIL import Image
import pickle

def decompose_homography(H, image0_shape):
    """ホモグラフィ行列から回転角度、スケール、並進成分を抽出
    
    Args:
        H: 3x3 ホモグラフィ行列
        image0_shape: (height, width) 元画像のサイズ
        
    Returns:
        rotation_deg: 回転角度（度）
        scale_x: X方向のスケール
        scale_y: Y方向のスケール
        translation: (tx, ty) 並進ベクトル（ピクセル単位）
        shear: せん断成分
    """
    # 正規化
    H_norm = H / H[2, 2]
    
    # アフィン部分を抽出
    A = H_norm[:2, :2]
    
    # 特異値分解でスケールと回転を分離
    U, S, Vt = np.linalg.svd(A)
    
    # 回転行列
    R = U @ Vt
    
    # 回転角度（ラジアンから度へ）
    rotation_rad = np.arctan2(R[1, 0], R[0, 0])
    rotation_deg = np.degrees(rotation_rad)
    
    # スケール
    scale_x = S[0]
    scale_y = S[1]
    
    # せん断（歪み）
    shear = np.arctan2(A[0, 1] + A[1, 0], A[0, 0] + A[1, 1])
    
    # 並進: 画像中心点を変換して実際の移動量を計算
    height, width = image0_shape
    center = np.array([[width / 2.0], [height / 2.0], [1.0]])
    transformed_center = H_norm @ center
    # 同次座標を正規化
    transformed_center = transformed_center[:2] / transformed_center[2]
    # 元の中心からの移動量
    translation = transformed_center.flatten() - np.array([width / 2.0, height / 2.0])
    
    return rotation_deg, scale_x, scale_y, translation, shear


def omniglue_inference(image0, image1, og_model, match_threshold=0.1, ransac_reproj_threshold=5.0, mmpp=0.0581):
    """OmniGlueによる画像マッチングと姿勢推定を実行
    
    Args:
        image0: numpy array, 1枚目の画像 (H, W, 3)
        image1: numpy array, 2枚目の画像 (H, W, 3)
        og_model: OmniGlueモデルインスタンス
        match_threshold: float, マッチングの信頼度閾値 [0.0, 1.0)
        ransac_reproj_threshold: float, RANSACの再投影誤差閾値（ピクセル単位）
        mmpp: float, ミリメートル毎ピクセルのスケールファクター
        
    Returns:
        results: dict, 推論結果を格納した辞書
            - match_kp0: numpy array, image0のマッチしたキーポイント座標
            - match_kp1: numpy array, image1のマッチしたキーポイント座標
            - match_confidences: numpy array, マッチの信頼度
            - num_matches: int, マッチング総数
            - num_filtered_matches: int, フィルタ後のマッチ数
            - H: numpy array or None, ホモグラフィ行列 (3x3)
            - inliers: int or None, インライア数
            - inlier_ratio: float or None, インライア比率
            - rotation_deg: float or None, 回転角度（度）
            - scale_x: float or None, X方向のスケール
            - scale_y: float or None, Y方向のスケール
            - translation: numpy array or None, 並進ベクトル (tx, ty)
            - shear: float or None, せん断角度
    """
    results = {}
    
    # Perform inference
    print("> Finding matches...")
    start = time.time()
    match_kp0, match_kp1, match_confidences = og_model.FindMatches(image0, image1)
    num_matches = match_kp0.shape[0]
    print(f"> \tFound {num_matches} matches.")
    print(f"> \tTook {time.time() - start} seconds.")
    
    # Filter by confidence
    print("> Filtering matches...")
    keep_idx = []
    for i in range(match_kp0.shape[0]):
        if match_confidences[i] > match_threshold:
            keep_idx.append(i)
    num_filtered_matches = len(keep_idx)
    match_kp0 = match_kp0[keep_idx]
    match_kp1 = match_kp1[keep_idx]
    match_confidences = match_confidences[keep_idx]
    print(f"> \tFound {num_filtered_matches}/{num_matches} above threshold {match_threshold}")
    
    results['match_kp0'] = match_kp0
    results['match_kp1'] = match_kp1
    results['match_confidences'] = match_confidences
    results['num_matches'] = num_matches
    results['num_filtered_matches'] = num_filtered_matches
    
    # Compute homography matrix (pose transformation)
    if num_filtered_matches >= 4:  # Minimum 4 points needed for homography
        print("> Computing homography matrix...")
        H, mask = cv2.findHomography(match_kp0, match_kp1, cv2.RANSAC, ransac_reproj_threshold)
        
        if H is not None:
            inliers = np.sum(mask)
            inlier_ratio = inliers / num_filtered_matches
            print(f"> \tHomography matrix found with {inliers}/{num_filtered_matches} inliers:")
            print(f"> \tInlier ratio: {inlier_ratio:.2%}")
            
            # Decompose homography
            print("> Decomposing homography...")
            rotation_deg, scale_x, scale_y, translation, shear = decompose_homography(H, image0.shape[:2])
            
            results['H'] = H
            results['inliers'] = inliers
            results['inlier_ratio'] = inlier_ratio
            results['rotation_deg'] = rotation_deg
            results['scale_x'] = scale_x
            results['scale_y'] = scale_y
            results['translation'] = translation * mmpp  # Convert to mm
            results['shear'] = shear
        else:
            print("> \tFailed to compute homography matrix")
            results['H'] = None
            results['inliers'] = None
            results['inlier_ratio'] = None
            results['rotation_deg'] = None
            results['scale_x'] = None
            results['scale_y'] = None
            results['translation'] = None
            results['shear'] = None
    else:
        print(f"> \tNot enough matches ({num_filtered_matches}) to compute homography (need at least 4)")
        results['H'] = None
        results['inliers'] = None
        results['inlier_ratio'] = None
        results['rotation_deg'] = None
        results['scale_x'] = None
        results['scale_y'] = None
        results['translation'] = None
        results['shear'] = None
    
    return results


def main(argv) -> None:
    if len(argv) != 3:
        raise ValueError("Incorrect command line usage - usage: python demo.py <img1_fp> <img2_fp>")
    image0_fp = argv[1]
    image1_fp = argv[2]
    for im_fp in [image0_fp, image1_fp]:
        if not os.path.exists(im_fp) or not os.path.isfile(im_fp):
            raise ValueError(f"Image filepath '{im_fp}' doesn't exist or is not a file.")

    # Load images
    print("> Loading images...")
    image0 = np.array(Image.open(argv[1]).convert("RGB"))
    image1 = np.array(Image.open(argv[2]).convert("RGB"))

    # Load models
    print("> Loading OmniGlue (and its submodules: SuperPoint & DINOv2)...")
    start = time.time()
    og = omniglue.OmniGlue(
        og_export="./models/og_export",
        sp_export="./models/sp_v6",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    print(f"> \tTook {time.time() - start} seconds.")

    # Run inference
    match_threshold = 0.001  # Choose any value [0.0, 1.0)
    mmpp = 0.0581  # [mm/pixel]
    ransac_reproj_threshold = 5.0
    results = omniglue_inference(image0, image1, og, match_threshold, ransac_reproj_threshold, mmpp)
    
    # Display and save results
    if results['H'] is not None:
        print("\nHomography Matrix (3x3):")
        print(results['H'])
        
        # Save homography matrix
        np.savetxt("./homography_matrix.txt", results['H'], fmt='%.6f')
        print("\n> \tSaved homography matrix to ./homography_matrix.txt")
        
        print(f"\n=== 姿勢変換パラメータ ===")
        print(f"回転角度: {results['rotation_deg']:.2f}°")
        print(f"スケール: X={results['scale_x']:.3f}, Y={results['scale_y']:.3f}")
        print(f"並進成分: X={results['translation'][0]:.2f}mm, Y={results['translation'][1]:.2f}mm")
        print(f"せん断角度: {np.degrees(results['shear']):.2f}°")
        
        # Save transformation parameters as JSON
        params_dict = {
            "rotation_deg": float(results['rotation_deg']),
            "scale_x": float(results['scale_x']),
            "scale_y": float(results['scale_y']),
            "translation_x_mm": float(results['translation'][0]),
            "translation_y_mm": float(results['translation'][1]),
            "shear_deg": float(np.degrees(results['shear'])),
            "inliers": int(results['inliers']),
            "num_filtered_matches": int(results['num_filtered_matches']),
            "inlier_ratio": float(results['inlier_ratio'])
        }
        with open(f"./results/pose_estimation/data.pkl", "wb") as f:
            pickle.dump(params_dict, f)
        print("> \tSaved transformation parameters to ./results/pose_estimation/data.pkl")

    # Visualize
    print("\n> Visualizing matches...")
    viz = utils.visualize_matches(
        image0,
        image1,
        results['match_kp0'],
        results['match_kp1'],
        np.eye(results['num_filtered_matches']),
        show_keypoints=True,
        highlight_unmatched=True,
        title=f"{results['num_filtered_matches']} matches",
        line_width=2,
    )
    plt.figure(figsize=(20, 10), dpi=100, facecolor="w", edgecolor="k")
    plt.axis("off")
    plt.imshow(viz)
    plt.imsave("./demo_output.png", viz)
    print("> \tSaved visualization to ./demo_output.png")


if __name__ == "__main__":
    main(sys.argv)
