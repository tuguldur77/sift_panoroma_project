# 202213827 투굴두루 Computer Vision SIFT-based Custom Panorama Stitching
# Code configuration based on teachers given test imgs

import os
import cv2
import numpy as np

NUM_IMAGES = 5
INPUT_TEMPLATE = "input_{}.jpg"
OUTPUT_NAME = "panorama_result.jpg"
MAX_DIM = 900
RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 4.0
MIN_MATCHES = 10

# Final output mode. "planar" uses only homographies; the others pre-warp to a cylinder.
FINAL_MODE = "35mm"

# Retry with these modes if the chosen one diverges.
AUTO_FALLBACK_MODES = ["35mm", "24mm", "planar"]

# Write one panorama per mode when comparing focal lengths.
COMPARE_ALL_MODES = False

FOCAL_FACTORS = {"24mm": 24.0 / 36.0, "35mm": 35.0 / 36.0,
                 "50mm": 50.0 / 36.0, "85mm": 85.0 / 36.0}

# Crop mode for the stitched panorama.
CROP_MODE = "fullwidth"
CROP_BLACK_TOLERANCE = 0.05
CROP_CENTER_FRACTION = 0.55

MAX_CANVAS_DIM = 20000
MAX_CANVAS_PIXELS = 80_000_000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def load_images():
    """Load input_1..input_5 from BASE_DIR and resize for stable, fast matching."""
    images = []
    for i in range(1, NUM_IMAGES + 1):
        path = os.path.join(BASE_DIR, INPUT_TEMPLATE.format(i))
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read '{path}'. "
                                    f"Make sure {INPUT_TEMPLATE.format(i)} is in the same folder.")
        h, w = img.shape[:2]
        scale = MAX_DIM / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(img)
    return images
def cylindrical_warp(img, focal_px):
    """Project the image onto a cylinder before matching or stitching."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    x_idx, y_idx = np.meshgrid(np.arange(w), np.arange(h))
    theta = (x_idx - cx) / focal_px
    height = (y_idx - cy) / focal_px

    X, Y, Z = np.sin(theta), height, np.cos(theta)
    src_x = (focal_px * X / Z) + cx
    src_y = (focal_px * Y / Z) + cy

    map_x = src_x.astype(np.float32)
    map_y = src_y.astype(np.float32)
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    cols = np.where(gray.max(axis=0) > 0)[0]
    if len(cols) > 0:
        warped = warped[:, cols.min():cols.max() + 1]
    return warped


def match_pair(img_a, img_b):
    """Return matched point pairs between two images."""
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=20)
    kp_a, des_a = sift.detectAndCompute(gray_a, None)
    kp_b, des_b = sift.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None:
        return np.empty((0, 2)), np.empty((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_L2)

    good_ab = {}
    for pair in bf.knnMatch(des_a, des_b, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_ab[m.queryIdx] = m.trainIdx

    good_ba = {}
    for pair in bf.knnMatch(des_b, des_a, k=2):
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_ba[m.queryIdx] = m.trainIdx

    pts_a, pts_b = [], []
    for a_idx, b_idx in good_ab.items():
        if good_ba.get(b_idx, -1) == a_idx:
            pts_a.append(kp_a[a_idx].pt)
            pts_b.append(kp_b[b_idx].pt)

    return np.array(pts_a, dtype=np.float32), np.array(pts_b, dtype=np.float32)


def estimate_homography(pts_src, pts_dst):
    """Estimate a 3x3 homography from source points to destination points."""
    if len(pts_src) < MIN_MATCHES:
        raise ValueError(f"Only {len(pts_src)} matches; need >= {MIN_MATCHES}. "
                         f"Adjacent images may not overlap enough.")
    H, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, RANSAC_THRESHOLD)
    if H is None:
        raise ValueError("RANSAC failed to find a homography for an adjacent pair.")
    return H


def is_sane_homography(H, w, h):
    """Reject homographies that are degenerate or stretch the image too much."""
    if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-8:
        return False
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    p = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.all(np.isfinite(p)):
        return False
    area = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    orig = float(w * h)
    return 0.04 * orig < area < 25.0 * orig


def chain_homographies(images, ref_index):
    """Map every image into the middle image's frame."""
    n = len(images)
    H_to_ref = [None] * n
    H_to_ref[ref_index] = np.eye(3, dtype=np.float64)

    def pair_homography(i, j):
        """Estimate the transform from image i to image j."""
        pts_i, pts_j = match_pair(images[i], images[j])
        if len(pts_i) < MIN_MATCHES:
            raise ValueError(
                f"input_{i+1}.jpg and input_{j+1}.jpg share only {len(pts_i)} matches "
                f"(need >= {MIN_MATCHES}). These two images do not overlap enough to be "
                f"stitched. Re-shoot so consecutive images overlap by ~30-50%.")
        H = estimate_homography(pts_i, pts_j)                         # [Practice 03]
        h_i, w_i = images[i].shape[:2]
        if not is_sane_homography(H, w_i, h_i):
            raise ValueError(
                f"The transform between input_{i+1}.jpg and input_{j+1}.jpg is degenerate "
                f"(extreme stretching). This usually means a very wide rotation or strong "
                f"parallax for 'planar' mode. Try a cylindrical mode (set FINAL_MODE = \"24mm\").")
        return H

    for i in range(ref_index - 1, -1, -1):
        H_to_ref[i] = H_to_ref[i + 1] @ pair_homography(i, i + 1)

    for i in range(ref_index + 1, n):
        H_to_ref[i] = H_to_ref[i - 1] @ pair_homography(i, i - 1)

    return H_to_ref


def compute_canvas(images, H_to_ref):
    """Find the output canvas and translation that keeps coordinates positive."""
    all_corners = []
    for img, H in zip(images, H_to_ref):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H))
    all_corners = np.concatenate(all_corners, axis=0).reshape(-1, 2)

    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    canvas_w, canvas_h = int(x_max - x_min), int(y_max - y_min)

    if (canvas_w > MAX_CANVAS_DIM or canvas_h > MAX_CANVAS_DIM
            or canvas_w * canvas_h > MAX_CANVAS_PIXELS):
        raise ValueError(
            f"Computed canvas {canvas_w}x{canvas_h} is too large -> a transform diverged "
            f"(typical for 'planar' mode on a wide sweep). Try a cylindrical mode "
            f"(set FINAL_MODE = \"24mm\"), or re-shoot with less rotation per step.")

    offset = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    return offset, (canvas_w, canvas_h)


def feather_weight(h, w):
    """Build a center-weighted weight map for feather blending."""
    wx = 1.0 - np.abs(np.linspace(-1, 1, w))
    wy = 1.0 - np.abs(np.linspace(-1, 1, h))
    return np.clip(np.outer(wy, wx).astype(np.float32), 1e-3, 1.0)


def warp_and_blend(images, H_to_ref, offset, canvas_size):
    """Warp each image into the canvas and blend them with weighted averaging."""
    canvas_w, canvas_h = canvas_size
    accum = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w),    dtype=np.float32)

    for img, H in zip(images, H_to_ref):
        h, w = img.shape[:2]
        M = offset @ H
        warped_img = cv2.warpPerspective(img.astype(np.float32), M, (canvas_w, canvas_h))
        warped_w = cv2.warpPerspective(feather_weight(h, w), M, (canvas_w, canvas_h))
        accum += warped_img * warped_w[..., None]
        weight += warped_w

    weight = np.maximum(weight, 1e-6)
    return np.clip(accum / weight[..., None], 0, 255).astype(np.uint8)


def crop_black(image):
    """Trim the outer black border."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def crop_to_rectangle(image):
    """Crop to the largest rectangle with no black pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask_full = gray > 8
    mask_full = cv2.erode(mask_full.astype(np.uint8),
                          np.ones((5, 5), np.uint8)).astype(bool)
    H, W = mask_full.shape

    scale = 600.0 / W if W > 600 else 1.0
    if scale < 1.0:
        small = cv2.resize(mask_full.astype(np.uint8), (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        small = mask_full
    sh, sw = small.shape

    best_area, best = 0, (0, 0, sh, sw)
    heights = np.zeros(sw, dtype=int)
    for r in range(sh):
        heights = np.where(small[r], heights + 1, 0)
        stack = []
        for c in range(sw + 1):
            cur = heights[c] if c < sw else 0
            start = c
            while stack and stack[-1][1] > cur:
                idx, hgt = stack.pop()
                area = hgt * (c - idx)
                if area > best_area:
                    best_area, best = area, (r - hgt + 1, idx, hgt, c - idx)
                start = idx
            stack.append((start, cur))

    top, left, hh, ww = best
    inv = 1.0 / scale
    top, left = int(top * inv), int(left * inv)
    hh, ww = int(hh * inv), int(ww * inv)
    cropped = image[top:top + hh, left:left + ww]
    return cropped if cropped.size > 0 else image


def crop_relaxed(image, tolerance=0.05):
    """Grow the tight crop while keeping the black fraction below tolerance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 600.0 / image.shape[1] if image.shape[1] > 600 else 1.0
    if scale < 1.0:
        small_gray = cv2.resize(gray, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)
    else:
        small_gray = gray
    valid = small_gray > 8
    sh, sw = valid.shape
    black_integral = cv2.integral((~valid).astype(np.uint8))   # (sh+1, sw+1)

    heights = np.zeros(sw, dtype=int); best_area, best = 0, (0, 0, sh, sw)
    for r in range(sh):
        heights = np.where(valid[r], heights + 1, 0)
        stack = []
        for c in range(sw + 1):
            cur = heights[c] if c < sw else 0
            start = c
            while stack and stack[-1][1] > cur:
                idx, hgt = stack.pop()
                area = hgt * (c - idx)
                if area > best_area:
                    best_area, best = area, (r - hgt + 1, idx, hgt, c - idx)
                start = idx
            stack.append((start, cur))
    t, l, hh, ww = best; b, rr = t + hh, l + ww

    def black_frac(t, b, l, r):
        if b <= t or r <= l:
            return 1.0
        s = (black_integral[b, r] - black_integral[t, r]
             - black_integral[b, l] + black_integral[t, l])
        return s / float((b - t) * (r - l))

    changed = True
    while changed:
        changed = False
        if t > 0  and black_frac(t - 1, b, l, rr) <= tolerance: t -= 1;  changed = True
        if b < sh and black_frac(t, b + 1, l, rr) <= tolerance: b += 1;  changed = True
        if l > 0  and black_frac(t, b, l - 1, rr) <= tolerance: l -= 1;  changed = True
        if rr < sw and black_frac(t, b, l, rr + 1) <= tolerance: rr += 1; changed = True

    inv = 1.0 / scale
    out = image[int(t * inv):int(b * inv), int(l * inv):int(rr * inv)]
    return out if out.size > 0 else image


def crop_fullwidth(image, center_fraction=0.6):
    """Keep the full width and trim only the top and bottom."""
    bb = crop_black(image)
    g = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY) > 8
    h, w = g.shape

    l = int(w * (1 - center_fraction) / 2)
    r = w - l
    top_first = np.argmax(g, axis=0)
    bot_last  = h - 1 - np.argmax(g[::-1], axis=0)
    top_cut = int(top_first[l:r].max())
    bot_cut = int(bot_last[l:r].min())
    if bot_cut <= top_cut:
        return crop_relaxed(image, 0.1)

    band = g[top_cut:bot_cut + 1, :]
    col_ok = band.mean(axis=0) > 0.98
    xs = np.where(col_ok)[0]
    if len(xs) == 0:
        return bb[top_cut:bot_cut + 1, :]
    out = bb[top_cut:bot_cut + 1, xs.min():xs.max() + 1]
    return out if out.size > 0 else bb


def apply_crop(image):
    """Apply the configured crop mode."""
    if CROP_MODE == "fullwidth":
        return crop_fullwidth(image, CROP_CENTER_FRACTION)
    if CROP_MODE == "keep_all":
        return crop_black(image)
    if CROP_MODE == "relaxed":
        return crop_relaxed(image, CROP_BLACK_TOLERANCE)
    return crop_to_rectangle(image)


def build_panorama(base_images, mode):
    """Build a panorama for one warping mode."""
    if mode == "planar":
        images = base_images
    else:
        images = [cylindrical_warp(img, FOCAL_FACTORS[mode] * img.shape[1]) for img in base_images]

    ref_index = len(images) // 2
    H_to_ref = chain_homographies(images, ref_index)
    offset, canvas_size = compute_canvas(images, H_to_ref)
    panorama = warp_and_blend(images, H_to_ref, offset, canvas_size)
    panorama = apply_crop(panorama)
    return panorama


def main():
    print("Loading images ...")
    base_images = load_images()

    modes_to_try = [FINAL_MODE] + [m for m in AUTO_FALLBACK_MODES if m != FINAL_MODE]
    final, used_mode, last_err = None, None, None
    for mode in modes_to_try:
        try:
            print(f"Building final panorama (mode = {mode}) ...")
            final = build_panorama(base_images, mode)
            used_mode = mode
            break
        except Exception as e:
            last_err = e
            print(f"  mode '{mode}' failed: {e}")

    if final is None:
        raise SystemExit(f"\nCould not build a panorama in any mode. Last error:\n{last_err}")

    out_path = os.path.join(BASE_DIR, OUTPUT_NAME)
    cv2.imwrite(out_path, final)
    print(f"Saved: {out_path}   (mode used: {used_mode})")
    print("Done.")


if __name__ == "__main__":
    main()
