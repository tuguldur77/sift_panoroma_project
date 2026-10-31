# SIFT Panorama Stitching

A Computer Vision course project that stitches five overlapping photos into one wide panorama. It builds on the course's practice exercises and adds cylindrical warping and fallback modes for real photos.

## How it works

1. **Match features:** finds SIFT keypoints in neighboring photos and filters matches with a ratio test and cross-check.
2. **Align photos:** estimates homographies with RANSAC and aligns every photo to the middle one.
3. **Warp:** projects photos onto a cylinder so that wide sweeps don't stretch out of shape.
4. **Blend:** uses feathered weights to hide the seams.
5. **Crop:** removes the empty black edges.

## Run it

Place `input_1.jpg` to `input_5.jpg`, ordered left to right, next to the script.

```bash
pip install opencv-python numpy
python panorama.py
```

The result is saved as `panorama_result.jpg`. See [report.pdf](report.pdf) for the write-up and sample results.
