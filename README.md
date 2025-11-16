
# 🧭 Automatic Image Coregistration in R Using AI-Based Keypoints (OpenCV)

This guide shows how to **coregister two images** in R by automatically detecting **common points** using AI-powered feature detectors (ORB/SIFT via OpenCV through `reticulate`).

---

## 📦 Requirements

Install the R package:

```r
install.packages("reticulate")
```

Install Python dependencies:

```bash
pip install opencv-python numpy
```

---

## 📁 R Script (Markdown Safe)

```r
library(reticulate)

# Use python environment (edit path if needed)
use_virtualenv("~/.venv", required = TRUE)

cv <- import("cv2")
np <- import("numpy")

# ----------------------------------------------------
# 1. Load images
# ----------------------------------------------------
img1 <- cv$imread("image_reference.jpg", cv$IMREAD_COLOR)
img2 <- cv$imread("image_to_align.jpg", cv$IMREAD_COLOR)

if (is.null(img1) || is.null(img2)) stop("Could not load images.")

# ----------------------------------------------------
# 2. Detect keypoints using AI-based ORB
# ----------------------------------------------------
orb <- cv$ORB_create(5000L)

kp1_desc <- orb$detectAndCompute(img1, NULL)
kp1 <- kp1_desc[[1]]
desc1 <- kp1_desc[[2]]

kp2_desc <- orb$detectAndCompute(img2, NULL)
kp2 <- kp2_desc[[1]]
desc2 <- kp2_desc[[2]]

# ----------------------------------------------------
# 3. Match descriptors
# ----------------------------------------------------
bf <- cv$BFMatcher(cv$NORM_HAMMING, crossCheck = TRUE)
matches <- bf$match(desc1, desc2)

# Sort by descriptor distance (best first)
matches_sorted <- matches[order(sapply(matches, function(x) x$distance))]

# Keep the best N matches
N <- 50
good_matches <- matches_sorted[1:min(N, length(matches_sorted))]

# ----------------------------------------------------
# 4. Extract matched point coordinates
# ----------------------------------------------------
pts1 <- lapply(good_matches, function(m) kp1[[m$queryIdx + 1L]]$pt)
pts2 <- lapply(good_matches, function(m) kp2[[m$trainIdx + 1L]]$pt)

pts1_np <- np$array(do.call(rbind, pts1), dtype="float32")
pts2_np <- np$array(do.call(rbind, pts2), dtype="float32")

# ----------------------------------------------------
# 5. Compute transformation matrix (Homography)
# ----------------------------------------------------
H <- cv$findHomography(pts2_np, pts1_np, cv$RANSAC)[[1]]

# ----------------------------------------------------
# 6. Warp second image to align with first
# ----------------------------------------------------
h <- img1$shape[[1]]
w <- img1$shape[[2]]

aligned_img <- cv$warpPerspective(img2, H, tuple(w, h))

# ----------------------------------------------------
# 7. Save result
# ----------------------------------------------------
cv$imwrite("image_aligned.jpg", aligned_img)

cat("Coregistration complete → saved as image_aligned.jpg\n")
```

---

## 📝 How It Works

* **ORB AI detector** extracts thousands of keypoints.
* OpenCV automatically **matches** corresponding keypoints between the two images.
* A **homography matrix** is computed using RANSAC.
* The second image is **warped** to match the geometry of the reference image.

---

## 🔧 Options

You can replace ORB with:

* `cv$SIFT_create()` – more accurate
* `cv$AKAZE_create()`
* Deep-learning detectors (SuperPoint, R2D2) — available if you want them

---
