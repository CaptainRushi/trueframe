import cv2
import numpy as np
from config import PATCH_COUNT


class PatchVoter:
    """
    Multi-patch voting for single images.
    Splits a face crop into overlapping patches and runs inference on each.
    Averaging patch scores reduces false positives from localized artifacts.
    """

    def extract_patches(self, face_crop, count=None):
        if count is None:
            count = PATCH_COUNT

        h, w = face_crop.shape[:2]
        if h < 64 or w < 64:
            return []

        patches = []
        patch_h, patch_w = h // 2, w // 2
        step_h = max(1, (h - patch_h) // max(1, int(count ** 0.5) - 1))
        step_w = max(1, (w - patch_w) // max(1, int(count ** 0.5) - 1))

        for y in range(0, h - patch_h + 1, step_h):
            for x in range(0, w - patch_w + 1, step_w):
                patch = face_crop[y:y + patch_h, x:x + patch_w]
                patches.append(patch)
                if len(patches) >= count:
                    return patches

        return patches
