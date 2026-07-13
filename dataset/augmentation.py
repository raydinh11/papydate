import numpy as np
import cv2 as cv
from PIL import Image
import random


class ElasticDeformation:
    def __init__(self, alpha=30, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]

        dx = cv.GaussianBlur(np.random.randn(h, w).astype(np.float32) * self.alpha,
                             (0, 0), self.sigma)
        dy = cv.GaussianBlur(np.random.randn(h, w).astype(np.float32) * self.alpha,
                             (0, 0), self.sigma)

        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        warped = cv.remap(img_np, map_x, map_y, cv.INTER_LINEAR,
                          borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(warped)


class RandomAffine:
    def __init__(self, p=0.5, rotation=10, scale=(0.9, 1.1)):
        self.p = p
        self.rotation = rotation
        self.scale = scale

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img_np = np.array(img)
        h, w = img_np.shape[:2]
        cx, cy = w / 2, h / 2

        angle = random.uniform(-self.rotation, self.rotation)
        scale = random.uniform(*self.scale)

        M = cv.getRotationMatrix2D((cx, cy), angle, scale)
        warped = cv.warpAffine(img_np, M, (w, h),
                               borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return Image.fromarray(warped)
