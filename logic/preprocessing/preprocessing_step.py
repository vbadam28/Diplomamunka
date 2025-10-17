import cv2
import numpy as np
class PreprocessingStep:
    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def process(self, ctx):
        img = self.func(ctx.get('roi',ctx.get('image')), **self.kwargs)
        ctx.set('roi',img)
        return ctx


def enhanceImage(image, gamma=0.4, c=1):
    norm_img = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return c * np.power(norm_img, gamma)

def inverseEnhanceImage(image, gamma=0.4, c=1, isNorm = False):
    norm_img = np.array(image).copy()
    if isNorm:
        norm_img = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return c * np.power(norm_img, 1 / gamma)

def normalize(image):
    return cv2.normalize(image,None,0,1,cv2.NORM_MINMAX)

def gaussianBlur(image, ksize=(5,5), sigmaX=0):
    return cv2.GaussianBlur(image,ksize,sigmaX)