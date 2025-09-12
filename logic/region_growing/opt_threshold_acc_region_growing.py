import cv2
import numpy as np
from collections import deque

from logic.region_growing.region_growing import RegionGrowingStrategy


class OptThresholdAccRegionGrowing(RegionGrowingStrategy):
    def __init__(self):
        self.debug=False
    def regionGrowing(self, image, seed, optimal_threshold):
        firstSeed = seed

        regionMean = image[firstSeed[0], firstSeed[1]]

        segmented = np.zeros_like(image, dtype=np.uint8)
        segmented[firstSeed[0], firstSeed[1]] = 1

        queue = deque([tuple(firstSeed)])  # list(map(tuple, [firstSeed]))

        intensitySum = regionMean
        count = 1
        while len(queue) > 0:
            y, x = queue.popleft()  # queue.pop(0)

            for dy, dx in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                if 0 <= (y + dy) < image.shape[0] and 0 <= (x + dx) < image.shape[1] and not segmented[
                    y + dy, x + dx]:  # valid offset to check
                    intensity = image[y + dy, x + dx]
                    if (np.abs(intensity - regionMean) >= np.abs(regionMean - optimal_threshold)):
                        continue
                    queue.append((y + dy, x + dx))
                    segmented[y + dy, x + dx] = 1
                    intensitySum += intensity
                    count += 1
            regionMean = intensitySum / count
        return segmented

    def run(self, ctx):
        self.debug = ctx.get('debug',False)
        image = ctx.get('image')
        seeds = ctx.get('seeds')
        optimal_threshold = ctx.get('optimal_threshold')
        '''5,6,7 region growing'''
        accumulatedMask = np.zeros_like(image)
        for seed in seeds:
            if accumulatedMask[
                seed[0], seed[1]] == 1:  # Ha még nem tartozik hozzá egy segmented régióhoz se akkor kell csak számolni
                continue
            segmented = self.regionGrowing(image, seed, optimal_threshold)
            accumulatedMask[segmented == 1] = 1
            # accumulatedMask |= segmented
        if self.debug:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 8))

            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(accumulatedMask, cmap="binary")
            plt.title("Segmented version")
        return [accumulatedMask]