import cv2
import numpy as np

from logic.preprocessing.preprocessing_step import inverseEnhanceImage


class QuadTree:
    hyperintenseRange = (0.8, 1.0)  # (0.47, 0.8)  # Egész datasetből megállapítnai
    hypointenseRange = (0.05, 0.14)  # (0.1, 0.25)  # Egész datasetből megállapítnai
    meanHyperThres = 0.495  # based on average from samples
    meanHypoThres = 0.2  # based on average from samples
    sumHyperThres = 30  # from our dataset
    sumHypoThres = 100  # from our dataset

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def __init__(self, depth=3, options = None):

        self.depth = depth
        self.options = options
        self.regionFeatures = []
        self.leafs = []
        self.toMerge = []


    def process(self,ctx):
        self.debug = ctx.get('debug',False)
        image = ctx.get('roi',ctx.get('image'))
        self.image = image
        self.modifiedImage = cv2.cvtColor(
            cv2.normalize(inverseEnhanceImage(self.image).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)



        self.split((0, 0), self.image.shape, self.depth)
        self.result = self.merge()

        if self.debug:
            self.showDebug()
        ctx.set('roi',self.result)
        return ctx
    def merge(self):
        mask = np.zeros_like(self.image).astype(np.uint8)
        for area in self.toMerge:
            row_start, col_start = area[0], area[1]
            row_length, col_length = area[2] + 1, area[3] + 1
            # cv2.rectangle(self.modifiedImage,(area[1],area[0]),(area[1]+area[3],area[0]+area[2]),(255,0,0),2)
            mask[row_start:row_start + row_length, col_start:col_start + col_length] = 1
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(self.modifiedImage, [c], -1, (255, 0, 0), thickness=2)
        t = self.image.copy()
        t[mask == 0] = 0
        return t

    def split(self, start, length, depth, position='Root'):
        start_row, start_col = start
        row_length, col_length = length

        if depth <= 0:  # 3 mély lehet emrgelni. lentről felfele
            if not self.isHomogen(start, length, position, depth - 1):
                self.toMerge.append((start_row, start_col, row_length, col_length))

            return
        if self.isHomogen(start, length, position, depth):
            return

        # cv2.calcHist(self.image[start_row:start_row+row_length,start_col:start_col+col_length])

        if self.debug:
            self.drawOnImage(start_row,row_length,start_col,col_length,depth)

        self.split((start_row, start_col), (row_length // 2, col_length // 2), depth - 1, 'top-left')
        self.split((start_row, start_col + col_length // 2), (row_length // 2, col_length // 2), depth - 1,
                   'top-right')
        self.split((start_row + row_length // 2, start_col), (row_length // 2, col_length // 2), depth - 1,
                   'bottom-left')
        self.split((start_row + row_length // 2, start_col + col_length // 2), (row_length // 2, col_length // 2),
                   depth - 1, 'bottom-right')

    def getStat(self, start, length, position='', depth=100):
        start_row, start_col = start
        row_length, col_length = length

        region = self.image[start_row:start_row + row_length, start_col:start_col + col_length]

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0), density=True)
        norm_hist, norm_bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))

        bin_width = bin_edges[1] - bin_edges[0]
        P = hist * bin_width

        hyperMask = P[
            ((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
        hypoMask = hist[int(self.hypointenseRange[0] * 256): int(self.hypointenseRange[1] * 256)]

        hyperMaskNorm = norm_hist[
            ((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
        hypoMaskNorm = norm_hist[int(self.hypointenseRange[0] * 256): int(self.hypointenseRange[1] * 256)]

        meanHyper = np.mean(hyperMask)  # 0.495
        meanHypo = np.mean(hypoMask)  # 0.2

        return {"mean": meanHyper, \
                "numof": np.sum(hyperMaskNorm), \
                "std": np.std(hyperMask), \
                "entropy": -np.sum(P[P > 0] * np.log2(P[P > 0])), \
                "split 1": (meanHyper > self.meanHyperThres), \
                "split 2": (meanHyper <= self.meanHyperThres and np.sum(hyperMaskNorm) > self.sumHyperThres), \
                "split": ((meanHyper > self.meanHyperThres) or (
                        meanHyper <= self.meanHyperThres and np.sum(hyperMaskNorm) > self.sumHyperThres)) \
                }

    def isHomogen(self, start, length, position='', depth=100):

        start_row, start_col = start
        row_length, col_length = length

        region = self.image[start_row:start_row + row_length, start_col:start_col + col_length]

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0), density=True)
        norm_hist, norm_bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))

        bin_width = bin_edges[1] - bin_edges[0]
        P = hist * bin_width

        # hyperMask = hist[int((self.hyperintenseRange[0] * 256)): int((self.hyperintenseRange[1] * 256))]
        # hypoMask = hist[int(self.hypointenseRange[0] * 256): int(self.hypointenseRange[1] * 256)]
        hyperMask = P[
            ((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
        hypoMask = P[((bin_edges >= self.hypointenseRange[0])[:-1] & (bin_edges <= self.hypointenseRange[1])[:-1])]

        hyperMaskNorm = norm_hist[
            ((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
        hypoMaskNorm = norm_hist[int(self.hypointenseRange[0] * 256): int(self.hypointenseRange[1] * 256)]

        meanHyper = np.mean(hyperMask)  # 0.495
        meanHypo = np.mean(hypoMask)  # 0.2

        if (meanHyper > self.meanHyperThres) or (
                meanHyper <= self.meanHyperThres and np.sum(hyperMaskNorm) > self.sumHyperThres):
            return False  # SPLITELNI SZABAD
        if (meanHypo <= self.meanHypoThres and meanHypo != 0) or (
                meanHypo > self.meanHypoThres and np.sum(hypoMaskNorm) > self.sumHypoThres):
            return False  # SPLITELNI SZABAD

        return True

    def drawOnImage(self,start_row,row_length,start_col,col_length,depth):
        top_left = self.image[start_row:start_row + row_length // 2, start_col:start_col + col_length // 2]
        top_right = self.image[start_row:start_row + row_length // 2,
                    start_col + col_length // 2: start_col + col_length]
        bottom_left = self.image[start_row + row_length // 2: start_row + row_length,
                      start_col:start_col + col_length // 2]
        bottom_right = self.image[start_row + row_length // 2: start_row + row_length,
                       start_col + col_length // 2: start_col + col_length]

        color = (255, 0, 0) if depth == 3 else (0, 255, 0)
        color = color if depth >= 2 else (0, 0, 255)

        offset = 2 if depth < 3 else 0

        cv2.line(self.modifiedImage, (start_col + col_length // 2, start_row),
                 (start_col + col_length // 2, start_row + row_length), self.colors[depth - 1], 1)
        cv2.line(self.modifiedImage, (start_col, start_row + row_length // 2),
                 (start_col + col_length, start_row + row_length // 2), self.colors[depth - 1], 1)

    def showDebug(self):
        from matplotlib import pyplot as plt
        oImage = inverseEnhanceImage(self.image)
        correctedImage = self.image
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(oImage, cmap="gray")
        plt.title("Original Image")
        plt.subplot(2, 2, 2)
        plt.imshow(correctedImage, cmap="gray")
        plt.title("Enhanced Image")

        plt.subplot(2, 2, 3)
        h, b = np.histogram(cv2.normalize(oImage.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX), bins=256,
                            range=(0.0, 1.0))
        h[0] = 0
        plt.plot(h)
        plt.title("Original Histogram")
        plt.xticks(np.arange(0, 256, 50), np.round(b[np.arange(0, 256, 50)], 2))
        plt.subplot(2, 2, 4)
        h2, b2 = np.histogram(correctedImage, bins=256, range=(0.0, 1.0))
        h2[0] = 0
        plt.plot(h2)
        plt.xticks(np.arange(0, 256, 50), np.round(b2[np.arange(0, 256, 50)], 2))
        plt.title("Enhanced Histogram")

        plt.show()

        plt.figure()
        plt.imshow(self.modifiedImage)
        plt.show()