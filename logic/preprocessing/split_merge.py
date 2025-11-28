import cv2
import numpy as np

from logic.preprocessing.preprocessing_step import inverseEnhanceImage


class SplitMerge:
    hyperintenseRange = (0.82, 1.0)  # (0.47, 0.8)  # Egész datasetből megállapítnai
    hypointenseRange = (0.05, 0.14)  # (0.1, 0.25)  # Egész datasetből megállapítnai
    meanHyperThres =  0.842 #0.495  # based on average from samples
    meanHypoThres = 0.09  # based on average from samples
    sumHyperThres = 300  # from our dataset
    sumHypoThres = 100  # from our dataset

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def __init__(self, depth=3):

        self.depth = depth
        self.toMerge = []


    def process(self,ctx):
        self.toMerge = []
        self.debug = ctx.get('debug',False)
        image = ctx.get('roi',ctx.get('image'))
        self.loadParams(ctx)
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

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))

        bin_centers = (bin_edges[:-1]+bin_edges[1:]) / 2
        low, high = self.hyperintenseRange
        maskHyper = (bin_centers>=low) & (bin_centers<=high)
        #maskHyper = (bin_edges[:-1] >= low) & (bin_edges[1:] <= high)
        sum = np.sum(bin_centers[maskHyper] * hist[maskHyper])
        N = np.sum(hist[maskHyper])
        avgHyper = stddev = entropy = 0
        if N!=0:
            avgHyper = sum / N
            #szórás
            meanSq = np.sum((bin_centers[maskHyper] ** 2) * hist[maskHyper]) / N
            var = meanSq - avgHyper **2
            stddev = np.sqrt(var)
            #entropy
            nonzero = hist[maskHyper]>0
            p = hist[maskHyper][nonzero] / np.sum(hist[maskHyper][nonzero])
            entropy = -np.sum(p * np.log2(p))

        low, high = self.hypointenseRange
        maskHypo = (bin_centers >= low) & (bin_centers <= high)
        #maskHypo = (bin_edges[:-1] >= low) & (bin_edges[1:] <= high)
        sum = np.sum(bin_centers[maskHypo] * hist[maskHypo])
        N = np.sum(hist[maskHypo])
        avgHypo = 0

        if N != 0:
            avgHypo = sum / N
        return {"mean": avgHyper, \
                "numof": np.sum(hist[maskHyper]), \
                "std": np.std(stddev), \
                "entropy": entropy, \
                "split 1": (avgHyper > self.meanHyperThres), \
                "split 2": (avgHyper <= self.meanHyperThres and np.sum(hist[maskHyper]) > self.sumHyperThres), \
                "split": ((avgHyper > self.meanHyperThres) or (
                        avgHyper <= self.meanHyperThres and np.sum(hist[maskHyper]) > self.sumHyperThres)) \
                }

    def isHomogen(self, start, length, position='', depth=100):

        start_row, start_col = start
        row_length, col_length = length

        region = self.image[start_row:start_row + row_length, start_col:start_col + col_length]

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))
        hist[0]=0
        avgHyper, sumHyper = self.getMeanSumIntensity(hist,bin_edges,self.hyperintenseRange)
        avgHypo, sumHypo = self.getMeanSumIntensity(hist,bin_edges,self.hypointenseRange)

        if (avgHyper > self.meanHyperThres) or (
                avgHyper <= self.meanHyperThres and sumHyper > self.sumHyperThres):

            return False
        if (avgHypo <= self.meanHypoThres and avgHypo > 1e-9 and sumHypo>10) or (
                avgHypo > self.meanHypoThres and sumHypo > self.sumHypoThres):

            return False

        return True

    def getMeanSumIntensity(self,hist,bin_edges,intensityRange):
        low, high = intensityRange
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mask = (bin_centers >= low) & (bin_centers <= high)
        mask = (bin_edges[:-1] >= low) & (bin_edges[1:] <= high)
        sum = np.sum(bin_centers[mask] * hist[mask])
        N = np.sum(hist[mask])
        avgHyper = 0
        if N != 0:
            avgHyper = sum / N

        return avgHyper, N

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

        plt.figure(figsize=(12, 8))
        plt.imshow(self.modifiedImage)
        plt.show()

    def loadParams(self,ctx):
        self.depth = ctx.params.get("qt:depth",self.depth)

        hyperMin= ctx.params.get("qt:hyperMin",self.hyperintenseRange[0])
        hyperMax = ctx.params.get("qt:hyperMax",self.hyperintenseRange[1])
        self.hyperintenseRange = (hyperMin,hyperMax)

        hypoMin = ctx.params.get("qt:hypoMin",self.hypointenseRange[0])
        hypoMax = ctx.params.get("qt:hypoMax",self.hypointenseRange[1])
        self.hypointenseRange = (hypoMin,hypoMax)

        self.meanHyperThres = ctx.params.get("qt:meanHyperTresh",self.meanHyperThres)
        self.meanHypoThres = ctx.params.get("qt:meanHypoTresh", self.meanHypoThres)

        self.sumHyperThres = ctx.params.get("qt:sumHyperThresh", self.sumHyperThres)
        self.sumHypoThres = ctx.params.get("qt:sumHypoThresh", self.sumHypoThres)

        return