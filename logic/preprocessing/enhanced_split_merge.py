import cv2
import numpy as np

from logic.preprocessing.preprocessing_step import inverseEnhanceImage


class EnhancedSplitMerge:
    hyperintenseRange = (0.82, 1.0)  # (0.47, 0.8)
    hypointenseRange = (0.05, 0.14)  # (0.1, 0.25)
    meanHyperThres =  0.842
    meanHypoThres = 0.09
    sumHyperThres = 300
    sumHypoThres = 100


    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def __init__(self, depth=3, options = None):

        self.depth = depth
        self.options = options
        self.regionFeatures = []
        self.leafs = []
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
        self.toMerge =  self.filterBlocks()
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

    def filterBlocks(self):
        modifiedImage = cv2.cvtColor(
            cv2.normalize(inverseEnhanceImage(self.image).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)

        best, bestScore, rest = self.selectBestArea()
        print()

        sortedList  = sorted(rest, key=self.compareHelper(best, bestScore), reverse=True)
        print()
        return self.chooseBlocks(best,bestScore, sortedList)





    def chooseBlocks(self,best,bestArea,sortedList):
        mergedRegion = np.zeros_like(self.image)
        row_start, col_start, row_length, col_length = best
        mergedRegion[row_start:row_start + row_length, col_start:col_start + col_length] = self.image[row_start:row_start + row_length, col_start:col_start + col_length]



        t=[best]
        for area in sortedList:
            mergedRegionTest = mergedRegion.copy()

            row_start, col_start, row_length, col_length = area
            mergedRegionTest[row_start:row_start + row_length, col_start:col_start + col_length] = self.image[row_start:row_start + row_length,col_start:col_start + col_length]


            #ez nem feltétlenül jó hisz volt már egy max intenzitás, ami legfeljebb nőni tud
            #a legkisebb max intenzitást érdemes nézni?
            mergedScore = self.score(None, mergedRegionTest)[0]
            print(mergedScore)
            print()
            if mergedScore>0.4:
                mergedRegion = mergedRegionTest.copy()
                t.append(area)

        '''from matplotlib import pyplot as plt
        plt.close()
        plt.figure(figsize=(12,8))
        plt.imshow(mergedRegion, cmap="gray")
        plt.show()'''

        return  t
    def compareHelper(self, bestArea, bestScore):
        bScore, bMask = bestScore
        bRow, bCol, bRowLength, bColLength = bestArea
        bCenter = (bCol+(bColLength/2) ,bRow +(bRowLength/2)   )
        bestRegion = self.image[bRow:bRow + bRowLength, bCol:bCol + bColLength]
        bestRegion2 = inverseEnhanceImage(self.image)[bRow:bRow + bRowLength, bCol:bCol + bColLength]


        def cmp(area):
            score, mask = self.score(area)
            row_start, col_start, row_length, col_length = area
            center = (col_start+(col_length/2), (row_start+(row_length/2)) )
            region = self.image[row_start:row_start + row_length, col_start:col_start + col_length]
            region2 = inverseEnhanceImage(self.image)[row_start:row_start + row_length,
                          col_start:col_start + col_length]

            mergedMask = np.zeros_like(self.image).astype(np.uint8)
            mergedMask[bRow:bRow + bRowLength, bCol: bCol+ bColLength] = bMask
            mergedMask[row_start:row_start + row_length, col_start: col_start+ col_length] = mask

            mergedRegion = np.zeros_like(self.image)
            mergedRegion[bRow:bRow + bRowLength, bCol: bCol+ bColLength] = bestRegion2
            mergedRegion[row_start:row_start + row_length, col_start: col_start+ col_length] = region2

            score/bScore

            1-np.abs(np.max(region2) -np.max(bestRegion2))
            alfa = 1
            np.exp((-alfa)*np.abs(self.compactness(mask)[0] - self.compactness(bMask)[0]))


            d_max = np.diag(self.image).shape[0]
            k = 5#self.image.shape[0]// region.shape[0] #5

            d_i = np.linalg.norm((np.array(bCenter)//[bColLength,bRowLength]) - (np.array(center)//[col_length,row_length])) #/d_max
            #d_i = np.linalg.norm(np.array(bCenter) - np.array(center)) #/d_max
            tau = 1.41#-np.log2(2) / 0.7 # d_max / k #0.7#np.diag(region).shape[0] / d_max  # -> kis tau = kis táv->nagyhiba
            distanceScore = np.exp(-d_i / tau) #büntessük azt aki távolabb esik  e^(-d/t)

            mergedScore = self.score(None,mergedRegion)[0]
            mergedMaskCompactness = self.compactness(mergedMask)[0]
            print(f"DISTANCE {distanceScore}, d_i: {d_i},\t  tau: {tau},    updatedScore: {score * distanceScore}  oldScore: {score}, mergedRegionSCore: {mergedScore}, combinedScore: {mergedScore * distanceScore}  mergedMaskCompactness: {mergedMaskCompactness}")


            return d_i, score * distanceScore


        return cmp
    def selectBestArea(self):
        best = self.toMerge[0]
        bestScore = self.score(best)

        rest = self.toMerge
        j = 0
        for i, area in enumerate(rest):
            score, mask = self.score(area, None)
            if score > bestScore[0]:
                bestScore = (score, mask)
                best = area
                j = i
        # rest.pop(j)

        return best, bestScore, rest
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

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        low, high = self.hyperintenseRange
        maskHyper = (bin_centers >= low) & (bin_centers <= high)
        # maskHyper = (bin_edges[:-1] >= low) & (bin_edges[1:] <= high)
        sum = np.sum(bin_centers[maskHyper] * hist[maskHyper])
        N = np.sum(hist[maskHyper])
        avgHyper = stddev = entropy = 0
        if N != 0:
            avgHyper = sum / N
            # szórás
            meanSq = np.sum((bin_centers[maskHyper] ** 2) * hist[maskHyper]) / N
            var = meanSq - avgHyper ** 2
            stddev = np.sqrt(var)
            # entropy
            nonzero = hist[maskHyper] > 0
            p = hist[maskHyper][nonzero] / np.sum(hist[maskHyper][nonzero])
            entropy = -np.sum(p * np.log2(p))

        low, high = self.hypointenseRange
        maskHypo = (bin_centers >= low) & (bin_centers <= high)
        # maskHypo = (bin_edges[:-1] >= low) & (bin_edges[1:] <= high)
        sum = np.sum(bin_centers[maskHypo] * hist[maskHypo])
        N = np.sum(hist[maskHypo])
        avgHypo = 0

        if N != 0:
            avgHypo = sum / N
        return {"mean": avgHyper,
                "numof": np.sum(hist[maskHyper]),
                "std": np.std(stddev),
                "entropy": entropy,
                "split 1": (avgHyper > self.meanHyperThres),
                "split 2": (avgHyper <= self.meanHyperThres and np.sum(hist[maskHyper]) > self.sumHyperThres),
                "split": ((avgHyper > self.meanHyperThres) or (
                        avgHyper <= self.meanHyperThres and np.sum(hist[maskHyper]) > self.sumHyperThres))
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



    def score(self,area, region=None,):
        if region is None:
            row_start, col_start, row_length, col_length = area
            region = self.image[row_start:row_start + row_length, col_start:col_start + col_length]
            region2 = inverseEnhanceImage(self.image)[row_start:row_start + row_length, col_start:col_start + col_length]
        else:
            region2=region
        rMax  = np.max(region2)
        rMean = np.mean(region2)
        rStd = np.std(region2)
        rP95Int= np.percentile(region2, 95)
        hist, _ = np.histogram(region2, bins=32, range=(region2.min(), region2.max()))
        p = hist / np.sum(hist) if np.sum(hist) > 0 else hist
        entropy = -np.sum(p * np.log2(p + 1e-9))



        norm_region = cv2.normalize(region2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        thresh, _ = cv2.threshold(norm_region[norm_region>0],0,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        mask = np.zeros_like(norm_region)
        mask[norm_region>thresh] = 255




        compactness, perimeter, area, contourImage = self.compactness(mask)

        alfa = 1 #[0-1]
        C_score = np.exp(-alfa * np.abs(compactness-1))   #---> 1 helyen 1 minden máshol 1 alatt vesz fel értéket

        beta=2#3.6
        gamma=0.8

        score = ( np.power(rMax,beta) ) * np.power(C_score,gamma)   # beta,gamma---> fontossága a változóknak --> kisebb 1 ->drasztikusan csökken az érték


        '''if self.debug:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(8, 6), frameon=False)
            plt.subplot(1,4,1)
            plt.imshow(inverseEnhanceImage(region), cmap="gray")
            plt.title(f"thresh {thresh}")#(f"rMax {rMax}, rMean {rMean}, rStd {rStd}, rP95Int {rP95Int}")
            plt.subplot(1,4,2)
            plt.imshow(mask, cmap="gray")
            plt.subplot(1, 4, 3)
            plt.title(str("%.2f" % perimeter) + "  a   "+ str("%.2f" % area))
            plt.imshow(contourImage, cmap="gray")
            #plt.subplot(1,4,4)
            #plt.imshow(edges, cmap="gray")
            #plt.subplot(1,4,4)
            #plt.imshow(edges)
            plt.show(block = True)
        '''
        print(f"rMax {rMax}, rMean {rMean}, rStd {np.std(region2[mask==255])}, rP95Int {rP95Int},  ENTROPY {entropy}, thresh {thresh}, compactness {'%.2f'%compactness}, SCORE {score}")



        return score, mask
    def compactness(self, mask):

        contourImage = np.zeros_like(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        perimeter = 0
        for c in contours:
            perimeter += cv2.arcLength(c, True)
        cv2.drawContours(contourImage, contours, -1, 255, thickness=cv2.FILLED)

        area = np.count_nonzero(mask)

        compactness = np.inf
        if area > 0:
            compactness = (perimeter ** 2) / (
                        4 * np.pi * area)  # normalized version-->close to 1 (orig: p^2 / a --> close to 4pi)
        return compactness, perimeter, area, contourImage


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

