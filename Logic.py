import nibabel as nib
import cv2
import numpy as np
from collections import deque


class Logic:
    def __init__(self):
        pass

    def loadNifti(self, path):
        img = nib.load(path)
        return img.get_fdata()

    def select5Seed(self, img, segImg=None):

        skullStrippedImg = self.skullStripping(img)
        seeds = self.seedSelection(skullStrippedImg)
        binary_seg = (segImg != 0).astype(int)
        masks = self.evaluate(skullStrippedImg, seeds, binary_seg)

        self.score(masks, binary_seg, seeds)


    def skullStripping(self, img):
        ''' Calculate Otsu’s Threshold '''
        norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thres, res = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Az Otsu csak normalizált
        orig_thres = (thres / 255) * (img.max() - img.min()) + img.min()



        ''' Threshold the image '''
        img_c = img.copy()
        img_c[img_c < orig_thres] = 0
        norm_img_c = norm_img.copy()
        norm_img_c[norm_img_c < thres] = 0

        _, bw = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)

        ''' Open the binary image using a disk structuring Seed '''
        radius = 4
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, se)

        '''5: Dilate the binary image'''
        bw = cv2.morphologyEx(bw, cv2.MORPH_DILATE, se)


        '''6: Select the largest binary image'''

        contours, _ = cv2.findContours(bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        largest_blob_mask = np.zeros(bw.shape, dtype=np.uint8)

        cv2.drawContours(largest_blob_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)


        ''' 7: Dilate the binary image ''' '''     BW ← imclose(BW,se) --->?'''
        bw = cv2.morphologyEx(largest_blob_mask, cv2.MORPH_CLOSE, se)

        ''' 8: Fill holes on the binary image ''' '''BW ← imfill(BW,se)'''
        start_point = (0, 0)
        mask = np.zeros((bw.shape[0] + 2, bw.shape[1] + 2), dtype=np.uint8)
        retval, resImage, bw, rect = cv2.floodFill(bw, mask, start_point, 255, 30, 30,
                                                   flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                                                               255 << 8))

        bw = cv2.morphologyEx(largest_blob_mask, cv2.MORPH_CLOSE, se)

        ''' 9: Remove the skull '''   ''''stripped ← im(!BW) = 0'''
        norm_img[~bw.astype(bool)] = 0

        img[~bw.astype(bool)] = 0
        norm_img[~bw.astype(bool)] = 0

        return img  # norm_img

    def seedSelection(self, img):
        '''  Resize the Image''' '''im ← imresize(im, [256, 256]) '''
        resized_im = np.array(img).copy()  # cv2.resize(img,[256,256])

        ''' Iterate through each 8 × 8 block 
                Collect the mean of each block
                Collect the centers of each block
        '''
        width = resized_im.shape[0] if resized_im.shape[0] % 8 == 0 else 256  # btw resize img kéne
        height = resized_im.shape[1] if resized_im.shape[1] % 8 == 0 else 256
        if width == 256 or height == 256:
            resized_im = cv2.resize(img, [256, 256])
            width = 256
            height = 256

        mIs = []
        cBs = []
        for i in range(0, width, 8):
            for j in range(0, height, 8):
                block = resized_im[i:i + 8, j:j + 8]
                mIs.append(np.mean(block))
                cBs.append((j + 3, i + 3))

        ''' 10: Select top 5 blocks based on the intensity '''

        top5Indicies = np.argsort(mIs)[-5:][::-1]
        seeds = np.array(cBs)[top5Indicies]
        return seeds

    def evaluate(self, img, seeds, ref_mask=None):
        width = 240
        height = 240

        masks = []

        seedImg = cv2.cvtColor(
            cv2.resize(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), (width, height)),
            cv2.COLOR_GRAY2BGR)

        mask = np.zeros((seedImg.shape[0] + 2, seedImg.shape[1] + 2), dtype=np.uint8)
        for i, seed in enumerate(seeds):
            cv2.circle(seedImg, seed, 2, [0, 0, 255], 1)
            mask = np.zeros((seedImg.shape[0] + 2, seedImg.shape[1] + 2), dtype=np.uint8)
            floodFill = img.copy()

            retval, resImage, mask, rect = cv2.floodFill(
                cv2.normalize(floodFill, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mask, seed, 255, 75,
                75,
                flags=4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                        255 << 8))
            '''retval, resImage, mask, rect = cv2.floodFill(cv2.normalize(floodFill.astype(np.float32), None, 0, 1.0, cv2.NORM_MINMAX), mask, seed, 1.0,  0.3,
                                                         0.3,
                                                         flags=4  | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (
                                                                     255 << 8))'''


            masks.append(mask)



        return masks

    def score(self, masks, ref_mask, seeds):
        GREEN = [0, 255, 0]  # TP
        BLACK = [0, 0, 0]  # TN
        RED = [255, 0, 0]  # FP
        BLUE = [0, 0, 255]  # FN

        scores = []


        for i in range(len(masks)):
            mask = masks[i][1:-1, 1:-1]

            TP_mask = (mask == 255) & (ref_mask == 1)
            TN_mask = (mask == 0) & (ref_mask == 0)
            FP_mask = (mask == 255) & (ref_mask == 0)
            FN_mask = (mask == 0) & (ref_mask == 1)

            TP = np.sum(TP_mask)
            TN = np.sum(TN_mask)
            FP = np.sum(FP_mask)
            FN = np.sum(FN_mask)
            scores.append({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
            diff_img = np.zeros((ref_mask.shape[0], ref_mask.shape[1], 3), dtype=np.uint8)
            diff_img[TP_mask] = GREEN
            diff_img[TN_mask] = BLACK
            diff_img[FP_mask] = RED
            diff_img[FN_mask] = BLUE

            DSS = TP / (0.5 * (2 * TP + FP + FN))


        return scores

    '''
    12: for m = 1 : 5 do
    13: ROIm= Region-growing(seedm)
    14: end for
    15: Compare each ROIm against GT using evaluation parameters for m = 1 : 5
    16: Select the best ROI as a final segmentation output.
    '''

    '''In preprocessing'''
    ''' image convert  to Double Prescision min 0 max 1 value'''
    ''' Remove backgorund value threshold 0.023'''
    '''  lényegében előállítja aképet amink már van.
        - majd gamma-law transformation algorhitm  
        to expand the narrow range of low input gray level values of the DW images to a wider range

        s = cr**gamma gamma = 0.4
            c is amplitude


     '''


    def findSeedPoints(self, region, image, groundTruth=None):
        image = enhanceImage(image)

        ''' 1. Calc Histogram '''
        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))
        hist[0] = 0  # ha üres terület belelóg nem érdekel (divergence-nél is negatív lenne)
        ''' 2. Calc Divergence '''
        div = np.gradient(hist)
        ''' 3. Set optimal threshold'''

        idx = np.argmax(div)
        tail = div[idx + 1:]

        nearest_zero = np.argmin(np.abs(tail))
        optimal_threshold_idx = nearest_zero + 1 + idx
        optimal_threshold = bin_edges[optimal_threshold_idx]


        ''' 4. Automatic seed selection'''
        mask = np.zeros_like(image)
        mask[region >= bin_edges[nearest_zero + 1 + idx]] = 1
        seeds = np.argwhere(mask == 1)  # (row,col ) (y,x)

        coloredImage = cv2.cvtColor(
            cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        regionCopy = region
        regionCopy[regionCopy != 0] = 1
        regionCopy = regionCopy.astype(np.uint8)
        contours, _ = cv2.findContours(regionCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(coloredImage, [c], -1, (255, 0, 0), thickness=2)


        '''5,6,7 region growing'''
        accumulatedMask = np.zeros_like(image)
        for seed in seeds:
            if accumulatedMask[
                seed[0], seed[1]] == 1:  # Ha még nem tartozik hozzá egy segmented régióhoz se akkor kell csak számolni
                continue
            segmented = self.regionGrowing(image, seed, optimal_threshold)
            accumulatedMask[segmented == 1] = 1
            # accumulatedMask |= segmented

        if groundTruth is not None:

            binary_seg = (groundTruth != 0).astype(int)




        if groundTruth is not None:
            binary_seg = (groundTruth != 0).astype(int)
            self.score2(accumulatedMask, binary_seg)


        return accumulatedMask

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

    def score2(self, mask, ref_mask):
        GREEN = [0, 255, 0]  # TP
        BLACK = [0, 0, 0]  # TN
        RED = [255, 0, 0]  # FP
        BLUE = [0, 0, 255]  # FN

        scores = []

        TP_mask = (mask == 1) & (ref_mask == 1)
        TN_mask = (mask == 0) & (ref_mask == 0)
        FP_mask = (mask == 1) & (ref_mask == 0)
        FN_mask = (mask == 0) & (ref_mask == 1)

        TP = np.sum(TP_mask)
        TN = np.sum(TN_mask)
        FP = np.sum(FP_mask)
        FN = np.sum(FN_mask)
        scores.append({"TP": TP, "TN": TN, "FP": FP, "FN": FN})
        diff_img = np.zeros((ref_mask.shape[0], ref_mask.shape[1], 3), dtype=np.uint8)
        diff_img[TP_mask] = GREEN
        diff_img[TN_mask] = BLACK
        diff_img[FP_mask] = RED
        diff_img[FN_mask] = BLUE

        DSS = TP / (0.5 * (2 * TP + FP + FN))

        return scores

    def executeAlg2(self, img, img_seg=None):
        QTree = QuadTree(img)
        treeRes = QTree.result

        mask = self.findSeedPoints(treeRes, img, img_seg)
        return mask

class QuadTree():
    hyperintenseRange = (0.8, 1.0)  # (0.47, 0.8)  # Egész datasetből megállapítnai
    hypointenseRange = (0.05, 0.14)  # (0.1, 0.25)  # Egész datasetből megállapítnai
    meanHyperThres = 0.495  # based on average from samples
    meanHypoThres = 0.2  # based on average from samples
    sumHyperThres = 30  # from our dataset
    sumHypoThres = 100  # from our dataset

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    def __init__(self, image, depth=3):

        self.image = image  # image.swapaxes(-2,-1)[...,::-1]#image
        self.depth = depth

        self.regionFeatures = []
        self.leafs = []
        self.toMerge = []

        self.modifiedImage = cv2.cvtColor(
            cv2.normalize(self.image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)

        self.correctedImage = enhanceImage(self.image)

        h, b = np.histogram(cv2.normalize(self.image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX), bins=256,
                            range=(0.0, 1.0))
        h[0] = 0

        h2, b2 = np.histogram(self.correctedImage, bins=256, range=(0.0, 1.0))
        h2[0] = 0


        self.split((0, 0), self.image.shape, depth)
        self.result = self.merge()

        cv2.waitKey(0)

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
        t = self.correctedImage.copy()
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

        self.split((start_row, start_col), (row_length // 2, col_length // 2), depth - 1, 'top-left')
        self.split((start_row, start_col + col_length // 2), (row_length // 2, col_length // 2), depth - 1, 'top-right')
        self.split((start_row + row_length // 2, start_col), (row_length // 2, col_length // 2), depth - 1,
                   'bottom-left')
        self.split((start_row + row_length // 2, start_col + col_length // 2), (row_length // 2, col_length // 2),
                   depth - 1, 'bottom-right')

    def getStat(self, start, length, position='', depth=100):
        start_row, start_col = start
        row_length, col_length = length

        region = self.correctedImage[start_row:start_row + row_length, start_col:start_col + col_length]

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0), density=True)
        norm_hist, norm_bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))

        bin_width = bin_edges[1] - bin_edges[0]
        P = hist * bin_width

        hyperMask = P[((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
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

        region = self.correctedImage[start_row:start_row + row_length, start_col:start_col + col_length]

        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0), density=True)
        norm_hist, norm_bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))

        bin_width = bin_edges[1] - bin_edges[0]
        P = hist * bin_width

        # hyperMask = hist[int((self.hyperintenseRange[0] * 256)): int((self.hyperintenseRange[1] * 256))]
        # hypoMask = hist[int(self.hypointenseRange[0] * 256): int(self.hypointenseRange[1] * 256)]
        hyperMask = P[((bin_edges >= self.hyperintenseRange[0])[:-1] & (bin_edges <= self.hyperintenseRange[1])[:-1])]
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


''' A WIKIS JÓ LEÍRÁSNAK TŰNIK'''
'''
1.Define the criterion to be used for homogeneity
2.Split the image into equal size regions
3.Calculate homogeneity for each region
4.If the region is homogeneous, then merge it with neighbors
5.The process is repeated until all regions pass the homogeneity test
'''

'''BTIW Ő IS JÓNAK TŰNIK'''
''' https://stackoverflow.com/questions/7050164/image-segmentation-split-and-merge-quadtrees '''


def enhanceImage(image, gamma=0.4, c=1):
    norm_img = cv2.normalize(image.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
    return c * np.power(norm_img, gamma)