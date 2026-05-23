import cv2
import numpy as np
from scipy.signal import find_peaks

from logic.preprocessing.preprocessing_step import enhanceImage
from logic.seed_selector.seed_selector import SeedSelector

class DivergenceSeedSelector(SeedSelector):
    def __init__(self, enhanced=False):
        self.debug=False
        self.ctx={}
        self.enhanced = enhanced

    def calcRoiHistogram(self,region):
        hist, bin_edges = np.histogram(region, bins=256, range=(0.0, 1.0))
        hist[0] = 0  # ha üres terület belelóg nem érdekel

        return  hist, bin_edges

    def setOptimalThreshold(self,div,bin_edges):
        idx = np.argmax(div)
        tail = div[idx + 1:]

        nearest_zero = np.argmin(np.abs(tail))
        #if nearest_zero == len(tail)-1 and len(tail)>0:
        #    nearest_zero = np.argmin(np.abs(tail[:-1]))


        optimal_threshold_idx = nearest_zero + 1 + idx
        optimal_threshold = bin_edges[optimal_threshold_idx]

        return idx,tail,nearest_zero, optimal_threshold_idx,optimal_threshold
    def setOptimalThresholdEnhanced(self,div,bin_edges):

        img_hist, img_bin_edges = np.histogram(self.ctx.get("image"), bins=256, range=(0.0, 1.0))
        img_hist[0] = 0
        globalMaxIdxImg =np.argmax(img_hist)


        peaks, props = find_peaks(div, prominence=0.1 * np.max(div))
        #candidatePeaks = peaks[peaks>globalMaxIdxImg-10]
        #choosenPeak = candidatePeaks[np.argmin(np.abs(candidatePeaks-globalMaxIdxImg))]


        score = (0.7*np.abs(peaks-globalMaxIdxImg)/len(div)) +  0.3* (1 - (div[peaks]/np.max(div)))
        score[score<=0]=np.inf
        choosenPeak = peaks[np.argmin(score)]


        #score = (np.abs(peaks-globalMaxIdxImg) / len(div) - 0.3 * props["prominences"] / np.max(props["prominences"]))
        #choosenPeak = peaks[np.argmin(score)]

        idx = choosenPeak
        tail = div[idx + 1:]

        nearest_zero = np.argmin(np.abs(tail))

        '''eps = 1e-3
        for i in range(len(tail)):
            if abs(tail[i])< eps:
                nearest_zero = i
                break
        '''
        eps = 0.03 * np.max(div)
        window=3
        for i in range(len(tail) - window):
            if np.mean(abs(tail[i:i+window])) < eps:
                nearest_zero = i
                break


        '''window = 7
        eps_mean = 0.03 * np.max(div)
        eps_std = 0.02 * np.max(div)

        for i in range(len(tail)- window):
            seg = tail[i:i+window]
            if np.mean(np.abs(seg)) < eps_mean and np.std(seg) < eps_std:
                nearest_zero = i
                break

        for i in range(len(tail)-window):
            if np.std(tail[i:i+window])<eps_std:
                nearest_zero = i
                break
        '''

        optimal_threshold_idx = nearest_zero + 1 + idx
        optimal_threshold = bin_edges[optimal_threshold_idx]

        return idx,tail,nearest_zero, optimal_threshold_idx,optimal_threshold, choosenPeak

    def setOptimalThresholdEnhanced2(self,div,bin_edges):


        peaks, props = find_peaks(div, prominence=0.1 * np.max(div))

        maxVal = np.max(div)
        candidates = peaks[div[peaks] > 0.7 * maxVal]
        choosenPeak = np.min(candidates)



        idx = choosenPeak
        tail = div[idx + 1:]

        nearest_zero = np.argmin(np.abs(tail))

        eps = 1e-3
        for i in range(len(tail)):
            if abs(tail[i])< eps:
                nearest_zero = i
                break


        optimal_threshold_idx = nearest_zero + 1 + idx
        optimal_threshold = bin_edges[optimal_threshold_idx]

        return idx,tail,nearest_zero, optimal_threshold_idx,optimal_threshold


    def select(self,ctx):
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import savgol_filter
        self.debug=ctx.get('debug',False)
        self.ctx = ctx
        region = ctx.get('roi',ctx.get('image'))
        image = enhanceImage(ctx.get('image'))
        ctx.set('image',image)
        ''' 1. Calc Histogram '''
        hist, bin_edges = self.calcRoiHistogram(region)

        #peaks,_ = find_peaks(hist, prominence=0.05 * np.max(hist))
        #peakDensity = len(peaks)/len(hist)
        #if peakDensity>0.05:
        if self.enhanced:
            hist = gaussian_filter1d(hist, sigma=1.1)
            #win = max(7, int(0.05 * len(hist)) | 1)
            #hist = savgol_filter(hist, win, polyorder=2)

        ''' 2. Calc Divergence '''

        P=hist
        s = np.sum(hist)
        if s>0:
            P = hist / s

        div = np.gradient(hist) * P

        '''from matplotlib import pyplot as plt
        plt.figure()

        plt.plot(np.gradient(hist) * P,alpha=0.5)
        '''

        '''smooth_sigma = 1
        r = int(3 * smooth_sigma) +1
        x = np.arange(-r,r+1)
        kernel = np.exp(-(x**2) / (2.0 * smooth_sigma **2))
        kernel = kernel / kernel.sum()
        P = np.convolve(P, kernel, mode='same' )
        '''

        '''
        plt.plot(div*10000, alpha=0.5)
        plt.plot(np.gradient(hist) * P,alpha=0.5)

        plt.legend(["gradient","div","b","c"])

        plt.title("histogram")
        plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))
        plt.show()
        '''

        idx, tail, nearest_zero, optimal_threshold_idx, optimal_threshold = 0, 0, 0, 0, 0
        ''' 3. Set optimal threshold'''
        if self.enhanced:
            idx,tail, nearest_zero, optimal_threshold_idx,optimal_threshold, choosenMaxPeak = self.setOptimalThresholdEnhanced(div,bin_edges)

            if optimal_threshold_idx >= len(div)-20 and optimal_threshold_idx-choosenMaxPeak<= 20:
                div[choosenMaxPeak:]= -div[choosenMaxPeak]
                #div[optimal_threshold_idx] = div[choosenMaxPeak]
                idx, tail, nearest_zero, optimal_threshold_idx, optimal_threshold,_ = self.setOptimalThresholdEnhanced(div, bin_edges)
            elif optimal_threshold_idx >= len(div)-20:
                div[-20:] = -div[choosenMaxPeak]
                idx, tail, nearest_zero, optimal_threshold_idx, optimal_threshold,_ = self.setOptimalThresholdEnhanced(div, bin_edges)

        else:
            idx,tail, nearest_zero, optimal_threshold_idx,optimal_threshold = self.setOptimalThreshold(div,bin_edges)

        if self.debug:
            self.showOptThreshold(hist,bin_edges,div,idx,optimal_threshold_idx)
        ''' 4. Automatic seed selection'''
        mask = np.zeros_like(image)
        mask[region >= bin_edges[nearest_zero + 1 + idx]] = 1
        seeds = np.argwhere(mask == 1)  # (row,col ) (y,x)
        seeds= seeds[:, ::-1]  # (x,y)

        if self.debug:
            self.showSeeds(image, region,mask)

        ctx.set('optimal_threshold',optimal_threshold)
        ctx.set('seeds',seeds)
        return ctx

    def showOptThreshold(self,hist,bin_edges,div,idx,optimal_threshold_idx):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(hist)
        plt.title("histogram")
        plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))

        plt.subplot(1, 2, 2)
        plt.plot(div)
        plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))
        plt.scatter(idx, div[idx], c="green")
        plt.scatter(optimal_threshold_idx, div[optimal_threshold_idx], c="red")

        plt.title(f"Divergence, Thres: ({bin_edges[optimal_threshold_idx]},{div[optimal_threshold_idx]:.4f})")

        plt.show()
    def showSeeds(self, image, region,mask):
        from matplotlib import pyplot as plt
        coloredImage = cv2.cvtColor(
            cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        regionCopy = region
        regionCopy[regionCopy != 0] = 1
        regionCopy = regionCopy.astype(np.uint8)
        contours, _ = cv2.findContours(regionCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cv2.drawContours(coloredImage, [c], -1, (255, 0, 0), thickness=2)

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title("teljes kép")
        plt.imshow(coloredImage)
        plt.subplot(1, 2, 2)
        plt.title("seedpontok")
        plt.imshow(mask, cmap="binary")
        plt.show()