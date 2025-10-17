from collections import namedtuple
import numpy as np


Window = namedtuple("Window", ["x", "y", "w", "h", "score"])
class SlidingWindow:
    def __init__(self,windowSize=(8,8), step=4, overlapping=True, iou=0.5,kTop=5, mode='std'):
        self.debug = True
        self.windowSize = windowSize
        self.step = step
        self.overlapping = overlapping
        self.iouThres=iou
        self.kTop=kTop
        self.mode = mode

    def sliding(self,image):
        H, W = image.shape[:2]
        wH, wW = self.windowSize

        if isinstance(self.step, int):
            sh, sw = self.step, self.step
        else:
            sh, sw = self.step

        if not self.overlapping:
            sh, sw = wH, wW

        for y in range(0, H- wH+1, sh):
            for x in range(0, W-wW+1, sw):
                yield (x,y, image[y:y+wH, x:x+wW])

    def scoreWindow(self, image, window, mode="mean"):
        score=0
        if mode=="mean":
            score = np.mean(window)
        elif mode=="max":
            score = np.max(window)
        elif mode=="std":
            score = np.std(window)

            globalMean = np.mean(image[image>0])
            values = window[window>0]
            score = 0.0
            if values.size != 0:
                score = np.sqrt(np.mean((values-globalMean)**2))
        elif mode=="blob":
            #from scipy.ndimage import gaussian_laplace,gaussian_filter
            score = 0.0
            vals = image[image>0]
            if len(window[window>0])>0:
                score = np.abs(window[window>0].mean() - vals.mean()) / (vals.std() + 1e-6)
        elif mode=="entropy":
            #from skimage.measure import shannon_entropy
            #score = shannon_entropy(window)
            score = 0.0

            globalHist  =self.computeNormHist(image)
            localHist = self.computeNormHist(window)

            mask = (localHist>0) & (globalHist >0)
            #-summa p * log p helyett
            #summa P * log(P/Q) ahol P a window a Q az image
            score = np.sum(localHist[mask] * np.log(localHist[mask] / globalHist[mask]))




        elif mode=="all":
            score = 0.0

        return score
    def computeNormHist(self,data, bins=256):
        hist, _ = np.histogram(data[data > 0], bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-10
        return hist

    def iou(self,win1,win2):
        x1, y1, w1, h1,_ = win1
        x2, y2, w2, h2,_ = win2

        xx1 = max(x1,x2)
        yy1 = max(y1,y2)
        xx2 = min(x1+w1,x2+w2)
        yy2 = min(y1+h1,y2+h2)

        inter_w = max(0, xx2 - xx1)
        inter_h = max(0, yy2 - yy1)
        inter = inter_w*inter_h

        area1 = w1*h1
        area2 = w2*h2

        union = area1 + area2 - inter

        return inter/union if union>0 else 0


    def nonMaxSupression(self,windows, iou_threshold=0.5):
        windows =  sorted(windows, key=lambda w: w.score, reverse=True)
        kept=[]
        while windows:
            best = windows.pop(0)
            kept.append(best)
            windows = [w for w in windows if self.iou(best,w) < iou_threshold]

        return kept
    def nonMaxSupression2(self,scoreMap,size=3, th=0):
        from scipy.ndimage import  maximum_filter
        localMax = maximum_filter(scoreMap,size=size) == scoreMap
        maxima = np.argwhere(localMax & (scoreMap>th))

        return maxima



    def process(self, ctx):

        image = ctx.get('image')
        self.debug=ctx.get('debug')
        self.loadParams(ctx)

        windows = []
        scoreMap=np.zeros((image.shape[0],image.shape[1]), dtype=float)

        for (x, y, window) in self.sliding(image):
            score = self.scoreWindow(image,window,mode=self.mode)
            windows.append(Window(x,y,window.shape[1],window.shape[0], score))
            scoreMap[y+window.shape[0]//2,x + window.shape[1]//2] = score


        selected = []#self.nonMaxSupression(windows,self.iouThres)
        #selected = sorted(selected, key=lambda w: w.score, reverse=True)[:self.kTop]

        #ctx.set('windows',selected)

        maxima = self.nonMaxSupression2(scoreMap)
        h,w = self.windowSize
        candidates = [(x,y,scoreMap[y,x]) for y,x in maxima]
        candidates.sort(key=lambda c: c[2], reverse=True)

        selected2 = []
        for cx,cy, score in candidates:
            #if all(abs(cx - win.x+w//2) >= w // 2 or abs(cy-win.y+h//2) >= h//2 for win in selected2):
            selected2.append((Window(cx-w//2,cy-h//2,w,h,score)))
            if len(selected2) >= self.kTop:
                break
        ctx.set('windows', selected2)

        if self.debug:
            self.showDebug(scoreMap,image,selected, selected2)
        return ctx


    def showDebug(self, scoreMap,image,selected, selected2):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.title("scoreMap")
        plt.imshow(scoreMap, cmap="gray")

        import cv2
        i = cv2.cvtColor(cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                         cv2.COLOR_GRAY2RGB)
        i2 = cv2.cvtColor(cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                         cv2.COLOR_GRAY2RGB)

        for w in selected:
            i[w.y + w.h // 2, w.x + w.w // 2] = (255, 0, 0)
            cv2.circle(i, center=(w.x + np.round(w.w / 2).astype(np.int32), w.y + np.round(w.h / 2).astype(np.int32)),
                       radius=4, color=(255, 0, 0))
            cv2.rectangle(i,(w.x,w.y),(w.x+w.w,w.y+w.h),(0,0,255))
        for w in selected2:
            i2[w.y + w.h // 2, w.x + w.w // 2] = (255, 0, 0)
            cv2.circle(i2, center=(w.x + np.round(w.w / 2).astype(np.int32), w.y + np.round(w.h / 2).astype(np.int32)),
                       radius=4, color=(255, 0, 0))
            cv2.rectangle(i2,(w.x,w.y),(w.x+w.w,w.y+w.h),(0,0,255))

        plt.subplot(1, 3, 2)
        plt.title('image')
        plt.imshow(i)
        plt.subplot(1, 3, 3)
        plt.title('image2')
        plt.imshow(i2)
        plt.show()
        print(scoreMap.max(), scoreMap[scoreMap > 0].min(), scoreMap[scoreMap > 0].mean())

    def loadParams(self,ctx):
        h = ctx.params.get("sw:w_height",self.windowSize[0])
        w = ctx.params.get("sw:w_width",self.windowSize[1])
        self.windowSize = (h,w)

        self.step = ctx.params.get("sw:step",self.step)
        #self.overlapping = overlapping
        #self.iouThres=iou
        self.kTop=ctx.params.get("sw:top_k",self.kTop)
        self.mode = ctx.params.get("sw:mode",self.mode)