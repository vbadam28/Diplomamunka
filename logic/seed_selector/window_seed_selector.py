from logic.seed_selector.seed_selector import SeedSelector
import numpy as np

class WindowSeedSelector(SeedSelector):
    def __init__(self,mode='center'):
        self.debug=True
        self.mode=mode

    def select(self,ctx):
        self.debug = ctx.get('debug')
        image = ctx.get('image')
        windows = ctx.get('windows')
        seeds=[]
        if self.mode=="center":
            for w in windows:
                seeds.append( (w.x + w.w//2, w.y+w.h//2) )
        elif self.mode=="max":
            for w in windows:
                windowSeed = np.argwhere(image[w.y:w.y+w.h,w.x:w.x+w.w]==np.max(image[w.y:w.y+w.h,w.x:w.x+w.w]))[0]

                seed = (windowSeed[1]+w.x,windowSeed[0]+w.y)
                seeds.append(seed)
        ctx.set('seeds',seeds)

        if self.debug:
            self.showSeeds(ctx.get('image'),seeds, windows)

        return ctx

    def showSeeds(self,image,seeds, windows):
        from matplotlib import pyplot as plt
        import cv2
        coloredImage = cv2.cvtColor(
            cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)

        for seed in seeds:
            cv2.circle(coloredImage,center=seed,radius=4,color=(255,0,0))

        plt.figure()
        plt.title('seeds')
        plt.imshow(coloredImage)
        plt.show()
