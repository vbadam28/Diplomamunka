from sklearn.mixture import GaussianMixture
from logic.seed_selector.seed_selector import SeedSelector
import numpy as np

class GmmSeedSelector(SeedSelector):
    def __init__(self,n_components=2, random_state=0):
        self.n_components=n_components
        self.random_state = random_state
        self.debug = False

    def select(self, ctx):

        self.debug = ctx.get('debug')
        img = ctx.get('image')
        X =  ctx.get('roi')

        '''Egyszerűbb a opt thresholdos region growing mint az opencv-s'''
        from logic.seed_selector.divergence_seed_selector import DivergenceSeedSelector as dss
        ctx = dss().select(ctx)


        roi_mask = X>0
        roi_pixels = img[roi_mask].reshape(-1,1)

        gmm = GaussianMixture(n_components=self.n_components, random_state=self.random_state) # sima + elváltozott területre osztályozunk valség modellel

        gmm.fit(roi_pixels)

        probs = gmm.predict_proba(roi_pixels) #milyen valségel tartozik az egyik kompoenesbe bele

        means = gmm.means_.flatten() # 2 komp közül egyiknek nagyobb lesz az átlaga (világosabb terület )
        brighter_component = np.argmax(means)

        high_conf_mask = probs[:, brighter_component] > 0.9 # magas valószínűséggel elváltozott területhez tartozik
        seed_pixels = roi_pixels[high_conf_mask]

        roi_indices = np.argwhere(roi_mask)
        seed_indices = roi_indices[high_conf_mask]

        if self.debug:
            seed_img = np.zeros_like(img)
            for row, col in seed_indices:
                seed_img[row, col] = 255
            from matplotlib import pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.title("orig")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(seed_img, cmap='gray')
            plt.title("seedpoints")
            plt.axis('off')

            plt.show()

        ctx.set('seeds', seed_indices)
        return ctx