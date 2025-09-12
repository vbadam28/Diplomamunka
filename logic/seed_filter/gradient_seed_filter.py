import cv2
import numpy as np

'''
Az lenne a cél, hogy a szélekhez eső seedpontokat kivesszük,
 egy homogénebb/simább területeken levőket hagyjuk meg
'''
class gradientSeedFilter:

    def __init__(self):
        self.debug=False


    def filter(self,ctx):
        self.debug = ctx.get('debug')
        img = ctx.get('image')
        seeds = ctx.get('seeds')

        grad_x_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        # grad_y_sobel = ndimage.sobel(img, axis=0)
        sobel_magnitude = np.hypot(grad_x_sobel, grad_y_sobel)

        edge_threshold = np.percentile(sobel_magnitude, 80)
        brightness_threshold = np.percentile(img, 60)

        # filter seeds
        selected_seeds = []
        for y, x in seeds:
            # flat area and bright area
            if sobel_magnitude[y, x] < edge_threshold and img[y, x] > brightness_threshold:
                selected_seeds.append((x, y))

        selected_seeds = np.array(selected_seeds)

        grad_x_sharr = cv2.Scharr(img, cv2.CV_64F, 1, 0)
        grad_y_sharr = cv2.Scharr(img, cv2.CV_64F, 0, 1)
        scharr_magnitude = np.hypot(grad_x_sharr, grad_y_sharr)

        edge_threshold = np.percentile(scharr_magnitude, 80)
        brightness_threshold = np.percentile(img, 60)
        # filter seeds
        selected_seeds2 = []
        for y, x in seeds:
            # flat area and bright area
            if scharr_magnitude[y, x] < edge_threshold and img[y, x] > brightness_threshold:
                selected_seeds2.append((x, y))

        selected_seeds2 = np.array(selected_seeds2)

        if self.debug:
            self.showDebug(img,sobel_magnitude,scharr_magnitude,selected_seeds,selected_seeds2)

        ctx.set('seeds',selected_seeds)

        return ctx

    def showDebug(self,img,sobel_magnitude,scharr_magnitude,selected_seeds,selected_seeds2):
        from matplotlib import pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(sobel_magnitude)
        plt.title("sobel img")

        plt.subplot(2, 2, 2)
        plt.imshow(scharr_magnitude)
        plt.title("scharr img")

        if len(selected_seeds) > 0:
            plt.subplot(2, 2, 3)
            plt.imshow(img, cmap="gray")
            plt.scatter(x=selected_seeds[:, 0], y=selected_seeds[:, 1], c="red")
        if len(selected_seeds2) > 0:
            plt.subplot(2, 2, 4)
            plt.imshow(img, cmap="gray")
            plt.scatter(x=selected_seeds2[:, 0], y=selected_seeds2[:, 1], c="red")
