from logic.pipeline.pipelineFactory import PipelineFactory
import pandas as pd
import nibabel as n
import numpy as np
from logic.pipeline.pipelineContext import PipelineContext
from logic.preprocessing.enhanced_split_merge import EnhancedSplitMerge
from logic.preprocessing.preprocessing_step import enhanceImage, normalize, PreprocessingStep
from logic.preprocessing.split_merge import SplitMerge
from logic.seed_selector.divergence_seed_selector import DivergenceSeedSelector
from calc_metrics import score, getConfMtx

# '''

slices = [(223, 79), (66, 104), (327, 52), (44, 103), (8, 70), (224, 101), (100, 58), (44, 62)]  # original better
slices = [
    (224,101),
    (8,70),
    (180, 105),
    (330, 56), (330, 61), (230, 72), (253, 120), (230, 71), (230, 73), (229, 100), (331, 120),
          (321, 37)]  # enahnced better
slices_close_better_original = [(8, 64), (16, 58), (308, 50),  # low ones
                                (128, 47), (190, 88), (260, 128), (283, 113)]  # better ones
slices_close_better_enhanced = [(361, 35), (320, 109), (325, 95), (15, 68), (282, 87), (264, 94)]
img = n.load("../../content/dataset/brain_339/339_flair.nii").get_fdata()[:, :, 85]  # ezen nem működik hisz contrast PICI
seg = n.load("../../content/dataset/brain_339/339_seg.nii").get_fdata()[:, :, 85]

# img = n.load("../content/dataset/brain_355/355_flair.nii").get_fdata()[:, :,84]
# seg = n.load("../content/dataset/brain_355/355_seg.nii").get_fdata()[:, :, 84]
pl = PipelineFactory.divergenceSeeds()
pl = PipelineFactory.slidingWindows()
ctx = PipelineContext()
ctx.data = {"image": img.copy(), "roi": img.copy(), "debug": False, "params":{'sw_mode':"max", }}
mask = pl.run(ctx)[0]
# from matplotlib import pyplot as pt
# pt.figure()
# pt.imshow()
# pt.show()

tmp = normalize(img)
print("contrast", np.abs(np.mean(tmp[seg.astype(bool)]) - np.mean(tmp[(~seg.astype(bool)) & (tmp > 0)])), "meanColor",
      np.mean(tmp[tmp > 0]), "median", np.median(tmp[tmp > 0]))
for img_idx, slice_idx in slices:  # slices_close_better_enhanced:
    img = n.load(f"../../content/dataset/brain_{img_idx}/{img_idx}_flair.nii").get_fdata()[:, :, slice_idx]
    seg = n.load(f"../../content/dataset/brain_{img_idx}/{img_idx}_seg.nii").get_fdata()[:, :, slice_idx]
    ctx = PipelineContext()
    ctx.data = {"image": img.copy(), "roi": img.copy(), "debug": True}
    ctxE = PipelineContext()
    ctxE.data = {"image": img.copy(), "roi": img.copy(), "debug": True}

    ctxP = PipelineContext()
    ctxP.data = {"image": img.copy(), "roi": img.copy(), "debug": False}
    ctxEP = PipelineContext()
    ctxEP.data = {"image": img.copy(), "roi": img.copy(), "debug": False}
    # tmp = normalize(img)
    # print("contrast",np.abs(np.mean(tmp[seg.astype(bool)]) - np.mean(tmp[(~seg.astype(bool)) & (tmp>0) ])), "meanColor",np.mean(tmp[tmp>0]),"median", np.median(tmp[tmp>0]))

    # from matplotlib import pyplot
    # pyplot.figure()
    # pyplot.imshow(img,cmap="gray")
    # pyplot.show()
    # eImg = enhanceImage(normalize(img))
    from matplotlib import pyplot as plt

    original_show = plt.show


    def no_show(*args, **kwargs):
        plt.close('all')  # kill any figures created so far


    plt.show = no_show

    ctx = PreprocessingStep(normalize).process(ctx)
    ctx = PreprocessingStep(enhanceImage).process(ctx)
    SM = SplitMerge()
    ctx = SM.process(ctx)

    DSS = DivergenceSeedSelector()
    DSS.ctx = ctx

    region = ctx.get('roi', ctx.get('image'))
    image = enhanceImage(ctx.get('image'))
    ctx.set('image', image)

    hist, bin_edges = DSS.calcRoiHistogram(region)
    P = hist
    s = np.sum(hist)
    if s > 0:
        P = hist / s
    div = np.gradient(hist) * P

    idx, tail, nearest_zero, optimal_threshold_idx, optimal_threshold = DSS.setOptimalThreshold(div, bin_edges)
    # DSS.showOptThreshold(hist,bin_edges,div,idx,optimal_threshold_idx)

    ctxE = PreprocessingStep(normalize).process(ctxE)
    ctxE = PreprocessingStep(enhanceImage).process(ctxE)
    ESM = EnhancedSplitMerge()
    ctxE = ESM.process(ctxE)

    DSSE = DivergenceSeedSelector(enhanced=True)
    DSSE.ctx = ctxE
    from scipy.ndimage import gaussian_filter1d

    region = ctxE.get('roi', ctxE.get('image'))
    image = enhanceImage(ctxE.get('image'))
    ctxE.set('image', image)
    histE, bin_edgesE = DSSE.calcRoiHistogram(region)
    histE = gaussian_filter1d(histE, sigma=1.1)
    P = histE
    s = np.sum(histE)
    if s > 0:
        P = histE / s
    divE = np.gradient(histE) * P

    idxE, tail, nearest_zero, optimal_threshold_idxE, optimal_threshold, choosenMaxPeak = DSSE.setOptimalThresholdEnhanced(
        divE, bin_edgesE)

    if optimal_threshold_idxE >= len(divE) - 20 and optimal_threshold_idxE - choosenMaxPeak <= 20:
        divE[choosenMaxPeak:] = -divE[choosenMaxPeak]
        # divE[optimal_threshold_idxE] = divE[choosenMaxPeak]
        idxE, tail, nearest_zero, optimal_threshold_idxE, optimal_threshold, _ = DSSE.setOptimalThresholdEnhanced(divE,
                                                                                                                  bin_edgesE)
    elif optimal_threshold_idxE >= len(divE) - 20:
        divE[-20:] = -divE[choosenMaxPeak]
        idxE, tail, nearest_zero, optimal_threshold_idxE, optimal_threshold, _ = DSSE.setOptimalThresholdEnhanced(divE,
                                                                                                                  bin_edgesE)
    # DSSE.showOptThreshold(histE,bin_edgesE,divE,idxE,optimal_threshold_idxE)

    mP = np.array(PipelineFactory.divergenceSeeds().run(ctxP)[0], dtype=np.uint8)
    mEP = np.array(PipelineFactory.enhancedDivergence().run(ctxEP)[0], dtype=np.uint8)
    mP[mP != 0] = 255
    mEP[mEP != 0] = 255
    import cv2

    contoursP, _ = cv2.findContours(mP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursEP, _ = cv2.findContours(mEP, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    SMImage = SM.modifiedImage.copy()
    ESMImage = ESM.modifiedImage.copy()

    cv2.drawContours(SMImage, contoursP, -1, (255, 255, 0), thickness=1)
    cv2.drawContours(ESMImage, contoursEP, -1, (255, 255, 0), thickness=1)

    # from matplotlib import pyplot as plt
    plt.show = original_show
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, (1, 4))
    plt.plot(hist, alpha=0.5)
    plt.plot(histE, alpha=0.75)
    plt.title("histogram")
    plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))

    plt.subplot(2, 3, (2, 5))
    plt.plot(div, label="divergence", alpha=0.5)
    plt.plot(divE, label="enhanced", alpha=0.75)
    plt.xticks(np.arange(0, 256, 50), np.round(bin_edges[np.arange(0, 256, 50)], 2))
    plt.scatter(idx, div[idx], c="green")
    plt.scatter(optimal_threshold_idx, div[optimal_threshold_idx], c="red")

    plt.scatter(idxE, divE[idxE], c="green")
    plt.scatter(optimal_threshold_idxE, divE[optimal_threshold_idxE], c="red")

    plt.title(f"Divergence,  Thres: ({bin_edges[optimal_threshold_idx]:.4f},{div[optimal_threshold_idx]:.4f})\n"
              f"EDivergence, Thres: ({bin_edgesE[optimal_threshold_idxE]:.4f},{divE[optimal_threshold_idxE]:.4f})")
    plt.legend()
    plt.subplot(2, 3, 3)
    plt.title(f"{img_idx}/{slice_idx}")
    plt.imshow(SMImage)
    plt.subplot(2, 3, 6)
    plt.imshow(ESMImage)
    plt.show()

exit()
# '''
'''pl = PipelineFactory.select5Seeds()
ctx = PipelineContext()
ctx.data = {"image":img.copy(),"roi":img.copy(),"debug":False}

masks = pl.run(ctx)
print(ctx.get("seeds"))
import cv2
import matplotlib.pyplot as plt


coloredImage = cv2.cvtColor(
    cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLOR_GRAY2BGR)

bestdx=0
bestDs=0
for i,seed in enumerate(ctx.get("seeds")):
    cv2.circle(coloredImage, center=seed, radius=4, color=(255, 0, 0))
    ds = score(masks[i].astype(bool),seg.astype(bool),img)[0]["DS"]
    if ds>bestDs:
        bestDs=ds
        bestdx=i

contours, _ = cv2.findContours(masks[bestdx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    cv2.drawContours(coloredImage, [c], -1, (255, 255, 0), thickness=1)

plt.figure()
plt.subplot(1,2,1)
plt.axis("off")

plt.title('Eredeti')
plt.imshow(coloredImage)

plt.subplot(1,2,2)
plt.title("Sliding window")
pl = PipelineFactory.slidingWindows()
ctx = PipelineContext()
ctx.data = {"image":img.copy(),"roi":img.copy(),"debug":False}

masks = pl.run(ctx)
print(ctx.get("seeds"))
import cv2
import matplotlib.pyplot as plt


coloredImage = cv2.cvtColor(
    cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLOR_GRAY2BGR)

bestdx=0
bestDs=0
for i,seed in enumerate(ctx.get("seeds")):
    cv2.circle(coloredImage, center=seed, radius=4, color=(255, 0, 0))
    ds = score(masks[i].astype(bool),seg.astype(bool),img)[0]["DS"]
    if ds>bestDs:
        bestDs=ds
        bestdx=i

contours, _ = cv2.findContours(masks[bestdx], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    cv2.drawContours(coloredImage, [c], -1, (255, 255, 0), thickness=1)
plt.imshow(coloredImage)
plt.axis("off")
plt.tight_layout()
plt.show()

exit()
'''
'''
path ="evaluate/csv/biratu_parallel_all_slices.csv"
df = pd.read_csv(path,sep=";")

dfIds = df.groupby(["brain ID", "slice ID"]).agg(brainID=("brain ID","first"),sliceID=("slice ID","first"))
cache = {}
for _, row in dfIds.iterrows():
    brainID, sliceID = row["brainID"], row["sliceID"]

    if brainID not in cache:
        cache[brainID] = n.load(f"../content/dataset/brain_{brainID}/{brainID}_seg.nii")
        cache[f"{brainID}-img"] = n.load(f"../content/dataset/brain_{brainID}/{brainID}_flair.nii")
    seg = cache[brainID].dataobj[:,:,sliceID].astype(np.uint8, copy=True)
    img = cache[f"{brainID}-img"].dataobj[:,:,sliceID].astype(np.float32, copy=True)
    seg = seg[img>0]

    df.loc[(df["brain ID"] ==brainID) & (df["slice ID"]==sliceID),"gtArea"] = np.count_nonzero(seg)

#df.to_csv(path ,sep=";")
print("kész")
exit()
'''
# dfBiratu, dfBrain = run(PipelineFactory.select5Seeds, ['../content/dataset/brain_355/355_flair.nii', '../content/dataset/brain_355/355_seg.nii'])

# dfSWMean = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'],{"sw:mode":"mean"})

# dfSWMax = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"max"})
# dfSWstd = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"std"})
# dfSWblob = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"blob"})
# dfSWentropy = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"entropy"})

# dfSWMean.to_csv("evaluate/csv/sliding_window_mean.csv", sep=";")
# dfSWMax.to_csv("evaluate/csv/sliding_window_max.csv", sep=";")
# dfSWstd.to_csv("evaluate/csv/sliding_window_std.csv", sep=";")
# dfSWblob.to_csv("evaluate/csv/sliding_window_blob.csv", sep=";")
# dfSWentropy.to_csv("evaluate/csv/sliding_window_entropy.csv", sep=";")

# dfBiratu.to_csv("evaluate/csv/biratu.csv", sep=";")

# dfSaad = run(PipelineFactory.divergenceSeeds, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'])

# with pd.option_context("display.max_rows",None,"display.max_columns", None):
# pd.options.display.max_columns = None # 0
# diff = dfBiratu.compare(dfSaad,result_names=("biratu","saad"))
# print(diff)
# diff.to_csv("evaluate/csv/basic_compare.csv",sep=";")
# print(dfBrain)
