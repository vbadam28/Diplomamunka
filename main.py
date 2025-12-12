import sys

from PySide6.QtWidgets import QApplication

from evaluate.calc_metrics import score, getConfMtx
from gui.window import MainWindow
from logic.pipeline.pipelineContext import PipelineContext
from logic.preprocessing.enhanced_split_merge import EnhancedSplitMerge
from logic.preprocessing.preprocessing_step import enhanceImage, normalize

if __name__ == '__main__':
    app = QApplication(sys.argv)
    wd = MainWindow()
    wd.show()
    app.exec()
    exit()

    from logic.pipeline.pipelineFactory import PipelineFactory
    #from evaluate.calc_metrics import run, runAll
    import pandas as pd
    import nibabel as n
    import numpy as np
    '''
    img = n.load("../content/dataset/brain_339/339_flair.nii").get_fdata()[:,:,85]#ezen nem működik hisz contrast PICI
    seg = n.load("../content/dataset/brain_339/339_seg.nii").get_fdata()[:,:,85]
    #img = n.load("../content/dataset/brain_355/355_flair.nii").get_fdata()[:, :,84]
    #seg = n.load("../content/dataset/brain_355/355_seg.nii").get_fdata()[:, :, 84]
    pl = PipelineFactory.divergenceSeeds()
    ctx = PipelineContext()
    ctx.data={"image":img.copy(),"roi":img.copy(), "debug":False}
    mask =pl.run(ctx)[0]
    #from matplotlib import pyplot as pt
    #pt.figure()
    #pt.imshow()
    #pt.show()

    tmp = normalize(img)
    print("contrast",np.abs(np.mean(tmp[seg.astype(bool)]) - np.mean(tmp[(~seg.astype(bool)) & (tmp>0) ])), "meanColor",np.mean(tmp[tmp>0]), "median", np.median(tmp[tmp>0]))

    img = n.load("../content/dataset/brain_354/354_flair.nii").get_fdata()[:, :,84]
    seg = n.load("../content/dataset/brain_354/354_seg.nii").get_fdata()[:, :, 84]
    tmp = normalize(img)
    print("contrast",np.abs(np.mean(tmp[seg.astype(bool)]) - np.mean(tmp[(~seg.astype(bool)) & (tmp>0) ])), "meanColor",np.mean(tmp[tmp>0]),"median", np.median(tmp[tmp>0]))

    #from matplotlib import pyplot
    #pyplot.figure()
    #pyplot.imshow(img,cmap="gray")
    #pyplot.show()
    eImg = enhanceImage(normalize(img))

    EnhancedSplitMerge().findHyperRangeAndAvg(eImg)

    exit()
    '''
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
    #dfBiratu, dfBrain = run(PipelineFactory.select5Seeds, ['../content/dataset/brain_355/355_flair.nii', '../content/dataset/brain_355/355_seg.nii'])

    #dfSWMean = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'],{"sw:mode":"mean"})

    #dfSWMax = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"max"})
    #dfSWstd = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"std"})
    #dfSWblob = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"blob"})
    #dfSWentropy = run(PipelineFactory.slidingWindows, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], {"sw:mode":"entropy"})

    #dfSWMean.to_csv("evaluate/csv/sliding_window_mean.csv", sep=";")
    #dfSWMax.to_csv("evaluate/csv/sliding_window_max.csv", sep=";")
    #dfSWstd.to_csv("evaluate/csv/sliding_window_std.csv", sep=";")
    #dfSWblob.to_csv("evaluate/csv/sliding_window_blob.csv", sep=";")
    #dfSWentropy.to_csv("evaluate/csv/sliding_window_entropy.csv", sep=";")

    #dfBiratu.to_csv("evaluate/csv/biratu.csv", sep=";")

    #dfSaad = run(PipelineFactory.divergenceSeeds, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'])

    #with pd.option_context("display.max_rows",None,"display.max_columns", None):
        #pd.options.display.max_columns = None # 0
        #diff = dfBiratu.compare(dfSaad,result_names=("biratu","saad"))
        #print(diff)
        #diff.to_csv("evaluate/csv/basic_compare.csv",sep=";")
        #print(dfBrain)

    '''
    dfBiratu = pd.read_csv("evaluate/csv/biratu.csv", sep=";")

    dfSWMean = pd.read_csv("evaluate/csv/sliding_window_mean.csv", sep=";")
    dfSWMax = pd.read_csv("evaluate/csv/sliding_window_max.csv", sep=";")
    dfSWstd = pd.read_csv("evaluate/csv/sliding_window_std.csv", sep=";")
    dfSWblob = pd.read_csv("evaluate/csv/sliding_window_blob.csv", sep=";")
    dfSWentropy = pd.read_csv("evaluate/csv/sliding_window_entropy.csv", sep=";")

    from matplotlib import pyplot as plt
    import numpy as np
    plt.figure()
    plt.plot(dfBiratu["DS"])

    plt.plot(dfSWMean["EF"], alpha=0.5)
    plt.plot(dfSWMax["EF"], alpha=0.5)
    plt.plot(dfSWstd["EF"], alpha=0.5)
    plt.plot(dfSWblob["EF"], alpha=0.5)
    #plt.plot(dfSWentropy["EF"], alpha=0.5)
    plt.legend(["Biratu","mean","max","std","blob","entropy", ])
    plt.show()
'''
    pls = [ #(PipelineFactory.select5Seeds, {}, "biratu"), #pár perc

            #(PipelineFactory.slidingWindows, {"sw:mode":"mean","wss:mode":"center"}, "sw"), 4db kb fél óra
            #(PipelineFactory.slidingWindows, {"sw:mode":"max","wss:mode":"center"}, "sw"),
            #(PipelineFactory.slidingWindows, {"sw:mode":"std","wss:mode":"center"}, "sw"),
            #(PipelineFactory.slidingWindows, {"sw:mode":"blob","wss:mode":"center"}, "sw"),

            #kimarad(PipelineFactory.slidingWindows, {"sw:mode":"entropy","wss:mode":"center"}, "sw"),

            #(PipelineFactory.slidingWindows, {"sw:mode": "mean", "wss:mode": "max"}, "sw"),  4db kb fél óra
            #(PipelineFactory.slidingWindows, {"sw:mode": "max", "wss:mode": "max"}, "sw"),
            #(PipelineFactory.slidingWindows, {"sw:mode": "std", "wss:mode": "max"}, "sw"),
            #(PipelineFactory.slidingWindows, {"sw:mode": "blob", "wss:mode": "max"}, "sw"),

            #kimarad(PipelineFactory.slidingWindows, {"sw:mode": "entropy", "wss:mode": "max"}, "sw"),

            #(PipelineFactory.divergenceSeeds, {"qt:sumHyperThresh": 300}, "saad"),
            #(PipelineFactory.divergenceSeeds, {"qt:sumHyperThresh": 30}, "saad"),
            #(PipelineFactory.divergenceSeeds, {"qt:sumHyperThresh": 100}, "saad"),
            #(PipelineFactory.divergenceSeeds, {"qt:sumHyperThresh": 200}, "saad"),
            #(PipelineFactory.divergenceSeeds, {"qt:sumHyperThresh": 400}, "saad"),

            (PipelineFactory.enhancedDivergence, {}, "enhanced div"),
            #(PipelineFactory.enhancedDivergence, {"qt:sumHyperThresh": 300}, "enhanced div"),
            #(PipelineFactory.enhancedDivergence, {"qt:sumHyperThresh": 30}, "enhanced div"),
            #(PipelineFactory.enhancedDivergence, {"qt:sumHyperThresh": 100}, "enhanced div"),
            #(PipelineFactory.enhancedDivergence, {"qt:sumHyperThresh": 200}, "enhanced div"),
            #(PipelineFactory.enhancedDivergence, {"qt:sumHyperThresh": 400}, "enhanced div"),
            ]
    from evaluate.calc_metrics_parallel import runAll, processTumorStatistics
    import time
    start = time.time()
    #df = runAll(pls)#None,processTask=processTumorStatistics)
    end = time.time()
    #df.reset_index(drop=True).to_csv("evaluate/csv/all_slices_stats.csv",sep=";")
    #df.to_csv("evaluate/csv/enhanced_div_parallel_all_slices.csv",sep=";")
    #df.to_csv("evaluate/csv/enhanced_div_parallel_all_slices_old.csv",sep=";")
    #df.to_csv("evaluate/csv/saad_parallel_all_slices.csv",sep=";")
    #df.to_csv("evaluate/csv/biratu_parallel_all_slices.csv",sep=";")
    #dfSw = pd.read_csv("evaluate/csv/sw_all_slices.csv", sep=";")
    #df = pd.concat([dfSw,df],ignore_index=True)
    #df.to_csv("evaluate/csv/sw_all_slices.csv",sep=";")
    sec = end-start
    print(f"runtime: {sec} sec ---> {int(sec/60)}:{int(sec-(int(sec/60)*60))} ")
    #"biratu_all_slices"

    '''dfBiratu = pd.read_csv("evaluate/csv/biratu_all_slices.csv",sep=";")
    dfBiratuParallel = pd.read_csv("evaluate/csv/biratu_parallel_all_slices.csv",sep=";")

    
    with pd.option_context("display.max_columns", None):
        keyCols = ["brain ID","slice ID","alg","params"]
        dfBiratuIndexed = dfBiratu.set_index(keyCols)
        dfBiratuParallelIndexed = dfBiratuParallel.set_index(keyCols)

        dfBiratuAligned, dfBiratuParallelAligned = dfBiratuIndexed.align(dfBiratuParallelIndexed, join="outer")

        diff = dfBiratuAligned.compare(dfBiratuParallelAligned, result_names = ("biratu","parallel"))
        #print(dfBiratuParallel.info())
        #diff = dfBiratu.compare(dfBiratuParallel,result_names=("biratu","biratuparallel"))
        print(diff)

        #added_rows = df2_indexed.index.difference(df1_indexed.index)
        #removed_rows = df1_indexed.index.difference(df2_indexed.index)
        #print(df2i.loc[added_rows])
        #print(df1i.loc[removed_rows])
        #result = df1_aligned.compare(df2_aligned, keep_shape=True, keep_equal=True)
'''

    dfSw = pd.read_csv("evaluate/csv/sw_all_slices.csv", sep=";")
