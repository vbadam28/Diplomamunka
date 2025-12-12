import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

import pandas as pd
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np

from evaluate.calc_metrics import score
from logic.pipeline.pipelineContext import PipelineContext
import time

_globalCache={}
def loadNiftiCached(path):
    if path not in _globalCache:
        _globalCache[path]= nib.load(path)
    return _globalCache[path]

def runAll(pipelines,processTask=None, basePath="../content/dataset",n_jobs=None, chunksize=4):
    if n_jobs is None:
        n_jobs = max(1,multiprocessing.cpu_count()-1)

    tasks = list(generateTasks(pipelines,basePath))
    total = len(tasks)

    if total == 0:
        return pd.DataFrame([])

    results = []
    if processTask is None:
        processTask = processPipelineTask
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        iterator = ex.map(processTask, generateTasks(pipelines,basePath), chunksize=chunksize)

        for res in tqdm(iterator, total=total, desc="Running pipelines"):
            if res is not None:
                results.append(res)
    df = pd.DataFrame(results)
    return df



def generateTasks(pipelines,basePath):
    folders = sorted(os.listdir(basePath), key=lambda s: int(s.split("_")[1]))

    for folder in folders:
        idx = folder.split("_")[1]
        flairPath = os.path.join(basePath,folder, f"{idx}_flair.nii")
        segPath = os.path.join(basePath,folder, f"{idx}_seg.nii")

        imgNifti = loadNiftiCached(flairPath) #.get_fdata()
        #imgSegNifti = nib.load(segPath).get_fdata()
        depth = imgNifti.shape[2]
        for layer in range(depth):
            #img = imgNifti[:,:,layer]
            #gt = imgSegNifti[:, :, layer]
            #gt[gt>1] = 1

            #if np.sum(gt)<10:
            #    continue
            #tumor = img[gt==1]
            #if tumor.size < 10 and np.min(tumor) < 1e-11:
            #    continue
            if pipelines is not None:
                for plFactory, params,algname in pipelines:
                    yield (flairPath, segPath,idx, layer, plFactory, params, algname)
            else:
                yield (flairPath, segPath, idx, layer)
def processPipelineTask(task):

    try:
        imgPath, gtPath, idx, layer, plFactory, params, algname = task
        import time as _time

        img = loadNiftiCached(imgPath).dataobj[:,:,layer].astype(np.float32, copy=True)
        gt = loadNiftiCached(gtPath).dataobj[:,:,layer].astype(np.uint8, copy=True)
        gt[gt>1] = 1

        if np.sum(gt)<10:
            return None
        tumor = img[gt==1]
        if tumor.size < 10 and np.min(tumor) < 1e-11:
            return None

        ctx = PipelineContext()

        ctx.data = {"image":img, "debug":False, "roi":img}
        ctx.params = params


        pl = plFactory()
        t0 = _time.perf_counter_ns()
        masks = pl.run(ctx)
        t1 = _time.perf_counter_ns()

        pred = masks[0]
        data, conf = score(pred.astype(bool), gt.astype(bool), img)

        for mask in masks[1:]:
            tmp, tmpConf = score(mask.astype(bool), gt.astype(bool), img)
            if tmp.get("DS",-1) > data.get("DS",-1):
                data, conf = tmp, tmpConf
                pred = mask
        data.update({
            "brain ID":idx,
            "slice ID":layer,
            "alg":algname,
            "params":params,
            "time(ms)":(t1 - t0)/1000000,
            "gtArea": np.count_nonzero(gt[img>0].astype(bool))
        })

        return data
    except Exception as e:
        return {
            "brain ID": idx if 'idx' in locals() else None,
            "slice ID": layer if 'layer' in locals() else None,
            "alg": algname if 'algname' in locals() else None,
            "params": params if 'params' in locals() else None,
            "error": str(e),
            "gtArea": np.count_nonzero(gt[img > 0].astype(bool)) if "gt" in locals() and "img" in locals() else None

        }


def processTumorStatistics(task):
    try:
        imgPath, gtPath, idx, layer = task

        img = loadNiftiCached(imgPath).dataobj[:, :, layer].astype(np.float32, copy=True)
        gt = loadNiftiCached(gtPath).dataobj[:, :, layer].astype(np.uint8, copy=True)
        gt[gt > 1] = 1

        if np.sum(gt) < 10:
            return None
        tumor = img[gt == 1]
        if tumor.size < 10 and np.min(tumor) < 1e-11:
            return None

        brain = img[(img>0) & (gt==0)]


        data={
            "brain ID" : idx,
            "slice ID" : layer,
            "tumorArea": np.count_nonzero(gt[img > 0].astype(bool)),
            "tumorStd" : np.std(tumor),
            "tumorMean": np.mean(tumor),
            "tumorMedian": np.median(tumor),
            "brainArea":np.count_nonzero(img[img>0]),
            "brainMean": np.mean(brain),
            "brainMedian": np.median(brain),
            "contrast": np.abs(np.mean(tumor) - np.mean(brain))


        }

        return data
    except Exception as e:
        return {
            "brain ID": idx if 'idx' in locals() else None,
            "slice ID": layer if 'layer' in locals() else None,
            "error": str(e),
        }







