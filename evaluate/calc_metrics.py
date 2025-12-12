try:
    from evaluate.metrics import *
except:
    from metrics import *
import pandas as pd
import nibabel as nibb
import time

def score(pred=None,gt=None,img=None, conf = None, addConf=True):
    data = {}
    if conf is None:
        brainMask = img>0 #np.argwhere(img!=0)
        conf, confMask = getConfMtx(pred[brainMask], gt[brainMask])
        #conf, confMask = getConfMtx(pred,gt)


    data['DS']  = DSS(conf)
    data['OF']  = OF(conf)
    data['EF']  = EF(conf)
    data['IoU'] = IoU(conf)
    data['Sn']  = Sn(conf)
    data['Sp']  = Sp(conf)
    data['Acc'] = Acc(conf)
    #data['PSNR']= PSNR(conf)

    data['AO']  = AO(conf)
    data['FPR'] = FPR(conf)
    data['FNR'] = FNR(conf)
    data['MA']  = MA(conf)
    data['MAPE']= MAPE(conf)
    data['Rerr']= Rerr(conf)
    if addConf:
        data["TP"] = conf["TP"]
        data["TN"] = conf["TN"]
        data["FP"] = conf["FP"]
        data["FN"] = conf["FN"]

    if pred is not None:
        data['MAPE2']= MAPE2(pred, gt)
        data['Rerr2']= Rerr2(pred, gt)

        data["PSNR"] = PSNR(pred,gt)
        data["PSNR3"] = PSNR3(pred,gt,img)
    data["PSNR2"] = PSNR2(conf)

    return data, conf

def run(plFactory,paths = ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], params = {}):
    from logic.pipeline.pipelineContext import PipelineContext
    path, seg_path = paths
    img = nibb.load(path)
    nifti_img = img.get_fdata()
    seg_img = nibb.load(seg_path)
    nifti_img_seg = seg_img.get_fdata()

    brainData = []
    brainDataNames = []
    brainConf = {"TP":0,"TN":0,"FP":0,"FN":0}
    for layer in range(nifti_img.shape[2]):
        #layer = 100

        img = nifti_img[:,:,layer]
        gt = nifti_img_seg[:,:,layer]
        name = f"layer: {layer}"

        if np.sum(gt)<10:
            continue

        # INIT PIPELINE
        ctx = PipelineContext()
        ctx.data = {"image":img.copy(), "debug":False, "roi": img.copy()}
        ctx.params = params

        pl = plFactory()
        masks = pl.run(ctx)


        pred = masks[0]
        data, conf = score(pred.astype(bool), gt.astype(bool), img)

        for mask in masks[1:]:
            tmp, tmpConf = score(mask.astype(bool),gt.astype(bool),img)
            if tmp["DS"]>data["DS"]:
                data, conf = tmp, tmpConf
                pred = mask

        brainData.append(data)
        brainDataNames.append(name)
        brainConf = {k: brainConf[k] + v for k, v in conf.items()}

    gData, gConf =  score(conf=brainConf)

    df = pd.DataFrame(brainData, index = brainDataNames)

    pd.options.display.max_columns = None # 0

    return df, pd.DataFrame(gData, index = ["brain 355"])


def runAll(pipelines, basePath = "../content/dataset"):
    from logic.pipeline.pipelineContext import PipelineContext
    from logic.pipeline.pipelineFactory import PipelineFactory
    import os

    folders = os.listdir(basePath)
    folders = sorted(folders, key=lambda s: int(s.split("_")[1]))

    layerData = []
    layerDataNames = []

    for folder in folders:
        idx = folder.split("_")[1]
        img = nibb.load(os.path.join(basePath, folder, f"{idx}_flair.nii"))
        nifti_img = img.get_fdata()
        seg_img = nibb.load(os.path.join(basePath, folder, f"{idx}_seg.nii"))
        nifti_img_seg = seg_img.get_fdata()

        for layer in range(nifti_img.shape[2]):
            img = nifti_img[:, :, layer]
            gt = nifti_img_seg[:, :, layer]
            gt[gt>0] = 1

            if np.sum(gt) < 10 or np.min(img[gt==1])<0.00000000001:
                continue

            for plFactory, params, algname in pipelines:
                ctx = PipelineContext()
                ctx.data = {"image": img.copy(), "debug": False, "roi": img.copy()}
                ctx.params = params

                pl = plFactory()
                start = time.time()
                try:
                    masks = pl.run(ctx)
                except Exception as e:
                    data = {}
                    data["brain ID"] = idx
                    data["slice ID"] = layer
                    data["alg"] = algname
                    data["params"] = params
                    data["error"] = str(e)
                    layerData.append(data)
                    continue

                end = time.time()
                pred = masks[0]
                data, conf = score(pred.astype(bool), gt.astype(bool), img)

                for mask in masks[1:]:
                    tmp, tmpConf = score(mask.astype(bool), gt.astype(bool), img)
                    if tmp["DS"] > data["DS"]:
                        data, conf = tmp, tmpConf
                        pred = mask
                data["brain ID"] = idx
                data["slice ID"] = layer
                data["alg"] = algname
                data["params"] = params
                data["time"] = end-start
                layerData.append(data)
                #brainDataNames.append(algname)
        if int(idx) % 50==0 or int(idx)==len(folders):
            print(f"ready {int(idx)+1} / {len(folders)}")
    return pd.DataFrame(layerData)