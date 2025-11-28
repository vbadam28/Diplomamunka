from evaluate.metrics import *
import pandas as pd
import nibabel as nibb


def score(pred,gt,img):
    data = {}
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

    data['MAPE2']= MAPE2(pred, gt)
    data['Rerr2']= Rerr2(pred, gt)

    data["PSNR"] = PSNR(pred,gt)
    data["PSNR2"] = PSNR2(conf)
    data["PSNR3"] = PSNR3(pred,gt,img)

    return data

def run(plFactory,paths = ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'], params = {}):
    path, seg_path = paths
    img = nibb.load(path)
    nifti_img = img.get_fdata()
    seg_img = nibb.load(seg_path)
    nifti_img_seg = seg_img.get_fdata()

    brainData = []
    brainDataNames = []

    for layer in range(nifti_img.shape[2]):
        #layer = 100

        img = nifti_img[:,:,layer]
        gt = nifti_img_seg[:,:,layer]
        name = f"layer: {layer}"

        if np.sum(gt)<10:
            continue

        from logic.pipeline.pipelineFactory import PipelineFactory
        from logic.pipeline.pipelineContext import PipelineContext
        # INIT PIPELINE
        ctx = PipelineContext()
        ctx.data = {"image":img.copy(), "debug":False, "roi": img.copy()}
        ctx.params = params

        pl = plFactory()
        masks = pl.run(ctx)


        pred = masks[0]
        data = score(pred.astype(bool), gt.astype(bool), img)

        for mask in masks[1:]:
            tmp = score(mask.astype(bool),gt.astype(bool),img)
            if tmp["DS"]>data["DS"]:
                data = tmp
                pred = mask

        brainData.append(data)
        brainDataNames.append(name)

    df = pd.DataFrame(brainData, index = brainDataNames)

    pd.options.display.max_columns = None # 0

    return df