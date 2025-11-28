import numpy as np

def getConfMtx(mask, ref):
    TP_mask = (mask == 1) & (ref == 1)
    TN_mask = (mask == 0) & (ref == 0)
    FP_mask = (mask == 1) & (ref == 0)
    FN_mask = (mask == 0) & (ref == 1)

    TP = np.sum(TP_mask)
    TN = np.sum(TN_mask)
    FP = np.sum(FP_mask)
    FN = np.sum(FN_mask)




    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}, {"TP": TP_mask, "TN": TN_mask, "FP": FP_mask, "FN": FN_mask}


def show(confMasks, ref):
    GREEN = [0, 255, 0]  # TP
    BLACK = [0, 0, 0]  # TN
    RED = [255, 0, 0]  # FP
    BLUE = [0, 0, 255]  # FN

    diffImg = np.zeros((ref.shape[0],ref.shape[1],3), dtype=np.uint8)

    diffImg[confMasks['TP']] = GREEN
    diffImg[confMasks['TN']] = BLACK
    diffImg[confMasks['FP']] = RED
    diffImg[confMasks['FN']] = BLUE

    from matplotlib import pyplot as plt
    from matplotlib import patches as mpatches
    plt.figure()
    plt.imshow(diffImg)
    legend_patches = [
        mpatches.Patch(color=np.array(GREEN) / 255, label='TP - True Positive (Green)'),
        mpatches.Patch(color=np.array(RED) / 255, label='FP - False Positive (Red)'),
        mpatches.Patch(color=np.array(BLUE) / 255, label='FN - False Negative (Blue)'),
        mpatches.Patch(color=np.array(BLACK) / 255, label='TN - True Negative (Black)')
    ]

    plt.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    #plt.axis('off')

    plt.show()


def DSS(conf): #dice similarity score

    divider = ((2 * conf["TP"] + conf["FP"] + conf["FN"]) / 2)

    return (conf["TP"] / divider)  if divider>0 else 0

def OF(conf): #overlap fraction
    divider = (conf["TP"] + conf["FN"])

    return conf["TP"] / divider if divider> 0 else 0

def EF(conf): #extra fraction
    divider = ( conf["TP"] + conf["FN"] )
    return conf["FP"] / divider if divider>0 else 0

def IoU(conf): #Intersection Over Union  = Jaccard Idx
    divider = conf["TP"] + conf["FN"] + conf["FP"]
    return conf["TP"] / divider if divider >  0 else 0

def Sn(conf): #sensitivity
    divider = (conf["TP"] + conf["FN"])
    return conf["TP"] / divider if divider>0 else 0

def Sp(conf): #specificity
    divider = (conf["TN"] + conf["FP"])
    return conf["TN"] / divider if divider>0 else 0
def Acc(conf): #accuracy
    return (conf["TP"] + conf["TN"]) / (conf["TP"]+ conf["TN"]+ conf["FP"]+ conf["FN"])

def PSNR(pred,gt): #Peak Signal to Noise Ratio
    MSE =  np.sum((pred.astype(np.uint8)-gt.astype(np.uint8))**2) / pred.size
    return 10 * np.log10(1 / MSE ) if MSE >0 else 0

def PSNR2(conf): #Peak Signal to Noise Ratio
    MSE = (conf["FP"] + conf["FN"]) / (conf["TP"] + conf["TN"]+conf["FP"] + conf["FN"])
    return 10*np.log10(1/MSE) if MSE>0 else 0

def PSNR3(pred,gt,img): #Peak Signal to Noise Ratio
    predImg = img.copy()
    predImg[pred==False] = 0
    gtImg = img.copy()
    gtImg[gt==False] = 0

    MSE =  np.sum((predImg-gtImg)**2) / pred.size
    return 10 * np.log10(1 / MSE ) if MSE >0 else 0

    return

#Acc, IoU, DSS, Sn, Sp, OF, EF, and PSNR



def AO(conf): #Area Overlap -> de, inkább IoU vagy Accuracy, mert egyébként TP lenne?
    return IoU(conf) #

def FPR(conf): #False positive rate
    divider = (conf["FP"] + conf["TN"])
    return conf["FP"] / divider if divider > 0 else np.nan
def FNR(conf): # false negative rate
    divider = (conf["FN"] + conf["TP"])
    return conf["FN"] / divider if divider>0 else np.nan
def MA(conf): # missclassified area
    return (conf["FP"] + conf["FN"]) / (conf["TP"]+ conf["TN"]+ conf["FP"]+ conf["FN"])

def MAPE(conf): #mean absolute percentage error
    divider = (conf["TP"] + conf["FN"])
    return abs(conf["FP"] - conf["FN"]) / divider if divider > 0 else 0

def Rerr(conf): # pixel absolute error ratio
    return (conf["FP"] - conf["FN"]) / (conf["TP"] + conf["FP"] + conf["FN"] + conf["TN"])

def MAPE2(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    S1 = pred.sum()
    S2 = gt.sum()

    return abs(S1-S2) / S2

def Rerr2(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    S1 = pred.sum()
    S2 = gt.sum()
    N = pred.size
    return (S1 - S2) / N
