from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
try:
    from evaluate.calc_metrics import score
except:
    from calc_metrics import score


def showHeatmap(groupPS, values=("DS_mean","Dice átlag"),index=("params","Paraméterezés"), columns=("lesionSize","Tumor méret kategória")):
    heatmapDf = groupPS.pivot(index=index[0], columns=columns[0], values=values[0])

    plt.figure(figsize=(8, 5))
    sns.heatmap(heatmapDf, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5)
    plt.title(values[1])
    plt.xlabel(columns[1])
    plt.ylabel(index[1])
    plt.tight_layout()
    plt.show()








def loadDataFrame(path,brainStatsPath=None):
    df = pd.read_csv(path, sep=";")
    dfError = None
    if "error" in df.columns:
        dfError = df[df["error"].notna()]
        dfError = dfError[["brain ID","slice ID","params","error"]]

        #df = df[df["error"].isna()]
        #df = df.drop(columns=["error"])

    df['gtArea2'] = df["TP"] + df["FN"]

    if brainStatsPath is not None:
        dfStat = pd.read_csv(brainStatsPath,sep=";")
        df = df.merge(dfStat,on=["brain ID","slice ID"], how="left")

    '''if dfError is not None:
        import nibabel as n
        cache = {}
        for _, row in dfError.iterrows():
            brainID, sliceID = row["brain ID"], row["slice ID"]
            if brainID not in cache:
                cache[brainID] = n.load(f"../../content/dataset/brain_{brainID}/{brainID}_seg.nii").get_fdata()
            seg = cache[brainID][:, :, sliceID]
           df.loc["gtArea"]  np.count_nonzero(seg)
    '''



    dfUnique = df.groupby(["brain ID", "slice ID"]).agg(gtArea=("gtArea", "first")).reset_index()

    areaBins = np.quantile(dfUnique["gtArea"], [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("area bins:", areaBins)
    df["lesionSize"] = pd.cut(
        df["gtArea"],
        bins=areaBins,  # [0,100,500,2000,np.inf],
        labels=["tiny", "small", "medium", "large", "huge"],
    )
    if brainStatsPath is not None:
        dfUnique = df.groupby(["brain ID", "slice ID"]).agg(brainArea=("brainArea", "first")).reset_index()
        areaBins = np.quantile(dfUnique["brainArea"], [0, 0.25, 0.5, 0.75, 0.9, 1.0])
        print("area bins:", areaBins)
        df["brainSize"] = pd.cut(
            df["brainArea"],
            bins=areaBins,  # [0,100,500,2000,np.inf],
            labels=["tiny", "small", "medium", "large", "huge"],
        )
        df = makeBins(df,["brainArea","contrast","tumorStd"],["brainSize","contrastGroup","tumorStdGroup"])

    df["params"] = df["params"].str.strip("{}").str.split(",").apply(
        lambda params: "-".join(pair.split(":")[-1].strip().strip(" '\"") for pair in params))

    return df,dfError


def groupByDataFrame(df, by, metrics, aggFuncs):
    grouped = df.groupby(by,observed=True)[metrics].agg(aggFuncs)
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped


def makeBins(df,cols,res):
    dfUnique = df.groupby(["brain ID", "slice ID"])[cols].agg("first").reset_index()
    for i,col in enumerate(cols):
        areaBins = np.quantile(dfUnique[col], [0, 0.25, 0.5, 0.75, 0.9, 1.0])
        print(f"{col} bins:", areaBins)
        df[res[i]] = pd.cut(
            df[col],
            bins=areaBins,  # [0,100,500,2000,np.inf],
            labels=["tiny", "small", "medium", "large", "huge"],
        )
    return df

def calcVolumetricDataFrame(df):
    volcm = df.groupby(["params", "brain ID"])[["TP", "TN", "FP", "FN","gtArea"]].sum().reset_index()


    areaBins = np.quantile(volcm["gtArea"], [0, 0.25, 0.5, 0.75, 0.9, 1.0])
    print("vol area bins", areaBins)
    volcm["lesionSize"] = pd.cut(
        volcm["gtArea"],
        bins=areaBins,  # [0,100,500,2000,np.inf],
        labels=["tiny", "small", "medium", "large", "huge"],
    )


    dfVol = volcm.apply(lambda row: pd.Series(
        score(conf={"TP": row["TP"], "TN": row["TN"], "FP": row["FP"], "FN": row["FN"], }, addConf=False)[0]), axis=1)

    dfVol = pd.concat([volcm, dfVol], axis=1)





    return dfVol


def calc(path,brainStatsPath=None):
    df,dfError = loadDataFrame(path,brainStatsPath)

    metrics = ["Acc", "DS", "IoU", "OF", "EF", "Sn", "Sp", "PSNR", "AO", "FPR", "FNR", "MA", "Rerr", "TP", "TN", "FP",
               "FN"]
    aggFuncs = ["min", "max", "mean", "median", "std", "count"]

    grouped = groupByDataFrame(df, by="params", metrics=metrics, aggFuncs=aggFuncs)
    # grouped.to_csv("csv/summary/sw_params_summary.csv",sep=";", index=False)

    groupPS = groupByDataFrame(df, by=["params", "lesionSize"], metrics=metrics, aggFuncs=aggFuncs)
    # groupPS.to_csv("csv/summary/sw_params_by_size_summary.csv",sep=";", index=False)

    dfVol = calcVolumetricDataFrame(df)
    # dfVol.to_csv("csv/summary/sw_vol_summary.csv",sep=";", index=False)

    return df, grouped, groupPS, dfVol, dfError




def runBiratu():
    df, grouped, groupPS, dfVol, _ = calc("csv/sw_all_slices.csv", "csv/all_slices_stats.csv")
    dfVolGrouped = groupByDataFrame(dfVol, by=["params", "lesionSize"], metrics=["DS", "EF", "OF", "Sp", "Sn"],
                                    aggFuncs=["mean", "median"])

    swParams = ["blob-center", "mean-center", "max-center"]

    dfBiratu, groupedBiratu, groupPSBiratu, dfVolBiratu, _ = calc("csv/biratu_parallel_all_slices.csv",
                                                                  "csv/all_slices_stats.csv")
    dfVolBiratuGrouped = groupByDataFrame(dfVolBiratu, by=["params", "lesionSize"],
                                          metrics=["DS", "EF", "OF", "Sp", "Sn", "FPR", "FNR"],
                                          aggFuncs=["mean", "median"])

    # df, grouped, groupPS, dfVol = calc("csv/saad_parallel_all_slices.csv")

    dfVolGrouped = groupByDataFrame(dfVol, by=["params", "lesionSize"],
                                    metrics=["DS", "EF", "OF", "Sp", "Sn", "FPR", "FNR"], aggFuncs=["mean", "median"])

    sel = groupPS[groupPS["params"].isin(["blob-center", "mean-center"])]

    groupPSBiratu["params"] = "biratu"
    groupCombined = pd.concat([sel, groupPSBiratu], ignore_index=True)

    groupCombined["params"] = pd.Categorical(
        groupCombined["params"],
        categories=["max-center", "std-center", "mean-center", "blob-center", "biratu"],
        ordered=True
    )

    groupCombined = groupCombined.sort_values("params").reset_index(drop=True)
    dfBiratu["params"] = "biratu"
    # ["max-center","std-center","blob-center","mean-center"]
    combined = pd.concat([df[df["params"].isin(["std-center", "blob-center", "mean-center"])], dfBiratu],
                         ignore_index=True)
    allSlicesDS = groupByDataFrame(combined, by=["params"], metrics=["DS", "FPR", "FNR", "EF", "OF", "Sn", "Sp"],
                                   aggFuncs=["mean", "median"]).reset_index(drop=True)
    # print(allSlicesDS)

    biratu = combined[combined["params"] == "biratu"][["brain ID", "slice ID", "DS", "lesionSize"]].rename(
        columns={"DS": "DS_biratu"})
    swmean = combined[combined["params"] == "mean-center"][["brain ID", "slice ID", "DS", "lesionSize"]].rename(
        columns={"DS": "DS_sw_mean"})
    swblob = combined[combined["params"] == "blob-center"][["brain ID", "slice ID", "DS", "lesionSize"]].rename(
        columns={"DS": "DS_sw_blob"})

    dfDiffs = (
        biratu
            .merge(swmean, on=["brain ID", "slice ID"], how="left")
            .merge(swblob, on=["brain ID", "slice ID"], how="left")
    )

    dfDiffs["diff_swmean"] = dfDiffs["DS_biratu"] - dfDiffs["DS_sw_mean"]
    dfDiffs["diff_swblob"] = dfDiffs["DS_biratu"] - dfDiffs["DS_sw_blob"]
    dfDiffs["diff_sw"] = dfDiffs["DS_sw_mean"] - dfDiffs["DS_sw_blob"]
    #dfDiffs.to_csv("csv/biratu_sw_diff.csv",sep=";")

    with pd.option_context("display.max_columns", None):#, "display.max_rows", None):
        print(dfDiffs[(dfDiffs["diff_swmean"]>0) & (dfDiffs["DS_sw_mean"]>0) & (dfDiffs["DS_biratu"]>0) & (dfDiffs["lesionSize"]!="tiny") & (dfDiffs["lesionSize"]!="small")].sort_values("diff_swmean",ascending=False))

    showHeatmap(groupPS)
    showHeatmap(groupPS, values=("FPR_mean", "FPR átlag"))
    showHeatmap(groupPS, values=("FNR_mean", "FNR átlag"))

    # showHeatmap(groupPS, values=("DS_median", "Dice medián"),columns=("brainSize","Agy(egészséges) méret kategória"))

    # showHeatmap(groupPS, values=("FPR_mean", "FPR átlag"), columns=("brainSize","Agy(egészséges) méret kategória"))
    # showHeatmap(groupPS, values=("FPR_median", "FPR medián"), columns=("brainSize","Agy(egészséges) méret kategória"))
    # showHeatmap(groupPS, values=("FNR_mean", "FNR átlag"), columns=("brainSize","Agy(egészséges) méret kategória"))
    # showHeatmap(groupPS, values=("FNR_median", "FNR medián"), columns=("brainSize","Agy(egészséges) méret kategória"))

    # showHeatmap(groupPS, values=("EF_mean", "EF átlag"))
    # showHeatmap(groupPS, values=("EF_median", "EF medián"))
    # showHeatmap(groupPS, values=("OF_mean", "OF átlag"))
    # showHeatmap(groupPS, values=("Sn_mean", "Sensitivity átlag"))
    # showHeatmap(groupPS, values=("Sp_mean", "Specificity átlag"))

    plotDf = allSlicesDS.melt(id_vars=["params"], value_vars=["DS_mean", "FPR_mean", "FNR_mean", "Sn_mean", "Sp_mean"],
                              var_name="Metric", value_name="Value")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(plotDf, x="params", y="Value", hue="Metric", errorbar="sd")
    plt.title("DS, FPR, FNR, Sn, Sp paraméterezés szerint")
    plt.ylabel("Érték")
    plt.xlabel("Paraméterezés")
    plt.legend(title="Metrika")
    plt.tight_layout()

    for p in ax.patches:
        if p.get_height() > 0.0:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="center",
                        fontsize=10, color="black",
                        xytext=(0, 5), textcoords="offset points"
                        )

    plt.show()