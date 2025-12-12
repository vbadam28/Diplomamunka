import pandas as pd
from matplotlib import  pyplot as plt
import seaborn as sns
import numpy as np

from calc_metrics import score
from calc_results import loadDataFrame, groupByDataFrame, showHeatmap, calcVolumetricDataFrame, calc, runBiratu

#runBiratu()




#"brainSize","contrastGroup","tumorStdGroup"

#groupPS = groupByDataFrame(df,by=["params","tumorStdGroup"],metrics=["DS","EF","OF","Sp","Sn","FPR","FNR"],aggFuncs=["mean","median"])
df, grouped, groupPS, dfVol, dfError = calc("csv/enhanced_div_parallel_all_slices_newer2.csv")
df, grouped, groupPS, dfVol, dfError = calc("csv/saad_parallel_all_slices.csv")

exit()
showHeatmap(groupPS)
showHeatmap(groupPS, values=("FPR_mean", "FPR átlag"))
showHeatmap(groupPS, values=("FNR_mean", "FNR átlag"))

#showHeatmap(groupPS, values=("DS_median", "Dice medián"),columns=("brainSize","Agy(egészséges) méret kategória"))

#showHeatmap(groupPS, values=("FPR_mean", "FPR átlag"), columns=("brainSize","Agy(egészséges) méret kategória"))
#showHeatmap(groupPS, values=("FPR_median", "FPR medián"), columns=("brainSize","Agy(egészséges) méret kategória"))
#showHeatmap(groupPS, values=("FNR_mean", "FNR átlag"), columns=("brainSize","Agy(egészséges) méret kategória"))
#showHeatmap(groupPS, values=("FNR_median", "FNR medián"), columns=("brainSize","Agy(egészséges) méret kategória"))

#showHeatmap(groupPS, values=("EF_mean", "EF átlag"))
#showHeatmap(groupPS, values=("EF_median", "EF medián"))
#showHeatmap(groupPS, values=("OF_mean", "OF átlag"))
#showHeatmap(groupPS, values=("Sn_mean", "Sensitivity átlag"))
#showHeatmap(groupPS, values=("Sp_mean", "Specificity átlag"))


plotDf = allSlicesDS.melt(id_vars=["params"], value_vars=["DS_mean","FPR_mean","FNR_mean","Sn_mean","Sp_mean"], var_name="Metric", value_name="Value")

plt.figure(figsize=(10,5))
ax = sns.barplot(plotDf, x="params",y="Value",hue="Metric", errorbar="sd")
plt.title("DS, FPR, FNR, Sn, Sp paraméterezés szerint")
plt.ylabel("Érték")
plt.xlabel("Paraméterezés")
plt.legend(title="Metrika")
plt.tight_layout()

for p in ax.patches:
    if p.get_height()>0.0:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha="center", va="center",
                    fontsize=10, color="black",
                    xytext=(0,5), textcoords="offset points"
                    )

plt.show()


exit()



sortedDf = df.sort_values(by=["DS","Sn","Sp","EF"], ascending=[False, False, False,True]).reset_index(drop=True)
sortedDf=sortedDf.drop(columns=[col for col in sortedDf.columns if "Unnamed" in col])
with pd.option_context("display.max_columns", None):
    #print(dfVol)
    print(sortedDf[sortedDf["lesionSize"]=="huge"])

    '''
        magas DS
        
        magas Sn + alacsony Sp → „mindent megeszik”
        magas Sp + alacsony Sn → „túl óvatos”
    '''




plotDf = groupPS.melt(id_vars=["params","lesionSize"], value_vars=["OF_mean","EF_mean"], var_name="Metric", value_name="Value")

plt.figure(figsize=(10,5))
sns.barplot(plotDf, x="params",y="Value",hue="Metric", errorbar="sd")
plt.title("Overlap és Extra Fraction paraméterezés szerint")
plt.ylabel("Érték")
plt.xlabel("Paraméterezés")
plt.legend(title="Metrika")
plt.tight_layout()
plt.show()


g = sns.catplot(plotDf[plotDf["lesionSize"]!="tiny"], kind="bar", x="params", y="Value", hue="Metric", col="lesionSize", errorbar="sd", height=4, aspect=1)
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle("OF és EF paraméterezés és tumor méret szerint")
g.set_axis_labels("Paraméterezés","Érték")
g.add_legend()
plt.show()












