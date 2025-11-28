import sys

from PySide6.QtWidgets import QApplication

from gui.window import MainWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    wd = MainWindow()
    wd.show()
    app.exec()

'''
    from logic.pipeline.pipelineFactory import PipelineFactory
    from evaluate.calc_metrics import run
    import pandas as pd


    df = run(PipelineFactory.select5Seeds, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'])
    df2 = run(PipelineFactory.divergenceSeeds, ['data_ni/BraTS20_Training_355_flair.nii', 'data_ni/W39_1998.09.19_Segm.nii'])

    with pd.option_context("display.max_rows",None,"display.max_columns", None):
        #pd.options.display.max_columns = None # 0
        diff = df.compare(df2,result_names=("biratu","saad"))
        print(diff)
        diff.to_csv("evaluate/csv/basic_compare.csv",sep=";")
'''