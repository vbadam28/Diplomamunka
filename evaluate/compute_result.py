from calc_metrics_parallel import runAll, processTumorStatistics
from logic.pipeline.pipelineFactory import PipelineFactory
import time


if __name__ == "__main__":
        datasetPath = "../../content/dataset"

        fileToSave = "csv/file_to_save.csv"
        '''
        4 fájl amiből dolgozik az result.py
        (mindegyik módszerre 1)
                "csv/biratu_parallel_all_slices.csv"
                "csv/sw_all_slices.csv"
                "csv/saad_parallel_all_slices.csv"
                "csv/enhanced_div_parallel_all_slices.csv"
        '''

        '''
        Pipeline-ok különböző paraméterezésekkel pls változó
                1 nagy módszerre 1 fájl általában ~3-5 perc 1 paraméterezés
                probléma: SlidingWindow(biratu továbbfejlezstés) 8 különböző paraméterezéssel  kb ~50percig fut
                ->nem ajánlott újra futattni
        '''
        pls = [
                (PipelineFactory.select5Seeds, {}, "biratu"), # ~3 perc

                #all sw 50 min
                #(PipelineFactory.slidingWindows, {"sw:mode":"mean","wss:mode":"center"}, "sw"), #4db kb fél óra
                #(PipelineFactory.slidingWindows, {"sw:mode":"max","wss:mode":"center"}, "sw"),
                #(PipelineFactory.slidingWindows, {"sw:mode":"std","wss:mode":"center"}, "sw"),
                #(PipelineFactory.slidingWindows, {"sw:mode":"blob","wss:mode":"center"}, "sw"),
                #(PipelineFactory.slidingWindows, {"sw:mode": "mean", "wss:mode": "max"}, "sw"),  #4db kb fél óra
                #(PipelineFactory.slidingWindows, {"sw:mode": "max", "wss:mode": "max"}, "sw"),
                #(PipelineFactory.slidingWindows, {"sw:mode": "std", "wss:mode": "max"}, "sw"),
                #(PipelineFactory.slidingWindows, {"sw:mode": "blob", "wss:mode": "max"}, "sw"),


                #(PipelineFactory.divergenceSeeds, {}, "saad"), # ~4perc

                #(PipelineFactory.enhancedDivergence, {}, "enhanced div"), # ~4perc
                ]

        start = time.time()
        df = runAll(pls, basePath=datasetPath)#None,processTask=processTumorStatistics)
        end = time.time()
        df.to_csv(fileToSave,sep=";")


        sec = end-start
        print(f"runtime: {sec} sec ---> {int(sec/60)}:{int(sec-(int(sec/60)*60))} ")