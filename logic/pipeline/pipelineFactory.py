from logic.pipeline.pipeline import Pipeline
from logic.preprocessing.preprocessing_step import PreprocessingStep, enhanceImage, normalize, gaussianBlur
from logic.preprocessing.skull_stripping import SkullStripping
from logic.preprocessing.quad_tree import QuadTree
from logic.preprocessing.sliding_window import SlidingWindow
from logic.region_growing.acc_opencv_region_growing import AccOpenCVRegionGrowing
from logic.region_growing.opencv_region_growing import OpenCVRegionGrowing
from logic.region_growing.opt_threshold_acc_region_growing import OptThresholdAccRegionGrowing
from logic.seed_selector.block_based_seed_selector import BlockBasedSeedSelector
from logic.seed_selector.divergence_seed_selector import DivergenceSeedSelector
from logic.seed_selector.gmm_seed_selector import GmmSeedSelector
from logic.seed_selector.manual_seed_selector import ManualSeedSelector
from logic.seed_selector.window_seed_selector import WindowSeedSelector


class PipelineFactory:
    @staticmethod
    def manualSeeds():
        return Pipeline(
            preprocessing=[],
            seedSelector=ManualSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )

    @staticmethod
    def select5Seeds():
        return Pipeline(
            preprocessing=[SkullStripping()],
            seedSelector=BlockBasedSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )

    ''' 
        BlockBased selec5Seeds hez képest sliding window különöző window választási stratégiákkal.
    '''
    @staticmethod
    def slidingWindows():
        return Pipeline(
            preprocessing=[SlidingWindow()],
            seedSelector=WindowSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )
    @staticmethod
    def divergenceSeeds():
        return Pipeline(
            preprocessing=[PreprocessingStep(normalize), PreprocessingStep(enhanceImage),QuadTree()],
            seedSelector=DivergenceSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()
        )

    ''' 
        Preporcessingben,
        (akár lehetne a gradient nél is, mert elég zajos az a histogram)
    '''
    @staticmethod
    def divergenceSeedsWithGaussianBlur():
        return Pipeline(
            preprocessing=[PreprocessingStep(normalize), PreprocessingStep(gaussianBlur), PreprocessingStep(enhanceImage), QuadTree()],
            seedSelector=DivergenceSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()
        )

    '''
        SplitnMerge + Gmm
    '''
    @staticmethod
    def splitmergeGmm():

        return Pipeline(
            preprocessing=[PreprocessingStep(normalize),PreprocessingStep(enhanceImage), QuadTree()],
            seedSelector=GmmSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()#AccOpenCVRegionGrowing()
        )
