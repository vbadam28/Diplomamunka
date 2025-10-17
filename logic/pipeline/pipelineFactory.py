from gui.controls.cv_region_growing_control import CVRegionGrowingControl
from gui.controls.quad_tree_control import QuadTreeControl
from gui.controls.sliding_window_control import SlidingWindowControl
from gui.controls.window_seed_selector_control import WindowSeedSelectorControl
from logic.pipeline.pipeline import Pipeline
from logic.preprocessing.enhanced_quad_tree import EnhancedQuadTree
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
        p = Pipeline(
            preprocessing=[],
            seedSelector=ManualSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )
        p.controls = [CVRegionGrowingControl]
        return p

    @staticmethod
    def select5Seeds():
        p = Pipeline(
            preprocessing=[SkullStripping()],
            seedSelector=BlockBasedSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )
        p.controls = [CVRegionGrowingControl]
        return p


    ''' 
        BlockBased selec5Seeds hez képest sliding window különöző window választási stratégiákkal.
    '''
    @staticmethod
    def slidingWindows():
        p = Pipeline(
            preprocessing=[SlidingWindow()],
            seedSelector=WindowSeedSelector(),
            seedFilters=[],
            regionGrowing=OpenCVRegionGrowing()
        )

        p.controls = [SlidingWindowControl, WindowSeedSelectorControl, CVRegionGrowingControl]
        return p

    @staticmethod
    def divergenceSeeds():
        p = Pipeline(
            preprocessing=[PreprocessingStep(normalize), PreprocessingStep(enhanceImage),QuadTree()],
            seedSelector=DivergenceSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()
        )
        p.controls = [QuadTreeControl]
        return p

    @staticmethod
    def enhancedDivergence():
        p = Pipeline(
            preprocessing=[PreprocessingStep(normalize), PreprocessingStep(enhanceImage), EnhancedQuadTree(),],
            seedSelector=DivergenceSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()
        )
        p.controls = [QuadTreeControl]
        return p

    ''' 
        Preporcessingben,
        (akár lehetne a gradient nél is, mert elég zajos az a histogram)
    '''
    @staticmethod
    def divergenceSeedsWithGaussianBlur():
        p = Pipeline(
            preprocessing=[PreprocessingStep(normalize), PreprocessingStep(gaussianBlur), PreprocessingStep(enhanceImage), QuadTree()],
            seedSelector=DivergenceSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()
        )
        p.controls = []
        return p

    '''
        SplitnMerge + Gmm
    '''
    @staticmethod
    def splitmergeGmm():

        p = Pipeline(
            preprocessing=[PreprocessingStep(normalize),PreprocessingStep(enhanceImage), QuadTree()],
            seedSelector=GmmSeedSelector(),
            seedFilters=[],
            regionGrowing=OptThresholdAccRegionGrowing()#AccOpenCVRegionGrowing()
        )

        p.controls = []
        return p

