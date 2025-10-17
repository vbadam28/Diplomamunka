class Pipeline:
    def __init__(self, preprocessing, seedSelector, seedFilters, regionGrowing):
        self.preprocessing = preprocessing
        self.seedSelector = seedSelector
        self.seedFilters = seedFilters
        self.regionGrowing = regionGrowing
        self.controls = []
    def run(self, ctx):
        #ctx = PipelineContext(image)
        for step in self.preprocessing:
            ctx = step.process(ctx)

        ctx =  self.seedSelector.select(ctx)

        for step in self.seedFilters:
            ctx = step.filter(ctx)

        mask = self.regionGrowing.run(ctx)

        return mask
