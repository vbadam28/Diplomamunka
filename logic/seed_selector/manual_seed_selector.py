from logic.seed_selector.seed_selector import SeedSelector


class ManualSeedSelector(SeedSelector):
    def __init__(self):
        self.debug = False
    def select(self,ctx):
        seeds = ctx.get('seeds')
        if seeds is None:
            raise Exception("Missing seeds")
        seeds = [p.toTuple() for p in seeds]
        ctx.set('seeds',seeds)
        return ctx
