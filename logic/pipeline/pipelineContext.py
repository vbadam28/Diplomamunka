class PipelineContext:
    def __init__(self,image,debug=False,seeds=None):
        self.data = {'image':image,'debug':debug,'roi':image}
        if seeds is not None:
            self.set('seeds', seeds)

    def get(self, key, default=None):
        return self.data.get(key,default)
    def set(self,key,value):
        self.data[key]=value
