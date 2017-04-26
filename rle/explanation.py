from rle.util.verbose_object import VerboseObject


class Explanation(VerboseObject):

    def __init__(self,
                 model,
                 explainer,
                 sampler,
                 depicter,
                 verbose=False):

        self.model = model
        self.explainer = explainer
        self.sampler = sampler
        self.depicter = depicter

        super().__init__(verbose)
