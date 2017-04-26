import abc


class VerboseObject(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose):
        self.verbose = verbose

    def verboselly_say(self,
                       string):
        print(string) if self.verbose else None
