import abc


class VerboseObject(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 verbose):
        """
        Object with parameter for verbosity.

        :param verbose: if true, allows verbose execution.
        :return: Nothing.
        """
        self.verbose = verbose

    def verboselly_say(self,
                       string):
        print(string) if self.verbose else None
