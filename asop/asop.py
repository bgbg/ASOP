import variableTypes
import scaling
import types

MINIMIZE, MAXIMIZE = (-1, 1)

class ASOP:
    '''The main class for ASOP algorithm

    '''


    def __init__(self, func, dimensions, direction=MINIMIZE,
                 scaling=None):
        '''

        @param func: callable. The objective function that needs to be optimized
        @param dimensions: a number or list
            of Variable type objects.
            If `dimensions` is a number, then this number of
            dimensions will be used. `ContinuousVariable`
            will be used will be used in this case.
            One may also pass a list of  variable objects to be used.
        @param direction: either MINIMIZE or MAXIMIZE. Default: minimize
        @param scaling: scaling function, None (default) or "auto". The scaling
            function is a function that receives a number and returns another
            number. If "auto" is passed as argument, scaling function will
            be created during the first call to `learn` by passing the
            `values` arguments to `scaling.tanhScalingFromValueExtrema`
        '''

        assert callable(func)
        self.func = func #FIXME Add support for no func at all. In this
            #case the "train" function will be disabled and the training
            #will be done by doing obj.sample(N), evaluating the samples
            #externally and then updating the hyperspace by the means of
            #obj.learn(samples, values)
        self.dimensions = self._parseDimensionsArgument(dimensions)
        assert \
            len(map(id, self.dimensions)) ==\
            len(set(map(id, self.dimensions))), \
            "ASOP dimensions contain at least two variables that point to"\
            " the same object. You don't want that."
        assert direction in (MINIMIZE, MAXIMIZE)
        self.direction = direction
        if (not scaling is None) and (scaling != 'auto'):
            assert callable(scaling)
        self.scaling = scaling



    def _parseDimensionsArgument(self, dimensions):
        if dimensions is None:
            msg = 'Dimensions cannot be `None`'
            raise ValueError(msg)
        nargs = None
        try:
            iter(dimensions)
        except:
            nargs = int(dimensions)
        else:
            if isinstance(dimensions, types.StringTypes):
                nargs = int(float(dimensions))
            else:
                dimensions = list(dimensions)
                attributes = ['random', 'alterSamplingDistribution',
                          'applySamplingScore']
                for d in dimensions:
                    for a in attributes:
                        if not hasattr(d, a):
                            msg = 'Variable does not have attribute "%s"'%a
                            raise ValueError(msg)

        if nargs is not None:
            if nargs < 1:
                msg = 'Illegal number of argument. Must be 1 or more'
                raise ValueError(msg)
            dimensions = [variableTypes.ContinuousVariable(name='X%d'%i) \
                          for i in range(nargs)]


        return dimensions

    def train(self, n=1, nToReturn=0):
        ''' Perform `n` training iterations
        @param n: number of iterations
        @param nToReturn: maximum number of (solution, value) pairs to return
        @return: list of best (solution, value) pairs seen by the function.
            The returned solutions are selected according to value and to
            self.direction (minimization or maximization)
        '''

        assert nToReturn >= 0

        theSample = self.sample(n)

        #if func accepts N arguments, map expects N iterables. theSample
        #above is a single iterable, in which each element is N-tuple.
        #Thus, need to transpose
        theValues = map(self.func, theSample)

        self.learn(theSample, theValues)
        if nToReturn > 0:
            ret = [(s, v) for (s, v) in zip(theSample, theValues)]
            reverse = (self.direction == MAXIMIZE)
            ret.sort(cmp=lambda a, b: cmp(a[1], b[1]), reverse=reverse)
            ret = ret[0:nToReturn]
        else:
            ret = []
        return ret



    def learn(self, solutions, values):
        '''Update the hyper-space with the given solutions and function values

        Note that this function bypasses the object's objective function
        '''

        assert len(solutions) == len(values)
        if self.scaling:
            if self.scaling == 'auto':
                try:
                    self.scaling = scaling.tanhScalingFromValueExtrema(values,
                                                                       0.8)
                except AssertionError:
                    msg = 'Could not create scaling function using the'\
                    ' specified values'
                    print msg
                    raise
            scaled = map(self.scaling, values)
        else:
            scaled = values

        scaled = map(lambda x: self.direction * x, scaled)

        for solution, value in zip(solutions, scaled):
            for x, dimension in zip(solution, self.dimensions):
                dimension.alterSamplingDistribution(value, x,
                                                    dimension.samplingStd,
                                                    immediateApply=False)
        #note the delayed apply above. Need to explicitly apply the score
        for dimension in self.dimensions:
            dimension.applySamplingScore()






    def sample(self, n=1):
        '''Draw n samples from the hyperspace'''

        assert n > 0
        components = []
        for d in self.dimensions:
            values = d.random(n)
            components.append(values)
        #matrix transpose magic http://stackoverflow.com/a/4937526/17523
        ret = zip(*components)
        return ret


