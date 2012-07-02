'''
Various variable types to be used by ASOP
'''
from abc import ABCMeta, abstractmethod
from scipy.stats.distributions import norm
import numpy as np
from copy import copy
import sys

try:
    import randomArbitrary
except ImportError, e:
    type_, value, traceback = sys.exc_info()
    msg = 'Failed to import moodule named randomArbitrary. '\
    'If you clone ASOP from a git repository, note that randomArbitrary is a'\
    'submodule. Use recursive cloning. Alternatively, `cd` to asop root '\
    'directory and type: \n'\
    'git submodule init\n'\
    'git submodule update\n\n'
    raise ImportError(msg + value.message)



def inverseLogit(logit):
    logit = np.array(logit)
    finfo = np.finfo(float)
    mx = -np.log(finfo.eps)
    mn = np.log(finfo.eps)
    ret = np.empty(logit.shape)
    selAboveMax = np.greater(logit, mx)
    selBelowMin = np.less(logit, mn)
    sel = (-selAboveMax) * (-selBelowMin)
    ret[selAboveMax] = 1.0
    ret[selBelowMin] = 0.0
    ret[sel] = 1 / (1.0 + np.exp(-logit[sel]))
    return ret


def asciiXYplot(x, y, strTitle=None, marker='|'):
    '''Create a primitive XY plot

    Unlike the wide convention, this function creates the plot
    such that the horizontal axis is Y and the vertical axis is X
    @param x: x values. May be either numerical or strings
    @param y: y values. Only numerical
    @param marker: the character to be used to plot
    @param strTitle: (optional) chart title
    '''

    Y_WIDTH = 60 #The width (in characters) of Y axis
    X_WIDTH = 20

    assert len(x) == len(y)
    assert len(marker) == 1

    outRight = []
    outLeft = []
    if strTitle is not None:
        outRight.append('%s'%strTitle)
        outLeft.append(' ' * X_WIDTH)
    else:
        pass

    mn = min(y)
    mx = max(y)
    if mn == mx:
        mn = mn - 0.2 * abs(mn)
        mx = mx + 0.2 * abs(mx)
    theRange = mx - mn
    md = (mx - mn) / 2.0
    strMin = str(mn)
    strMax = str(mx)
    strMid = str(md)
    strSpacer = ' ' * ((Y_WIDTH - len(strMin) - len(strMax) - len(strMid)) / 2)
    outRight.append(''.join((strMin, strSpacer, strMid, strSpacer, strMax)))
    outLeft.append(' ' * X_WIDTH)

    strSpacer = '_' * ((Y_WIDTH - 3)/2)
    outRight.append('|%s|_%s|'%(strSpacer, strSpacer))
    outLeft.append('_' * X_WIDTH)

    for xValue, yValue in zip(x, y):
        line = [' '] * Y_WIDTH
        loc = int( (Y_WIDTH - 1) * float(yValue - mn) / theRange)
        line[loc] = marker
        outRight.append(''.join(line))

        strX = '%s'%(xValue)
        strX = strX[0:X_WIDTH]
        outLeft.append('%s%s'%( (' '*(X_WIDTH-len(strX)),
                                 strX)))

    assert len(outLeft) == len(outRight)
    out = ['%s|%s'%(ll, lr) for (ll, lr) in zip(outLeft, outRight)]
    return '\n'.join(out)


class VariableBase(object):
    '''Abstract class for every variable type'''
    __metaclass__ = ABCMeta

    @staticmethod
    def probabilityFromScore(score):
        '''Convert the sampling score to sampling probability

        The conversion is done as follows: xxxxxxxxxxxxxx
        we than solve this equation for p
        '''

        p = inverseLogit(score)
        if np.sum(p) > 0:
            p = p / np.sum(p)
        else:
            p = np.ones(len(p), dtype=float) / float(len(p))
        return p



    def __init__(self, samplingValues, samplingScores=None,
                 samplingStd=None,
                 name=None,
                 **kwparam):
        '''Initialize the variable

        @param samplingValues: the values over which the sampling distribution
            is defined.
        @param samplingScores: the scores associated with the sampling values.
            If a single value is passed, it will be used `len(samplingValues)`
            times
        @param samplingStd: standard deviation of sampling. If None (default)
            this value is estimated as the range of sampling values divided
            by 100.0

        Keyword parameters:
        _probabilityCalculationStrategy: EXPERIMENTAL parameter. One of the
        following: "RAW", "CENTERED", "STANDARDIZED".
        Default: "STANDARDIZED"


        If specified, the two parameters have to be iterable and of the same
        length. If omitted, a default action will be performed. The
        default behavior is controlled by the inheriting classes

        '''

        if samplingValues is not None:
            self._x = list(samplingValues)
        else:
            self._x = self._defaultSamplingValues()

        if samplingScores is not None:
            try:
                self._scores = list(samplingScores)
            except TypeError:
                self._scores = [samplingScores] * len(self.x)
        else:
            self._scores = self._defaultScores(self.x)

        assert len(self.x) == len(self.scores),\
            'Length of sampling values (%d) should be equal to the '\
            'length of the sampling scores (%d)'%(len(self.x),
                                                      len(self.scores))

        if name is None:
            name = ''

        strategy = kwparam.pop('_probabilityCalculationStrategy',
                               'STANDARDIZED')
        self._probabilityCalculationStrategy = strategy
        self.name = name
        pValues = self._probabilityFromScore()
        self._pdfValues = pValues
        self._rng = self._createRNG()
        self.rand = self.random #alias
        if samplingStd is None:
            samplingStd = (max(self.x) - min(self.x)) / 4.0
            #the above line means that we assume the entire
            #range covers more than 99.9% of a normally distributed
            #random variable
        self.samplingStd = samplingStd

        self._nUpdates = 0

    def get_pdf_values(self):
        return copy(self._pdfValues)


    def get_x_values(self):
        return copy(self._x)


    x = property(get_x_values)
    pdfValues = property(get_pdf_values)


    @abstractmethod
    def _createRNG(self):
        '''Create self._rng - random number generator object

        '''
        pass


    def get_x(self):
        return self._x


    def get_scores(self):
        return self._scores

    x = property(get_x, None, None, None)
    scores = property(get_scores, None, None, None)



    def random(self, times=None):
        '''Randomly sample the variable specified amount of times

        Randomly sample the variable, using the underlying distribution.
        @param times: sample this number of times. If None (default), return
            a single number
        @return: if times is None, return a single number. Else, return a list
            with `times` numbers in it
        '''
        return self._rng.random(times)

    def applySamplingScore(self):
        '''Synchronize the internal PDF with the sampling score

        Make sure the internal probability density function corresponds to
        the sampling score
        '''

        pdfValues = np.multiply(self.pdfValues,
                                      self._probabilityFromScore())
        if np.all(pdfValues==0):
            pdfValues = np.ones(len(pdfValues), dtype=float) / len(pdfValues)
        self._pdfValues = pdfValues
        self._rng.set_pdf(self.x, self._pdfValues)
        self._scores = self._defaultScores(self.x)


    def alterSamplingDistribution(self, amount, location, width,
                                   immediateApply=True):
        '''Update the underlying distribution function

        @param amount: update the distribution function by this amount
        @param location : where should the distribution be updated
        @param width: how wide should the update be
        @param immediateApply: should the score be immediately applied to
            the internal PDF. Default: True. Set this argument to False if
            you plan to perform multiple updates in order to save time. Make
            sure to call `applySamplingScore` when done.
        '''

        assert(width >= 0)
        self._nUpdates += 1
        self._updateInternalSamplingScore(amount, location, width)
        if immediateApply:
            self.applySamplingScore()


    @abstractmethod
    def _updateInternalSamplingScore(self, amount, location, width):
        '''This function performs the actual changes to the sampling score
        and is unique to different types of variables'''

        pass

    @classmethod
    def _defaultScores(cls, x):
        return [0.0] * len(x)


    @staticmethod
    @abstractmethod
    def _defaultSamplingValues():
        pass


    def _probabilityFromScore(self):

        #implementation note: there are two very similar
        #functions: the static function probabilityFromScore
        #and this one. This separation is intentional.
        #Among others, it allows more convenient testing

#        scores = (self.scores - np.mean(self.scores))
        strategy = self._probabilityCalculationStrategy
        if strategy == 'RAW':
            scores = self.scores
        elif strategy == 'CENTERED':
            scores = self.scores - np.mean(self.scores)
        elif strategy == 'STANDARDIZED':
            scores = self.scores - np.mean(self.scores)
            if np.std(self.scores) != 0:
                scores /= np.std(self.scores)
        else:
            raise ValueError('_probabilityCalculationStrategy parameter has an illegal value of "%s"'%strategy)
        return self.probabilityFromScore(scores)

    def __repr__(self):
        if self.name is None:
            strName = ''
        else:
            strName = self.name
        ret = '<%s> "%s"'%(self.__class__.__name__, strName)
        return ret

    def strAsciiPlot(self):
        '''Return a string with ASCII representation of the variable'''
        return asciiXYplot(self.x, self.pdfValues, self.name)



class QuantitativeVariableBase(VariableBase):
    '''Base class for every quantitative variable type'''
    __metaclass__ = ABCMeta

    def __init__(self, samplingValues=None, samplingScores=None,
                 name=None,
                 **kwparam):
        '''See the documentation of `VariableBase`.
        '''

        samplingValues = self.parseSamplingValuesArgument(samplingValues)

        VariableBase.__init__(self, samplingValues=samplingValues,
                              samplingScores=samplingScores,
                              name=name,
                              **kwparam)


    @classmethod
    def parseSamplingValuesArgument(cls, samplingValues):
        '''Parse the samplingValuesArgument passed to __init__'''

        if samplingValues is None:
            samplingValues = cls._defaultSamplingValues()
        else:
            samplingValues = np.array(samplingValues)
            s = samplingValues[0:-1] - samplingValues[1:]
            if not np.all(np.sign(s) == -1):
                msg = 'Sampling values that are passed to a quantitative '\
                'variable have to be sorted ascendingly'
                raise ValueError(msg)
        return samplingValues


    def _updateInternalSamplingScore(self, amount, location, width):
        ''' Updating process is performed as follows:
        Normal distribution probability density is calculated such as its
        peak is located at `location`, its variance is defined by `width`
        such that the area under the PDF curve equals the absolute value of
        `amount`. The resulting curve is then added to the score
        '''
        values = norm.pdf(self.x, location, width) * amount
        self._scores = [v1 + v2 for (v1, v2) in zip(self._scores, values)]






class ContinuousVariable(QuantitativeVariableBase):
    '''Continuous variable'''

    def _createRNG(self):
        rng = randomArbitrary.RandomArbitrary(self.x,
                                          self.pdfValues)
        return rng

    @staticmethod
    def _defaultSamplingValues():
        return np.arange(0, 1.0, .01)



class IntegerVariable(QuantitativeVariableBase):
    '''Discrete quantitative variable that is limited to integer values only

    Any value that is passed to the constructor as the `samplingValues`
    argument will be truncated to int.
    If value truncation results in non-unique values, an exception will be
    raised
    '''

    def __init__(self, samplingValues=None, samplingScores=None,
                 name=None,
                 **kwparam):
        samplingValues = self.parseSamplingValuesArgument(samplingValues)
        (samplingValues, samplingScores) = \
            self._prepareSamplingValuesAndScores(samplingValues,
                                                     samplingScores)
        QuantitativeVariableBase.__init__(self,
                                          samplingValues=samplingValues,
                                          samplingScores=samplingScores,
                                          name=name,
                                          **kwparam)

    @classmethod
    def parseSamplingValuesArgument(cls, samplingValues):

        samplingValues = \
            QuantitativeVariableBase.parseSamplingValuesArgument(samplingValues)
        if samplingValues is None:
            samplingValues = cls._defaultSamplingValues()
        else:
            samplingValues = np.array(samplingValues, dtype=int)
        if len(samplingValues) != len(set(samplingValues)):
            msg = '''%s: the sampling values contains conflicts'''%\
                    cls.__name__
            raise ValueError(msg)
        return samplingValues


    @classmethod
    def _prepareSamplingValuesAndScores(cls, x, scores):
        '''Generate sampling scores'''
        DEFAULT_ABSCENT_INT_SCORE = -1000.0
        if scores is None:
            scores = cls._defaultScores(x)
        mn = np.min(x)
        mx = np.max(x)
        fullRange = np.arange(mn, mx + 1, 1)
        fullScores = []
        for xVal in fullRange:
            tmp = np.where(x == xVal)[0]
            if len(tmp):
                fullScores.append(scores[tmp[0]])
            else:
                fullScores.append(DEFAULT_ABSCENT_INT_SCORE)
        return (fullRange, np.array(fullScores))

    def _updateInternalSamplingScore(self, amount, location, width):
        if width == 0:
            ix = self.x.index(location)
            self._scores[ix] += amount
        else:
            QuantitativeVariableBase._updateInternalSamplingScore(self, amount, location, width)



    def _createRNG(self):
        rng = randomArbitrary.RandomArbitraryInteger(self.x, self.pdfValues)
        return rng

    @staticmethod
    def _defaultSamplingValues():
        return range(0, 100)




class QualitativeVariableBase(VariableBase):
    '''Abstract class for every qualitative variable type'''
    #This variable type isn't supported yet
    __metaclass__ = ABCMeta
    def __init__(self, samplingValues, samplingScores=None,
                 name=None,
                 **kwparam):
        raise NotImplementedError()




if __name__ == '__main__':
    pass


