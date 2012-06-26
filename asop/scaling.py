import numpy as np

'''Scaling functions

All the functions in this module create scaling functions - functions that
numerically scale input values
'''


def TanhScaling(x50, steepness):
    ''' Sigmoid tanh scaling

    Scaled value y is calculated as
        y = tanh( s * (x50 - v) )
    where v is the original value,  d is the reference point and s is the
    steepness factor
    '''

    assert steepness != 0
    return lambda inp: np.tanh(steepness * np.subtract(inp, x50))


def logisticScalingFromValueExtrema(values, yHigh):
    '''Create logistic scaling function using a list of values

    The resulting scaling function is designed such that `max(values)`
    will be scaled to `yHigh` and `min(values)` will be scaled to
    `1 - yHigh`. The midpoint is set to the middle of the range
    `[min(values), max(values)]`
    '''

    assert 0 < yHigh < 1.0
    yLow = 1.0 - yHigh
    assert 0 < yLow < 1.0

    mn = np.min(values)
    mx = np.max(values)
    assert mn != mx
    md = mn + (mx - mn) * 0.5
    assert md != 0
    assert mx != 0

    steepnessH = (md + np.log(-yHigh/(yHigh-1))) / mx
    steepnessL = steepnessH # = (-md + np.log(-yLow/(yLow-1))) / mn

    print 'steepnessH', steepnessH
    print 'steepnessL', steepnessL

    steepness = 0.5 * (steepnessH + steepnessL)
    return LogisticScaling(md, steepness)


def tanhScalingFromValueExtrema(values, yHigh):
    '''Create tanh scaling function using a list of values

    The resulting scaling function is designed such that `max(values)` will
    be scaled to `yHigh` and `min(values)` will be scaled to `-yHigh`.
    The parameters of tanh scaling are calculated as follows:
    the midpoint (x50) ix the middle of the [min(values), max(values)] range.
    The steepness,s, is:

        s = -arctanh(yHigh) / (x50 - max(values))

    '''

    assert 0 < yHigh < 1

    if len(values) < 2:
        msg = 'Automatic scaling parameters can be obtained only '\
        'when more than one value is supplied. '
        raise ValueError(msg)
    mn = np.min(values)
    mx = np.max(values)

    if mn == mx:
        msg = 'Automatic scaling parameters can be obtained only '\
        'when the supplied numbers have different values'
        raise ValueError(msg)
    md = mn + (mx - mn) / 2.0

    steepness = - np.arctanh(yHigh) / (md - mx)
    return TanhScaling(md, steepness)



def LinearScaling(a, b):
    '''Simple linear scaling

    y = ax + b
    '''

    assert a != 0
    return lambda inp: np.multiply(a, inp) + b



def LogisticScaling(x50, steepness):
    '''Logistic sigmoid scaling

    y = 1.0 / (1 + exp(-x50 - s * x))

    where x50 is the point of x, at which the scaled value is 0.5,
    s is the steepness factor
    '''

    assert steepness != 0
    return lambda inp: 1.0 / (1 + np.exp(-steepness * np.subtract(inp, x50)))




