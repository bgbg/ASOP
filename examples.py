from matplotlib import pylab as plt
import numpy as np

import asop

def rosenbrock((x, y)):
        '''Rosenbrock function

        Rosenbrock(x, y) = (1 - x)^2 + 100(y - x^2)^2
        '''
        return(1 - x) ** 2 + 100 * (y - x**2)**2

def rastrigin((x, y)):
    '''Rastrigin 2D function

    rastrigin(x, y) = 20 + x^2 + y^2 + 10(cos(2pi * x) + cos(2pi * y))
    '''

    ret = 20.0 + x**2 + y **2 + \
        10.0 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
    return ret

def sines2Dfunc((x, y)):
    return 1.0 + np.sin(x)**2 + np.sin(y)**2 - 0.1 * np.exp(-(x**2) - (y**2))


def humpFunction(v):
    '''Hump test function

    F(n) = \begin{cases} -h_k\left[1-\left(\frac{d(x,k)}{r_k} \right )^{\alpha_k} \right ], & \mbox{if } d(x,k) \le r_k \\
3n+1, & \mbox{otherwise}
\end{cases}

    '''

def demoOptimizeFunction2params(TIMES, SIZE, func, strTitle=None):
    if strTitle is None:
        strTitle = ''

    dimensions = [asop.variableTypes.ContinuousVariable(np.linspace(-2, 2, 1000),
                                                   samplingStd=.05)
                  for i in range(2)] #@UnusedVariable
    scalingFunc =  asop.scaling.TanhScaling(78, 0.015)
    optimizer = asop.ASOP(func, dimensions, scaling=scalingFunc)
    d0 = optimizer.dimensions[0]
    d1 = optimizer.dimensions[1]

    iteration = 0
#    pop = optimizer.train(100, )
#    pop0 = pop
    populations = []
#    populations = [pop0]
#    iteration += 1
    fig  = plt.figure()
    R = 3

    for i in range(TIMES): #@UnusedVariable
        iteration += 1
        pop = optimizer.train(SIZE, 10)
        populations.append(pop)
        ax = fig.add_subplot(R, 1, R)
        ax.plot([iteration,]*2,
            (np.percentile([p[1] for p in pop], 20),
             np.percentile([p[1] for p in pop], 80)), '-k')
        ax.plot(iteration, np.median([p[1] for p in pop]), 'ok')

    ax = fig.add_subplot(R, 1, 1)
    ax.plot(d0.x, d0.pdfValues, '-g')

    ax = fig.add_subplot(R, 1, 2)
    ax.plot(d1.x, d1.pdfValues, '-g')

    for i in range(0): #@UnusedVariable
        iteration += 1
        pop = optimizer.train(100, 100)
        populations.append(pop)
        ax = fig.add_subplot(R, 1, R)
        ax.plot([iteration,]*2,
            (np.percentile([p[1] for p in pop], 20),
             np.percentile([p[1] for p in pop], 80)), '-k')
        ax.plot(iteration, np.median([p[1] for p in pop]), 'ok')


    ax = fig.add_subplot(R, 1, 1)
    ax.plot(d0.x, d0.pdfValues, '-b')

    ax = fig.add_subplot(R, 1, 2)
    ax.plot(d1.x, d1.pdfValues, '-b')

    ax = fig.add_subplot(R, 1, R)
    ax.set_xlim(0, iteration + 1)
    ax.set_yscale('log')
    plt.close(fig)



    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.colors import LogNorm

    fig = plt.figure()
    ax = Axes3D(fig, azim = -128, elev = 43)
    s = .1
    X = np.arange(-3, 3.+s, s)
    Y = np.arange(-3, 3.+s, s)
    X, Y = np.meshgrid(X, Y)
    Z = map(func, zip(X, Y))
    ax.plot_surface(X, Y, Z, rstride = 4, cstride = 4, norm = LogNorm(),
                    cmap = cm.jet) #@UndefinedVariable
    pop = optimizer.sample(100)
    z = map(optimizer.func, pop)
    x = [p[0] for p in pop]
    y = [p[1] for p in pop]
    ax.plot(x, y, z, '*k')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('%s %d %d'%(strTitle, TIMES, SIZE))
    fig.savefig('%s_%s_%s.png'%(asop.variableTypes.DEBUG_FLAG, TIMES, SIZE))
#    plt.close(fig)
    return (optimizer, z)


if __name__ == '__main__':
    for func in (rastrigin, rosenbrock, sines2Dfunc):
        for i in (3,): # (1, 2, 3):
            asop.variableTypes.DEBUG_FLAG = i
            lValues = []
            fig = plt.figure(100 + i)
            ax = fig.add_subplot(111)
            for (t, s) in [(1, 100)]: #[ (10, 100), (100, 10)]:
                print (t, s)
                optimizer, values = demoOptimizeFunction2params(t, s, func, func.__name__)
                values.sort()
                lValues.append(values)
                ax.plot(values, '-', label='%s, %d - %d, %d'%(func.__name__,
                                                              i, t, s))
            ax.legend(loc=0)
            ax.set_yscale('log')


    plt.show()