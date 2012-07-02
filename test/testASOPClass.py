
import unittest
import asop
from asop import variableTypes
ASOP = asop.ASOP
import numpy as np


class TestInstatination(unittest.TestCase):
    '''Various functions that test object instatination'''

    def setUp(self):

        def funcOneParam(x):
            return x * x

        def funcTwoParams(x, y):
            #Rosenbrock function
            return(1 - x) ** 2 + 100 * (y - x**2)**2

        def funcThreeParams(x, y, z):
            return funcTwoParams(x, y) + z * z
        self.lFunctions = [funcOneParam, funcTwoParams, funcThreeParams]

    def testDimensionsAsNone(self):
        '''Supply None as dimensions argument'''
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            optimizer = ASOP(f)
            self.assertTrue(len(optimizer.dimensions) == n)


    def testDimensionsAsNumber(self):
        '''Supply a number as dimensions argument'''
        #also supply a string that can be converted to int
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            optimizer = ASOP(f, n)
            self.assertTrue(len(optimizer.dimensions) == n)

            optimizer = ASOP(f, str(n))
            self.assertTrue(optimizer.nDimensions == n)

    def testDimensionsAsVariables(self):
        '''Suppliy list of variables as dimensions argument'''
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            variables = [variableTypes.ContinuousVariable()
                         for j in range(n)] #@UnusedVariable
            optimizer = ASOP(f, variables)
            self.assertTrue(len(optimizer.dimensions) == n)

    def testDimensionsInvalidNumber(self):
        #supply negative int or non-int string
        for p in [-1, -2, 0, 1.2, '1.23']:
            for f in self.lFunctions:
                try:
                    optimizer = ASOP(f, p) #@UnusedVariable
                except ValueError:
                    pass
                else:
                    self.fail()


    def testDimensionsInvalidObjectsInList(self):
        '''Supply invalid object in the dimenstions argument'''
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            variables = [variableTypes.ContinuousVariable()
                         for j in range(n)] #@UnusedVariable
            variables[0] = None
            try:
                optimizer = ASOP(f, variables) #@UnusedVariable
            except ValueError:
                pass
            else:
                self.fail()

    def testDimensionsHaveNames(self):
        '''Dimensions created by default should have names'''
        for f in self.lFunctions:
            optimizer = ASOP(f)
            for dimension in optimizer.dimensions:
                self.assertTrue(hasattr(dimension, 'name'))
                self.assertTrue(bool(dimension.name))


class TestFunctionality(unittest.TestCase):
    '''Various functionality tests'''


    def createDummyObject(self):
        '''Create a dummy ASOP object'''
        def funcTwoParams(x, y):
            #Rosenbrock function
            return(1 - x) ** 2 + 100 * (y - x**2)**2
        optimizer = ASOP(funcTwoParams)
        return optimizer

    def testLearnArgumensDifferentLength(self):
        '''learn function accepts arguments with same length'''
        obj = self.createDummyObject()
        TIMES = 100
        for t in range(TIMES): #@UnusedVariable
            x = np.random.randint(1, 100)
            solutions = obj.sample(x)
            values = np.ones(x + 1)
            try:
                obj.learn(solutions, values)
            except AssertionError:
                pass
            else:
                self.fail()

    def testTrainFailsOnNegativeN(self):
        obj = self.createDummyObject()
        try:
            obj.train(-1)
        except AssertionError:
            pass
        else:
            self.fail()

    def testSampleFailsOnNonPositiveN(self):
        obj = self.createDummyObject()
        try:
            obj.sample(0)
        except AssertionError:
            pass
        else:
            self.fail()


    def testSampleManyTimes(self):
        #this is a stability test. sample many times
        #and make sure nothing crashes
        TIMES = 100
        SIZE = 20
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            for n in np.logspace(0, 4, SIZE).astype(int):
                s = obj.sample(SIZE)
                self.assertTrue(len(s) == n)



    def testTrainManyTimes(self):
        TIMES = 100
        SIZE = 20
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            for n in np.logspace(0, 4, SIZE).astype(int):
                nToReturn = max(1, int(n / 2))
                s = obj.train(n, nToReturn=nToReturn)
                self.assertTrue(len(s) == nToReturn)




if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()