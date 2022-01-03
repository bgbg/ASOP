
import unittest
import asop
from asop import variableTypes
ASOP = asop.ASOP
import numpy as np


class TestInstatination(unittest.TestCase):
    '''Various functions that test object instatination'''

    def setUp(self):

        def func1(p):
            return np.sum(np.square(p))

        def func2(x):
            return np.sum(1.0 / x)

        def func3(x):
            return np.sum(np.exp(-x))

        self.lFunctions = [func1, func2, func3]

    def testDimensionsEmptyOrNone(self):
        '''Supply None as dimensions argument'''
        for f in self.lFunctions:
            try:
                ASOP(f)
            except TypeError:
                pass
            else:
                self.fail()
            try:
                ASOP(f, None)
            except (ValueError, AssertionError):
                pass
            else:
                self.fail()


    def testDimensionsAsNumber(self):
        '''Supply a number as dimensions argument'''
        #also supply a string that can be converted to int
        for n in [1, 2, 4, 5]:
            for f in self.lFunctions:
                optimizer = ASOP(f, n)
                self.assertTrue(len(optimizer.dimensions) == n)
                optimizer = ASOP(f, str(n))
                self.assertTrue(len(optimizer.dimensions) == n)

    def testDimensionsAsVariables(self):
        '''Suppliy list of variables as dimensions argument'''
        for n in [1, 2, 4, 5]:
            for f in self.lFunctions:
                variables = [variableTypes.ContinuousVariable()
                             for j in range(n)] #@UnusedVariable
                optimizer = ASOP(f, variables)
                self.assertTrue(len(optimizer.dimensions) == n)

    def testDimensionsInvalidNumber(self):
        #supply negative int or non-int string
        for p in [-1, -2, 0, None, 'five']:
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
            for n in [2, 4, 6]:
                optimizer = ASOP(f, n)
                for dimension in optimizer.dimensions:
                    self.assertTrue(hasattr(dimension, 'name'))
                    self.assertTrue(bool(dimension.name))


class TestFunctionality(unittest.TestCase):
    '''Various functionality tests'''


    def createDummyObject(self, n=2):
        '''Create a dummy ASOP object'''
        def f1(p):
            return np.sum(np.square(p))

        optimizer = ASOP(f1, n)
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
                s = obj.sample(int(n)) #@UnusedVariable

    def testSampleSize(self):
        TIMES = 100
        SIZE = 20
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            for n in np.logspace(0, 4, SIZE).astype(int):
                s = obj.sample(int(n))
                self.assertTrue(len(s) == n)



    def testTrainManyTimes(self):
        '''Train many times. Test for stability and returned population size'''
        TIMES = 50
        SIZE = 10
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            for n in np.logspace(0, 3, SIZE).astype(int):
                nToReturn = max(1, int(n / 2))
                s = obj.train(n, nToReturn=nToReturn)
                self.assertTrue(len(s) == nToReturn)


    def testTrainWithReturnValueGreaterThanN(self):
        ''' On training, specify # to return greater than training size '''
        TIMES = 10
        SIZE = 10
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            for n in np.logspace(0, 3, SIZE).astype(int):
                nToReturn = n * 2
                s = obj.train(n, nToReturn=nToReturn)
                self.assertTrue(len(s) == n)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()