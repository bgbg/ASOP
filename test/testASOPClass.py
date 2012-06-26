
import unittest
import asop
ASOP = asop.ASOP
import numpy as np


class TestInstatination(unittest.TestCase):
    '''Various functions that test object instatination'''

    def testDimensionsAsNone(self):
        '''Supply None as dimensions argument'''
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            optimizer = ASOP(f)
            self.assertTrue(optimizer.nDimensions == n)


    def testDimensionsAsNumber(self):
        '''Supply a number as dimensions argument'''
        #also supply a string that can be converted to int
        for (i, f) in enumerate(self.lFunctions):
            n = i + 1
            optimizer = ASOP(f, n)
            self.assertTrue(optimizer.nDimensions == n)

            optimizer = ASOP(f, str(n))
            self.assertTrue(optimizer.nDimensions == n)

    def testDimensionsAsVariables(self):
        '''Suppliy list of variables as dimensions argument'''
        raise NotImplementedError()

    def testDimensionsInvalidNumber(self):
        #supply negative int or non-int string
        raise NotImplementedError()

    def testDimensionsInvalidObjectsInList(self):
        '''Supply invalid object in the dimenstions argument'''
        raise NotImplementedError()

    def testDimensionsHaveNames(self):
        '''Dimensions created by default should have names'''
        raise NotImplementedError()



class TestFunctionality(unittest.TestCase):
    '''Various functionality tests'''


    def createDummyObject(self):
        '''Create a dummy ASOP object'''
        raise NotImplementedError()


    def testScalingWorks(self):
        raise NotImplementedError()



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
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            n = np.logspace(0, 4).astype(int)
            s = obj.sample(100)
            self.assertTrue(len(s) == n)

    def testTrainManyTimes(self):
        TIMES = 100
        obj = self.createDummyObject()
        for t in range(TIMES): #@UnusedVariable
            n = np.logspace(0, 4).astype(int)
            nToReturn = max(1, int(n / 2))
            s = obj.train(n, nToReturn=nToReturn)
            self.assertTrue(len(s) == nToReturn)

            
    def testAutoScalingIsSupported(self):
        '''test scaling='auto' '''
        raise NotImplementedError()
    
    






if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()