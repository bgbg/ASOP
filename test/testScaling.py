import numpy as np
import unittest
from abc import ABCMeta, abstractmethod

import asop.scaling as scaling

NDIGITS = 2


class CaseBase(object):
    '''Tests each class should perform'''
    
    __metaclass__ = ABCMeta
    lClasses = [scaling.LinearScaling, 
                scaling.TanhScaling,
                scaling.LogisticScaling]


    def testStability(self):
        '''stability tests'''
        TIMES = 100
        SIZE = 1000
        for i in range(TIMES): #@UnusedVariable
            x = np.exp(np.random.randn(SIZE) * 4) #wide range of values
            self.obj(x)
        
    def testCallWithSingleValue(self):
        ret = self.obj(1.2)
        self.assertTrue(np.isscalar(ret))
        
    def testCallWithIterable(self):
        inp = np.linspace(0, 1.0)
        ret = self.obj(inp)
        try:
            iter(ret)
        except:
            self.fail()
            
        
    def testOutputIsReal(self):
        '''Make sure the scaler return real values only'''
        TIMES = 100
        SIZE = 1000
        for i in range(TIMES): #@UnusedVariable
            x = np.exp(np.random.randn(SIZE) * 4) #wide range of values
            ret = self.obj(x)
            self.assertTrue(np.all(np.isreal(ret)))
            
    
    def testIsCallable(self):
        self.assertTrue(callable(self.obj))
        
    @abstractmethod
    def testValues(self):
        pass
    
    
class TestTanhScaling(CaseBase, unittest.TestCase):
    
    def setUp(self):
        obj = scaling.TanhScaling(x50=0.5, steepness=1)
        self.obj = obj
    
    
    def testValues(self):
        ref = [
               [-10.00, 0.01, -11.00, -0.010], 
[-10.00, 0.01, -5.50, 0.045], 
[-10.00, 0.01, 0.00, 0.100], 
[-10.00, 0.01, 5.50, 0.154], 
[-10.00, 0.01, 11.00, 0.207], 
[-10.00, 0.51, -11.00, -0.466], 
[-10.00, 0.51, -5.50, 0.979], 
[-10.00, 0.51, 0.00, 1.000], 
[-10.00, 0.51, 5.50, 1.000], 
[-10.00, 0.51, 11.00, 1.000], 
[-10.00, 1.00, -11.00, -0.762], 
[-10.00, 1.00, -5.50, 1.000], 
[-10.00, 1.00, 0.00, 1.000], 
[-10.00, 1.00, 5.50, 1.000], 
[-10.00, 1.00, 11.00, 1.000], 
[0.00, 0.01, -11.00, -0.110], 
[0.00, 0.01, -5.50, -0.055], 
[0.00, 0.01, 0.00, 0.000], 
[0.00, 0.01, 5.50, 0.055], 
[0.00, 0.01, 11.00, 0.110], 
[0.00, 0.51, -11.00, -1.000], 
[0.00, 0.51, -5.50, -0.992], 
[0.00, 0.51, 0.00, 0.000], 
[0.00, 0.51, 5.50, 0.992], 
[0.00, 0.51, 11.00, 1.000], 
[0.00, 1.00, -11.00, -1.000], 
[0.00, 1.00, -5.50, -1.000], 
[0.00, 1.00, 0.00, 0.000], 
[0.00, 1.00, 5.50, 1.000], 
[0.00, 1.00, 11.00, 1.000], 
[10.00, 0.01, -11.00, -0.207], 
[10.00, 0.01, -5.50, -0.154], 
[10.00, 0.01, 0.00, -0.100], 
[10.00, 0.01, 5.50, -0.045], 
[10.00, 0.01, 11.00, 0.010], 
[10.00, 0.51, -11.00, -1.000], 
[10.00, 0.51, -5.50, -1.000], 
[10.00, 0.51, 0.00, -1.000], 
[10.00, 0.51, 5.50, -0.979], 
[10.00, 0.51, 11.00, 0.466], 
[10.00, 1.00, -11.00, -1.000], 
[10.00, 1.00, -5.50, -1.000], 
[10.00, 1.00, 0.00, -1.000], 
[10.00, 1.00, 5.50, -1.000], 
[10.00, 1.00, 11.00, 0.762], 
               ]
        squaredErrors = 0
        for x50, steepness, x, y in ref:
            obj = scaling.TanhScaling(x50, steepness)
            scaled = obj(x)
            diff2 = (y - scaled) ** 2
            squaredErrors += diff2
        self.assertAlmostEqual(squaredErrors / len(ref), 0, NDIGITS)
    
    
    def testFailsOnZeroSteepness(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            x50 = np.random.randn() * 10
            try:
                scaling.TanhScaling(x50, 0)
            except AssertionError:
                pass
            else:
                self.fail()
        
    
    def testX50(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            x50 = np.random.randn() * 20
            steepness = np.random.randn() * 5
            if steepness == 0:
                continue
            obj = scaling.TanhScaling(x50, steepness)
            self.assertAlmostEqual(obj(x50), 0, NDIGITS)

    def testScalingFromExtrema(self):
        TIMES = 1000
        SIZE = 100
        for i in range(TIMES): #@UnusedVariable
            if np.random.random() > 0.5:
                values = np.random.random(SIZE) * 10.0
            else:
                values = np.exp(np.random.randn(SIZE) * 10.0)
            yHigh = np.random.random() + 1e-4
            scaler = scaling.tanhScalingFromValueExtrema(values, yHigh)
            self.assertAlmostEqual(yHigh, scaler(np.max(values)), NDIGITS)
            self.assertAlmostEqual(-yHigh, scaler(np.min(values)), NDIGITS)
            
    def testScalingFromExtremaFailsOnSameInputValues(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            values = [np.random.random()] * 10
            yHigh = np.random.random() + 1e-4
            self.assertRaises(AssertionError, scaling.tanhScalingFromValueExtrema,
                              values, yHigh)
               
    def testScalingFromExtremaFailsOnBadYHigh(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            values = [np.random.random()] * 10
            for yHigh in [-0.5, 0.0, 1.0, 1.5]:
                self.assertRaises(AssertionError, scaling.tanhScalingFromValueExtrema,
                                  values, yHigh)

class TestLinearScaling(CaseBase, unittest.TestCase):
    
    def setUp(self):
        self.obj = scaling.LinearScaling(1, 0)
    
    def testValues(self):
        ref = [
               [-10.15, -1.00, -11.00, 110.650], 
                [-10.15, -1.00, -5.50, 54.825], 
                [-10.15, -1.00, 0.00, -1.000], 
                [-10.15, -1.00, 5.50, -56.825], 
                [-10.15, -1.00, 11.00, -112.650], 
                [-10.15, 0.00, -11.00, 111.650], 
                [-10.15, 0.00, -5.50, 55.825], 
                [-10.15, 0.00, 0.00, 0.000], 
                [-10.15, 0.00, 5.50, -55.825], 
                [-10.15, 0.00, 11.00, -111.650], 
                [-10.15, 1.00, -11.00, 112.650], 
                [-10.15, 1.00, -5.50, 56.825], 
                [-10.15, 1.00, 0.00, 1.000], 
                [-10.15, 1.00, 5.50, -54.825], 
                [-10.15, 1.00, 11.00, -110.650], 
                [-0.08, -1.00, -11.00, -0.175], 
                [-0.08, -1.00, -5.50, -0.587], 
                [-0.08, -1.00, 0.00, -1.000], 
                [-0.08, -1.00, 5.50, -1.413], 
                [-0.08, -1.00, 11.00, -1.825], 
                [-0.08, 0.00, -11.00, 0.825], 
                [-0.08, 0.00, -5.50, 0.413], 
                [-0.08, 0.00, 0.00, 0.000], 
                [-0.08, 0.00, 5.50, -0.413], 
                [-0.08, 0.00, 11.00, -0.825], 
                [-0.08, 1.00, -11.00, 1.825], 
                [-0.08, 1.00, -5.50, 1.413], 
                [-0.08, 1.00, 0.00, 1.000], 
                [-0.08, 1.00, 5.50, 0.587], 
                [-0.08, 1.00, 11.00, 0.175], 
                [10.00, -1.00, -11.00, -111.000], 
                [10.00, -1.00, -5.50, -56.000], 
                [10.00, -1.00, 0.00, -1.000], 
                [10.00, -1.00, 5.50, 54.000], 
                [10.00, -1.00, 11.00, 109.000], 
                [10.00, 0.00, -11.00, -110.000], 
                [10.00, 0.00, -5.50, -55.000], 
                [10.00, 0.00, 0.00, 0.000], 
                [10.00, 0.00, 5.50, 55.000], 
                [10.00, 0.00, 11.00, 110.000], 
                [10.00, 1.00, -11.00, -109.000], 
                [10.00, 1.00, -5.50, -54.000], 
                [10.00, 1.00, 0.00, 1.000], 
                [10.00, 1.00, 5.50, 56.000], 
                [10.00, 1.00, 11.00, 111.000], 
               ]
        
        squaredErrors = 0
        for a, b, x, y in ref:
            obj = scaling.LinearScaling(a, b)
            scaled = obj(x)
            diff2 = (y - scaled) ** 2
            squaredErrors += diff2
        self.assertAlmostEqual(squaredErrors / len(ref), 0, NDIGITS)
    
    def testIntercept(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            b = np.random.randn() * 10
            a = np.random.randn() * 5
            if a == 0:
                continue
            try:
                scaling.LinearScaling(0, b)
            except AssertionError:
                pass
            else:
                self.fail()
    
    def testFailsOnZeroSlope(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            b = np.random.randn() * 10
            try:
                scaling.LinearScaling(0, b)
            except AssertionError:
                pass
            else:
                self.fail()
            

class TestLogisticScaling(CaseBase, unittest.TestCase):
    
    def setUp(self):
        self.obj = scaling.LogisticScaling(0, 1.0)
    
    def testX50(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            x50 = np.random.randn() * 20
            steepness = np.random.randn() * 5
            if steepness == 0:
                continue
            obj = scaling.LogisticScaling(x50, steepness)
            self.assertAlmostEqual(obj(x50), 0.5, NDIGITS)
    
    def testValues(self):
        ref = [
               [-10.00, 0.01, -11.00, 0.498], 
                [-10.00, 0.01, -5.50, 0.511], 
                [-10.00, 0.01, 0.00, 0.525], 
                [-10.00, 0.01, 5.50, 0.539], 
                [-10.00, 0.01, 11.00, 0.552], 
                [-10.00, 0.51, -11.00, 0.376], 
                [-10.00, 0.51, -5.50, 0.907], 
                [-10.00, 0.51, 0.00, 0.994], 
                [-10.00, 0.51, 5.50, 1.000], 
                [-10.00, 0.51, 11.00, 1.000], 
                [-10.00, 1.00, -11.00, 0.269], 
                [-10.00, 1.00, -5.50, 0.989], 
                [-10.00, 1.00, 0.00, 1.000], 
                [-10.00, 1.00, 5.50, 1.000], 
                [-10.00, 1.00, 11.00, 1.000], 
                [0.00, 0.01, -11.00, 0.473], 
                [0.00, 0.01, -5.50, 0.486], 
                [0.00, 0.01, 0.00, 0.500], 
                [0.00, 0.01, 5.50, 0.514], 
                [0.00, 0.01, 11.00, 0.527], 
                [0.00, 0.51, -11.00, 0.004], 
                [0.00, 0.51, -5.50, 0.059], 
                [0.00, 0.51, 0.00, 0.500], 
                [0.00, 0.51, 5.50, 0.941], 
                [0.00, 0.51, 11.00, 0.996], 
                [0.00, 1.00, -11.00, 0.000], 
                [0.00, 1.00, -5.50, 0.004], 
                [0.00, 1.00, 0.00, 0.500], 
                [0.00, 1.00, 5.50, 0.996], 
                [0.00, 1.00, 11.00, 1.000], 
                [10.00, 0.01, -11.00, 0.448], 
                [10.00, 0.01, -5.50, 0.461], 
                [10.00, 0.01, 0.00, 0.475], 
                [10.00, 0.01, 5.50, 0.489], 
                [10.00, 0.01, 11.00, 0.502], 
                [10.00, 0.51, -11.00, 0.000], 
                [10.00, 0.51, -5.50, 0.000], 
                [10.00, 0.51, 0.00, 0.006], 
                [10.00, 0.51, 5.50, 0.093], 
                [10.00, 0.51, 11.00, 0.624], 
                [10.00, 1.00, -11.00, 0.000], 
                [10.00, 1.00, -5.50, 0.000], 
                [10.00, 1.00, 0.00, 0.000], 
                [10.00, 1.00, 5.50, 0.011], 
                [10.00, 1.00, 11.00, 0.731], 
               ]
        squaredErrors = 0
        for x50, steepness, x, y in ref:
            obj = scaling.LogisticScaling(x50, steepness)
            scaled = obj(x)
            diff2 = (y - scaled) ** 2
            squaredErrors += diff2
        self.assertAlmostEqual(squaredErrors / len(ref), 0, NDIGITS)
        
    def testFailsOnZeroSteepness(self):
        TIMES = 100
        for i in range(TIMES): #@UnusedVariable
            x50 = np.random.randn() * 10
            try:
                scaling.LogisticScaling(x50, 0)
            except AssertionError:
                pass
            else:
                self.fail()
                
                

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()