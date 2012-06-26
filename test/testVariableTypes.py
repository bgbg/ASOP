'''
Created on 27 Nov 2011

@author: boris
'''
import unittest
import numpy as np

from asop import variableTypes

class TestGeneral(unittest.TestCase):
    NDIGITS = 8

    lAbstractClasses = [variableTypes.VariableBase,
                    variableTypes.QualitativeVariableBase,
                    variableTypes.QuantitativeVariableBase]
    
    lConcreteClasses = [variableTypes.ContinuousVariable,
                        variableTypes.IntegerVariable]
    
    

    def testAbstractClasses(self):
        '''Make sure abstract classes are abstract'''

        
        for cls_ in self.lAbstractClasses:
            self.assertRaises(TypeError, cls_)
            
    
    def testConcreteClassesCanBeInstatinated(self):
        '''Concrete classes instatination w/out arguments'''
        lErrors = []
        for cls_ in self.lConcreteClasses:
            try:
                cls_()
            except:
                lErrors.append(cls_.__name__)
        if lErrors:
            msg = 'Failed to instatinate %s'%(','.join(e for e in lErrors))
            self.fail(msg)
            
    def testSingleValueAsSamplingScores(self):
        '''Pass a single value as samplingScores argument'''
        for cls_ in self.lConcreteClasses:
            x = np.arange(100) 
            samplingScores = 1
            cls_(x, samplingScores)
    
            
    def testMultipleScoreUpdates(self):
        #for each concrete class perform multiple score updates,
        #make sure no exception is raised
        TIMES = 500
        for cls_ in self.lConcreteClasses:
            obj = cls_()
            x = obj.x
            theRange = np.max(x) - np.min(x)
            n = len(x)
            lix = np.random.randint(0, n, TIMES)
            locations = np.array(x)[lix]
            amount = np.random.randn(TIMES) * 10.0
            width = np.random.sample(TIMES) * theRange
            for (a, loc, w) in \
                zip(amount, locations, width):
                obj.alterSamplingDistribution(a, loc, w)
        
                
            
    
    def testScoreUpdateIsReversible(self):
        
        TIMES = 100
        
        #for each concrete class: record the pdf,
        for cls_ in self.lConcreteClasses:
            obj = cls_()
            x = obj.x
            pdfBefore = np.array(obj.pdfValues)
            theRange = np.max(x) - np.min(x)
            n = len(x)
            lix = np.random.randint(0, n, TIMES)
            locations = np.array(x)[lix]
            amount = np.random.randn(TIMES) * 10.0
            width = np.random.sample(TIMES) * theRange
            for (a, loc, w) in \
                zip(amount, locations, width):
                obj.alterSamplingDistribution(a, loc, w, immediateApply=False)
            obj.applySamplingScore()
            pdfBetween = np.array(obj.pdfValues)
            
            #now revert the changes
            for (a, loc, w) in  \
                zip(amount, locations, width):
                a = -a 
                obj.alterSamplingDistribution(a, loc, w, immediateApply=False)
            obj.applySamplingScore()
            
            pdfAfter = np.array(obj.pdfValues)
            
            d1 = np.sum(np.square(pdfBefore - pdfBetween))
            d2 = np.sum(np.square(pdfBefore - pdfAfter))
            self.assertNotAlmostEquals(d1, 0, max(1, self.NDIGITS - 7))
            self.assertAlmostEquals(d2, 0, self.NDIGITS)
    
        
    def testDelayedScoreUpdate(self):
        '''Delayed score update has to be identical to immediate one'''
        
        TIMES = 100
        #for each concrete class: record the pdf,
        for cls_ in self.lConcreteClasses:
            obj = cls_()
            x = obj.x
            theRange = np.max(x) - np.min(x)
            n = len(x)
            lix = np.random.randint(0, n, TIMES)
            locations = np.array(x)[lix]
            amount = np.random.randn(TIMES) * 10.0
            width = np.random.sample(TIMES) * theRange
            
            #delayed update
            for (a, loc, w) in \
                zip(amount, locations, width):
                obj.alterSamplingDistribution(a, loc, w, 
                                               immediateApply=False)
            obj.applySamplingScore()
            pdfAfterDelayed = np.array(obj.pdfValues)
            
            #IMMEDIATE update
            obj = cls_() #reset the object
            for (a, loc, w) in \
                zip(amount, locations, width):
                obj.alterSamplingDistribution(a, loc, w, 
                                               immediateApply=True)
            #note: not calling obj.applySamplingScore()
            pdfAfterImmediate = np.array(obj.pdfValues)
            
            d = np.sum(np.square(pdfAfterDelayed - pdfAfterImmediate))
            self.assertAlmostEquals(d, 0, self.NDIGITS)
    
        
        
        
        
            
                  

    def testFailOnUnequalParameters(self):
        values = [1,2,3]
        scores = [1,2,3,4]
        self.assertRaises(AssertionError, variableTypes.ContinuousVariable,
                            values, scores)

    def testProbabilityFromScore(self):
        NDIGITS = self.NDIGITS
        r = np.random.random(100)
        scores = np.array([0], dtype=float)
        scores = np.hstack((scores, r))
        scores = np.hstack((scores, -r))
        pValues = variableTypes.VariableBase.probabilityFromScore(scores)
        self.assertAlmostEqual(pValues[0], 0.5, NDIGITS)
        self.assertAlmostEqual(np.mean(scores), 0, places=NDIGITS)
        for i in range(100): #@UnusedVariable
            s = np.random.randn() * np.random.randint(0, 100)
            s = scores + s
            p = variableTypes.VariableBase.probabilityFromScore(s)
            self.assertAlmostEqual(p[0], .5, NDIGITS)
            self.assertAlmostEqual(np.sum(p - pValues), 0.0, NDIGITS)

    def testInverseLogit(self):
        nPoints = 100
        pValues = np.random.rand(nPoints)
        pValues = pValues[pValues.astype(bool)] #remove zeros
        logitValues = np.log(pValues / (1.0 - pValues))
        pValuesFromLogit = variableTypes.inverseLogit(logitValues)
        for pOrig, pFromLogit in zip(pValues, pValuesFromLogit):
            self.assertAlmostEqual(pOrig, pFromLogit, self.NDIGITS)

    def testRandFunctionWorks(self):
        TIMES = 100

        for cls in self.lConcreteClasses:
            v = variableTypes.ContinuousVariable(range(10))
            r = v.rand()
            self.assertTrue(np.isscalar(r),
                '%s: rand with a no parameter should return a scalar'%cls)

            for i in range(TIMES): #@UnusedVariable
                n = np.random.randint(1, 1000)
                r = v.rand(n)
                self.assertTrue(len(r)==n,
                    '%s rand(%d) should return %d values'%(cls, n, n))




class TestContinuousVariable(unittest.TestCase):
    def testInitializationDefault(self):
        v = variableTypes.ContinuousVariable(range(10))
        del v

    def testUnsortedSamplingValues(self):
        cases = ([1., 2., 3., 0],
                 [9., 8., 7., 6],
                 [1., 1, 1., 1])
        for values in cases:
            self.assertRaises(
                          ValueError,
                          variableTypes.ContinuousVariable,
                          values)



class TestIntegerVariable(unittest.TestCase):
    def testRaiseErrorOnImproperInitialization(self):
        values = [1, 1.01, 2, 3]
        self.assertRaises(ValueError, variableTypes.IntegerVariable,
                          values)

    def testEveryValueIsSampled(self):
        TESTS = 100
        SIZE = 100
        for t in range(TESTS): #@UnusedVariable
            x = set(np.random.randint(0, 1000, size=SIZE))
            values = list(x)
            values.sort()
            var = variableTypes.IntegerVariable(values)
            ATTEMPTS = 10 * len(x)
            TIMES = 10 * len(x)
            sampled = set()
            for i in range(ATTEMPTS): #@UnusedVariable
                s = var.rand(TIMES)
                sampled.update(s)
                if sampled == x:
                    break
            if sampled != x:
                lMsg = []
                if len(x.difference(sampled)) > 0:
                    msg  = 'Failed to cover the entire range of '\
                        'IntegerVariable'
                    lMsg.append(msg)
                if len(sampled.difference(x)) > 0:
                    strSortedSet = lambda s: ','.join([str(v) for v in sorted(s)])
                    lines = ['IntegerVariable sampled unexpected '\
                        'values']
                    lines.append('Expected: %s'%strSortedSet(x))
                    lines.append('Sampled:  %s'%strSortedSet(sampled))
                    lines.append('Sampled - Expected: %s'%strSortedSet(sampled.difference(x)))
                    lines.append('Expected - Sampled: %s'%strSortedSet(x.difference(sampled)))
                    msg = '\n'.join(lines)
                    lMsg.append(msg)
                self.fail('.\n'.join(lMsg))





if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testAbstractClasses']
    unittest.main()

    #test