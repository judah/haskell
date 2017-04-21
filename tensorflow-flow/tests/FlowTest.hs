module Main where

import Google.Test (googleTest)
import TensorFlow.Flow
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))

simpleTest :: Test
simpleTest = testCase "simpleTest" $ runSession $ return ()
    
main :: IO ()
main = googleTest [simpleTest]
