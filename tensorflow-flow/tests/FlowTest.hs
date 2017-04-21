module Main where

import Control.Monad.IO.Class (liftIO)
import Google.Test (googleTest)
import TensorFlow.Flow
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))

simpleTest :: Test
simpleTest = testCase "simpleTest" $ runSession $ do
    result <- fetchFlow $ do
        v <- initializedVariable 3
        x <- readValue v
        assignAdd v 5
        y <- readValue v
        return (x * y)
    liftIO $ unScalar result @=? (3 * (3 + 5) :: Float)


main :: IO ()
main = googleTest [simpleTest]
