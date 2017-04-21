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

deferTest :: Test
deferTest = testCase "deferTest" $ runSession $ do
    v <- runFlow $ initializedVariable 3
    a <- deferFlow $ readValue v >>= assignAdd v
    let expect x = do
            x' <- fetchFlow $ readValue v
            liftIO $ x @=? unScalar x'
    expect (3 :: Float)
    runDeferred a
    expect 6
    runDeferred a
    expect 12

main :: IO ()
main = googleTest [simpleTest, deferTest]
