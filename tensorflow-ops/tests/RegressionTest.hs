{-# LANGUAGE OverloadedStrings #-}
-- | Simple linear regression example for the README.

import Control.Monad (replicateM, replicateM_, zipWithM)
import Control.Monad.IO.Class
import Lens.Family2 ((.~))
import System.Random (randomIO)
import Test.HUnit (assertBool)

import qualified TensorFlow.Core as TF
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF

main :: IO ()
main = do
    -- Generate data where `y = x*3 + 8`.
    xData <- replicateM 100 randomIO
    let yData = [x*3 + 8 | x <- xData]
    -- Fit linear regression model.
    (w, b) <- fit xData yData
    assertBool "w == 3" (abs (3 - w) < 0.001)
    assertBool "b == 8" (abs (8 - b) < 0.001)

fit :: [Float] -> [Float] -> IO (Float, Float)
fit xData yData = TF.runSession $ do
    -- Create tensorflow constants for x and y.
    let x = TF.vector xData
        y = TF.vector yData
    -- Create scalar variables for slope and intercept.
    w <- TF.initializedVariable' (TF.opName .~ "w") 0
    b <- TF.initializedVariable' (TF.opName .~ "b") 8

    -- Define the loss function.
    let yHat = x * TF.readVar w + TF.readVar b
        loss = TF.square (yHat - y)
    -- Optimize with gradient descent.
    trainStep <- gradientDescent 0.001 loss [w, b]
    replicateM_ 1000 (TF.run trainStep)
    -- Return the learned parameters.
    (TF.Scalar w', TF.Scalar b') <- TF.run (TF.readVar w, TF.readVar b)
    return (w', b')

gradientDescent :: Float
                -> TF.Tensor TF.Build Float
                -> [TF.Variable Float]
                -> TF.Session TF.ControlNode
gradientDescent alpha loss params = do
    let applyGrad param grad =
            TF.assign param (TF.readVar param - TF.scalar alpha `TF.mul` grad)
    -- TODO: this is suspect...
    reads <- mapM (TF.render . TF.readVar) params
    TF.group =<< zipWithM applyGrad params
        =<< TF.gradients loss reads
