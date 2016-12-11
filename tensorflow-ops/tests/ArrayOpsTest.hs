-- Copyright 2016 TensorFlow authors.
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import Control.Monad.IO.Class (liftIO)
import Data.Int (Int64, Int32)
import Google.Test (googleTest)
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import qualified Data.Vector as V

import qualified TensorFlow.Build as TF
import qualified TensorFlow.Ops as TF
import qualified TensorFlow.Session as TF
import qualified TensorFlow.Tensor as TF
import qualified TensorFlow.Types as TF
import qualified TensorFlow.GenOps.Core as CoreOps
import Lens.Family2 ((&), (.~))
import Data.Default.Class (def)
import Data.Monoid ((<>))

import System.IO( stderr)
import Data.ByteString.Builder (hPutBuilder)

-- TODO: ExprType inference isn't great...

-- | Test split and concat are inverses.
testSplit :: Test
testSplit = testCase "testSplit" $ TF.runSessionWithOptions
        (def & TF.sessionTracer .~ (hPutBuilder stderr . (<> "\n"))) $ do
    original <- TF.build $ TF.render $ TF.constant [2, 3] [0..5 :: Float]
    liftIO $ putStrLn "===ORIGINAL==="
    TF.run original >>= liftIO . print . V.toList
    let dim = 1  -- dimension to split along (with size of 3 in original)
    TF.build (TF.render (dim :: TF.TensorExpr Int32)) >>= TF.run >>= liftIO . print . V.toList
    splitList <- TF.build $ TF.render $ CoreOps.split 3 dim (TF.expr original)
    liftIO $ putStrLn "===SplitList==="
    TF.run splitList >>= liftIO . print . map V.toList
    liftIO $ putStrLn "===Restored==="
    restored <- TF.build $ TF.render $ CoreOps.concat dim (map TF.expr splitList)
    liftIO $ 3 @=? length splitList
    (x, y, z) <- TF.run (original, restored, splitList !! 1)
    liftIO $ x @=? (y :: V.Vector Float)
    liftIO $ V.fromList [1, 4] @=? z

testShapeN :: Test
testShapeN = testCase "testShapeN" $ TF.runSession $ do
    let shapes = map TF.Shape [[1],[2,3]]
    let tensors = map TF.zeros shapes :: [TF.TensorExpr Float]
    result <- TF.buildAnd TF.run $ TF.render $ CoreOps.shapeN tensors
    liftIO $ [V.fromList [1], V.fromList [2,3]] @=? (result :: [V.Vector Int64])

main :: IO ()
main = googleTest [ testSplit
                  , testShapeN
                  ]
