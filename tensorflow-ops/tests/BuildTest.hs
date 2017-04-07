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

{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Main where

import Control.Monad.IO.Class (liftIO)
import Lens.Family2 ((^.), (.~))
import Data.Int (Int64)
import Data.List (sort)
import Proto.Tensorflow.Core.Framework.Graph
    ( node )
import Proto.Tensorflow.Core.Framework.NodeDef
    ( NodeDef
    , device
    , name
    , op )
import TensorFlow.Build
    ( Build
    , BuildT
    , asGraphDef
    , evalBuildT
    , flushNodeBuffer
    , withDevice
    , withNameScope
    , opName
    )
import TensorFlow.Types (unScalar)
import TensorFlow.Ops
    ( add
    , assign
    , constant
    , initializedVariable
    , readVar
    , scalar
    , variable
    , variable'
    , Variable(..)
    )
import TensorFlow.Output (Device(..))
import TensorFlow.Tensor
    ( colocateWith
    , render
    , Tensor
    , Value
    , Ref
    )
import TensorFlow.Session
    ( run
    , runSession
    , run_
    )
import Test.Framework (Test)
import Test.Framework.Providers.HUnit (testCase)
import Test.HUnit ((@=?))
import Google.Test (googleTest)
import qualified Data.Vector as V

-- | Test 'opName' behavior.
testOpName :: Test
testOpName = testCase "testOpName" $ do
    let graph = variable' (opName .~ "foo") [] :: Build (Variable Float)
        nodeDef :: NodeDef
        nodeDef = head $ asGraphDef graph ^. node
    "VarHandleOp" @=? (nodeDef ^. op)
    "foo" @=? (nodeDef ^. name)

-- | Test that "run" will render and extend any pure ops that haven't already
-- been rendered.
testPureRender :: Test
testPureRender = testCase "testPureRender" $ runSession $ do
    result <- run $ 2 `add` 2
    liftIO $ 4 @=? (unScalar result :: Float)

-- | Test that "run" assigns any previously accumulated initializers.
testInitializedVariable :: Test
testInitializedVariable =
    testCase "testInitializedVariable" $ runSession $ do
        (formula, reset) <- do
            v <- initializedVariable 42
            r <- assign v 24
            return (1 + readVar v, r)
        result <- run formula
        liftIO $ 43 @=? (unScalar result :: Float)
        run_ reset  -- Updates v to a different value
        rerunResult <- run formula
        liftIO $ 25 @=? (unScalar rerunResult :: Float)

testInitializedVariableShape :: Test
testInitializedVariableShape =
    testCase "testInitializedVariableShape" $ runSession $ do
        vector <- initializedVariable (constant [1] [42 :: Float])
        result <- run (readVar vector)
        liftIO $ [42] @=? (result :: V.Vector Float)

-- | Test nameScoped behavior.
testNameScoped :: Test
testNameScoped = testCase "testNameScoped" $ do
    let graph = withNameScope "foo" $ variable [] :: Build (Variable Float)
        nodeDef :: NodeDef
        [nodeDef] = asGraphDef graph ^. node
    "foo/VarHandleOp_0" @=? (nodeDef ^. name)  -- TODO: Check prefix.
    "VarHandleOp" @=? (nodeDef ^. op)

-- | Test combined opName and nameScoped behavior.
testNamedAndScoped :: Test
testNamedAndScoped = testCase "testNamedAndScoped" $ do
    let graph :: Build (Variable Float)
        graph = withNameScope "foo1" (variable' (opName .~ "bar1") [])
        nodeDef :: NodeDef
        nodeDef = head $ asGraphDef graph ^. node
    "VarHandleOp" @=? (nodeDef ^. op)
    "foo1/bar1" @=? (nodeDef ^. name)

-- | Flush the node buffer and sort the nodes by name (for more stable tests).
flushed :: Ord a => (NodeDef -> a) -> BuildT IO [a]
flushed field = sort . map field <$> flushNodeBuffer

-- | Test the interaction of rendering, CSE and scoping.
testRenderDedup :: Test
testRenderDedup = testCase "testRenderDedup" $ evalBuildT $ do
   renderNodes
   names <- flushed (^. name)
   liftIO $ ["Const_1", "VarHandleOp_0", "VarHandleOp_2"] @=? names
   -- Render the nodes in a different scope, which should cause them
   -- to be distinct from the previous ones.
   withNameScope "foo" renderNodes
   scopedNames <- flushed (^. name)
   liftIO $ ["foo/Const_4", "foo/VarHandleOp_3", "foo/VarHandleOp_5"]
                @=? scopedNames
  where
    renderNodes = do
        -- A stateful op and a pure op.
        _ :: Variable Float <- variable []
        _ :: Tensor Value Float <- render 3
        -- Another stateful op, and a pure op which should be
        -- deduped with the previous one.
        _ :: Variable Float <- variable []
        _ :: Tensor Value Float <- render 3
        return ()

-- | Test the interaction of rendering, CSE and scoping.
testDeviceColocation :: Test
testDeviceColocation = testCase "testDeviceColocation" $ evalBuildT $ do
   renderNodes
   devices <- flushed (\x -> (x ^. name, x ^. device))
   liftIO $ [ ("Add_3","dev0")
            , ("Const_1","dev0")
            , ("ReadVariableOp_2","dev0")
            , ("VarHandleOp_0","dev0")] @=? devices
  where
    renderNodes = do
        -- A stateful op and a pure op.
        var :: Variable Float <- withDevice (Just $ Device "dev0") $ variable []
        -- Uses render to cause the expression be added to the graph.
        _ <- colocateWith (varHandle var) $ render $ 3 + readVar var
        return ()

-- | See https://github.com/tensorflow/haskell/issues/92.
-- Even though we're not explicitly evaluating `f0` until the end,
-- it should hold the earlier value of the variable.
testRereadRef :: Test
testRereadRef = testCase "testReRunAssign" $ runSession $ do
    w <- initializedVariable 0
    f0 <- run (readVar w)
    run_ =<< assign w (scalar (0.1::Float))
    f1 <- run (readVar w)
    liftIO $ (0.0, 0.1) @=? (unScalar f0, unScalar f1)

testMultipleInitialized :: Test
testMultipleInitialized = testCase "testMultipleInitialized"
                            $ runSession $ do
    w <- initializedVariable (2 :: Tensor Build Int64)
    v <- initializedVariable (3 :: Tensor Build Int64)
    (w', v') <- run (readVar w, readVar v)
    liftIO $ (2, 3) @=? (unScalar w', unScalar v')


main :: IO ()
main = googleTest [ testInitializedVariable
                  , testInitializedVariableShape
                  , testDeviceColocation
                  , testOpName
                  , testNameScoped
                  , testNamedAndScoped
                  , testPureRender
                  , testRenderDedup
                  , testRereadRef
                  , testMultipleInitialized
                  ]
