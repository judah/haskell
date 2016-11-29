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
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}

module TensorFlow.BuildOp
    ( BuildResult
    , buildOp
    , ExprResult(..)
    , exprOp
    , eqLengthGuard
    )
  where

import Control.Monad (replicateM)
import Control.Monad.Reader (ReaderT(..), runReaderT, ask)
import Control.Monad.State.Strict (State, evalState, runState, get, put)
import Control.Monad.Trans (lift)
import Data.Int (Int64)
import Lens.Family2 ((&), (<>~), (^.))

import TensorFlow.Build
import TensorFlow.Output
import TensorFlow.Tensor
import Proto.Tensorflow.Core.Framework.NodeDef (name)

data ResultState = ResultState !OutputIx [Int64] deriving Show

type Result = ReaderT Op (State ResultState)

-- | Class of types that can be used as op outputs.
class BuildResult a where
    buildResult :: Result a

instance (BuildResult a1, BuildResult a2) => BuildResult (a1, a2) where
    buildResult = (,) <$> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3) => BuildResult (a1, a2, a3) where
    buildResult = (,,) <$> buildResult <*> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3, BuildResult a4)
         => BuildResult (a1, a2, a3, a4) where
    buildResult = (,,,) <$> buildResult <*> buildResult <*> buildResult <*> buildResult

instance (BuildResult a1, BuildResult a2, BuildResult a3, BuildResult a4, BuildResult a5)
         => BuildResult (a1, a2, a3, a4, a5) where
    buildResult = (,,,,) <$> buildResult
                      <*> buildResult
                      <*> buildResult
                      <*> buildResult
                      <*> buildResult

instance ( BuildResult a1
         , BuildResult a2
         , BuildResult a3
         , BuildResult a4
         , BuildResult a5
         , BuildResult a6
         )
         => BuildResult (a1, a2, a3, a4, a5, a6) where
    buildResult = (,,,,,)
               <$> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult
               <*> buildResult

tensorResult :: TensorKind v -> Result (Tensor v a)
tensorResult v = Tensor v <$> recordResult

recordResult :: Result Output
recordResult = do
    o <- ask
    ResultState i ns <- get
    put $! ResultState (i+1) ns
    return $! output i o

instance BuildResult (ResourceHandle a) where
    buildResult = ResourceHandle <$> recordResult

instance BuildResult (Tensor Value a) where
    buildResult = tensorResult ValueKind

instance BuildResult (Tensor Ref a) where
    buildResult = tensorResult RefKind

instance BuildResult ControlNode where
    buildResult = ControlNode <$> ask

instance BuildResult a => BuildResult [a] where
    buildResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in buildResult. " ++
                          "Likely misuse of buildListOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) buildResult

-- TODO: better for these to just take OpDef.  (Similarly for the IsResult class.)

buildOp :: BuildResult a => [Int64] -> OpDef -> Build a
buildOp sizes o = do
    n <- Op . NodeName . (^. name) <$> addNewOp o
    return $ flip evalState (ResultState 0 sizes) (runReaderT buildResult n)

-- | Returns true if all the integers in each tuple are identical.
-- Throws an error with a descriptive message if not.
eqLengthGuard :: [(String, [(String, Int)])] -> Bool
eqLengthGuard = all eachOk
  where
    eachOk (_, []) = True
    -- The next line has (== 1) . length . nub in disguise
    eachOk (numberAttrName, pairs@((_, x) : zs)) = all (\z -> snd z == x) zs ||
        error ("number_attr " ++ numberAttrName ++
               " contains tensors with different length " ++ show pairs)

----------------------

class ExprResult a where
    exprResult :: ReaderT (Build OpDef) (State ResultState) a

instance (ExprResult a1, ExprResult a2) => ExprResult (a1, a2) where
    exprResult = (,) <$> exprResult <*> exprResult

instance (ExprResult a1, ExprResult a2, ExprResult a3) => ExprResult (a1, a2, a3) where
    exprResult = (,,) <$> exprResult <*> exprResult <*> exprResult

instance (ExprResult a1, ExprResult a2, ExprResult a3, ExprResult a4) => ExprResult (a1, a2, a3, a4) where
    exprResult = (,,,) <$> exprResult <*> exprResult <*> exprResult <*> exprResult

instance (ExprResult a1, ExprResult a2, ExprResult a3, ExprResult a4, ExprResult a5)
    => ExprResult (a1, a2, a3, a4, a5) where
    exprResult = (,,,,) <$> exprResult <*> exprResult <*> exprResult <*> exprResult <*> exprResult

instance (ExprResult a1, ExprResult a2, ExprResult a3, ExprResult a4, ExprResult a5, ExprResult a6)
    => ExprResult (a1, a2, a3, a4, a5, a6) where
    exprResult = (,,,,,) <$> exprResult <*> exprResult <*> exprResult <*> exprResult <*> exprResult
                            <*> exprResult

instance ExprResult a => ExprResult [a] where
    exprResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in buildResult. " ++
                          "Likely misuse of buildListOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) exprResult

instance ExprResult (TensorExpr a) where
    exprResult = do
        ResultState i ns <- get
        put $! ResultState (i+1) ns
        makeOp <- ask
        return $ TensorExpr $ do
            o <- makeOp
            output i . Op <$> getOrAddOp o

-- TODO: the list sizes might also depend on the attrs to come...
exprOp :: (ExprResult a)
    => [Int64] -> Build OpDef -> a
exprOp sizes o = flip evalState (ResultState 0 sizes)
                                    (runReaderT exprResult o)
