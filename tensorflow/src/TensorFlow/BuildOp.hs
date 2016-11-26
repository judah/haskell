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

{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}

module TensorFlow.BuildOp
    ( OpResult
    , buildResult
    , eqLengthGuard
    , BuildResult
    , IsResult(..)
    , ExprResult
    , exprResult
    )
  where

import Control.Monad (replicateM)
import Control.Monad.Reader (ReaderT, runReaderT, ask)
import Control.Monad.State.Strict (State, runState, get, put)
import Data.Int (Int64)
import Lens.Family2 ((&), (<>~), (^.))

import TensorFlow.Build
import TensorFlow.Output
import TensorFlow.Tensor
import Proto.Tensorflow.Core.Framework.NodeDef (name)

data ResultState = ResultState !OutputIx [Int64] deriving Show

type Result = ReaderT Op (State ResultState)

-- | Class of types that can be used as op outputs.
class OpResult a where
    toResult :: Result a

instance (OpResult a1, OpResult a2) => OpResult (a1, a2) where
    toResult = (,) <$> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3) => OpResult (a1, a2, a3) where
    toResult = (,,) <$> toResult <*> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3, OpResult a4)
         => OpResult (a1, a2, a3, a4) where
    toResult = (,,,) <$> toResult <*> toResult <*> toResult <*> toResult

instance (OpResult a1, OpResult a2, OpResult a3, OpResult a4, OpResult a5)
         => OpResult (a1, a2, a3, a4, a5) where
    toResult = (,,,,) <$> toResult
                      <*> toResult
                      <*> toResult
                      <*> toResult
                      <*> toResult

instance ( OpResult a1
         , OpResult a2
         , OpResult a3
         , OpResult a4
         , OpResult a5
         , OpResult a6
         )
         => OpResult (a1, a2, a3, a4, a5, a6) where
    toResult = (,,,,,)
               <$> toResult
               <*> toResult
               <*> toResult
               <*> toResult
               <*> toResult
               <*> toResult

tensorResult :: TensorKind v -> Result (Tensor v a)
tensorResult v = Tensor v <$> recordResult

recordResult :: Result Output
recordResult = do
    o <- ask
    ResultState i ns <- get
    put $! ResultState (i+1) ns
    return $! output i o

instance OpResult (ResourceHandle a) where
    toResult = ResourceHandle <$> recordResult

instance OpResult (Tensor Value a) where
    toResult = tensorResult ValueKind

instance OpResult (Tensor Ref a) where
    toResult = tensorResult RefKind

instance OpResult ControlNode where
    toResult = ControlNode <$> ask

instance OpResult a => OpResult [a] where
    toResult = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in toResult. " ++
                          "Likely misuse of buildListOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) toResult

runResult :: OpResult a => [Int64] -> Op -> a
runResult ns o =
    case runState (runReaderT toResult o) (ResultState 0 ns) of
        (x, ResultState _ []) -> x
        (_, ns') -> error $ "Ununsed length in runResult attributes: " ++
                            show (ns, ns')

-- TODO: better for these to just take OpDef.  (Similarly for the IsResult class.)

-- | Make a new "stateful" op, which will not be deduped with otherwise
-- identical ops.
buildResult :: OpResult a => [Int64] -> Build OpDef -> BuildResult a
buildResult ns = liftResult $ \o ->
        runResult ns . Op . NodeName . (^. name) <$> addNewOp o

exprResult :: OpResult a => [Int64] -> Expr OpDef -> ExprResult a
exprResult ns = liftResult $ \o ->
        runResult ns . Op . NodeName . (^. name) <$> unsafeToExpr (addNewOp o)

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


type family ResultType f where
    ResultType (Build a) = a
    ResultType (Expr a) = a
    ResultType ((OpDef -> OpDef) -> f) = ResultType f

class Monad m => IsResult m f where
    liftResult :: (OpDef -> m (ResultType f)) -> m OpDef -> f

instance IsResult Build (Build a) where
    liftResult f m = m >>= f

instance IsResult m f => IsResult m ((OpDef -> OpDef) -> f) where
    liftResult f o g = liftResult f (g <$> o)

-- TODO: better naming
type BuildResult a = forall f . (IsResult Build f, a ~ ResultType f) => f
type ExprResult a = forall f . (IsResult Expr f, a ~ ResultType f) => f

instance IsResult Expr (Build a) where
    liftResult f o = expr $ o >>= f

instance IsResult Expr (Expr a) where
    liftResult f m = m >>= f
