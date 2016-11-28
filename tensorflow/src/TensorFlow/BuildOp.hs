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
    ( OpResult
    , buildResult
    , eqLengthGuard
    , BuildResult
    , IsResult(..)
    , MonadOp(..)
    , (&>>)
    , MakeExprOp(..)
    , exprOp
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
buildResult :: OpResult a => [Int64] -> OpDef -> BuildResult a
buildResult ns o = do
    modifier <- askOpModifier
    liftResult $
        runResult ns . Op . NodeName . (^. name) <$> addNewOp (modifier o)

{-
exprResult :: OpResult a => [Int64] -> OpDef -> ExprResult a
exprResult ns o = do
    modifier <- askOpModifier
    liftResult $
        runResult ns . Op <$> unsafeToExpr (getOrAddOp $ modifier o)
-}

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

class Monad m => MonadOp m where
    askOpModifier :: m (OpDef -> OpDef)

instance MonadOp m => MonadOp (ReaderT (OpDef -> OpDef) m) where
    askOpModifier = (.) <$> ask <*> lift askOpModifier

instance MonadOp Build where
    askOpModifier = pure id

-- instance MonadOp Expr where
--    askOpModifier = pure id

class (Monad m, MonadOp f) => IsResult m f where
    liftResult :: m a -> f a

instance IsResult Build Build where
    liftResult = id

instance IsResult m f => IsResult m (ReaderT (OpDef -> OpDef) f) where
    liftResult = lift . liftResult

-- TODO: better naming
type BuildResult a = forall f . (IsResult Build f) => f a

-- TODO: better name for this op
(&>>) :: Monad m => ReaderT (OpDef -> OpDef) m a -> (OpDef -> OpDef) -> m a
f &>> g = runReaderT f g

infixl 1 &>>

----------------------

class MakeExprOp a where
    makeExprOp :: ReaderT (Build OpDef) (State ResultState) a

instance (MakeExprOp a1, MakeExprOp a2) => MakeExprOp (a1, a2) where
    makeExprOp = (,) <$> makeExprOp <*> makeExprOp

instance (MakeExprOp a1, MakeExprOp a2, MakeExprOp a3) => MakeExprOp (a1, a2, a3) where
    makeExprOp = (,,) <$> makeExprOp <*> makeExprOp <*> makeExprOp

instance (MakeExprOp a1, MakeExprOp a2, MakeExprOp a3, MakeExprOp a4) => MakeExprOp (a1, a2, a3, a4) where
    makeExprOp = (,,,) <$> makeExprOp <*> makeExprOp <*> makeExprOp <*> makeExprOp

instance (MakeExprOp a1, MakeExprOp a2, MakeExprOp a3, MakeExprOp a4, MakeExprOp a5)
    => MakeExprOp (a1, a2, a3, a4, a5) where
    makeExprOp = (,,,,) <$> makeExprOp <*> makeExprOp <*> makeExprOp <*> makeExprOp <*> makeExprOp

instance (MakeExprOp a1, MakeExprOp a2, MakeExprOp a3, MakeExprOp a4, MakeExprOp a5, MakeExprOp a6)
    => MakeExprOp (a1, a2, a3, a4, a5, a6) where
    makeExprOp = (,,,,,) <$> makeExprOp <*> makeExprOp <*> makeExprOp <*> makeExprOp <*> makeExprOp
                            <*> makeExprOp

instance MakeExprOp a => MakeExprOp [a] where
    makeExprOp = do
        ResultState i ns <- get
        case ns of
            [] -> error $ "Ran out of counts in toResult. " ++
                          "Likely misuse of buildListOp."
            (n : rest) -> do
                put $! ResultState i rest
                replicateM (fromIntegral n) makeExprOp

instance MakeExprOp (TensorExpr a) where
    makeExprOp = do
        ResultState i ns <- get
        put $! ResultState (i+1) ns
        makeOp <- ask
        return $ TensorExpr $ do
            o <- makeOp
            output i . Op <$> getOrAddOp o

-- TODO: the list sizes might also depend on the attrs to come...
exprOp :: (IsExprOp f a, MakeExprOp a)
    => [Int64] -> Build OpDef -> f
exprOp sizes = liftExprOp $ \o -> flip evalState (ResultState 0 sizes)
                                    (runReaderT makeExprOp o)
