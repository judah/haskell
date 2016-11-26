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

-- | This module contains definitions for some built-in TensorFlow operations.
--
-- Note that certain, "stateful" ops like 'variable' and 'assign' return a
-- 'Build' action (e.g., @Build (Tensor Ref a)@ instead of a pure value; the
-- returned 'Tensor's are always rendered in the current 'Build' context.  This
-- approach helps us avoid problems with inlining or common subexpression
-- elimination, by writing
--
-- > do
-- >     v <- variable []
-- >     w <- assign v 3
-- >     render $ w * w
--
-- instead of
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- > in w * w
--
-- since the latter could be reasonably transformed by the compiler into (or
-- vice versa)
--
-- > let
-- >    v = variable []
-- >    w = assign v 3
-- >    w' = assign v 3
-- > in w * w'
--
-- Ops should return a 'Build' action if their original 'OpDef' marks them as
-- stateful, or if they take any Refs as input.  (This mirrors the rules that
-- TensorFlow uses to avoid common subexpression elimination.)
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fno-warn-orphans #-}

module TensorFlow.Ops
    ( CoreOps.add
    , CoreOps.abs
    , CoreOps.addN
    , CoreOps.argMax
    , CoreOps.assign
    , CoreOps.broadcastGradientArgs
    , CoreOps.cast
    , CoreOps.concat
    , constant
    , CoreOps.equal
    , expandDims
    , initializedVariable
    , zeroInitializedVariable
    , CoreOps.fill
    , CoreOps.oneHot
    , CoreOps.matMul
    , matTranspose
    , CoreOps.mean
    , CoreOps.mul
    , CoreOps.neg
    , CoreOps.pack
    , CoreOps.placeholder
    , CoreOps.range
    , reducedShape
    , CoreOps.relu
    , CoreOps.reluGrad
    , CoreOps.reshape
    , restore
    , restoreFromName
    , save
    , scalar
    , shape
    , CoreOps.sign
    , CoreOps.size
    , CoreOps.softmax
    , CoreOps.softmaxCrossEntropyWithLogits
    , CoreOps.sparseToDense
    , CoreOps.sub
    , CoreOps.sum
    , CoreOps.transpose
    , CoreOps.truncatedNormal
    , CoreOps.variable
    , vector
    , zeros
    , CoreOps.zerosLike
    , scalarize
    ) where

import Data.ByteString (ByteString)
import Data.Complex (Complex)
import Data.Int (Int32, Int64)
import Prelude hiding (abs, sum, concat)
import Data.ProtoLens (def)
import Data.Text.Encoding (encodeUtf8)
import Lens.Family2 ((.~), (&), (^.))
import Text.Printf (printf)
import Proto.Tensorflow.Core.Framework.Tensor
    ( TensorProto
    , dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape
  as TensorShape
import TensorFlow.Build
import TensorFlow.BuildOp
import TensorFlow.ControlFlow (group)
import TensorFlow.Output (renderOutput)
import TensorFlow.Tensor
import TensorFlow.Types

import qualified TensorFlow.GenOps.Core as CoreOps

import qualified Prelude (abs)

-- TODO: Look into hs-boot refactoring to allow mutually recursive imports.
-- | Must be defined as an orphan because of the dependency order between Ops
-- and Tensor.
--
-- The indirect constraint "v ~ Value" helps disambiguate types, for example in
-- "neg 1 :: Tensor Value Float", it helps find the type of the subexpression
-- "1".
instance ( TensorType a
         , Num a
         , v ~ Value
         , OneOf '[ Double, Float, Int32, Int64
                  , Complex Float, Complex Double] a) => Num (Expr (Tensor v a)) where
    (+) = CoreOps.add
    (*) = CoreOps.mul
    (-) = CoreOps.sub
    abs = CoreOps.abs
    fromInteger = scalar . fromInteger
    signum = CoreOps.sign
    negate = CoreOps.neg

matTranspose :: forall a v . TensorType a
             => Expr (Tensor v a) -> ExprResult (Tensor Value a)
matTranspose = flip CoreOps.transpose (vector [1, 0 :: Int32])


-- | Creates a variable initialized to the given value.
-- Initialization happens next time session runs.
initializedVariable :: forall a . TensorType a
                    => Expr (Tensor Value a) -> Build (Tensor Ref a)
initializedVariable initializer = do
    v <- CoreOps.variable []  -- The shape is not known initially.
    (i :: Tensor Ref a) <-
        CoreOps.assign v initializer (opAttr "validate_shape" .~ False)
    addInitializer =<< group i
    return v

-- | Creates a zero-initialized variable with the given shape.
zeroInitializedVariable
  :: (TensorType a, Num a) =>
     TensorFlow.Types.Shape -> Build (Tensor TensorFlow.Tensor.Ref a)
zeroInitializedVariable = initializedVariable . zeros

-- TODO: the tensors might need the "foo:k" format...
-- TODO: Support heterogeneous list of tensors.
save :: forall a v . TensorType a
        => ByteString     -- ^ File path.
        -> Expr [Tensor v a]  -- ^ Tensors to save.
        -> Build ControlNode
save path xs = 
    expr xs >>= \xs' -> let
        f = (\t -> scalar $ encodeUtf8 $ renderOutput $ t ^. tensorOutput)
                    :: Tensor v a -> Expr (Tensor Value ByteString)
        names = mapM f xs' :: Expr [Tensor Value ByteString]
        types = replicate (length xs') (tensorType (undefined :: a))
        in scalar path >>= \p ->
            (CoreOps.pack names :: Build (Tensor Value ByteString)) >>=| \n ->
            buildResult [] $ opDef "Save"
                                    & opAttr "T" .~ types
                                & opInputs .~ ([p ^. tensorOutput, n ^. tensorOutput]
                                            ++ map (^. tensorOutput) xs')

-- | Restore a tensor's value from a checkpoint file.
--
-- This version allows restoring from a checkpoint file that uses a different
-- tensor name than the variable.
restoreFromName :: forall a . TensorType a
                => ByteString    -- ^ File path.
                -> ByteString    -- ^ Tensor name override.
                -> Tensor Ref a  -- ^ Tensor to restore.
                -> Build ControlNode
restoreFromName path name x =
    CoreOps.assign x (CoreOps.restore (scalar path) (scalar name))
        >>= group

-- | Restore a tensor's value from a checkpoint file.
restore :: forall a . TensorType a
        => ByteString    -- ^ File path.
        -> Tensor Ref a  -- ^ Tensor to restore.
        -> Build ControlNode
restore path x = do
    let name = encodeUtf8 $ renderOutput (x ^. tensorOutput)
    restoreFromName path name x

-- | Create a constant tensor.
--
-- The values should be in row major order, e.g.,
--
--   element 0:   index (0, ..., 0)
--   element 1:   index (0, ..., 1)
--   ...
constant :: forall a . TensorType a => Shape -> [a] -> ExprResult (Tensor Value a)
constant (Shape shape') values
    | invalidLength = error invalidLengthMsg
    | otherwise = CoreOps.const (opAttr "value" .~ typedNode)
  where
    invalidLength = product shape' /= fromIntegral (length values)
    invalidLengthMsg = printf "invalid tensor length: expected %d got %d"
                              (product shape')
                              (length values)
    nodeType = tensorType (undefined :: a)
    typedNode :: TensorProto
    typedNode = def
                & dtype .~ nodeType
                & tensorShape.TensorShape.dim .~
                      [def & TensorShape.size .~ x | x <- shape']
                & tensorVal .~ values

-- | Reshape a N-D tensor down to a scalar.
-- 
-- See `TensorFlow.GenOps.Core.reshape`.
scalarize :: (TensorType a) => Expr (Tensor v a) -> ExprResult (Tensor Value a)
scalarize t = CoreOps.reshape t (vector scalarShape)
    where
        scalarShape = [] :: [Int32]


-- | Create a constant vector.
vector :: TensorType a => [a] -> ExprResult (Tensor Value a)
vector xs = constant [fromIntegral $ length xs] xs

-- | Create a constant scalar.
scalar :: forall a . TensorType a => a -> ExprResult (Tensor Value a)
scalar x = constant [] [x]

zeros :: forall a . (Num a, TensorType a) => Shape -> Expr (Tensor Value a)
zeros (Shape shape') = CoreOps.fill (vector $ map fromIntegral shape') (scalar 0)

shape :: (TensorType t) => Expr (Tensor v1 t) -> ExprResult (Tensor Value Int32)
shape = CoreOps.shape

expandDims :: (TensorType t) => Expr (Tensor v1 t) -> Expr (Tensor v2 Int32) -> ExprResult (Tensor Value t)
expandDims = CoreOps.expandDims

-- | Helper function for reduction ops (translation of math_ops.reduced_shape).
reducedShape :: (OneOf '[ Int32, Int64 ] t1, OneOf '[ Int32, Int64 ] t2) =>
                Expr (Tensor v1 t1) -> Expr (Tensor v2 t2) -> Expr (Tensor Value Int32)
reducedShape inputShape axes =
    let inputShape32 = toInt32 inputShape         -- [2, 3, 5, 7]
        axes32 = toInt32 axes                     -- [1, 2]
        toInt32 x = CoreOps.cast x :: Expr (Tensor Value Int32)
        inputRank = CoreOps.size inputShape32     -- 4
        axesMod = (axes32 + inputRank) `CoreOps.mod` inputRank
        axesShape = shape axesMod                 -- [2]
    in CoreOps.dynamicStitch                      -- [2, 1, 1, 7]
         (sequence [CoreOps.range 0 inputRank 1,            -- [0, 1, 2, 3]
           axesMod])                               -- [1, 2]
         (sequence [inputShape32,                           -- [2, 3, 5, 7]
           CoreOps.fill axesShape 1])              -- [1, 1]
