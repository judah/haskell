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

{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}  -- For the Render class

module TensorFlow.Tensor where

import Data.ByteString (ByteString)
import Data.String (IsString(..))
import qualified Data.Text as Text
import Lens.Family2 ((^.))
import Lens.Family2.State ((%=), use)

import Proto.Tensorflow.Core.Framework.NodeDef (device)
import TensorFlow.Build
import TensorFlow.Output (Output, NodeName, outputNodeName, Device(..))
import TensorFlow.Types
    ( TensorData(..)
    , ListOf(..)
    )
import qualified TensorFlow.Internal.FFI as FFI

-- | A named output of a TensorFlow operation.
--
-- The type parameter @a@ is the type of the elements in the 'Tensor'.  The
-- parameter @v@ is either:
--
--   * 'Build': An unrendered, immutable value.
--   * 'Value': A rendered, immutable value.
--   * 'Ref': A rendered stateful handle (e.g., a variable).
--
-- Note that 'expr', 'value', 'render' and 'renderValue' can help convert between
-- the different types of 'Tensor'.
newtype Tensor v a = Tensor {tensorOutput :: Output}

data Ref
data Value

-- | Cast a 'Tensor Ref' into a 'Tensor Value'. This behaves like a no-op.
value :: Tensor Ref a -> Tensor Value a
value (Tensor o) = Tensor o

-- | A pair of a 'Tensor' and some data that should be fed into that 'Tensor'
-- when running the graph.
data Feed = Feed Output FFI.TensorData

tensorNodeName :: Tensor v a -> NodeName
tensorNodeName = outputNodeName . tensorOutput


-- | Create a 'Feed' for feeding the given data into a 'Tensor' when running
-- the graph.
--
-- Note that if a 'Tensor' is rendered, its identity may change; so feeding the
-- rendered 'Tensor' may be different than feeding the original 'Tensor'.
feed :: Tensor v a -> TensorData a -> Feed
feed t (TensorData td) = Feed (tensorOutput t) td

-- | Create a 'Tensor' for a given name.  This can be used to reference nodes
-- in a 'GraphDef' that was loaded via 'addGraphDef'.
-- TODO(judahjacobson): add more safety checks here.
tensorFromName :: Text.Text -> Tensor v a
tensorFromName = Tensor . fromString . Text.unpack

-- | Like 'tensorFromName', but type-restricted to 'Value'.
tensorValueFromName :: Text.Text -> Tensor Value a
tensorValueFromName = tensorFromName

-- | Like 'tensorFromName', but type-restricted to 'Ref'.
tensorRefFromName :: Text.Text -> Tensor Ref a
tensorRefFromName = tensorFromName

type TensorList v = ListOf (Tensor v)

tensorListOutputs :: TensorList v as -> [Output]
tensorListOutputs Nil = []
tensorListOutputs (t :/ ts) = tensorOutput t : tensorListOutputs ts

-- | Places all nodes rendered in the given 'Build' action on the same
-- device as the given Tensor (see also 'withDevice'). Make sure that
-- the action has side effects of rendering the desired tensors. A pure
-- return would not have the desired effect.
colocateWith :: MonadBuild m => Tensor v b -> m a -> m a
colocateWith t x = do
    d <- build $ Device . (^. device)
               <$> lookupNode (outputNodeName $ tensorOutput t)
    withDevice (Just d) x


-- | Records the given summary action in Build for retrieval with
-- Summary protocol buffer in string form. For safety, use the
-- pre-composed functions: Logging.scalarSummary and
-- Logging.histogramSummary.
addSummary :: MonadBuild m => Tensor v ByteString -- ^ A 'SummaryTensor'
                        -> m ()
addSummary t = build $ do
    -- TODO: more generic way
    let o = tensorOutput t
    summaries %= (o :)

-- | Retrieves the summary ops collected thus far. Typically this only
-- happens once, but if 'TensorFlow.Session.buildWithSummary' is used
-- repeatedly, the values accumulate.
collectAllSummaries :: MonadBuild m => m [SummaryTensor]
collectAllSummaries = build $ map Tensor <$> use summaries

-- | Synonym for the tensors that return serialized Summary proto.
type SummaryTensor = Tensor Value ByteString
