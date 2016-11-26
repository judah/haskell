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

{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE ViewPatterns #-}

module TensorFlow.Gradient
    ( gradients
    ) where

import Control.Monad (forM, zipWithM)
import Control.Monad.State.Strict (State, evalState, gets, modify)
import Data.ByteString (ByteString)
import Data.Complex (Complex)
import Data.Default (def)
import Data.Int (Int32, Int64)
import Data.Foldable (foldlM)
import Data.List (foldl', sortBy)
import Data.Map.Strict (Map)
import Data.Maybe (fromMaybe, maybeToList, mapMaybe)
import Data.Ord (comparing)
import Data.ProtoLens.TextFormat (showMessage)
import Data.Set (Set)
import Data.Text (Text)
import Data.Tuple (swap)
import Lens.Family2 (Lens', (&), (^.), (.~), (%~), to)
import Lens.Family2.State.Strict (uses)
import Lens.Family2.Stock (at, intAt)
import Lens.Family2.Unchecked (lens, iso)
import Prelude hiding (sum)
import Text.Printf (printf)
import qualified Data.Graph.Inductive.Basic as FGL
import qualified Data.Graph.Inductive.Graph as FGL
import qualified Data.Graph.Inductive.PatriciaTree as FGL
import qualified Data.Graph.Inductive.Query.DFS as FGL
import qualified Data.IntMap.Strict as IntMap
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import qualified Data.Text as Text

import qualified TensorFlow.GenOps.Core as CoreOps
import TensorFlow.Build
    ( Build
    , renderedNodeDefs
    , opDef
    , opAttr
    , opInputs
    , Expr
    , expr
    , unsafeToExpr
    )
import TensorFlow.BuildOp
import TensorFlow.Ops
    ( addN
    , broadcastGradientArgs
    , expandDims
    , fill
    , matMul
    , reducedShape
    , reluGrad
    , reshape
    , scalar
    , shape
    , softmaxCrossEntropyWithLogits
    , sum
    , scalarize
    , vector
    , zerosLike
    )
import TensorFlow.Output
    ( NodeName(..)
    , Op (Op)
    , Output(..)
    , OutputIx(..)
    , outputIndex
    , outputOp
    , unOp
    )
import TensorFlow.Tensor
    ( Tensor(..)
    , TensorKind (ValueKind)
    , Value
    , tensorOutput
    )
import TensorFlow.Types (Attribute, OneOf, TensorType, attrLens)
import Proto.Tensorflow.Core.Framework.NodeDef
    (NodeDef, attr, input, op, name)

type GradientCompatible a =
    -- TODO(fmayle): MaxPoolGrad doesn't support Double for some reason.
    (Num a, OneOf '[ Float, Complex Float, Complex Double ] a)

-- TODO(fmayle): Support control flow.
-- TODO(fmayle): Support gate_gradients-like option to avoid race conditions.
-- TODO(fmayle): Do we need to consider control inputs? See _PendingCount in
-- tensorflow/python/ops/gradients.py.
-- TODO(fmayle): Maybe store the gradient functions and numOutputs on the OpDef.


-- | Gradient of @y@ w.r.t. each element of @xs@.
gradients :: forall a v1 v2 . ( Num (Tensor v1 a)
                                -- TODO(gnezdo): remove indirect constraint.
                               -- It's a wart inherited from Num instance.
                              , v1 ~ Value
                              , GradientCompatible a
                              )
          => Tensor v1 a  -- ^ The output of the graph.
          -> [Tensor v2 a]  -- ^ Tensors for which gradients are computed.
          -> Expr [Tensor Value a]
gradients y xs = do
    -- The gradients are computed using "reverse accumulation", similarly to
    -- what is described here:
    -- https://en.wikipedia.org/wiki/Automatic_differentiation#The_chain_rule.2C_forward_and_reverse_accumulation
    --
    -- The code is summarised as follows:
    --
    -- 1. Create an fgl graph of the relevant nodes (ops) and edges (tensors).
    -- 2. Initialize the gradient of y to 1 (∂y/∂y = 1) and the rest of tensor's
    --    gradients to nothing.
    -- 3. Process the nodes in reverse topological order (i.e. each node comes
    --    after all of its outputs so that the output gradients for a node have
    --    been completely calculated before it is processed):
    --      a. Record the gradient for each of the node's output tensors (∂y/∂w
    --         for each output tensor w).
    --      b. Calculate the gradient of y w.r.t. each of the node's input
    --         tensors using the gradients of the node's output tensors.
    --
    --         Written differently, for each output tensor w and input tensor v:
    --           ∂y/∂w = ...            (calculated in previous steps)
    --           ∂w/∂v = ...            (op specific)
    --           ∂y/∂v = ∂y/∂w * ∂w/∂v  (technically, if tensor v is an input
    --                                   to multiple nodes, then this is only
    --                                   part of ∂y/∂v)
    --
    -- 4. Lookup the recorded gradient for each x in xs.

    let yName = y ^. tensorOutput . outputOp . to unOp 
    -- TODO(fmayle): Move this into Build.hs and call it unsafeNodeDefFromName?
    nodeDefLookup :: (NodeName -> NodeDef) <- unsafeToExpr $ uses renderedNodeDefs $
        (\f x -> fromMaybe (error $ "no NodeDef found for " ++ show x) (f x))
        . flip Map.lookup
    let (gr, nodeMap) = createGraph yName nodeDefLookup
    -- Set gradient of y to one.
    let initPending :: Expr (Map.Map FGL.Node (PendingGradients a))
        initPending = do
            f <- fill (shape $ pure y) 1
            pure $ Map.empty & at (nodeMap Map.! yName)
                                . nonEmpty
                                . outputIxAt (y ^. tensorOutput . outputIndex)
                                . nonEmpty
                                .~ [f]
    -- Calculate the gradients of y w.r.t. each node in the graph.
    gradientMap <- initPending >>= graphGrads gr
    -- Lookup the gradients for each x.
    forM xs $ \x -> do
        let xName = x ^. tensorOutput . outputOp . to unOp
        fromMaybe (zerosLike $ pure x) $ do
            n <- nodeMap ^. at xName
            let i = x ^. tensorOutput . outputIndex
            pure <$> gradientMap ^. at n . nonEmpty . outputIxAt i

outputIxAt :: OutputIx -> Lens' (IntMap.IntMap v) (Maybe v)
outputIxAt = intAt . unOutputIx

-- | Incomplete gradients of a node's outputs.
--
-- The lists represent partial sums. The key is an OutputIx sans newtype.
type PendingGradients a = IntMap.IntMap [Tensor Value a]

-- | Gradients of a node's outputs. The key is an OutputIx sans newtype.
type Gradients a = IntMap.IntMap (Tensor Value a)

-- | Graph of TensorFlow operations.
type Graph = FGL.Gr NodeDef EdgeLabel

-- | Data associated with an edge.
--
-- Pair of
--   1. Output index of a tensor from the source node.
--   2. Input index that the tensor connects to on the destination node.
type EdgeLabel = (OutputIx, OutputIx)


-- | State used for calculating gradients.
data GradientsState a = GradientsState
                      { _gradientsPending :: !(Map FGL.Node (PendingGradients a))
                      , _gradientsResult  :: !(Map FGL.Node (Gradients a))
                      }

gradientsPending :: Lens' (GradientsState a) (Map FGL.Node (PendingGradients a))
gradientsPending = lens _gradientsPending (\x y -> x { _gradientsPending = y })

gradientsResult :: Lens' (GradientsState a) (Map FGL.Node (Gradients a))
gradientsResult = lens _gradientsResult (\x y -> x { _gradientsResult = y })


-- TODO(fmayle): Use something like Data.List.Safe.
-- | Safe version of (!!).
safeIndex :: [a] -> Int -> Maybe a
_      `safeIndex` n | n < 0 = Nothing
[]     `safeIndex` _         = Nothing
(x:_)  `safeIndex` 0         = Just x
(_:xs) `safeIndex` n         = xs `safeIndex` (n-1)

-- Copy of http://hackage.haskell.org/package/lens-3.9.0.2/docs/Control-Lens-Iso.html#v%3anon
anon :: a -> (a -> Bool) -> Lens' (Maybe a) a
anon a p = iso (fromMaybe a) go where
  go b | p b       = Nothing
       | otherwise = Just b

non :: Eq a => a -> Lens' (Maybe a) a
non a = anon a (a==)

-- | Lens that defaults Nothing to mempty.
nonEmpty :: (Monoid (t v), Foldable t) => Lens' (Maybe (t v)) (t v)
nonEmpty = anon mempty null

-- | Calculate the gradients for every node in a graph.
graphGrads :: forall a. GradientCompatible a
           => Graph
           -> Map FGL.Node (PendingGradients a)
           -- ^ Initial gradients (usually just 1 for the node of interest).
           -> Expr (Map FGL.Node (Gradients a))
-- TODO: foldlM'
graphGrads gr initPending = fmap (^. gradientsResult) (foldlM go initState nodeOrder)
  where
    initState = GradientsState initPending Map.empty
    -- Reverse topological sort.
    -- TODO(fmayle): Filter out nodes that are not successors of any x in xs to
    -- avoid calculating gradients that won't be used.
    nodeOrder = FGL.topsort $ FGL.grev gr
    go state node = do
        -- Aggregate the accumulated gradients for this node.
        outputGrads <- 
                sumPendingGradient (state ^. gradientsPending . at node . nonEmpty)
        if null outputGrads
           then pure state
           else do
                -- Calculate the gradients for each of the node's inputs.
                let nextState = state & gradientsResult %~ Map.insert node outputGrads
                let ctx = FGL.context gr node
                gs <- calculateInputGrads ctx outputGrads gr
                pure $ updatePendingGradients ctx gs nextState

-- | Reduce accumulated gradients for each output to one Tensor.
sumPendingGradient :: GradientCompatible a
                   => PendingGradients a -> Expr (Gradients a)
sumPendingGradient = fmap (IntMap.mapMaybe id) . traverse f
  where
    f [] = pure Nothing
    f [x] = pure $ Just x
    f xs = Just <$> addN (pure xs)


-- | Calculate the gradients of a node's input tensors.
--
-- This is mostly just a wrapper around opGrad.
calculateInputGrads :: forall a. GradientCompatible a
                    => FGL.Context NodeDef EdgeLabel
                    -> Gradients a  -- ^ Output gradients of the node.
                    -> Graph
                    -> Expr [Maybe (Tensor Value a)]
calculateInputGrads (inputEdges, _, nodeDef, _) outputGrads gr =
    opGrad (nodeDef ^. op) nodeDef inputTensors fullOutGrads
  where
    fullOutGrads =
        fullOutputGrads (numOutputs nodeDef) (Op $ NodeName $ nodeDef ^. name) outputGrads
    -- Create a tensor from an edge (technically an Output, but it seems less
    -- confusing to refer to it as a tensor here).
    edgeToTensor :: (EdgeLabel, FGL.Node) -> Output
    edgeToTensor ((i, _), n) =
        case FGL.lab gr n of
            Just edgeNodeDef -> Output i (Op $ NodeName $ edgeNodeDef ^. name)
            Nothing -> error $ "calculateInputGrads: missing input node for "
                               ++ Text.unpack (nodeDef ^. name)
    -- Input tensors, sorted by input index.
    inputTensors = map edgeToTensor $ sortBy (comparing (snd . fst)) inputEdges

-- | Convert a Map of gradients to a list, with zeros for missing outputs.
fullOutputGrads :: (TensorType a, Num a)
                => OutputIx  -- ^ Number of outputs.
                -> Op
                -> Gradients a
                -> [Expr (Tensor Value a)]
fullOutputGrads n o gs =
    map (\i -> fromMaybe (zero i) (fmap pure $ gs ^. outputIxAt i)) [0..n-1]
  where
    -- A tensor of zeros with the same shape as the i'th output.
    zero i = zerosLike $ toT (Output i o)


-- | Update the pending gradients of a node's inputs.
updatePendingGradients :: forall a. (TensorType a, Num a)
                       => FGL.Context NodeDef EdgeLabel
                       -> [Maybe (Tensor Value a)]
                       -- ^ Gradient of each input tensor.
                       -> GradientsState a
                       -> GradientsState a
updatePendingGradients (inputEdges, _, nodeDef, _) inputGrads initState =
    foldl' go initState inputEdges
  where
    go :: GradientsState a -> (EdgeLabel, FGL.Node) -> GradientsState a
    go state ((outIndex, OutputIx inIndex), node) =
        case maybeGradient of
            Nothing -> state
            Just g ->
                -- Add to the list of pending gradients for this tensor.
                state & gradientsPending
                      . at node
                      . nonEmpty
                      . outputIxAt outIndex
                      . nonEmpty
                      %~ (g:)
      where
        badSizeErr = error $ printf "updatePendingGradients: bad input index \
                                    \%d for inputGrads of length %d in %s"
                                    inIndex (length inputGrads)
                                    (show (nodeDef ^. name))
        maybeGradient = fromMaybe badSizeErr (safeIndex inputGrads inIndex)


-- | Create a graph that includes a node and its transitive dependencies.
createGraph :: NodeName -> (NodeName -> NodeDef)
            -> (Graph, Map NodeName FGL.Node)
createGraph nodeName nodeDefLookup = (FGL.nmap nodeDefLookup graph, nodeMap)
  where
    -- Parse a tensor name.
    parseTensorName :: Text -> Maybe (NodeName, OutputIx)
    parseTensorName n
        | Text.null n        = error "parseTensorName: empty name"
        | Text.head n == '^' = Nothing  -- Control edge
        | otherwise          =
            let (nm, indexStr) = Text.breakOn ":" n
                index | Text.null indexStr = 0
                      | otherwise = read $ Text.unpack $ Text.tail indexStr
            in Just (NodeName nm, OutputIx index)

    -- Build a map from node name to outward edges.
    --
    -- The state is the set of visited nodes.
    collect :: Maybe (NodeName, OutputIx, OutputIx)
            -> NodeName
            -> State (Set NodeName)
                     (Map NodeName [(NodeName, OutputIx, OutputIx)])
    collect outgoingEdge nm = do
        let nextLookup = Map.singleton nm (maybeToList outgoingEdge)
        seen <- gets (Set.member nm)
        modify (Set.insert nm)
        if seen
            then pure nextLookup
            else do
                let inputs = nodeDefLookup nm ^. input
                    recurse inIndex (parentName, outIndex) =
                        collect (Just (nm, outIndex, inIndex)) parentName
                subEdgeLookups <-
                    zipWithM recurse [0..] $ mapMaybe parseTensorName inputs
                pure $ Map.unionsWith (++) (nextLookup:subEdgeLookups)

    edgeLookup = evalState (collect Nothing nodeName) Set.empty
    -- Associate an ID with each node name.
    nodeMap = Map.fromList $ zip (Map.keys edgeLookup) [0..]
    -- Create the graph.
    graph = FGL.mkGraph (swap <$> Map.toList nodeMap)
                        [ (nodeMap Map.! n, nodeMap Map.! m, (i, j))
                        | (n, edges) <- Map.toList edgeLookup
                        , (m, i, j) <- edges
                        ]

-- | Function to compute the gradient of y w.r.t. each input.
--
-- Let y be an arbitrary tensor
-- and [w_0, ..., w_n] be the output tensors of a node
-- and [v_0, ..., v_n] be the input tensors of the same node.
--
-- Given [∂y/∂w_0, ..., ∂y/∂w_n] and [v_0, ..., v_n], a GradientFunc computes
-- [∂y/∂v_0, ..., ∂y/∂v_n] for a particular op type.
--
-- A Nothing gradient is equivalent to zero (but allows for short circuiting
-- computation when all the gradients for something are Nothing).
type GradientFunc a = NodeDef
                    -> [Output]
                    -- ^ Input tensors.
                    -> [Expr (Tensor Value a)]
                    -- ^ Gradient of y w.r.t. each output tensor.
                    -> Expr [Maybe (Tensor Value a)]
                    -- ^ Gradient of y w.r.t. each input tensor.

grads :: [Maybe (Expr (Tensor Value a))] -> Expr [Maybe (Tensor Value a)]
grads = sequence . map swapJust
  where
    swapJust :: Maybe (Expr a) -> Expr (Maybe a)
    swapJust Nothing = pure Nothing
    swapJust (Just e) = fmap Just e

-- TODO(fmayle): Assert the type is correct.
-- | Create a Tensor from an Output.
-- TODO: maybe just return Tensor, and add "pure"s everywhere?
toT :: Output -> Expr (Tensor Value a)
toT = pure . Tensor ValueKind


-- | Wrapper around `TensorFlow.GenOps.Core.slice` that builds vectors from scalars for
-- simple slicing operations.
flatSlice :: forall v1 t . (TensorType t)
         => Expr (Tensor v1 t)    -- ^ __input__
         -> Int32          -- ^ __begin__: specifies the offset into the first dimension of
                           -- 'input' to slice from.
         -> Int32          -- ^ __size__: specifies the number of elements of the first dimension
                           -- of 'input' to slice. If size is -1, all remaining elements in the dimension
                           -- are included in the slice (i.e. this is equivalent to setting
                           -- size = input.dim_size(0) - begin).
         -> Expr (Tensor Value t) -- ^ __output__
flatSlice t begin size = CoreOps.slice t (vector [begin]) (vector [size])


-- | The gradient function for an op type.
--
-- These implementations should match their python counterparts in:
-- third_party/tensorflow/python/ops/*_grad.py
opGrad :: forall a . GradientCompatible a => Text -> GradientFunc a

opGrad "Abs" _ [toT -> x] [dz] = do {g <- dz * signum x; pure [Just g]}
opGrad "Neg" _ [_] [dz] = do { g <- -dz; pure [Just g]}
opGrad "Relu" _ [toT -> x] [dz] = do { g <- reluGrad dz x; pure [Just g]}

opGrad "Square" _ [toT -> x] [dz] =
    -- TODO(fmayle): Handle complex numbers.
    -- TODO(fmayle): The python code makes dz a control dependency of the 2*x
    -- (for performance reasons?). Will need to put these functions in the Build
    -- monad to replicate that.
    grads [Just $ dz * (2 * x)]

opGrad "Gather" _ [toT -> x, toT -> indices] [dz] =
    -- TODO(fmayle): The python version uses a better performance implementation
    -- when the shape is known without having to run the graph.
    -- TODO(fmayle): We shouldn't convert the result to a dense tensor. Sparse
    -- tensor support will require some thinking.
    grads [ Just $ CoreOps.unsortedSegmentSum values indices' numRows
    , Nothing
    ]
  where
    -- TODO(gnezdo): Use colocateWith but it requires Build monad.
    denseShape = shape (x :: Expr (Tensor Value a))
    numRows = scalarize $ flatSlice denseShape 0 1
    valuesShape = CoreOps.concat 0 $ sequence [ allDimensions
                                   , flatSlice denseShape 1 (-1)
                                   ]
    values = reshape dz valuesShape
    -- TODO(fmayle): This could be either Int32 or Int64.
    indices' = reshape indices allDimensions :: Expr (Tensor Value Int32)

opGrad "Max" _ [toT -> x, toT -> indices] [dz] =
    grads [Just $ indicators `CoreOps.div` numSelected * dz', Nothing]
  where
    sx = shape (x :: Expr (Tensor Value a))
    outputShapeKeptDims = reducedShape sx (indices :: Expr (Tensor Value Int32))
    x' = reshape x outputShapeKeptDims
    dz' = reshape dz outputShapeKeptDims
    indicators = CoreOps.cast $ CoreOps.equal x' x
    numSelected = reshape (sum indicators indices) outputShapeKeptDims

-- Min and Max have identical gradient implementations.
opGrad "Min" u v w = opGrad "Max" u v w

opGrad "Sum" _ [toT -> x, toT -> indices] [dz] =
    grads [ Just $ CoreOps.tile grad tileScaling, Nothing ]
  where
    -- TODO(gnezdo): Implement the fast-path from math_grad._SumGrad.
    sx = shape (x :: Expr (Tensor Value a))
    outputShapeKeptDims = reducedShape sx (indices :: Expr (Tensor Value Int32))
    tileScaling = safeShapeDiv sx outputShapeKeptDims
    grad = reshape dz outputShapeKeptDims

opGrad "Mean" u v@[toT -> x, _] w = do
    [Just dz, Nothing] <- opGrad "Sum" u v w
    let inputShape = shape (x :: Expr (Tensor Value a))
    let outputShape = shape $ pure (dz :: Tensor Value a)
    -- TODO(fmayle): Add fast path when shape is known.
    let inputSize = CoreOps.prod inputShape $ rangeOfRank inputShape
    let outputSize = CoreOps.prod outputShape $ rangeOfRank outputShape
    let factor = safeShapeDiv inputSize outputSize
    grads [Just $ pure dz `CoreOps.div` CoreOps.cast factor, Nothing]

opGrad "Add" _ [toT -> x, toT -> y] [dz] = do
    (rx, ry) <- broadcastGradientArgs sx sy
    grads [ Just $ reshape (sum dz (pure rx)) sx
        , Just $ reshape (sum dz (pure ry)) sy ]
  where
    sx = shape (x :: Expr (Tensor Value a))
    sy = shape (y :: Expr (Tensor Value a))

opGrad "Sub" u v w = do
    [Just x, Just y] <- opGrad "Add" u v w
    grads [Just (pure x), Just (-pure y)]

opGrad "SoftmaxCrossEntropyWithLogits" _ [toT -> x, toT -> y] [dz, _] =
    grads [ Just $ expandDims dz (-1) * fmap snd (softmaxCrossEntropyWithLogits x y)
    , Nothing ]

opGrad "Mul" _ [toT -> x, toT -> y] [dz] = do
    (rx, ry) <- broadcastGradientArgs sx sy
    -- TODO(fmayle): Handle complex numbers.
    grads [ Just $ reshape (sum (dz * y) (pure rx)) sx
            , Just $ reshape (sum (x * dz) (pure ry)) sy ]
  where
    sx = shape (x :: Expr (Tensor Value a))
    sy = shape (y :: Expr (Tensor Value a))

opGrad "Div" _ [toT -> x, toT -> y] [dz] = do
    (rx, ry) <- broadcastGradientArgs sx sy
    -- TODO(fmayle): Handle complex numbers.
    -- TODO(gnezdo): Provide Fractional instance and use '/' instead of div.
    grads [ Just $ reshape (sum (dz `CoreOps.div` y) (pure rx)) sx
            , Just $ reshape (sum (dz * (negate x `CoreOps.div` (y * y))) (pure ry)) sy
            ]
  where
    sx = shape (x :: Expr (Tensor Value a))
    sy = shape (y :: Expr (Tensor Value a))

opGrad "MatMul" nodeDef [toT -> x, toT -> y] [dz] =
    let transposeA = lookupAttr nodeDef "transpose_a"
        transposeB = lookupAttr nodeDef "transpose_b"
        transAttrs a b =
            (opAttr "transpose_a" .~ a) . (opAttr "transpose_b" .~ b)
    in grads $ case (transposeA, transposeB) of
       (False, False) ->
           [ Just $ (dz `matMul` y) &>> transAttrs False True
           , Just $ (x `matMul` dz) &>> transAttrs True False ]
       (False, True) ->
           [ Just $ dz `matMul` y
           , Just $ (x `matMul` dz) &>> transAttrs True False ]
       (True, False) ->
           [ Just $ (dz `matMul` y) &>> transAttrs False True
           , Just $ x `matMul` dz ]
       (True, True) ->
           [ Just $ (dz `matMul` y) &>> transAttrs True True
           , Just $ (x `matMul` dz) &>> transAttrs True True ]

opGrad "Transpose" _ [_, toT -> p] [dz] =
    grads [ Just $ CoreOps.transpose dz
            (CoreOps.invertPermutation p :: Expr (Tensor Value Int32))
    , Nothing
    ]

opGrad "Conv2D" nodeDef [toT -> x, toT -> y] [dz] =
    grads [ Just $ CoreOps.conv2DBackpropInput (shape x) y dz
          &>> opAttr "strides" .~ strides
          &>> opAttr "padding" .~ padding
          &>> opAttr "use_cudnn_on_gpu" .~ useCudnnOnGpu
          &>> opAttr "data_format" .~ dataFormat
    , Just $ CoreOps.conv2DBackpropFilter x (shape y) dz
          &>> opAttr "strides" .~ strides
          &>> opAttr "padding" .~ padding
          &>> opAttr "use_cudnn_on_gpu" .~ useCudnnOnGpu
          &>> opAttr "data_format" .~ dataFormat
    ]
  where
    strides = lookupAttr nodeDef "strides" :: [Int64]
    padding = lookupAttr nodeDef "padding" :: ByteString
    useCudnnOnGpu = lookupAttr nodeDef "use_cudnn_on_gpu" :: Bool
    dataFormat = lookupAttr nodeDef "data_format" :: ByteString

opGrad "MaxPool" nodeDef [toT -> x] [dz] =
    grads [ Just $ CoreOps.maxPoolGrad x output dz
          &>> opAttr "ksize" .~ ksize
          &>> opAttr "strides" .~ strides
          &>> opAttr "padding" .~ padding
          &>> opAttr "data_format" .~ dataFormat
    ]
  where
    output :: Expr (Tensor Value a)
    output = toT $ Output 0 (Op $ NodeName $ nodeDef ^. name)
    ksize = lookupAttr nodeDef "ksize" :: [Int64]
    strides = lookupAttr nodeDef "strides" :: [Int64]
    padding = lookupAttr nodeDef "padding" :: ByteString
    dataFormat = lookupAttr nodeDef "data_format" :: ByteString

opGrad "Reshape" _ [toT -> x, _] [dz] =
    grads [Just $ reshape dz $ shape (x :: Expr (Tensor Value a)), Nothing]

opGrad "OneHot" _ _ _ = pure [Nothing, Nothing, Nothing, Nothing]
opGrad "TruncatedNormal" _ _ _ = pure [Nothing]

opGrad "RefIdentity" _ _ [dz] = grads [Just dz]
opGrad "Cast" nodeDef _ [dz] = grads [Just reverseCast]
  where
    -- TODO(gnezdo): too permissive, python only allows float types as src_type.
    reverseCast = do
            dz' <- dz
            exprResult [] (opDef "Cast"
                 & opAttr "DstT" .~ (lookupAttr nodeDef "SrcT" :: ByteString)
                 & opAttr "SrcT" .~ (lookupAttr nodeDef "DstT" :: ByteString)
                 & opInputs .~ [dz' ^. tensorOutput])

opGrad "DynamicStitch" nodeDef inputs [dz] =
    (replicate halfLen Nothing ++) . map Just <$> sequence valuesGrads
  where
    halfLen =
        let len = length inputs
            half = len `div` 2
        in if 2 * half == len
           then half
           else error ("Uneven input size " ++ show (len, showMessage nodeDef))
    valuesGrads = [ CoreOps.gather dz (toT idx :: Expr (Tensor Value Int32))
                  | idx <- take halfLen inputs
                  ]

opGrad "DynamicPartition" nodeDef [toT -> xs, toT -> indices] dz = do
    reconstructed <-
            CoreOps.reshape stitched
                    (CoreOps.shape (xs :: Expr (Tensor Value a))
                        :: Expr (Tensor Value Int32))
    pure [ Just reconstructed, Nothing ]
  where
    stitched = CoreOps.dynamicStitch partitionedIndices (sequence dz)
    partitionedIndices = CoreOps.dynamicPartition np originalIndices indices
    np = lookupAttr nodeDef "num_partitions" :: Int64
    originalIndices =
        CoreOps.reshape (CoreOps.range 0 (CoreOps.size indices) 1) prefixShape
    prefixShape = shapeInt32 indices
    shapeInt32 = CoreOps.shape :: Expr (Tensor Value Int32) -> Expr (Tensor Value Int32)

opGrad "Select" _ [toT -> c, toT -> x, _] [dz] = do
    g1 <- CoreOps.select c dz zeros
    g2 <- CoreOps.select c zeros dz
    pure [Nothing, Just g1, Just g2]
  where zeros = CoreOps.zerosLike x

-- TODO(gnezdo): Unlike Python, no control dependency on dz.
opGrad "Log" _ [toT -> x] [dz] = grads [Just $ dz * CoreOps.inv x]
-- TODO(gnezdo): Reuse the output instead of doing another exp,
-- though, it is probably CSE'd away anyway.
opGrad "Exp" _ [toT -> x] [dz] = grads [Just $ dz * CoreOps.exp x]
opGrad "SparseSegmentSum" _ [toT -> x, toT -> y, toT -> t] [dz] =
    grads [
    Just $ CoreOps.unsortedSegmentSum
             (CoreOps.gather dz (t :: Expr (Tensor Value Int32)))
             (y :: Expr (Tensor Value Int32)) inputRows,
    Nothing, Nothing]
  where inputRows = flatSlice (shape (x :: Expr (Tensor Value a))) 0 1

opGrad "LabelClasses" _ _ _ = pure [Nothing, Nothing]
opGrad "LabelWeights" _ _ _ = pure [Nothing]
opGrad "Size" _ _ _ = pure [Nothing]
opGrad "ZerosLike" _ _ _ = pure [Nothing]

-- TODO(fmayle): These can go away if we properly prune the graph.
opGrad "Const" _ _ _ = pure [Nothing, Nothing]
opGrad "Placeholder" _ _ _ = pure []
opGrad "Variable" _ _ _ = pure []

opGrad n nodeDef ins grads =
    error $ "no gradient implemented for " ++
            show (n, length ins, length grads, showMessage nodeDef, ins)

-- | The number of outputs for an op type.
numOutputs :: NodeDef -> OutputIx
numOutputs o =
    case o ^. op of
        "Abs" -> 1
        "Add" -> 1
        "Cast" -> 1
        "Const" -> 1
        "Conv2D" -> 1
        "Div" -> 1
        "DynamicStitch" -> 1
        "DynamicPartition" ->
            fromIntegral (lookupAttr o "num_partitions" :: Int64)
        "Exp" -> 1
        "Gather" -> 1
        "LabelClasses" -> 1
        "LabelWeights" -> 1
        "Log" -> 1
        "MatMul" -> 1
        "Max" -> 1
        "MaxPool" -> 1
        "Mean" -> 1
        "Min" -> 1
        "Mul" -> 1
        "Neg" -> 1
        "Placeholder" -> 1
        "OneHot" -> 1
        "RefIdentity" -> 1
        "Relu" -> 1
        "Reshape" -> 1
        "Select" -> 1
        "Size" -> 1
        "SoftmaxCrossEntropyWithLogits" -> 2
        "Square" -> 1
        "SparseSegmentSum" -> 1
        "Sub" -> 1
        "Sum" -> 1
        "Transpose" -> 1
        "TruncatedNormal" -> 1
        "Variable" -> 1
        "ZerosLike" -> 1
        _ -> error $ "numOuputs not implemented for " ++ show (o ^. op)

-- Divides `x / y` assuming `x, y >= 0`, treating `0 / 0 = 0`
safeShapeDiv :: Expr (Tensor v1 Int32) -> Expr (Tensor v2 Int32) -> Expr (Tensor Value Int32)
safeShapeDiv x y = x `CoreOps.div` (CoreOps.maximum y 1)

allDimensions :: Expr (Tensor Value Int32)
allDimensions = vector [-1 :: Int32]

rangeOfRank :: forall v1 t. TensorType t => Expr (Tensor v1 t) -> Expr (Tensor Value Int32)
rangeOfRank x = CoreOps.range 0 (CoreOps.rank x) 1

lookupAttr ::  Attribute a1 => NodeDef -> Text -> a1
lookupAttr nodeDef attrName = nodeDef ^. attr . at attrName . non def . attrLens
