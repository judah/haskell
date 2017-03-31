{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeSynonymInstances #-}
module TensorFlow.Flow where

import Data.Default (def)
import Data.Foldable (fold)
import Data.Monoid ((<>))
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Control.Monad.State.Strict
import Lens.Family2 ((&), (.~), (^.), (%~))
import TensorFlow.Build
import TensorFlow.BuildOp
import TensorFlow.Nodes
import TensorFlow.Output
import TensorFlow.Session
import TensorFlow.Tensor
import TensorFlow.Types
import Proto.Tensorflow.Core.Framework.Tensor
    ( TensorProto
    , dtype
    , tensorShape
    )
import qualified Proto.Tensorflow.Core.Framework.TensorShape
  as TensorShape

type Deps = Set.Set NodeName

-- TODO: nicer
instance Nodes Deps where
    getNodes = return

data Blocking = NonBlocking | Blocking

newtype Flow s a = Flow (StateT Deps Build a)
    deriving (Functor, Applicative, Monad)

newtype Expr t a = Expr (Tensor Value a)  -- TODO: Tensor Build a

runFlow :: (forall s . Flow s a) -> Session a
runFlow (Flow act) = do
    (result, deps) <- build $ runStateT act Set.empty
    run_ deps
    return result

instance OpResult (Expr t a) where
    toResult = Expr <$> toResult

instance BuildOp (Expr t a) where
    buildOp' counts o ts = Expr $ buildOp' counts o ts

instance BuildOp f => BuildOp (Expr t a -> f) where
    buildOp' rf o ts (Expr t) = buildOp' rf o ts t

-- TODO: this renders everything...is that OK?
instance (OpResult a, Nodes a) => BuildOp (Flow s a) where
    buildOp' ns o outputs = do
        prevDeps <- Flow get
        result <- Flow $ lift $ buildOp' ns (o & opControlInputs %~ (++ Set.toList prevDeps))
                                    outputs
        Flow $ lift (getNodes result) >>= put
        return result



instance (Num a, TensorType a) -- TODO: also the OneOf constraint
            => Num (Expr t a) where
    (+) = buildOp (opDef "Add")
    (*) = buildOp (opDef "Mul")
    (-) = buildOp (opDef "Sub")
    abs = buildOp (opDef "Abs")
    signum = buildOp (opDef "Sign")
    negate = buildOp (opDef "neg")
    fromInteger n = constant (Shape []) [fromInteger n]

constant :: forall a s . TensorType a => Shape -> [a] -> Expr s a
constant (Shape cShape) values
    | invalidLength = error invalidLengthMsg
    | otherwise = buildOp $ opDef "Const" & opAttr "value" .~ typedNode
  where
    invalidLength = product cShape /= fromIntegral (length values)
    invalidLengthMsg = "invalid tensor length: expected "
                            ++ show (product cShape)
                            ++ " got " ++ show (length values)
    typedNode :: TensorProto
    typedNode = def
                & dtype .~ tensorType (undefined :: a)
                & tensorShape.TensorShape.dim .~
                      [def & TensorShape.size .~ x | x <- cShape]
                & tensorVal .~ values


newtype Var a = Var ResourceHandle

newVar :: forall a s . TensorType a => Shape -> Flow s (Var a)
newVar sh = fmap Var $
    buildOp $ opDef "Var"
                                & opAttr "shape" .~ sh
                                & opAttr "dtype" .~ tensorType (undefined :: a)

assign :: forall a s . TensorType a => Var a -> Expr s a -> Flow s ()
assign (Var v) x = voidNode $ buildOp (opDef "AssignVariableOp"
                                & opAttr "dtype" .~ tensorType (undefined :: a))
                        v x

initializedVar :: forall a s . TensorType a => Expr s a -> Flow s (Var a)
initializedVar x = do
    v <- newVar (Shape [])  -- unknown shape at the start
    assign v x
    return v

voidNode :: Flow s ControlNode -> Flow s ()
voidNode _ = return ()

liftF2 :: (a -> b -> c) -> Flow s a -> Flow s b -> Flow s c
liftF2 f (Flow g) (Flow h) = Flow $ do
    deps <- get
    (x, deps') <- lift $ runStateT g deps
    (y , deps'') <- lift $ runStateT h deps
    put $ deps' <> deps''
    return $ f x y
