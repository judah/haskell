{-# LANGUAGE DataKinds #-}
{-# LANGUAGE EmptyDataDecls #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
module TensorFlow.Flow where

import Data.Complex (Complex)
import Data.Default (def)
import Data.Foldable (fold)
import Data.Int (Int32, Int64)
import Data.Monoid ((<>))
import qualified Data.Map.Strict as Map
import qualified Data.Set as Set
import Control.Monad.State.Strict
import Lens.Family2 ((&), (.~), (^.), (%~))
import TensorFlow.Build
import TensorFlow.ControlFlow
import TensorFlow.Ops ()
import TensorFlow.Variable (Variable)
import qualified TensorFlow.Variable as Variable
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

-- TODO: the Deps are already in GraphState's initializationNodes
newtype Flow s a = Flow (StateT Deps Build a)
    deriving (Functor, Applicative, Monad)

newtype Expr t a = Expr (Tensor Build a)

runFlow :: (forall s . Flow (Once s) a) -> Session a
runFlow (Flow act) = do
    (result, deps) <- build $ runStateT act Set.empty
    run_ deps
    return result

deferFlow :: (forall s . Flow s ()) -> Session Deferred
deferFlow (Flow act) = do
    deps <- build $ execStateT act Set.empty
    return $ Deferred deps

newtype Deferred = Deferred Deps

splice :: Deferred -> Flow s ()
splice (Deferred deps) = Flow $ modify (<> deps)

withDeps :: Nodes a => Build a -> Flow s a
withDeps m = do
    prevDeps <- Flow get
    result <- Flow $ lift $ withNodeDependencies prevDeps m
    Flow $ lift (getNodes result) >>= put
    return result

instance (Num a,
         OneOf '[ Double, Float, Int32, Int64,
                  Complex Float, Complex Double] a)
            => Num (Expr t a) where
    Expr a + Expr b = Expr (a + b)
    Expr a * Expr b = Expr (a * b)
    Expr a - Expr b = Expr (a - b)
    abs (Expr a) = Expr (abs a)
    signum (Expr a) = Expr (signum a)
    negate (Expr a) = Expr (negate a)
    fromInteger = Expr . fromInteger

newVariable :: forall a s . TensorType a => Shape -> Flow (Once s) (Variable a)
newVariable = Flow . lift . Variable.variable

assign :: forall a s . TensorType a => Variable a -> Expr s a -> Flow s ()
assign v (Expr x) = void $ withDeps $ Variable.assign v x

data Once s

initializedVariable :: TensorType a => Expr (Once s) a -> Flow (Once s) (Variable a)
initializedVariable x = do
    v <- newVariable (Shape [])  -- unknown shape at the start
    assign v x
    return v

value :: TensorType a => Variable a -> Expr s a
value = Expr . Variable.readValue

liftF2 :: (a -> b -> c) -> Flow s a -> Flow s b -> Flow s c
liftF2 f (Flow g) (Flow h) = Flow $ do
    deps <- get
    (x, deps') <- lift $ runStateT g deps
    (y , deps'') <- lift $ runStateT h deps
    put $ deps' <> deps''
    return $ f x y

-- TODO:
-- Fetch values
-- gradients: how, when things aren't fixed?
-- Automatic codegen?  Change MonadBuild?
-- devices and name scopes?
