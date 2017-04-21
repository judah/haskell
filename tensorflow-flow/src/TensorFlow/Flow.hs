{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
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

class Now s where

type Deps = Set.Set NodeName

-- TODO: nicer
instance Nodes Deps where
    getNodes = return

instance Fetchable Deps () where
    getFetch _ = return $ pure ()

-- TODO: the Deps are already in GraphState's initializationNodes
newtype Flow s a = Flow (StateT Deps Build a)
    deriving (Functor, Applicative, Monad)

newtype Expr t a = Expr (Tensor Build a)

runFlow :: forall a . (forall s . Now s => Flow s a) -> Session a
runFlow (Flow act :: Flow NowInstance a) = do
    (result, deps) <- build $ runStateT act Set.empty
    run_ deps
    return result

data NowInstance
instance Now NowInstance

deferFlow :: (forall s . Flow s ()) -> Session Deferred
deferFlow (Flow act) = do
    deps <- build $ execStateT act Set.empty
    return $ Deferred deps

-- TODO: more generic
fetchFlow :: forall a b . Fetchable (Tensor Build a) b
          => (forall s . Now s => Flow s (Expr s a))
          -> Session b
fetchFlow (Flow act :: Flow NowInstance (Expr NowInstance a)) = do
    -- TODO: should we avoid running the deps?
    (Expr t, deps) <- build (runStateT act Set.empty)
    (result, ()) <- run (t, deps)
    return result

newtype Deferred = Deferred Deps

splice :: Deferred -> Flow s ()
splice (Deferred deps) = Flow $ modify (<> deps)

buildDeps :: Nodes a => Build a -> Flow s ()
buildDeps m = do
    prevDeps <- Flow get
    result <- Flow $ lift $ withNodeDependencies prevDeps m
    Flow $ lift (getNodes result) >>= put

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

newVariable :: forall a s . (Now s, TensorType a) => Shape -> Flow s (Variable a)
newVariable = Flow . lift . Variable.variable

-- TODO: does this actually line up??
assign :: forall a s . TensorType a => Variable a -> Expr s a -> Flow s ()
assign v (Expr x) = buildDeps $ Variable.assign v x

-- initializedVariable :: TensorType a => Expr (Now s) a -> Flow (Now s) (Variable a)
initializedVariable x = do
    v <- newVariable (Shape [])  -- unknown shape at the start
    assign v x
    return v

readValue :: TensorType a => Variable a -> Flow s (Expr s a)
readValue v = do
    prevDeps <- Flow get
    result <- Flow $ lift $ withNodeDependencies prevDeps
                    $ render $ Variable.readValue v
    return $ Expr $ expr result

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
