{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
module TensorFlow.Flow
    ( -- * Session
      Session
    , runSession
      -- * The Flow monad
    , Flow
    , runFlow
    , fetchFlow
    , Initializer
    , liftF2
      -- ** Fetchable types
    , Scalar(..)
    -- * Deferred actions
    , deferFlow
    , Deferred
    , splice
    -- * Operations
    , Expr
    -- ** Variables
    , Variable
    , newVariable
    , assign
    , assignAdd
    , initializedVariable
    , readValue
    ) where

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

class Initializer s where

data Deps = Deps
    { latestWrites :: Set.Set NodeName
    , latestReads :: Set.Set NodeName
    }

writeDeps, readDeps :: Set.Set NodeName -> Deps
writeDeps d = Deps d mempty
readDeps = Deps mempty

instance Monoid Deps where
    Deps a b `mappend` Deps a' b' = Deps (a <> a') (b <> b')
    mempty = Deps Set.empty Set.empty

-- TODO: nicer
instance Nodes Deps where
    getNodes = return . latestWrites

instance Fetchable Deps () where
    getFetch _ = return $ pure ()

-- TODO: the Deps are already in GraphState's initializationNodes
newtype Flow s a = Flow (StateT Deps Build a)
    deriving (Functor, Applicative, Monad)

newtype Expr s a = Expr (Tensor Build a)

runFlow :: forall a . (forall s . Initializer s => Flow s a) -> Session a
runFlow (Flow act :: Flow InitializerInstance a) = do
    (result, deps) <- build $ runStateT act mempty
    run_ deps
    return result

data InitializerInstance
instance Initializer InitializerInstance

deferFlow :: (forall s . Flow s ()) -> Session Deferred
deferFlow (Flow act) = do
    deps <- build $ execStateT act mempty
    return $ Deferred deps

-- TODO: more generic
fetchFlow :: forall a b . Fetchable (Tensor Build a) b
          => (forall s . Initializer s => Flow s (Expr s a))
          -> Session b
fetchFlow (Flow act :: Flow InitializerInstance (Expr InitializerInstance a)) = do
    -- TODO: should we avoid running the deps?
    (Expr t, deps) <- build (runStateT act mempty)
    (result, ()) <- run (t, deps)
    return result

newtype Deferred = Deferred Deps
    deriving Monoid

splice :: Deferred -> Flow s ()
splice (Deferred deps) = Flow $ modify (<> deps)

buildWriteDeps :: Nodes a => Build a -> Flow s a
buildWriteDeps m = do
    prevDeps <- Flow get
    result <- Flow $ lift $ withNodeDependencies
                    (latestWrites prevDeps <> latestReads prevDeps)
                    m
    Flow $ lift (getNodes result) >>= put . writeDeps
    return result

buildReadDeps :: Nodes a => Build a -> Flow s a
buildReadDeps m = do
    prevDeps <- Flow get
    result <- Flow $ lift $ withNodeDependencies
                    (latestWrites prevDeps) m
    Flow $ lift (getNodes result) >>= put . (<> prevDeps) . readDeps
    return result

instance (Num a,
         OneOf '[ Double, Float, Int32, Int64,
                  Complex Float, Complex Double] a)
            => Num (Expr s a) where
    Expr a + Expr b = Expr (a + b)
    Expr a * Expr b = Expr (a * b)
    Expr a - Expr b = Expr (a - b)
    abs (Expr a) = Expr (abs a)
    signum (Expr a) = Expr (signum a)
    negate (Expr a) = Expr (negate a)
    fromInteger = Expr . fromInteger

newVariable :: forall a s . (Initializer s, TensorType a) => Shape -> Flow s (Variable a)
newVariable = isolated . Variable.variable

isolated :: Build a -> Flow s a
isolated = Flow . lift

assign :: forall a s . TensorType a => Variable a -> Expr s a -> Flow s ()
assign v (Expr x) = do
    x' <- isolated $ render x
    void $ buildWriteDeps $ Variable.assign v x'

assignAdd :: forall a s . TensorType a => Variable a -> Expr s a -> Flow s ()
assignAdd v (Expr x) = do
    x' <- isolated $ render x
    void $ buildWriteDeps $ Variable.assignAdd v x'

initializedVariable :: (TensorType a, Initializer s) => Expr s a -> Flow s (Variable a)
initializedVariable x = do
    v <- newVariable (Shape [])  -- unknown shape at the start
    assign v x
    return v

-- TODO: this sequences the read against everything else.  Is that too strict?
readValue :: TensorType a => Variable a -> Flow s (Expr s a)
readValue = fmap (Expr . expr) . buildReadDeps . render . Variable.readValue

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
