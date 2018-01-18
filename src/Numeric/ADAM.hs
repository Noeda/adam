{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE RankNTypes #-}

module Numeric.ADAM
  ( adamGradientDescent
  , adamGradientDescent'
  , adamGradientDescentPipe
  , AdamConfig(..)
  , defaultAdamConfig )
  where

import Control.DeepSeq
import Control.Monad.Trans.State.Lazy
import Data.Aeson
import Data.Binary hiding ( get, put )
import Data.Functor.Identity
import Data.Data
import Data.Reflection
import Data.Foldable
import Data.Traversable
import GHC.Generics
import Numeric.AD
import Numeric.AD.Internal.Reverse
import Pipes hiding ( for )

data AdamConfig a = AdamConfig
  { stepSize :: !a
  , b1       :: !a
  , b2       :: !a
  , epsilon  :: !a }
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Functor, Foldable, Traversable, Binary, FromJSON, ToJSON, NFData )

defaultAdamConfig :: Fractional a => AdamConfig a
defaultAdamConfig = AdamConfig
  { stepSize = 0.001
  , b1       = 0.9
  , b2       = 0.999
  , epsilon  = 10e-8 }

data AdamState a = AdamState
  { firstMoment  :: [a]
  , secondMoment :: [a]
  , b1t          :: !a
  , b2t          :: !a }
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Functor, Foldable, Traversable, Binary, FromJSON, ToJSON, NFData )

zipWithTraversable :: Traversable f => f a -> [b] -> (a -> b -> c) -> f c
zipWithTraversable structure zipping action = flip evalState zipping $
  for structure $ \value -> do
    (zip_item:rest) <- get
    put rest

    return (action value zip_item)
{-# INLINE zipWithTraversable #-}

-- | Same as `gradientDescent` but the descent function is ADAM.
adamGradientDescent :: (Traversable f, Fractional a, Floating a, NFData a)
                    => (forall s. Reifies s Tape => f (Reverse s a) -> Reverse s a)
                    -> f a
                    -> AdamConfig a
                    -> [f a]
adamGradientDescent evaluator structure config =
  go (adamGradientDescentPipe (\structure -> return $ grad evaluator structure)
                              structure
                              config)
 where
  go producer = case runIdentity $ next producer of
    Left _ -> []
    Right (value, next_producer) ->
      value:go next_producer

-- | Same as `adamGradientDescent` but uses `defaultAdamConfig` for
-- configuration.
adamGradientDescent' :: (Traversable f, Floating a, NFData a)
                     => (forall s. Reifies s Tape => f (Reverse s a) -> Reverse s a)
                     -> f a
                     -> [f a]
adamGradientDescent' evaluator structure = adamGradientDescent evaluator structure defaultAdamConfig
{-# INLINE adamGradientDescent' #-}

-- | A piped version of `adamGradientDescent`. This lets you thread a monad
-- through the process.
adamGradientDescentPipe :: (Traversable f, Floating a, NFData a, Monad m)
                        => (f a -> m (f a))
                        -> f a
                        -> AdamConfig a
                        -> Producer (f a) m void
adamGradientDescentPipe next_gradient structure config =
  flip evalStateT initial_state $ loop_it structure
 where
  initial_state = AdamState
    { firstMoment  = replicate num_parameters 0
    , secondMoment = replicate num_parameters 0
    , b1t = b1 config
    , b2t = b2 config }

  num_parameters = length $ toList structure

  loop_it structure = do
    gradient <- fmap toList $ lift $ lift $ next_gradient structure
    st <- get

    let new_first_moment = zipWith (\m g -> (b1 config * m) + (1 - b1 config) * g) (firstMoment st) gradient
        new_second_moment = zipWith (\m g -> (b2 config * m) + (1 - b2 config) * g * g) (secondMoment st) gradient
        bias_corrected_first_moment  = fmap (\v -> (v / (1 - b1t st))) new_first_moment
        bias_corrected_second_moment = fmap (\v -> (v / (1 - b2t st))) new_second_moment

        new_structure = zipWithTraversable structure (zip bias_corrected_first_moment bias_corrected_second_moment) $
                          \old_value (first_moment, second_moment) ->
                            let d = sqrt second_moment + epsilon config
                             in old_value - stepSize config * first_moment / d

    put st { b1t = b1t st * b1 config
           , b2t = b2t st * b2 config
           , firstMoment = new_first_moment
           , secondMoment = new_second_moment }

    lift $ yield new_structure
    st `deepseq` loop_it new_structure

