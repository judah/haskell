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
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{- | Rendering of TensorFlow operations as Haskell functions.

The basic type signature generated for each op is:

> {constraints} => {mandatory attrs} -> {input tensors} -> {output tensors}

where:

* @{mandatory attrs}@ is of the form @A_1 -> ... -> A_N@, where each @A@ is an
 op attribute that doesn't have a default and can't be inferred from other
 inputs.

* @{constraints}@ restrict the type parameters of the input and output tensors
 (for example: 'TensorType' or 'OneOf').

* @{input tensors}@ is of the form @T_1 -> ... -> T_N@, where each @T@ is of
the form @Tensor Ref a@, @Tensor v a@ or @ResourceHandle a@ (or a list of one
of those types), and @a@ is either a concrete type or a (constrained) type
variable.

* @{output tensors}@ is of the form @(T_1,...,T_N)@ for "pure" ops, and
@Build (T_1,...,T_N)@ for "stateful" ops.  An op is considered "stateful" if
it takes a @Tensor Ref@ or @ResourceHandle@ as input, or if it's explicitly
marked \"Stateful\" in its @REGISTER_OP@ definition.  (If there are no outputs,
it is either @ControlNode@ or @Build ControlNode@.)
-}

module TensorFlow.OpGen
  ( OpGenFlags(..)
  , docOpList
  , flagParser)
  where

import Data.Foldable (toList)
import Data.Maybe (fromMaybe, mapMaybe)
import Data.ProtoLens (def, showMessage)
import Data.List.NonEmpty (NonEmpty)
import qualified Data.List.NonEmpty as NE
import Lens.Family2 ((^.), (.~), (&), view)
import Options.Applicative (Parser, help, long, strOption, value)
import Proto.Tensorflow.Core.Framework.OpDef
  ( OpList
  , OpDef
  , attr
  , inputArg
  , name
  , op
  , outputArg
  )
import Proto.Tensorflow.Core.Framework.Types (DataType(..))
import System.FilePath (takeBaseName)
import TensorFlow.OpGen.ParsedOp
import Text.PrettyPrint.Mainland
  ( Doc
  , (<>)
  , (<+>)
  , (</>)
  , (<+/>)
  , brackets
  , comma
  , commasep
  , dquotes
  , empty
  , enclose
  , flatten
  , folddoc
  , hang
  , indent
  , parens
  , sep
  , stack
  , strictText
  , tuple
  )
import qualified Data.Set as Set
import qualified Data.Text as Text

data OpGenFlags = OpGenFlags
     { outputFile :: String
     , prefix :: String
     , excludeList :: String
     }

flagParser :: Parser OpGenFlags
flagParser = OpGenFlags
     <$> strOption (mconcat [ long "output"
                            , help "File to write."
                            ])
     <*> strOption (mconcat [ long "prefix"
                            , help "Haskell package prefix to use"
                            ])
     <*> strOption (mconcat [ long "exclude_list"
                            , value ""
                            , help "Comma separated Ops names to ignore"
                            ])


docOpList :: OpGenFlags -> OpList -> Doc
docOpList flags opList =
  stack [ "{-# LANGUAGE ConstraintKinds #-}"
        , "{-# LANGUAGE DataKinds #-}"
        , "{-# LANGUAGE FlexibleContexts #-}"
        , "{-# LANGUAGE FlexibleInstances #-}"
        , "{-# LANGUAGE OverloadedStrings #-}"
        , "{-# LANGUAGE RankNTypes #-}"
        , "{-# LANGUAGE ScopedTypeVariables #-}"
        , "{-# LANGUAGE TypeFamilies #-}"
          -- Avoids reports about shadowing standard library names.
        , "{-# OPTIONS_GHC -fno-warn-name-shadowing #-}"
          -- eqLengthGuard never returns false and dies instead.
        , "{-# OPTIONS_GHC -fno-warn-incomplete-patterns #-}"
        , "module" <+> strictText moduleName <+> "where"
        , empty
        , imports
        , empty
        , folddoc (\x y -> x </> empty </> y)
                  (map renderOpAndExtras $
                   filter (not . flip elem exclusions . view name) $
                   toList $ opList ^. op)
        ]
  where moduleName =
            Text.pack (prefix flags) <> "." <> camelCase
             -- Discards the optional trailing _ops_op_lib
            (fromMaybe shortName (Text.stripSuffix "_ops_op_lib" shortName))
        shortName = Text.pack (takeBaseName $ outputFile flags)
        exclusions = Text.splitOn "," $ Text.pack $ excludeList flags
        renderOpAndExtras o = renderOp (parseOp o) </> extras o

imports :: Doc
imports = stack [
      "import Data.ByteString (ByteString)"
    , "import Data.Complex (Complex)"
    , "import Data.Int (Int8, Int16, Int32, Int64)"
    , "import Data.Word (Word8, Word16)"
    , "import Lens.Family2 ((.~), (&), (^.))"
    , "import TensorFlow.Build"
    , "import TensorFlow.BuildOp"
    , "import TensorFlow.Output (ResourceHandle(..))"
    , "import TensorFlow.Tensor"
    , "import TensorFlow.Types"
    ]

renderHaskellName, renderTFName, renderQuotedTFName :: Name -> Doc
renderHaskellName = strictText . unHaskellName . haskellName
renderTFName = strictText . unTFName . tfName
renderQuotedTFName = dquotes . renderTFName


-- | Generate the source code for a single op.
-- For example:
--
-- -- | {haddock comment}
-- foo :: {type sig}
-- foo attr1 attr2 input1 input2 | eqLengthGuard [...] = {function body}
renderOp :: ParsedOp -> Doc
renderOp pOp = stack $
    [ haddocks
    , n <+> "::" <+> hang 0 (typeSig pOp)
    , n <+> hang 0 args <+> "|" <+> funcGuard listSizeAttrs
                <+> "=" </>  -- args are indented
                    -- the body needs to be indented wrt the name
                    indent indentation (functionBody pOp)
    ]
  where
    n = renderHaskellName $ parsedOpName pOp
    listSizeAttrs = inferredListSizeAttrs pOp
    args = sep $ map renderHaskellName
                $ map attrName (explicitInputAttrs pOp)
                ++ map parsedArgName (parsedInputs pOp)
    haddocks = "-- |" <+> multilineComment (parsedOpSummary pOp) (parsedOpDescription pOp)

-- | A check that all lists of the given size have the given length.
-- For example:
--   eqLengthGuard [("N", [("input1", length input1), ("input2", length input2)])]
funcGuard :: [Attr (NonEmpty Name)] -> Doc
funcGuard _ = "True"
-- TODO:
funcGuard attrs = "eqLengthGuard" <+> brackets (commasep entries)
      where
        entries =
            [ parens $ nAttr <> comma <+>
              brackets (commasep $ toList $
                            map renderTensorName (toList $ attrInfo a))
            | a <- attrs
            , let nAttr = renderQuotedTFName (attrName a)
            ]
        renderTensorName x = parens $ renderQuotedTFName x <> comma <+>
                        "length" <+> renderHaskellName x

functionBody :: ParsedOp -> Doc
functionBody pOp
    -- TODO: merge these two cases
    | parsedOpIsStateful pOp =
    bindListSizes (inferredListSizeAttrs pOp)
    <+> (if parsedOpIsStateful pOp then "buildResult" else "exprResult")
         <+> brackets (commasep $
                                    map renderHaskellName outputListsSizes)
        <+> parens (hang 0 (stack buildOpParts))
    | otherwise = exprOpBindings pOp (brackets (commasep $ map renderHaskellName outputListsSizes))
            $
            "pure" <+> parens (hang 0 (stack buildOpParts))
  where
    outputListsSizes = [ a
                       | ParsedArg { parsedArgCase = ListArg { argLength = a } }
                            <- parsedOutputs pOp]
    buildOpParts =
        "opDef" <+> renderQuotedTFName (parsedOpName pOp) :
        -- Renders tensor arguments.
        [ "& opAttr" <+> renderQuotedTFName n <+>
          ".~ tensorType (undefined ::" <+> renderHaskellName n <> ")"
        | a <- inferredTypeAttrs pOp, let n = attrName a
        ] ++
        -- Renders mandatory attributes as function parameters.
        [ "& opAttr" <+> renderQuotedTFName n <+> ".~" <+> renderHaskellName n
        | a <- explicitInputAttrs pOp, let n = attrName a
        ] ++
        -- Renders sizes of tensor list types having number_attr.
        [ "& opAttr" <+> renderQuotedTFName n <+> ".~" <+> renderHaskellName n
        | a <- inferredListSizeAttrs pOp, let n = attrName a
        ]
        ++ if null (parsedInputs pOp)
            then []
            else ["& opInputs .~ " <+> inputsList pOp (parsedInputs pOp)]

isExprInput :: ParsedArg -> Bool
isExprInput p = case parsedArgKind p of
    ArgTensorValue -> True
    ArgTensorEither _ -> True
    _ -> False

monadicVarName :: ParsedArg -> Doc
monadicVarName t
    | isExprInput t = renderHaskellName (parsedArgName t) <> "'"
    | otherwise = renderHaskellName (parsedArgName t)

-- TODO: generate a "do" block instead.
bindInputs o [] = ""
bindInputs o (t : ts)
    | isExprInput t = "liftResult" <+> liftedN <+> ">>= \\" <> n <+> "->" <+> bindInputs o ts
    | otherwise = bindInputs o ts
  where
    n = renderHaskellName $ parsedArgName t
    liftedN
        | parsedOpIsStateful o = parens $ "expr" <+> monadicVarName t
        | otherwise = monadicVarName t
    sequenced x
        | ListArg{} <- parsedArgCase t = "sequenceA" <+> x
        | otherwise = x

exprOpBindings :: ParsedOp -> Doc -> Doc -> Doc
exprOpBindings pOp sizes e
    = "let" <+> indent 2 (stack $ map listAssignment (inferredListSizeAttrs pOp))
      </> "in exprOp" <+> sizes <+> "$ do"
        <+> indent 2 (stack $ mapMaybe bindExprInput (parsedInputs pOp) ++ [e])
  where
    bindExprInput t
        | isExprInput t = Just $ monadicVarName t 
                                <+> "<-" <+> maybeMapM t <+> "exprOutput"
                                <+> renderHaskellName (parsedArgName t)
        | otherwise = Nothing
    maybeMapM t
        | ListArg{} <- parsedArgCase t  = "mapM"
        | otherwise = ""
                            
bindListSizes [] = ""
bindListSizes (a : as) = "let" <+> listAssignment a <+> "in" <+> bindListSizes as

-- | Define the implicit list length attributes.
-- For example:
--     n1 = fromIntegral (length input1) :: Int64
listAssignment :: Attr (NonEmpty Name) -> Doc
listAssignment a = renderHaskellName (attrName a) <+> "="
                            <+> "fromIntegral (length"
                            <+> renderHaskellName (NE.head $ attrInfo a)
                            <> ") :: Int64"


inputsList :: ParsedOp -> [ParsedArg] -> Doc
inputsList o [] = "[]"
inputsList o (t:ts) = case parsedArgCase t of
    SimpleArg{}
        | not (parsedOpIsStateful o) && isExprInput t
            -> outputSelector <+> n' <+> ":" <+> inputsList o ts
        | otherwise -> outputSelector <+> n <+> ":" <+> inputsList o ts
    ListArg{}
        | not (parsedOpIsStateful o) && isExprInput t -> "map" <+> outputSelector <+> n' <+> "++" <+> inputsList o ts
        | otherwise -> "map" <+> outputSelector <+> n <+> "++" <+> inputsList o ts
  where
    n' = monadicVarName t
    n = renderHaskellName $ parsedArgName t
    outputSelector = case parsedArgKind t of
                        -- TODO: clean this up
                        ArgResource -> "(\\(ResourceHandle h) -> h)"
                        _
                            | parsedOpIsStateful o -> "(^. tensorOutput)"
                            | otherwise -> "id"

-- | Write a comment with the inputs/outputs/attributes in proto format, for
-- debugging.
extras :: OpDef -> Doc
extras d = enclose "{-\n" "\n-}" $
            strictText $ Text.pack $
            showMessage ((def :: OpDef)
                        & inputArg .~ (d ^. inputArg)
                        & outputArg .~ (d ^. outputArg)
                        & attr .~ (d ^. attr))

-- | The type signature for an op.
-- Of the form:
-- forall t1 t2 v1 v2 . (TensorType t1, TensorType t2)
--      => Float -> Tensor t1 v1 -> Tensor t2 v2
-- where "Float" is an explicit input attribute, "Tensor t1 v1" is an input, and
-- "Tensor t2 v2" is an output.
typeSig :: ParsedOp -> Doc
typeSig pOp = constraints
            <+/> signatureFold (map attrInput (explicitInputAttrs pOp)
                                ++ map tensorArgAndComment (parsedInputs pOp)
                                ++ [outputs])
  where
    constraints
        | null (inferredTypeAttrs pOp) = empty
        | otherwise = "forall" <+> sep typeParams <+> "." <+> classConstraints <+> "=>"
    typeParams = [strictText v | k <- parsedInputs pOp ++ parsedOutputs pOp,
                  ArgTensorEither v <- [parsedArgKind k]]
                ++ [renderHaskellName $ attrName n | n <- inferredTypeAttrs pOp]
    classConstraints = tuple $ concatMap tensorArgConstraint
                    $ inferredTypeAttrs pOp
    signatureFold = folddoc (\x y -> x </> "->" <+> y)
    attrInput a = renderAttrType (attrInfo a) <+> hang 0 ("-- ^" <+> attrComment a)
    renderAttrType (AttrSingle a) = renderAttrBaseType a
    renderAttrType (AttrList a) = brackets $ renderAttrBaseType a
    renderAttrBaseType = \case
        AttrBytes -> "ByteString"
        AttrInt64 -> "Data.Int.Int64"
        AttrFloat -> "Float"
        AttrBool -> "Bool"
        AttrType -> "DataType"
        AttrShape -> "Shape"
        AttrTensor -> "TensorProto"

    tensorArgAndComment t = tensorArg pOp t <+> hang 0 ("-- ^" <+> argComment t)
    outputs = case parsedOutputs pOp of
        [] -> wrapBuild "ControlNode"
        -- TODO(judahjacobson): To improve indentation: `tensorArgAndComment a`
        [a] -> wrapBuild (tensorArg pOp a) <+> "-- ^" <+> argComment a
        as -> wrapBuild (tuple (map (tensorArg pOp) as)) <+/> resultComment as
    wrapBuild o
        | parsedOpIsStateful pOp = "BuildResult" <+> parens o
        | otherwise = "ExprOp" <+> parens o

-- | Render an op input or output.
-- For example: "Tensor Ref Int64", "Tensor v t", "ResourceHandle dtype"
tensorArg :: ParsedOp -> ParsedArg -> Doc
tensorArg pOp p = case parsedArgCase p of
    SimpleArg { argType = t } -> tensorType t
    ListArg { argType = t } -> brackets $ tensorType t
    MixedListArg {} -> "{{{tensorArg: can't handle heterogeneous lists}}}"
  where
    tensorType t = let
        a = case t of
                ArgTypeFixed dt -> strictText $ dtTypeToHaskell dt
                ArgTypeAttr n -> renderHaskellName n
        v = case parsedArgKind p of
                ArgTensorRef -> "Tensor Ref"
                -- TODO: more robust logic around TensorExpr;
                -- decide in parseOp?
                ArgTensorValue
                    | not (parsedOpIsStateful pOp) -> "TensorExpr"
                    | otherwise -> "Tensor Value"
                ArgTensorEither v'
                    | not (parsedOpIsStateful pOp) -> "TensorExpr"
                    | otherwise -> "Tensor" <+> strictText v'
                ArgResource -> "ResourceHandle"
        in v <+> a

attrComment :: Attr a -> Doc
attrComment a = argComment' (attrName a) (attrDescription a)

argComment :: ParsedArg -> Doc
argComment a = argComment' (parsedArgName a) (parsedArgDescription a)

argComment' :: Name -> Text.Text -> Doc
argComment' argName argDesc =
    bold (renderTFName argName) <> splitMultilineText (":" <+>) argDesc

bold :: Doc -> Doc
bold n = "__" <> n <> "__"

-- | Comment for the outputs of an op.
-- For example:
--   -- ^ (__output1__, __output2__)
--   --
--   -- * __output1__: description1
--   --
--   -- * __output2__: description2
resultComment :: [ParsedArg] -> Doc
resultComment os = stack $ flatten commentSummary : map commentDetails os
  where
    commentSummary = "-- ^" <+> tuple [bold (renderTFName $ parsedArgName o) | o <- os]
    commentDetails o =
        stack [ "--"
              , "-- *" <+> argComment o
              ]

-- | Constraints for a given type parameter.
-- E.g.: ["TensorType t"] or ["TensorType t", "OneOf [Int64, Float] t"]
tensorArgConstraint :: Attr [DataType] -> [Doc]
tensorArgConstraint a
    = ("TensorType" <+> n
        : if null typeList
            then []
            else ["OneOf" <+> "'" <> brackets (commasep typeList) <+> n])
  where
    n = renderHaskellName $ attrName a
    typeList = map strictText $
                    Set.toList $ Set.fromList $
                    map dtTypeToHaskell $ attrInfo a

-- NOTE: The cases of this function should be kept in sync with
-- TensorFlow.Types.AllTensorTypes.
dtTypeToHaskell :: DataType -> Text.Text
dtTypeToHaskell DT_BOOL = "Bool"
dtTypeToHaskell DT_BFLOAT16 = "Data.Word.Word16"
dtTypeToHaskell DT_COMPLEX128 = "(Data.Complex.Complex Double)"
dtTypeToHaskell DT_COMPLEX64 = "(Data.Complex.Complex Float)"
dtTypeToHaskell DT_DOUBLE = "Double"
dtTypeToHaskell DT_FLOAT = "Float"
dtTypeToHaskell DT_INT16 = "Data.Int.Int16"
dtTypeToHaskell DT_INT32 = "Data.Int.Int32"
dtTypeToHaskell DT_INT64 = "Data.Int.Int64"
dtTypeToHaskell DT_INT8 = "Data.Int.Int8"
dtTypeToHaskell DT_QINT32 = "Data.Int.Int32"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QINT8 = "Data.Word.Word8"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QINT16 = "Data.Int.Int16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QUINT16 = "Data.Word.Word16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_QUINT8 = "Data.Word.Word8"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_STRING = "Data.ByteString.ByteString"
dtTypeToHaskell DT_UINT16 = "Data.Word.Word16"
dtTypeToHaskell DT_HALF = "Data.Word.Word16"  -- TODO(gnezdo): make unique
dtTypeToHaskell DT_UINT8 = "Data.Word.Word8"
dtTypeToHaskell DT_RESOURCE =
    error "ResourceHandle must be prevented from getting here."
dtTypeToHaskell x =
    Text.pack $ "Unsupported type in dtTypeToHaskell: " ++ show x

-- | haddockComment escapes TensorFlow doc strings into haddock.
-- TODO(gnezdo): deal with the markup.
haddockComment :: Text.Text -> Doc
haddockComment = strictText

-- | Generate a multiline comment.  For example:
--   summary'
--   --
--   -- detail_line1
--   -- detail_line2
--   -- ...
multilineComment :: Text.Text -> Text.Text -> Doc
multilineComment summary' detail =
    haddockComment summary' </>
    splitMultilineText insertParagraphAndComment detail
  where insertParagraphAndComment x = "--" </> "--" <+> x

-- | Converts the given multi-line detail string into
-- a multi-line haddock. Applies the given lead to the
-- first line. Returns an empty document for empty detail.
splitMultilineText :: (Doc -> Doc) -> Text.Text -> Doc
splitMultilineText lead detail =
  case Text.lines detail of
    [] -> empty
    (l : ls) -> stack $ lead (haddockComment l)
                      : map (("--" <+>) . haddockComment) ls

indentation :: Int
indentation = 4
