/*
-- | This module provides convenience functions built from the core of the RASP-L language.
--
-- It is based on Listing 3 of
-- "What Algorithms Can Transformers Learn", https://arxiv.org/abs/2310.16028,
-- by Zhou et al.
*/

use crate::core::{
    from_bool, indices, kqv, max_kqv, min_kqv, sel_width, select_causal, seq_map, AggregationType,
    Predicate, Queries, Sequence, Token,
};
use crate::{core::indices_of, utils::filled_with};

pub fn question_where(bs: &[bool], xys: (&Sequence, &Sequence)) -> Sequence {
    /*
    -- | Use a boolean sequence to select between two sequences.
    -- Also known in Python RASP-L as "where", see `where'`.
    (?) :: [Bool] -> (Sequence, Sequence) -> Sequence
    bs ? (xs, ys) = seqMap (\xm ym -> if xm == 0 then ym else xm) xms yms
      where
        xms = seqMap (\bt x -> if bt == 1 then x else 0) bts xs
        yms = seqMap (\bt y -> if bt == 0 then y else 0) bts ys
        bts = fromBoolSeq bs
    */
    let selector = |xm, ym| {
        if xm == Token(0) {
            ym
        } else {
            xm
        }
    };
    let bts = bs.iter().map(|b| from_bool(*b)).collect();
    let xms = seq_map(
        |bt, x| {
            if bt == Token(1) {
                x
            } else {
                Token(0)
            }
        },
        &bts,
        xys.0,
    );
    let yms = seq_map(
        |bt, y| {
            if bt == Token(0) {
                y
            } else {
                Token(0)
            }
        },
        &bts,
        xys.1,
    );
    seq_map(selector, &xms, &yms)
}

#[allow(dead_code)]
fn question_where_curry(bs: &[bool], xs: &Sequence, ys: &Sequence) -> Sequence {
    /*
    -- | Use a boolean sequence to select between two sequences.
    -- Provided for compatibility with Listing 3, but with
    -- an apostrophe to avoid a name clash with the "where" keyword.
    where' :: [Bool] -> Sequence -> Sequence -> Sequence
    where' bs xs ys = bs ? (xs, ys)
    */
    question_where(bs, (xs, ys))
}

#[allow(dead_code)]
fn shift_right(filler: Token, shift_amt: i8, xs: &Sequence) -> Sequence {
    /*
    -- | Shift a sequence to the right by a given number of elements,
    -- filling the vacated positions with the provided `Token
    shiftRight ::
      -- | Filler `Token`
      Token ->
      -- | Number of positions to shift
      Int8 ->
      -- | Input `Sequence`
      Sequence ->
      Sequence
    shiftRight filler shift_amt xs = kqv filler Mean shiftedIdxs idxs (==) xs
      where
        shiftedIdxs = map (+ shift_amt) idxs
        idxs = indices xs
    */
    let idxs = indices(xs);
    let shifted_idxs = idxs.iter().map(|z| Token(z.0 + shift_amt)).collect();
    let equal_tokens: Predicate = |z, w| z == w;
    kqv(
        filler,
        AggregationType::Mean,
        &shifted_idxs,
        &idxs,
        &equal_tokens,
        xs,
    )
}

impl From<Token> for bool {
    /*
    -- | Maps tokens onto bools using Python's "truthiness" rules.
    toBool :: Token -> Bool
    toBool x
      | x == 0 = False
      | otherwise = True
    */
    fn from(value: Token) -> Self {
        value.0 != 0
    }
}

pub fn from_bool_seq(bs: &[bool]) -> Sequence {
    /*
    -- | Converts a list of bools to a sequence of tokens.
    fromBoolSeq :: [Bool] -> Sequence
    fromBoolSeq = map fromBool
    */
    bs.iter().map(|b| from_bool(*b)).collect()
}

#[allow(dead_code)]
fn cum_sum(bs: &[bool]) -> Sequence {
    /*
    -- | Computes the cumulative sum of a boolean sequence.
    cumSum :: [Bool] -> Sequence
    cumSum bs = selWidth (selectCausal bTokens bTokens first)
      where
        bTokens = fromBoolSeq bs
        first x _ = toBool x
    */
    let b_tokens = from_bool_seq(bs);
    let first: Predicate = |x: Token, _: Token| x.into();
    sel_width(select_causal(&b_tokens, &b_tokens, &first))
}

#[allow(dead_code)]
fn mask(mask_t: Token, bs: &[bool], xs: &Sequence) -> Sequence {
    /*
    -- | Masks a `Sequence` with a boolean sequence, using the provided `Token` as the mask.
    mask :: Token -> [Bool] -> Sequence -> Sequence
    mask maskT bs xs = bs ? (xs, xs `filledWith` maskT)
    */
    let ys = filled_with(xs, mask_t);
    question_where(bs, (xs, &ys))
}

fn maximum(xs: &Sequence) -> Sequence {
    /*
    -- | Computes the running maximum of a `Sequence`.
    maximum' :: Sequence -> Sequence
    maximum' xs = maxKQV xs xs always xs
      where
        always _ _ = True
    */
    let always: Predicate = |_, _| true;
    max_kqv(xs, xs, &always, xs)
}

fn minimum(xs: &Sequence) -> Sequence {
    /*
    -- | Computes the running minimum of a `Sequence`.
    minimum' :: Sequence -> Sequence
    minimum' xs = minKQV xs xs always xs
      where
        always _ _ = True
    */
    let always: Predicate = |_, _| true;
    min_kqv(xs, xs, &always, xs)
}

#[allow(dead_code)]
fn argmax(xs: &Sequence) -> Sequence {
    /*
    -- | Computes the indices of the running maximum values in a `Sequence`.
    argmax :: Sequence -> Sequence
    argmax xs = maxKQV xs maxs (==) (indicesOf xs)
      where
        maxs = maximum' xs
    */
    let maxs = maximum(xs);
    let equal_tokens: Predicate = |z, w| z == w;
    let indices = indices_of(xs);
    max_kqv(xs, &maxs, &equal_tokens, &indices)
}

#[allow(dead_code)]
fn argmin(xs: &Sequence) -> Sequence {
    /*
    -- | Computes the indices of the running minimum values in a `Sequence`.
    argmin :: Sequence -> Sequence
    argmin xs = maxKQV xs mins (==) (indicesOf xs)
      where
        mins = minimum' xs
    */
    let mins = minimum(xs);
    let equal_tokens: Predicate = |z, w| z == w;
    let indices = indices_of(xs);
    max_kqv(xs, &mins, &equal_tokens, &indices)
}

pub fn sample<F>(end_of_seq: Token, prog: F, mut xs: Sequence, n: u8) -> Sequence
where
    F: Fn(&Sequence) -> Sequence,
{
    /*
    -- | Greedily and autoregressively sample the output of a RASP-L program on a sequence.
    sample ::
      -- | End of sequence token
      Token ->
      -- | RASP-L program to extend the sequence
      (Sequence -> Sequence) ->
      -- | Initial/prompt sequence
      Sequence ->
      -- | Number of steps to decode
      Word8 ->
      -- | Output (including prompt)
      Sequence
    sample _ _ xs 0 = xs
    sample endOfSequence prog xs n
      | last xs == endOfSequence = xs
      | otherwise = sample endOfSequence prog (xs ++ [last $ prog xs]) (n - 1)
    */
    if n == 0 {
        return xs;
    }
    match xs.last() {
        Some(last) if *last == end_of_seq => xs,
        Some(_) => {
            xs.push(prog(&xs).last().unwrap().to_owned());
            sample(end_of_seq, prog, xs, n - 1)
        }
        None => {
            xs.push(prog(&xs).last().unwrap().to_owned());
            sample(end_of_seq, prog, xs, n - 1)
        }
    }
}

#[allow(dead_code)]
fn sample_autoregressive<F>(end_of_seq: Token, prog: F, xs: Sequence, n: u8) -> Sequence
where
    F: Fn(&Sequence) -> Sequence,
{
    /*
    -- | Greedily and autoregressively sample the output of a RASP-L program on a sequence.
    --
    -- Provided for compatibility with Listing 3.
    sample_autoregressive :: Token -> (Sequence -> Sequence) -> Sequence -> Word8 -> Sequence
    sample_autoregressive = sample
    */
    sample(end_of_seq, prog, xs, n)
}

#[allow(dead_code)]
fn num_previous(xs: &Sequence, queries: &Queries) -> Sequence {
    /*
    -- | Computes the number of previous tokens in a `Sequence` that are equal to each `Token` from `Queries`.
    numPrev :: Sequence -> Queries -> Sequence
    numPrev xs queries = selWidth (selectCausal xs queries (==))
    */
    let equality: Predicate = |a, b| a == b;
    sel_width(select_causal(xs, queries, &equality))
}

#[allow(dead_code)]
fn has_seen(xs: &Sequence, queries: &Queries) -> Sequence {
    /*
    -- | Returns 1s where the `Token` from the `Queries` has been seen before in the `Sequence`.
    hasSeen :: Sequence -> Queries -> Sequence
    hasSeen xs queries = kqv 0 Max xs queries (==) (queries `filledWith` 1)
    */
    let equality: Predicate = |a, b| a == b;
    let values = filled_with(queries, Token(1));
    kqv(
        Token(0),
        AggregationType::Max,
        xs,
        queries,
        &equality,
        &values,
    )
}

#[allow(dead_code)]
fn firsts(filler: Token, xs: &Sequence, queries: &Queries) -> Sequence {
    /*
    -- | Finds the first occurrence of each query token in a `Sequence`.
    firsts :: Token -> Sequence -> Queries -> Sequence
    firsts filler xs queries = kqv filler Min xs queries (==) (indicesOf xs)
    */
    let equality: Predicate = |a, b| a == b;
    let values = indices_of(xs);
    kqv(
        filler,
        AggregationType::Min,
        xs,
        queries,
        &equality,
        &values,
    )
}

#[allow(dead_code)]
fn lasts(filler: Token, xs: &Sequence, queries: &Queries) -> Sequence {
    /*
    -- | Finds the last occurrence of each query token in a `Sequence`.
    lasts :: Token -> Sequence -> Queries -> Sequence
    lasts filler xs queries = kqv filler Max xs queries (==) (indicesOf xs)
    */
    let equality: Predicate = |a, b| a == b;
    let values = indices_of(xs);
    kqv(
        filler,
        AggregationType::Max,
        xs,
        queries,
        &equality,
        &values,
    )
}

#[allow(dead_code)]
fn index_select(filler: Token, xs: &Sequence, idxs: &Sequence) -> Sequence {
    /*
    -- | Selects the tokens from a `Sequence` at the indices provided by another sequence.
    indexSelect :: Token -> Sequence -> Sequence -> Sequence
    indexSelect filler xs idxs = kqv filler Max (indicesOf xs) idxs (==) xs
    */
    let equality: Predicate = |a, b| a == b;
    let keys = indices_of(xs);
    kqv(filler, AggregationType::Max, &keys, idxs, &equality, xs)
}

mod tests {

    use super::{Sequence, Token};

    struct EqualLengthSequences {
        seq1: Sequence,
        seq2: Sequence,
    }

    trait Arbitrary {
        fn arbitrary() -> Self;
        fn shrink(&self) -> Box<dyn Iterator<Item = Self>>;
    }

    impl Arbitrary for Token {
        fn arbitrary() -> Self {
            use rand::{distributions::Uniform, prelude::Distribution};
            let do_low = rand::distributions::Bernoulli::new(0.2).unwrap();
            let mut rng = rand::thread_rng();
            let should_do_low = do_low.sample(&mut rng);
            if should_do_low {
                let between = Uniform::<i8>::from(0..3);
                Self(between.sample(&mut rng))
            } else {
                let between = Uniform::<i8>::from(0..i8::MAX);
                Self(between.sample(&mut rng))
            }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            Box::new(std::iter::empty::<_>())
        }
    }

    impl Arbitrary for Sequence {
        fn arbitrary() -> Self {
            use rand::{distributions::Uniform, prelude::Distribution};
            let n_max = 100;
            let between = Uniform::<usize>::from(0..n_max + 1);
            let mut rng = rand::thread_rng();
            let len = between.sample(&mut rng);
            (0..len).map(|_| Token::arbitrary()).collect()
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            todo!()
        }
    }

    impl Arbitrary for EqualLengthSequences {
        fn arbitrary() -> Self {
            /*
            -- Generate a pair of lists of the same length
            instance Arbitrary EqualLengthSequences where
            arbitrary = do
                len <- choose (0, 100) -- Choose a length between 0 and 100
                list1 <- vectorOf len arbitrary
                list2 <- vectorOf len arbitrary
                return $ EqualLengthSequences (list1, list2)
            */
            use rand::{distributions::Uniform, prelude::Distribution};
            let n_max = 100;
            let between = Uniform::<usize>::from(0..n_max + 1);
            let mut rng = rand::thread_rng();
            let len = between.sample(&mut rng);
            let seq1 = (0..len).map(|_| Token::arbitrary()).collect();
            let seq2 = (0..len).map(|_| Token::arbitrary()).collect();
            Self { seq1, seq2 }
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let shrinked1 = self.seq1.shrink();
            let iter = shrinked1
                .into_iter()
                .zip(self.seq2.shrink())
                .map(|(l1_new, l2_new)| EqualLengthSequences {
                    seq1: l1_new,
                    seq2: l2_new,
                });
            Box::new(iter)
        }
    }

    #[test]
    fn prop_where_all_true_is_id_left() {
        /*
        prop_where_allTrue_is_idLeft :: Sequence -> Property
        prop_where_allTrue_is_idLeft xs = xs === allTrue ? (xs, xs `filledWith` undefined)
        where
            allTrue = replicate (length xs) True
        */
        use super::question_where;
        use crate::core::BoolSequence;
        let num_trials = 10;
        for _ in 0..num_trials {
            let zs = EqualLengthSequences::arbitrary();
            let xs = zs.seq1;
            /*
            unlike above where could use undefined and rely on Haskell laziness
            instead we create a junk ys of the same length
            */
            let ys = zs.seq2;
            let all_true = (0..xs.len()).map(|_| true).collect::<BoolSequence>();
            let observed = question_where(&all_true, (&xs, &ys));
            for (a,b) in observed.iter().zip(xs) {
                assert_eq!((*a).0,b.0);
            }
        }
    }

    #[test]
    fn prop_where_all_false_is_id_right() {
        /*
        prop_where_allFalse_is_idRight :: Sequence -> Property
        prop_where_allFalse_is_idRight xs = xs === allFalse ? (xs `filledWith` undefined, xs)
        where
            allFalse = replicate (length xs) False
        */
        use super::question_where;
        use crate::core::BoolSequence;
        let num_trials = 10;
        for _ in 0..num_trials {
            let zs = EqualLengthSequences::arbitrary();
            let xs = zs.seq1;
            /*
            unlike above where could use undefined and rely on Haskell laziness
            instead we create a junk ys of the same length
            */
            let ys = zs.seq2;
            let all_false = (0..xs.len()).map(|_| false).collect::<BoolSequence>();
            let observed = question_where(&all_false, (&ys, &xs));
            for (a,b) in observed.iter().zip(xs) {
                assert_eq!((*a).0,b.0);
            }
        }
    }

    #[test]
    fn prop_where_alternating_alternates() {
        /*
        prop_where_alternating_alternates :: Sequence -> Property
        prop_where_alternating_alternates xs = take l (cycle [1, -1]) === alternating ? (xs `filledWith` 1, xs `filledWith` (-1))
        where
            alternating = cycle [True, False]
            l = length xs
        */
        use super::question_where;
        use crate::{core::BoolSequence,utils::filled_with};
        let num_trials = 10;
        for _ in 0..num_trials {
            let xs = Sequence::arbitrary();
            let ones = filled_with(&xs, Token(1));
            let neg_ones = filled_with(&xs, Token(-1));
            let alternating = (0..xs.len()).map(|idx| idx % 2 == 0).collect::<BoolSequence>();
            let observed = question_where(&alternating, (&ones,&neg_ones));
            let expected = alternating.iter().map(|z| if *z {Token(1)} else {Token(-1)}).collect::<Sequence>();
            for (a,b) in observed.iter().zip(expected) {
                assert_eq!((*a).0,b.0);
            }
        }
    }

    #[test]
    fn prop_shift_right_zero_is_id() {
        /*
        prop_shiftRight_zero_is_id :: Sequence -> Property
        prop_shiftRight_zero_is_id xs = xs === shiftRight 0 0 xs
        */
        todo!()
    }

    #[test]
    fn prop_shift_right_length_matches_replicate() {
        /*
        prop_shiftRight_length_matches_replicate :: Sequence -> Property
        prop_shiftRight_length_matches_replicate xs = replicate (fromIntegral l) 1 === shiftRight 1 l xs
        where
            l = fromIntegral . length $ xs
        */
        todo!()
    }

    #[test]
    fn prop_shift_right_matches_rotate_fill() {
        /*
        prop_shiftRight_matches_rotateFill :: Token -> Int8 -> Sequence -> Property
        prop_shiftRight_matches_rotateFill t n xs = n >= 0 && l > 0 ==> rotateFill xs === shiftRight t n xs
        where
            -- Uses normal list operations to shift the sequence.
            rotateFill :: Sequence -> Sequence
            rotateFill s = take l $ replicate n' t ++ take (l - n') s

            n' = fromIntegral n
            l = length xs
        */
        todo!()
    }

    #[test]
    fn prop_cum_sum_matches_scanl() {
        /*
        prop_cumSum_matches_scanl :: [Bool] -> Property
        prop_cumSum_matches_scanl bs = scanl1 (+) (map fromBool bs) === cumSum bs
        */
        todo!()
    }

    #[test]
    fn prop_mask_matches_zip_with() {
        /*
        prop_mask_matches_zipWith :: Token -> [Bool] -> Sequence -> Property
        prop_mask_matches_zipWith t bs xs = zipWith (\b x -> if b then x else t) bs xs === mask t bs xs
        */
        todo!()
    }

    #[test]
    fn prop_maximum_matches_scanl() {
        /*
        prop_maximum'_matches_scanl :: Sequence -> Property
        prop_maximum'_matches_scanl xs = scanl1 max xs === maximum' xs
        */
        todo!()
    }

    #[test]
    fn prop_minimum_matches_scanl() {
        /*
        prop_minimum'_matches_scanl :: Sequence -> Property
        prop_minimum'_matches_scanl xs = scanl1 min xs === minimum' xs
        */
        todo!()
    }

    #[test]
    fn prop_argmax_matches_scanl() {
        /*
        prop_argmax_matches_scanl :: Sequence -> Property
        prop_argmax_matches_scanl xs = map fst (scanl1 argmax' (enumerate xs)) === argmax xs
        where
            argmax' :: (Token, Token) -> (Token, Token) -> (Token, Token)
            argmax' (accIdx, accVal) (idx, val)
            | val >= accVal = (idx, val)
            | otherwise = (accIdx, accVal)

            enumerate = zip [0 ..]
        */
        todo!()
    }

    #[test]
    fn prop_argmin_matches_scanl() {
        /*
        prop_argmin_matches_scanl :: Sequence -> Property
        prop_argmin_matches_scanl xs = map fst (scanl1 argmin' (enumerate xs)) === argmin xs
        where
            argmin' :: (Token, Token) -> (Token, Token) -> (Token, Token)
            argmin' (accIdx, accVal) (idx, val)
            | val <= accVal = (idx, val)
            | otherwise = (accIdx, accVal)

            enumerate = zip [0 ..]
        */
        todo!()
    }

    #[test]
    fn prop_num_prev_matches_zip_with() {
        /*
        prop_numPrev_matches_zipWith :: EqualLengthSequences -> Property
        prop_numPrev_matches_zipWith (EqualLengthSequences (xs, qs)) =
        l > 0 ==> zipWith numPrev' (tail (inits xs)) qs === numPrev xs qs
        where
            numPrev' :: [Token] -> Token -> Int8
            numPrev' (x : xs) q = numPrev' xs q + fromBool (x == q)
            numPrev' [] _ = 0

            l = length qs
        */
        todo!()
    }

    #[test]
    fn prop_has_seen_matches_zip_with() {
        /*
        prop_hasSeen_matches_zipWith :: EqualLengthSequences -> Property
        prop_hasSeen_matches_zipWith (EqualLengthSequences (xs, qs)) =
        l > 0 ==> zipWith hasSeen' (tail (inits xs)) qs === hasSeen xs qs
        where
            hasSeen' :: [Token] -> Token -> Int8
            hasSeen' (x : xs) q = max (hasSeen' xs q) (fromBool (x == q))
            hasSeen' [] _ = 0

            l = length qs
        */
        todo!()
    }

    #[test]
    fn prop_firsts_matches_zip_with() {
        /*
        prop_firsts_matches_zipWith :: Token -> EqualLengthSequences -> Property
        prop_firsts_matches_zipWith filler (EqualLengthSequences (xs, qs)) =
        l > 0 ==> zipWith firsts' (tail (inits (enumerate xs))) qs === firsts filler xs qs
        where
            firsts' :: [(Int8, Token)] -> Token -> Int8
            firsts' ((idx, x) : xs) q =
            if x == q then idx else firsts' xs q
            firsts' [] _ = filler

            enumerate = zip [0 ..]

            l = length qs
        */
        todo!()
    }

    #[test]
    fn prop_lasts_matches_zip_with() {
        /*
        prop_lasts_matches_zipWith :: Token -> EqualLengthSequences -> Property
        prop_lasts_matches_zipWith filler (EqualLengthSequences (xs, qs)) =
        l > 0 ==> zipWith lasts' (tail (inits (enumerate xs))) qs === lasts filler xs qs
        where
            lasts' :: [(Int8, Token)] -> Token -> Int8
            lasts' xs q = case filter (\(_, x) -> x == q) xs of
            [] -> filler
            xs' -> fst $ last xs'

            enumerate = zip [0 ..]

            l = length qs
        */
        todo!()
    }

    #[test]
    fn prop_index_select_matches_zip_with() {
        /*
        prop_indexSelect_matches_zipWith :: Token -> EqualLengthSequences -> Property
        prop_indexSelect_matches_zipWith filler (EqualLengthSequences (xs, idxs)) =
        l > 0 ==> zipWith indexSelect' (tail (inits (enumerate xs))) idxs === indexSelect filler xs idxs
        where
            indexSelect' :: [(Int8, Token)] -> Token -> Token
            indexSelect' xs q = case filter (\(idx, _) -> idx == q) xs of
            [] -> filler
            xs' -> snd $ last xs'
            enumerate = zip [0 ..]

            l = length idxs
        */
        todo!()
    }
}
