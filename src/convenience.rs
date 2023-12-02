/*
-- | This module provides convenience functions built from the core of the RASP-L language.
--
-- It is based on Listing 3 of
-- "What Algorithms Can Transformers Learn", https://arxiv.org/abs/2310.16028,
-- by Zhou et al.
*/

use crate::core::{
    from_bool, indices, kqv, max_kqv, min_kqv, sel_width, select_causal, seq_map, AggregationType,
    Predicate, Sequence, Token,
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
            if bt == Token(1) {
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

fn sample<F>(end_of_seq: Token, prog: F, mut xs: Sequence, n: u8) -> Sequence
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
