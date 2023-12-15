use std::ops::{Add, AddAssign, DivAssign};

#[allow(unused_imports)]
use crate::utils::filled_with;
use crate::utils::{filter_by_list, safe_int8mean, safe_maximum, safe_minimum};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Token(pub i8);
impl Token {
    fn min() -> Self {
        Token(i8::MIN)
    }
}
impl Add for Token {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}
impl AddAssign for Token {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}
impl DivAssign<usize> for Token {
    fn div_assign(&mut self, rhs: usize) {
        self.0 /= rhs as i8;
    }
}

pub type Sequence = Vec<Token>;
type Keys = Sequence;
pub type Queries = Sequence;
type Values = Sequence;
pub type Predicate = fn(Token, Token) -> bool;
pub type BoolSequence = Vec<bool>;
type Selector = Vec<BoolSequence>;
pub enum AggregationType {
    Min,
    Max,
    Mean,
}

pub fn select_causal(keys: &Keys, queries: &Queries, predicate: &Predicate) -> Selector {
    /*
    -- | Compareis pairs of elements from sequences with a predicate subject to a causal constraint.
    selectCausal :: Keys -> Queries -> Predicate -> Selector
    selectCausal keys queries predicate =
    [ [ (keyIndex <= queryIndex) && predicate (keys !! keyIndex) (queries !! queryIndex)
        | keyIndex <- [0 .. length keys - 1]
        ]
        | queryIndex <- [0 .. length queries - 1]
    ]
    */
    queries
        .iter()
        .enumerate()
        .map(|(query_index, cur_query)| {
            keys.iter()
                .enumerate()
                .map(|(key_index, cur_key)| {
                    key_index < query_index && predicate(*cur_key, *cur_query)
                })
                .collect()
        })
        .collect()
}

fn aggr_max_by_row(filler: Token, v: &Sequence, a: &BoolSequence) -> Token {
    /*
    aggrMaxByRow :: Token -> Sequence -> BoolSequence -> Token
    aggrMaxByRow filler v a = fromMaybe filler maybeMax
    where
        maybeMax = safeMaximum (filterByList a v)
    */
    let maybe_max = safe_maximum(&filter_by_list(a, v));
    maybe_max.unwrap_or(filler)
}
fn aggr_mean_by_row(filler: Token, v: &Sequence, a: &BoolSequence) -> Token {
    /*
    aggrMeanByRow :: Token -> Sequence -> BoolSequence -> Token
    aggrMeanByRow filler v a = fromMaybe filler maybeMean
    where
        maybeMean = safeInt8Mean (filterByList a v)
    */
    let maybe_mean = safe_int8mean(&filter_by_list(a, v));
    maybe_mean.unwrap_or(filler)
}
fn aggr_min_by_row(filler: Token, v: &Sequence, a: &BoolSequence) -> Token {
    /*
    aggrMinByRow :: Token -> Sequence -> BoolSequence -> Token
    aggrMinByRow filler v a = fromMaybe filler maybeMin
    where
        maybeMin = safeMinimum (filterByList a v)
    */
    let maybe_mean: Option<Token> = safe_minimum(&filter_by_list(a, v));
    maybe_mean.unwrap_or(filler)
}

fn aggr_max(filler: Token) -> impl Fn(&Selector, &Values) -> Sequence {
    move |a: &Selector, v: &Values| {
        a.iter()
            .map(|row| aggr_max_by_row(filler, v, row))
            .collect()
    }
}

fn aggr_mean(filler: Token) -> impl Fn(&Selector, &Values) -> Sequence {
    move |a: &Selector, v: &Values| {
        a.iter()
            .map(|row| aggr_mean_by_row(filler, v, row))
            .collect()
    }
}

fn aggr_min(filler: Token) -> impl Fn(&Selector, &Values) -> Sequence {
    move |a: &Selector, v: &Values| {
        a.iter()
            .map(|row| aggr_min_by_row(filler, v, row))
            .collect()
    }
}

type Aggregator = Box<dyn Fn(&Selector, &Values) -> Sequence>;

fn aggregate(agg: AggregationType, token: Token) -> Aggregator {
    match agg {
        AggregationType::Min => Box::new(aggr_min(token)),
        AggregationType::Max => Box::new(aggr_max(token)),
        AggregationType::Mean => Box::new(aggr_mean(token)),
    }
}

pub fn kqv(
    filler: Token,
    agg: AggregationType,
    keys: &Keys,
    queries: &Queries,
    predicate: &Predicate,
    values: &Values,
) -> Sequence {
    // kqv filler agg keys queries predicate = aggregate agg filler $ selectCausal keys queries predicate
    let selected = select_causal(keys, queries, predicate);
    aggregate(agg, filler)(&selected, values)
}

pub fn max_kqv(keys: &Keys, queries: &Queries, predicate: &Predicate, values: &Values) -> Sequence {
    kqv(
        Token::min(),
        AggregationType::Max,
        keys,
        queries,
        predicate,
        values,
    )
}

pub fn min_kqv(keys: &Keys, queries: &Queries, predicate: &Predicate, values: &Values) -> Sequence {
    kqv(
        Token::min(),
        AggregationType::Min,
        keys,
        queries,
        predicate,
        values,
    )
}

pub fn sel_width(sel: Selector) -> Sequence {
    // Computes the "width", or number of nonzero entries, of the rows of a `Selector`.
    sel.iter()
        .map(|row| {
            row.iter()
                .map(|z| from_bool(*z))
                .fold(Token(0), |acc, x| acc + x)
        })
        .collect()
}

pub fn from_bool(val: bool) -> Token {
    if val {
        Token(1)
    } else {
        Token(0)
    }
}

#[allow(dead_code)]
fn tok_map(on_tokens: fn(Token) -> Token, seq: &Sequence) -> Sequence {
    seq.iter().map(|z| on_tokens(*z)).collect()
}

pub fn seq_map(on_tokens: fn(Token, Token) -> Token, seq1: &Sequence, seq2: &Sequence) -> Sequence {
    /*
    -- | Applies an elementwise operation for pairs of tokens on a pair of sequences.
    -- Alias for `zipWith`.
    */
    seq1.iter()
        .zip(seq2)
        .map(|(z1, z2)| on_tokens(*z1, *z2))
        .collect()
}

#[allow(dead_code)]
fn full(seq: Sequence, t: Token) -> Sequence {
    /*
    -- | Creates a sequence of the same length as the provided sequence filled with the provided token.
    -- Alias for `filledWith`.
    full :: Sequence -> Token -> Sequence
    full = filledWith
    */
    filled_with(&seq, t)
}

pub fn indices_of(seq: &Sequence) -> Sequence {
    /*
    -- | Extracts the indices of the elements in a sequence.
    indicesOf :: Sequence -> Sequence
    indicesOf x = [0 .. (fromIntegral (length x) - 1)]
    */
    let len = seq.len();
    (0..len).map(|x| Token(x as i8)).collect()
}

pub fn indices(seq: &Sequence) -> Sequence {
    /*
    -- | Extracts the indices of the elements in a sequence.
    -- Alias for `indicesOf`.
    indices :: Sequence -> Sequence
    indices = indicesOf
    */
    indices_of(seq)
}

#[allow(dead_code)]
fn aggr(ag_type: AggregationType, token: Token) -> Aggregator {
    /*
    -- | Creates an aggregator with a given aggregation type.
    -- Alias for `aggregate`.
    aggr :: AggregationType -> Token -> Aggregator
    aggr = aggregate
    */
    aggregate(ag_type, token)
}

#[allow(dead_code)]
fn select(is_causal: bool, keys: Keys, queries: Queries, predicate: Predicate) -> Selector {
    /*
    -- | Produces a selector indicating which pairs of `Keys` and `Queries` match.
    select ::
    -- | Whether to use causal selection
    Bool ->
    -- | A collection of `Keys` to check against `Queries`
    Keys ->
    -- | A collection of `Queries` to check against `Keys`
    Queries ->
    -- | A boolean predicate that determines whether a key and query match
    Predicate ->
    -- | A collection of boolean sequences indicating which pairs of `Keys` and `Queries` match
    Selector
    select True = selectCausal
    select False = selectAcausal
    */
    if is_causal {
        select_causal(&keys, &queries, &predicate)
    } else {
        select_acausal(&keys, &queries, &predicate)
    }
}

fn select_acausal(keys: &Keys, queries: &Queries, predicate: &Predicate) -> Selector {
    /*
    -- | Non-causal selection is included for some reason.
    selectAcausal :: Keys -> Queries -> Predicate -> Selector
    selectAcausal keys queries predicate = [[predicate keyIndex queryIndex | keyIndex <- keys] | queryIndex <- queries]
    */
    queries
        .iter()
        .map(|query_index| {
            keys.iter()
                .map(|key_index| predicate(*key_index, *query_index))
                .collect()
        })
        .collect()
}

mod tests {
    use super::{Selector, Sequence};

    #[allow(dead_code)]
    fn random_sequence() -> Sequence {
        todo!()
    }

    #[allow(dead_code)]
    fn random_selector() -> Selector {
        todo!()
    }

    #[test]
    fn max_kqv_is_maximum() {
        todo!()
    }

    #[test]
    fn min_kqv_is_minimum() {
        todo!()
    }

    #[test]
    fn prop_sel_width_is_num_true() {
        todo!()
    }
}
