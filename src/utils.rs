use std::ops::{AddAssign, DivAssign};

pub fn filled_with<T: Clone>(seq: &[T], filler: T) -> Vec<T> {
    /*
    -- | Creates a matched-length constant sequence with the provided token.
    filledWith :: Sequence -> Token -> Sequence
    filledWith = replicate . length
    */
    let len = seq.len();
    let mut ret_val = Vec::with_capacity(len);
    for _ in 0..len {
        ret_val.push(filler.clone());
    }
    ret_val
}

pub fn filter_by_list<T: Clone>(sel: &[bool], from_this: &[T]) -> Vec<T> {
    /*
    filterByList :: [Bool] -> [a] -> [a]
    filterByList (True : bs) (x : xs) = x : filterByList bs xs
    filterByList (False : bs) (_ : xs) = filterByList bs xs
    filterByList _ _ = []
    */
    sel.iter()
        .zip(from_this)
        .filter_map(
            |(selector, item)| {
                if *selector {
                    Some(item.clone())
                } else {
                    None
                }
            },
        )
        .collect()
}

pub fn safe_maximum<T: Ord + Clone>(v: &[T]) -> Option<T> {
    /*
    safeMaximum :: (Ord a) => [a] -> Maybe a
    safeMaximum [] = Nothing
    safeMaximum xs = Just (maximum xs)
    */
    if v.is_empty() {
        None
    } else {
        let mut maxed = v[0].clone();
        for v_cur in &v[1..] {
            if *v_cur > maxed {
                maxed = v_cur.clone();
            }
        }
        Some(maxed)
    }
}

pub fn safe_minimum<T: Ord + Clone>(v: &[T]) -> Option<T> {
    /*
    safeMinimum :: (Ord a) => [a] -> Maybe a
    safeMinimum [] = Nothing
    safeMinimum xs = Just (minimum xs)
    */
    if v.is_empty() {
        None
    } else {
        let mut mined = v[0].clone();
        for v_cur in &v[1..] {
            if *v_cur < mined {
                mined = v_cur.clone();
            }
        }
        Some(mined)
    }
}

pub fn safe_int8mean<T>(v: &[T]) -> Option<T>
where
    T: AddAssign + Clone + DivAssign<usize>,
{
    /*
    safeInt8Mean :: Sequence -> Maybe Token
    safeInt8Mean [] = Nothing
    safeInt8Mean xs = Just (sum xs `div` fromIntegral (length xs))
    */
    if v.is_empty() {
        None
    } else {
        let mut summed = v[0].clone();
        for v_cur in &v[1..] {
            summed += v_cur.clone();
        }
        summed /= v.len();
        Some(summed)
    }
}
