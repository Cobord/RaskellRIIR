use crate::{core::{BoolSequence,Predicate,Sequence,Token,from_bool, indices_of, max_kqv}, utils::filled_with, convenience::{question_where, from_bool_seq,sample}};

const SOS : Token = Token(-1);
const EOS : Token = Token(-2);

#[allow(dead_code)]
fn equals_token(t1: Token, t2: Token) -> Token {
    from_bool(t1==t2)
}

#[allow(dead_code)]
fn rasp_count(inputs : &Sequence) -> Sequence {
    let idxs = indices_of(inputs);
    let equality : Predicate = |a,b| a==b;
    let last_sos = max_kqv(inputs,&filled_with(inputs,SOS),&equality,&idxs);
    let start_counting = idxs.iter()
        .zip(last_sos.clone())
        .map(|(a,b)| *a==b+Token(2))
        .collect::<BoolSequence>();
    let count_tos = max_kqv(inputs,
        &last_sos.iter().map(|a| *a+Token(2)).collect(),
        &equality,inputs);
    let count_froms = max_kqv(inputs,
        &last_sos.iter().map(|a| *a+Token(1)).collect(),
        &equality,inputs);
    let transitions = inputs.iter().zip(count_tos).map(|(a,b)| *a==b).collect::<BoolSequence>();
    let succs = inputs.iter().map(|a| *a+Token(1)).collect::<Sequence>();
    let with_eos = question_where(&transitions,(&filled_with(inputs,EOS),&succs))
        .iter()
        .map(|z| *z>Token(0))
        .collect::<BoolSequence>();
    let final_counts = question_where(&start_counting, (&count_froms,&from_bool_seq(&with_eos)));
    final_counts
}

#[allow(dead_code)]
fn show_token(t : Token) -> String {
    match t {
        SOS => "SOS".to_string(),
        EOS => "EOS".to_string(),
        Token(x) => x.to_string(),
    }
}

#[allow(dead_code)]
fn count(xs : &mut Sequence) -> Sequence {
    let seq_length = 24;
    xs.insert(0, SOS);
    sample(EOS, rasp_count, xs.clone(), seq_length)
}

mod tests {
    #[test]
    fn example_count_test() {
        todo!()
    }
}