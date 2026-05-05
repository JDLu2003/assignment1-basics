use fancy_regex::Regex;
use std::fs;

use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

#[pyfunction]
pub fn train_bpe(
    input_path: String,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(HashMap<usize, Vec<u8>>, Vec<(Vec<u8>, Vec<u8>)>)> {
    #[cfg(debug_assertions)]
    {
        println!(
            "[rust] innput file: {}\nvocab_size: {}\nspecial_tokens: {:?}",
            input_path, vocab_size, special_tokens
        );
    }
    let _start = std::time::Instant::now();
    let file_bytes: Vec<u8> = fs::read(input_path).expect("can not read file from input_path");
    let content_str = String::from_utf8_lossy(&file_bytes).replace('\u{FFFD}', "");

    let mut vocab: HashMap<usize, Vec<u8>> = HashMap::new();

    for idx in 0..(256 + special_tokens.len()) {
        if idx < 256 {
            vocab.insert(idx, vec![idx as u8]);
        } else {
            vocab.insert(idx, special_tokens[idx - 256].clone().into_bytes());
        }
    }
    let mut special_token_map: HashMap<&str, usize> = HashMap::new();
    for (i, st) in special_tokens.iter().enumerate() {
        let token_str = st;
        special_token_map.insert(token_str, 256 + i);
    }

    let escaped_special_tokens: Vec<String> = special_tokens
        .iter()
        .map(|s: &String| fancy_regex::escape(s).into_owned())
        .collect();
    let special_token_pattern: String = escaped_special_tokens.join("|");

    let base_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let final_pattern: String = if special_tokens.is_empty() {
        base_pattern.to_string()
    } else {
        format!("(?:{})|{}", special_token_pattern, base_pattern)
    };
    let re = Regex::new(&final_pattern)
        .map_err(|e| PyValueError::new_err(format!("cannot convert regex in bpe.rs: {}", e)))?;

    let content_split: Vec<&str> = re
        .find_iter(&content_str)
        .map(|m| m.unwrap().as_str())
        .collect();

    // let mut sequneces: Vec<Vec<usize>> = Vec::with_capacity(content_split.len());
    let mut sequences_counting: HashMap<Vec<usize>, usize> = HashMap::new();

    for chunk in content_split {
        if let Some(&token_id) = special_token_map.get(chunk) {
            // sequneces.push(vec![token_id]);
            sequences_counting.insert(vec![token_id], 1);
        } else {
            let byte_seq: Vec<usize> = chunk.as_bytes().iter().map(|&c| c as usize).collect();
            // sequneces.push(byte_seq.clone());
            sequences_counting.entry(byte_seq).and_modify(|c| *c += 1).or_insert(1);
        }
    }

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut next_id: usize = 256 + special_tokens.len();

    while vocab.len() < vocab_size {
        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for (sequence, count) in &sequences_counting {
            if sequence.len() < 2 {
                continue;
            }
            for window in sequence.windows(2) {
                let pair: (usize, usize) = (window[0], window[1]);
                pair_counts.entry(pair).and_modify(|p| *p += count).or_insert(*count);
            }
        }
        if pair_counts.is_empty() {
            break;
        }
        let (&best_pair, &best_count) = pair_counts
            .iter()
            .max_by(
                |a: &(&(usize, usize), &usize), b: &(&(usize, usize), &usize)| {
                    let count_cmp = a.1.cmp(b.1);
                    if count_cmp == std::cmp::Ordering::Equal {
                        let a0_bytes = vocab.get(&a.0.0).unwrap();
                        let a1_bytes = vocab.get(&a.0.1).unwrap();
                        let b0_bytes = vocab.get(&b.0.0).unwrap();
                        let b1_bytes = vocab.get(&b.0.1).unwrap();
                        let tuple_a = (a0_bytes, a1_bytes);
                        let tuple_b = (b0_bytes, b1_bytes);
                        tuple_a.cmp(&tuple_b)
                    } else {
                        count_cmp
                    }
                },
            )
            .unwrap();

        if best_count <= 0 {
            break;
        }

        let bytes_left: Vec<u8> = vocab.get(&best_pair.0).unwrap().clone();
        let bytes_right: Vec<u8> = vocab.get(&best_pair.1).unwrap().clone();

        let mut new_bytes: Vec<u8> = Vec::with_capacity(bytes_left.len() + bytes_right.len());
        new_bytes.extend_from_slice(&bytes_left);
        new_bytes.extend_from_slice(&bytes_right);

        vocab.insert(next_id, new_bytes);
        merges.push((bytes_left, bytes_right));

        let mut sequences_counting_new: HashMap<Vec<usize>, usize> = HashMap::new();

        for (seqence, count) in &mut sequences_counting {
            let mut new_seqence: Vec<usize> = Vec::with_capacity(seqence.len());
            let mut i: usize = 0;
            while i < seqence.len() {
                if i < seqence.len() - 1
                    && seqence[i] == best_pair.0
                    && seqence[i + 1] == best_pair.1
                {
                    new_seqence.push(next_id);
                    i += 2;
                } else {
                    new_seqence.push(seqence[i]);
                    i += 1;
                }
            }
            sequences_counting_new.entry(new_seqence).and_modify(|c| *c += *count).or_insert(*count);
        }
        sequences_counting = sequences_counting_new;
        next_id += 1;
    }
    #[cfg(debug_assertions)]
    {
        println!("[rust] vocab.len: {}", vocab.len());
        let duration = _start.elapsed();
        println!("[rust] bpe_train duration: {:?}", duration);
    }
    Ok((vocab, merges))
}
