use aho_corasick::AhoCorasick;
use fancy_regex::Regex;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;

use pyo3::{exceptions::PyValueError, prelude::*};
use std::collections::HashMap;

type Vocab = HashMap<usize, Vec<u8>>;
type Merges = Vec<(Vec<u8>, Vec<u8>)>;

#[pyfunction]
pub fn train_bpe(
    input_path: String,
    vocab_size: usize,
    special_tokens: Vec<String>,
) -> PyResult<(Vocab, Merges)> {
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

    let base_pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    let re = Regex::new(base_pattern)
        .map_err(|e| PyValueError::new_err(format!("cannot convert regex in bpe.rs: {}", e)))?;

    let ac = AhoCorasick::new(special_tokens.clone()).unwrap();
    let spans: Vec<&str> = remove_special_tokens(&content_str, &ac);

    let mut sequences_counting: HashMap<Vec<usize>, usize> = HashMap::new();

    for sp in spans {
        if sp.is_empty() {
            continue;
        }
        for m in re.find_iter(sp) {
            let piece = m.unwrap().as_str();
            if piece.is_empty() {
                continue;
            }
            let byte_seq: Vec<usize> = piece.as_bytes().iter().map(|&c| c as usize).collect();
            sequences_counting
                .entry(byte_seq)
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }
    }

    let mut merges: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut next_id: usize = 256 + special_tokens.len();

    let pb: ProgressBar = ProgressBar::new(vocab_size as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {per_sec} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    while vocab.len() < vocab_size {
        Python::with_gil(|py| py.check_signals().is_err().then(|| panic!("kill by shell")));
        pb.set_position(vocab.len() as u64);
        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();

        for (sequence, count) in &sequences_counting {
            if sequence.len() < 2 {
                continue;
            }
            for window in sequence.windows(2) {
                let pair: (usize, usize) = (window[0], window[1]);
                pair_counts
                    .entry(pair)
                    .and_modify(|p| *p += count)
                    .or_insert(*count);
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

        if best_count == 0 {
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
            sequences_counting_new
                .entry(new_seqence)
                .and_modify(|c| *c += *count)
                .or_insert(*count);
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
    pb.finish_with_message("bpe train finished");
    Ok((vocab, merges))
}

fn remove_special_tokens<'a>(content_str: &'a str, ac: &AhoCorasick) -> Vec<&'a str> {
    let mut spans: Vec<&'a str> = Vec::new();
    let mut last_end: usize = 0;

    for mat in ac.find_iter(content_str) {
        let start: usize = mat.start();
        let end: usize = mat.end();

        if start > last_end {
            spans.push(&content_str[last_end..start]);
        }
        // 如果需要保留特殊 token 本身，可以在这里处理
        last_end = end;
    }

    if last_end < content_str.len() {
        spans.push(&content_str[last_end..]);
    }

    spans
}
