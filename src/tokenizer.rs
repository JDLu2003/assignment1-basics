use core::panic;
use std::{collections::HashMap, vec};

use aho_corasick::AhoCorasick;
use fancy_regex::Regex;
use pyo3::{exceptions::PyValueError, prelude::*};

macro_rules! println_special {
    ($($arg:tt)*) => {
        println!("###########");
        println!($($arg)*);
        println!("###########");
    };
}
#[pyclass]
pub struct BPETokenizer {
    vocab: HashMap<i32, Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
    special_tokens: Vec<String>,
}

#[pymethods]
impl BPETokenizer {
    #[new]
    #[pyo3(signature = (vocab, merges, special_tokens=None))]
    fn new(
        vocab: HashMap<i32, Vec<u8>>,
        merges: Vec<(Vec<u8>, Vec<u8>)>,
        special_tokens: Option<Vec<String>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            print!("\n===============================================\nArg:");
            println!("    vocab:");
            for (idx, elem) in vocab.iter().enumerate() {
                println!("        {:?}", elem);
                if idx > 5 {
                    break;
                }
            }
            println!("    merges:");
            for (idx, elem) in merges.iter().enumerate() {
                println!("        {:?}", elem);
                if idx > 5 {
                    break;
                }
            }
            println!("    special_tokens:");
            for (idx, elem) in special_tokens
                .clone()
                .unwrap_or_default()
                .iter()
                .enumerate()
            {
                println!("        {:?}", elem);
                if idx > 5 {
                    break;
                }
            }
        }
        Self {
            vocab,
            merges,
            special_tokens: special_tokens.unwrap_or_default(),
        }
    }
    fn encode(&self, text: &str) -> PyResult<Vec<i32>> {
        let content_str = String::from_utf8_lossy(text.as_bytes()).replace('\u{FFFD}', "");
        let ac = AhoCorasick::new(self.special_tokens.clone()).unwrap();
        let chunks = build_chunks(&content_str, &ac);

        let mut special_token_map_reverse: HashMap<Vec<u8>, i32> = HashMap::new();
        for st in &self.special_tokens {
            let (key, value) = self
                .vocab
                .iter()
                .find(|(_, v)| *v == &st.as_bytes())
                .expect(format!("cannot find special_token in vocab: {}", st).as_str());
            special_token_map_reverse.insert(value.to_vec(), *key);
        }
        let mut vocab_reverse: HashMap<Vec<u8>, i32> = HashMap::new();
        for (key, value) in &self.vocab {
            vocab_reverse.insert(value.to_vec(), *key);
        }

        let mut merge_map: HashMap<(Vec<u8>, Vec<u8>), usize> = HashMap::new();
        for (idx, pair) in self.merges.iter().enumerate() {
            merge_map.insert(pair.clone(), idx);
        }

        let mut res_buckets: Vec<Vec<i32>> = Vec::new();

        let convert_text = |chunk: &str| -> Result<Vec<i32>, fancy_regex::Error> {
            if chunk.is_empty() {
                return Ok(Vec::new());
            }
            let mut chunk_i32: Vec<i32> = Vec::new();
            let pattern =
                r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
            let re = Regex::new(pattern)?;
            for word in re.find_iter(chunk) {
                let word = word.unwrap().as_str().as_bytes().to_vec();
                let mut seq: Vec<Vec<u8>> = word.iter().map(|&b| vec![b]).collect();
                loop {
                    let best_pair_idx = seq
                        .windows(2)
                        .enumerate()
                        .filter_map(|(i, w)| {
                            merge_map
                                .get(&(w[0].clone(), w[1].clone()))
                                .map(|&rank| (i, rank))
                        })
                        .min_by_key(|&(_, rank)| rank)
                        .map(|(i, _)| i);

                    if let Some(best_i) = best_pair_idx {
                        let mut new_seq = Vec::with_capacity(seq.len() - 1);
                        let mut i = 0;
                        while i < seq.len() {
                            if i == best_i {
                                let mut merged = seq[i].clone();
                                merged.extend(&seq[i + 1]);
                                new_seq.push(merged);
                                i += 2;
                            } else {
                                new_seq.push(seq[i].clone());
                                i += 1;
                            }
                        }
                        seq = new_seq;
                    } else {
                        break;
                    }
                }
                for token_byte in seq {
                    let token_id = vocab_reverse
                        .get(&token_byte)
                        .expect("Token not found in vocab!");
                    chunk_i32.push(*token_id);
                }
            }

            Ok(chunk_i32)
        };

        for (idx, chunk) in chunks.iter().enumerate() {
            if idx % 2 == 0 {
                // 偶数是普通的文本
                let chunk_i32 = convert_text(chunk).map_err(|_| PyValueError::new_err(""))?;
                res_buckets.push(chunk_i32);
            } else {
                // 奇数是 special token
                let new_chunk: Vec<u8> = Vec::with_capacity(1);
                let token = special_token_map_reverse.get(chunk.as_bytes()).expect(
                    format!("cannot find special token in hashmap: {:?}", new_chunk).as_str(),
                );
                res_buckets.push(vec![*token]);
            }
        }
        let res = res_buckets.concat();
        #[cfg(debug_assertions)]
        {
            println!("input is:");
            println_special!("{}", text);
            println!("ouput is:");
            println_special!("{:?}", res);
            println!("decode is:");
            println_special!(
                "{}",
                self.decode(res.clone()).expect("error to decode in debug")
            );
        }
        Ok(res)
    }
    fn decode(&self, ids: Vec<i32>) -> PyResult<String> {
        let mut bytes_str: Vec<u8> = Vec::new();
        for id in ids {
            if let Some(bytes) = self.vocab.get(&id) {
                bytes_str.extend(bytes);
            } else {
                panic!("cannot find token ids in vocab: {}", id);
            }
        }

        let content_str = String::from_utf8_lossy(&bytes_str).replace('\u{FFFD}', "");
        Ok(content_str)
    }
    fn encode_iterable(&self, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<i32>> {
        let mut all_ids: Vec<i32> = Vec::new();
        for item in iterable.try_iter()? {
            let item = item?;
            let text: &str = item.extract()?;
            all_ids.extend(
                self.encode(text)
                    .expect(format!("cannot convert text:{}", text).as_str()),
            );
        }
        Ok(all_ids)
    }
}
/// 生成 chunks
/// 对于 <special_token> test1 <special_token> text2 <special_token>
/// 生成 "", "<special_token>", "test1", "<special_token>", "text2", "<special_token>", ""
/// 偶数是普通的文本
fn build_chunks<'a>(content_str: &'a str, ac: &'a AhoCorasick) -> Vec<&'a str> {
    let mut chunks: Vec<&'a str> = Vec::new();

    let mut last_end = 0;
    for mat in ac.find_iter(content_str) {
        let start = mat.start();
        chunks.push(&content_str[last_end..start]);

        chunks.push(&content_str[start..mat.end()]);
        last_end = mat.end();
    }
    chunks.push(&content_str[last_end..]);
    chunks
}
