use ahash::AHashMap;
use smallvec::SmallVec;
use indexmap::IndexMap;
use crate::decoder::TurkishDecoder;

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum TokenType {
    ROOT,
    SUFFIX,
    BPE,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Token {
    pub token: String,
    pub id: i32,
    pub token_type: TokenType,
}

/// Trie node for fast prefix matching - using array for common ASCII chars
struct TrieNode {
    children: AHashMap<char, Box<TrieNode>>,
    value: Option<(i32, usize)>, // (id, char_count)
}

impl TrieNode {
    #[inline]
    fn new() -> Self {
        Self {
            children: AHashMap::with_capacity(4),
            value: None,
        }
    }
    
    #[inline]
    fn get_child(&self, c: char) -> Option<&TrieNode> {
        self.children.get(&c).map(|b| b.as_ref())
    }
}

/// Trie for O(n) prefix matching
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    fn insert(&mut self, key: &str, value: i32) {
        let mut node = &mut self.root;
        let char_count = key.chars().count();
        for c in key.chars() {
            node = node.children.entry(c).or_insert_with(|| Box::new(TrieNode::new()));
        }
        node.value = Some((value, char_count));
    }

    /// Find all prefix matches - returns in descending length order
    #[inline(always)]
    fn find_all_prefixes<'a>(&self, s: &'a str) -> SmallVec<[(i32, usize, usize); 8]> {
        // Returns (id, byte_len, char_count)
        let mut matches = SmallVec::new();
        let mut node = &self.root;

        for (i, c) in s.char_indices() {
            if let Some(child) = node.get_child(c) {
                node = child;
                if let Some((id, char_count)) = node.value {
                    matches.push((id, i + c.len_utf8(), char_count));
                }
            } else {
                break;
            }
        }
        matches.reverse();
        matches
    }

    /// Find longest prefix match only
    #[inline(always)]
    fn find_longest_prefix_info(&self, s: &str) -> Option<(i32, usize, usize)> {
        let mut node = &self.root;
        let mut last_match: Option<(i32, usize, usize)> = None;
        for (i, c) in s.char_indices() {
            if let Some(child) = node.get_child(c) {
                node = child;
                // byte_pos not needed unless verified outside
                if let Some((id, char_count)) = node.value {
                    last_match = Some((id, i + c.len_utf8(), char_count));
                }
            } else {
                break;
            }
        }
        last_match
    }
}

pub struct TurkishTokenizer {
    roots_trie: Trie,
    suffixes_trie: Trie,
    bpe_trie: Trie,
    reverse_dict: AHashMap<i32, Vec<String>>,
    token_types: AHashMap<i32, TokenType>,
    decoder: TurkishDecoder,
    
    // Special tokens
    uppercase_id: i32,
    unknown_id: i32,
    space_id: i32,
    #[allow(dead_code)]
    pad_id: i32,
    #[allow(dead_code)]
    eos_id: i32,
}

impl TurkishTokenizer {
    pub fn from_files(
        roots_json: &str,
        ekler_json: &str,
        bpe_json: &str,
    ) -> Result<Self, serde_json::Error> {
        let roots: IndexMap<String, i32> = serde_json::from_str(roots_json)?;
        let suffixes: IndexMap<String, i32> = serde_json::from_str(ekler_json)?;
        let bpe_tokens: IndexMap<String, i32> = serde_json::from_str(bpe_json)?;
        
        // Build Tries
        let mut roots_trie = Trie::new();
        let mut suffixes_trie = Trie::new();
        let mut bpe_trie = Trie::new();
        
        for (k, &v) in &roots {
            roots_trie.insert(k, v);
        }
        for (k, &v) in &suffixes {
            suffixes_trie.insert(k, v);
        }
        for (k, &v) in &bpe_tokens {
            bpe_trie.insert(k, v);
        }

        // Build reverse dict with AHashMap and token types
        let mut reverse_dict: AHashMap<i32, Vec<String>> = AHashMap::new();
        let mut token_types: AHashMap<i32, TokenType> = AHashMap::new();

        let mut insert_rev = |map: &IndexMap<String, i32>, t_type: TokenType| {
            for (k, &v) in map {
                reverse_dict.entry(v).or_default().push(k.clone());
                token_types.insert(v, t_type);
            }
        };
        insert_rev(&roots, TokenType::ROOT);
        insert_rev(&suffixes, TokenType::SUFFIX);
        insert_rev(&bpe_tokens, TokenType::BPE);
        
        // Convert to std HashMap for decoder
        let std_reverse: std::collections::HashMap<i32, Vec<String>> = 
            reverse_dict.iter().map(|(&k, v)| (k, v.clone())).collect();
        let decoder = TurkishDecoder::new(std_reverse);
        
        let uppercase_id = *roots.get("<uppercase>").unwrap_or(&0);
        let unknown_id = *roots.get("<unknown>").unwrap_or(&1);
        let space_id = *roots.get(" ").unwrap_or(&2);
        let pad_id = *roots.get("<pad>").unwrap_or(&0);
        let eos_id = *roots.get("<eos>").unwrap_or(&0);

        Ok(Self {
            roots_trie,
            suffixes_trie,
            bpe_trie,
            reverse_dict,
            token_types,
            decoder,
            uppercase_id,
            unknown_id,
            space_id,
            pad_id,
            eos_id,
        })
    }
    
    /// Fast Turkish lowercase - inlined for performance
    #[inline(always)]
    fn tr_lower_char(c: char) -> char {
        match c {
            'İ' => 'i',
            'I' => 'ı',
            'A'..='Z' => ((c as u8) + 32) as char,
            'Ç' => 'ç',
            'Ğ' => 'ğ',
            'Ö' => 'ö',
            'Ş' => 'ş',
            'Ü' => 'ü',
            _ => c,
        }
    }
    
    /// Optimized tokenize for a segment (already lowercased)
    #[inline]
    fn tokenize_segment_fast(&self, s: &str, result: &mut SmallVec<[i32; 64]>) {
        let s_len = s.len();
        let mut pos_byte = 0;
        
        while pos_byte < s_len {
            let substr = &s[pos_byte..];
            
            // Try each trie and collect matches
            let r_matches = self.roots_trie.find_all_prefixes(substr);
            let b_matches = self.bpe_trie.find_all_prefixes(substr);
            let s_matches = self.suffixes_trie.find_all_prefixes(substr);
            
            let mut best_score = 0usize;
            let mut best_priority = 3i32;
            let mut best_byte_len = 0usize;
            let mut best_id = self.unknown_id;
            
            // Score Roots (priority 0 - highest)
            for &(id, byte_len, char_count) in r_matches.iter() {
                let mut score = char_count;
                if byte_len == substr.len() {
                    score += 5; // Full match bonus
                } else {
                    // Suffix lookahead bonus
                    let remainder = &substr[byte_len..];
                    if let Some((_, _, next_char_count)) = self.suffixes_trie.find_longest_prefix_info(remainder) {
                         if next_char_count >= 2 {
                             score += next_char_count;
                         }
                    }
                }

                if score > best_score || (score == best_score && 0 < best_priority) {
                    best_score = score;
                    best_priority = 0;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            // Score BPEs (priority 2 - was 1)
            for &(id, byte_len, char_count) in b_matches.iter() {
                let mut score = char_count;
                
                if byte_len == substr.len() {
                    score += 5; // Full match bonus
                } else {
                    // Suffix lookahead bonus
                    let remainder = &substr[byte_len..];
                    if let Some((_, _, next_char_count)) = self.suffixes_trie.find_longest_prefix_info(remainder) {
                        if next_char_count >= 2 {
                            score += next_char_count;
                        }
                    }
                }
                
                if score > best_score || (score == best_score && 2 < best_priority) {
                    best_score = score;
                    best_priority = 2;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            // Score Suffixes (priority 1 - was 2)
            for &(id, byte_len, char_count) in s_matches.iter() {
                let mut score = char_count;
                
                if byte_len == substr.len() {
                    score += 5; // Full match bonus
                } else {
                     // Suffix lookahead bonus
                    let remainder = &substr[byte_len..];
                    if let Some((_, _, next_char_count)) = self.suffixes_trie.find_longest_prefix_info(remainder) {
                        if next_char_count >= 2 {
                            score += next_char_count;
                        }
                    }
                }
                
                // Priority 1 is BETTER than 2, so we check if 1 < best_priority
                if score > best_score || (score == best_score && 1 < best_priority) {
                    best_score = score;
                    best_priority = 1;
                    best_byte_len = byte_len;
                    best_id = id;
                }
            }
            
            if best_priority == 3 {
                // No match found - emit unknown and skip one char
                result.push(self.unknown_id);
                if let Some(c) = substr.chars().next() { 
                    pos_byte += c.len_utf8();
                    // If no progress possible (EOS or weird char), break to avoid infinite loop
                    if pos_byte >= substr.len() { break; } 
                } else {
                    break; 
                }
            } else {
                result.push(best_id);
                pos_byte += best_byte_len;
            }
        }
    }
    
    /// Fast encode with minimal allocations
    /// Fast encode with minimal allocations
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let estimated_tokens = text.len() / 4;
        let mut all_ids: SmallVec<[i32; 64]> = SmallVec::with_capacity(estimated_tokens.min(64));
        
        // Reusable buffer for lowercased text
        let mut lower_buf = String::with_capacity(64);
        
        let parts: Vec<&str> = text.split_whitespace().collect();
        
        for (_i, part) in parts.iter().enumerate() {
            let part_trimmed: &str = part; 
            
            // CamelCase splitting logic matching Python's _camel_split_with_positions EXACTLY
            // Python: split on ANY uppercase char at i > 0
            let char_indices: Vec<(usize, char)> = part_trimmed.char_indices().collect();
            let count = char_indices.len();
            
            let mut start_idx = 0;
            let mut segments = Vec::new();

            for i in 1..count {
                let (_, c) = char_indices[i];
                if c.is_uppercase() {
                     let (byte_start, c_start) = char_indices[start_idx];
                     let (byte_end, _) = char_indices[i];
                     let segment_text = &part_trimmed[byte_start..byte_end];
                     let mut lower = String::with_capacity(segment_text.len());
                     for ch in segment_text.chars() { 
                         lower.push(Self::tr_lower_char(ch)); 
                     }
                     // Check if c_start is uppercase. 
                     // In Python loop, start is updated. The segment is word[start:i].
                     // Python's _tr_lower handles capitalization.
                     // IMPORTANT: We need to know if the segment SHOULD have Upper marker.
                     // In Python: NO special check for segment casing, BUT segments are formed from original text.
                     // The `starts_upper` flag in our Rust code expects us to tell it.
                     // If original segment started with Uppercase, we pass true.
                     segments.push((lower, c_start.is_uppercase()));
                     start_idx = i;
                }
            }
            
            if start_idx < count {
                let (byte_start, c_start) = char_indices[start_idx];
                let segment_text = &part_trimmed[byte_start..];
                let mut lower = String::with_capacity(segment_text.len());
                for ch in segment_text.chars() { 
                    lower.push(Self::tr_lower_char(ch)); 
                }
                segments.push((lower, c_start.is_uppercase()));
            }

            // Process segments
            for (idx, (seg, starts_upper)) in segments.into_iter().enumerate() {
                 let mut temp_ids: SmallVec<[i32; 64]> = SmallVec::new();
                 
                 lower_buf.clear();
                 if idx == 0 {
                     lower_buf.push(' ');
                 }
                 lower_buf.push_str(&seg);
                 
                 if !lower_buf.is_empty() {
                     self.tokenize_segment_fast(&lower_buf, &mut temp_ids);
                 }
                 
                 if idx == 0 {
                     // Handle Space Absorption Logic for First Segment
                     if !temp_ids.is_empty() {
                         if temp_ids[0] == self.space_id {
                             // Space was tokenized separately (e.g. " " + "Instagram")
                             all_ids.push(self.space_id);
                             if starts_upper {
                                 all_ids.push(self.uppercase_id);
                             }
                             // Append the rest
                             for k in 1..temp_ids.len() {
                                 all_ids.push(temp_ids[k]);
                             }
                         } else {
                             // Space was absorbed (e.g. " tetkik")
                             if starts_upper {
                                 all_ids.push(self.uppercase_id);
                             }
                             // Append all (including the absorbed space token)
                             all_ids.extend(temp_ids);
                         }
                     }
                 } else {
                     // Subsequent segments (No leading space prepended)
                     if starts_upper {
                         all_ids.push(self.uppercase_id);
                     }
                     all_ids.extend(temp_ids);
                 }
            }
        }
        
        // Space removal cleaning logic (matching Python tokenize_text)
        let mut final_ids: Vec<i32> = Vec::with_capacity(all_ids.len());
        
        for i in 0..all_ids.len() {
            let id = all_ids[i];
            
            // Logic 1: Remove space between Uppercase and Suffix (token > 19999)
            // Python: not (0<=id<=19999) and prev==Uppercase and prevprev==Space
            if id > 19999 {
                if final_ids.len() >= 2 {
                    let last = final_ids[final_ids.len()-1];
                    let second_last = final_ids[final_ids.len()-2];
                    if last == self.space_id && second_last == self.uppercase_id {
                        final_ids.pop(); 
                    }
                }
            }
            
            // Logic 2: Remove space before Uppercase if next token starts with space
            if id == self.uppercase_id && !final_ids.is_empty() {
                if let Some(&last_id) = final_ids.last() {
                    if last_id == self.space_id {
                        // Check lookahead at i+1
                        if i + 1 < all_ids.len() {
                            let next_id = all_ids[i+1];
                            if let Some(strs) = self.reverse_dict.get(&next_id) {
                                if !strs.is_empty() && strs[0].starts_with(' ') {
                                    final_ids.pop();
                                }
                            }
                        }
                    }
                }
            }
            
            final_ids.push(id);
        }
        
        final_ids
    }

pub fn tokenize_text(&self, text: &str) -> Vec<Token> {
        let ids = self.encode(text);
        ids.iter().map(|&id| {
            let token_str = if let Some(strs) = self.reverse_dict.get(&id) {
                if !strs.is_empty() { strs[0].clone() } else { format!("<id:{}>", id) }
            } else {
                format!("<id:{}>", id)
            };
            
            let token_type = if let Some(&t_type) = self.token_types.get(&id) {
                t_type
            } else {
                 if id == self.uppercase_id {
                     TokenType::ROOT // Uppercase marker
                 } else if id <= 20071 && id >= 20000 {
                     TokenType::SUFFIX // Fallback for suffix range if missing
                 } else if id < 20000 {
                     TokenType::ROOT // Fallback
                 } else {
                     TokenType::BPE // Fallback
                 }
            };
            
            Token { token: token_str, id, token_type }
        }).collect()
    }
    
    pub fn decode(&self, ids: Vec<i32>) -> String {
        self.decoder.decode(ids)
    }
}
