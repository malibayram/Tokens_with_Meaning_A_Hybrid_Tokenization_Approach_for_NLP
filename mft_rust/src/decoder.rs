use std::collections::HashMap;

pub struct TurkishDecoder {
    reverse_dict: HashMap<i32, Vec<String>>,
}

// Constants for vowel/consonant sets
const ALL_VOWELS: &str = "aeıioöuüâ";
const INCE_VOWELS: &str = "eiöü";
const AI_VOWELS: &str = "aıâ";
const EI_VOWELS: &str = "ei";
const OU_VOWELS: &str = "ou";
const HARD_CONSONANTS: &str = "fstkçşhp";
const WHITESPACE: &str = " \n\t";

impl TurkishDecoder {
    pub fn new(reverse_dict: HashMap<i32, Vec<String>>) -> Self {
        Self { reverse_dict }
    }

    fn has_vowel(s: &str) -> bool {
        s.chars().any(|c| ALL_VOWELS.contains(c))
    }

    fn starts_with_vowel(&self, word: &str) -> bool {
        word.chars().next().map_or(false, |c| ALL_VOWELS.contains(c))
    }

    fn ends_with_vowel(&self, word: &str) -> bool {
        word.chars().last().map_or(false, |c| ALL_VOWELS.contains(c))
    }

    fn ends_with_any(&self, word: &str, charset: &str) -> bool {
        for c in word.chars().rev() {
            if charset.contains(c) {
                return true;
            }
            if ALL_VOWELS.contains(c) {
                return false;
            }
        }
        false
    }
    
    fn ends_with_ince(&self, word: &str) -> bool {
        match word {
            "saat" | "kilovatsaat" | "ziraat" | "itaat" | "istikbal" => true,
            _ => self.ends_with_any(word, INCE_VOWELS),
        }
    }

    fn ends_with_sert_unsuz(&self, word: &str) -> bool {
        word.chars().last().map_or(false, |c| HARD_CONSONANTS.contains(c))
    }
    
    fn get_vowel_suffix_index(&self, prev_token: &str) -> usize {
        if self.ends_with_any(prev_token, AI_VOWELS) {
            0
        } else if self.ends_with_any(prev_token, EI_VOWELS) {
            1
        } else if self.ends_with_any(prev_token, OU_VOWELS) {
            2
        } else {
            3
        }
    }

    fn handle_la_le_suffix(&self, prev_token: &str, suffixes: &[String], end_of_word: bool) -> String {
        if self.ends_with_vowel(prev_token) && end_of_word {
            if self.ends_with_ince(prev_token) {
                suffixes[3].clone() // yle
            } else {
                suffixes[2].clone() // yla
            }
        } else {
            if self.ends_with_ince(prev_token) {
                suffixes[1].clone() // le
            } else {
                suffixes[0].clone() // la
            }
        }
    }

    fn handle_da_de_suffix(&self, prev_token: &str, suffixes: &[String]) -> String {
        if self.ends_with_sert_unsuz(prev_token) {
            if self.ends_with_ince(prev_token) {
                suffixes[3].clone() // te
            } else {
                suffixes[2].clone() // ta
            }
        } else {
            if self.ends_with_ince(prev_token) {
                suffixes[1].clone() // de
            } else {
                suffixes[0].clone() // da
            }
        }
    }

    fn handle_di_du_suffix(&self, prev_token: &str, suffixes: &[String]) -> String {
        let base_index = self.get_vowel_suffix_index(prev_token);
        if self.ends_with_sert_unsuz(prev_token) {
            suffixes[base_index + 4].clone()
        } else {
            suffixes[base_index].clone()
        }
    }
    
    fn handle_lik_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
          if i >= ids.len() - 1 {
              return suffixes[0].clone();
          }
          
          let next_token = &self.reverse_dict[&ids[i + 1]][0];
          let base_index = self.get_vowel_suffix_index(prev_token);
          
          if self.starts_with_vowel(next_token) {
              suffixes[base_index + 4].clone()
          } else {
              suffixes[base_index].clone()
          }
    }

    fn handle_cik_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        if i >= ids.len() - 1 {
            return suffixes[0].clone();
        }
        
        let next_token = &self.reverse_dict[&ids[i + 1]][0];
        let base_index = self.get_vowel_suffix_index(prev_token);
        
        let offset = if self.starts_with_vowel(next_token) {
            if self.ends_with_sert_unsuz(prev_token) { 12 } else { 8 }
        } else {
            if self.ends_with_sert_unsuz(prev_token) { 4 } else { 0 }
        };
        
        suffixes[base_index + offset].clone()
    }
    
    fn handle_mak_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        if i >= ids.len() - 1 {
            return suffixes[0].clone();
        }
        
        let next_token = &self.reverse_dict[&ids[i + 1]][0];
        let base_index = if self.ends_with_ince(prev_token) { 1 } else { 0 };
        
        if self.starts_with_vowel(next_token) {
            suffixes[base_index + 2].clone()
        } else {
            suffixes[base_index].clone()
        }
    }
    
    fn handle_acak_suffix(&self, i: usize, ids: &[i32], prev_token: &str, suffixes: &[String]) -> String {
        let is_vowel_ending = self.ends_with_vowel(prev_token);
        let is_ince = self.ends_with_ince(prev_token);
        
        let is_vowel_starting = if i < ids.len() - 1 {
             let next_token = &self.reverse_dict[&ids[i + 1]][0];
             self.starts_with_vowel(next_token)
        } else {
             false
        };
        
        if is_vowel_starting {
            if is_vowel_ending {
                suffixes[if is_ince { 7 } else { 6 }].clone()
            } else {
                 suffixes[if is_ince { 3 } else { 2 }].clone()
            }
        } else {
            if is_vowel_ending {
                 suffixes[if is_ince { 5 } else { 4 }].clone()
            } else {
                 suffixes[if is_ince { 1 } else { 0 }].clone()
            }
        }
    }
    
    fn select_correct_suffix(&self, i: usize, ids: &[i32], prev_token: &str) -> String {
        let token_id = ids[i];
        let suffixes = &self.reverse_dict[&token_id];
        
        if token_id < 20013 {
             if self.ends_with_ince(prev_token) { suffixes[1].clone() } else { suffixes[0].clone() }
        } else if token_id < 20023 {
             suffixes[self.get_vowel_suffix_index(prev_token)].clone()
        } else if token_id == 20023 { // la, le
             let mut end_of_word = true;
             if i < ids.len() - 1 {
                 let _next_token = &self.reverse_dict[&ids[i + 1]][0];
                 if !WHITESPACE.contains(_next_token.chars().next().unwrap_or(' ')) {
                     end_of_word = false;
                 }
             }
             self.handle_la_le_suffix(prev_token, suffixes, end_of_word)
        } else if token_id <= 20025 { // da, de, tan...
             self.handle_da_de_suffix(prev_token, suffixes)
        } else if token_id < 20029 { // di, du...
             self.handle_di_du_suffix(prev_token, suffixes)
        } else if token_id == 20029 { // lik
             self.handle_lik_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20030 { // cik
             self.handle_cik_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20031 { // mak
             self.handle_mak_suffix(i, ids, prev_token, suffixes)
        } else if token_id == 20032 { // acak
             self.handle_acak_suffix(i, ids, prev_token, suffixes)
        } else {
             suffixes[0].clone()
        }
    }

    fn select_correct_root(&self, i: usize, ids: &[i32]) -> String {
        let token_id = ids[i];
        let tokens = &self.reverse_dict[&token_id];
        
        if i >= ids.len() - 1 {
            return tokens[0].clone();
        }
        
        let _next_token = &self.reverse_dict[&ids[i + 1]][0];
        
        // === EXCEPTIONS: Roots that should NOT soften ===
        // 204 (hayat), 220 (belirt), 298 (meslek)
        if token_id == 204 || token_id == 220 || token_id == 298 {
             return tokens[0].clone();
        }

        // Special case: üçlü (2227) - always return üçlü (variant 1) unless specific context
        if token_id == 2227 {
             if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
        }

        // Akış (aka/akı) Exception (2199) - Default to "akı" (variant 1) 
        if token_id == 2199 {
            if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 let next_str = &self.reverse_dict[&next_id][0];
                 // Use "aka" only when followed by vowel-starting suffixes like "acak"
                 if next_str.starts_with('a') || next_str.starts_with('e') {
                      return tokens[0].clone(); // "aka" for "akacak"
                 }
            }
            // Default to "akı"
            if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
        }

        // Ata/Atı Exception (2212) - for "atılırsa", "atılmak", "atıyorlar" etc.
        // Use "atı" (variant 1) when followed by 'l' (passive) or 'y' (yor, yacak)
        if token_id == 2212 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 let next_str = &self.reverse_dict[&next_id][0];
                 if next_str.trim().starts_with('l') || next_str.trim().starts_with('y') {
                      return tokens[1].clone(); // "atı" + "lırsa" = "atılırsa"
                 }
            }
            return tokens[0].clone(); // "ata" by default
        }
        
        // Yaşına (yaşa/yaşı) Exception (2209)
        if token_id == 2209 {
             if i < ids.len() - 1 {
                 // 20188 = 'na'
                 if ids[i+1] == 20188 {
                      if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
                 }
             }
             return tokens[0].clone();
        }
        
        // Alın (alın/aln) Exception (182) - Default to "alın" (variant 0)
        if token_id == 182 {
             if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 // Only drop vowel for simple possessive suffixes (ı, i, u, ü)
                 if next_id == 20034 || next_id == 20033 || next_id == 20035 || next_id == 20036 {
                      if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
                 }
             }
             return tokens[0].clone();
        }

        // Ilim/Ilm Exception (166) - Default to "ilim" (variant 0)
        if token_id == 166 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                let next_id = ids[i+1];
                // Only use "ilm" for possessive/buffer case (ilmi, ilme) id 20033 ('i'), 20038 ('e')
                if next_id == 20033 || next_id == 20038 {
                     return tokens[1].clone(); // "ilm" + i = "ilmi"
                }
            }
            return tokens[0].clone(); // Default to "ilim"
        }

        // Boya/Boyu Exception (2220) - "boya" (paint) vs "boyu" (height)
        // Use "boyu" (variant 1) by default
        if token_id == 2220 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                let next_id = ids[i+1];
                let next_str = &self.reverse_dict[&next_id][0];
                // Use "boya" only when followed by actual suffix tokens starting with 'n', 'm', 'l', 'd'
                if next_id >= 20000 && !next_str.trim().is_empty() {
                    let first_char = next_str.trim().chars().next().unwrap();
                    if "nmld".contains(first_char) {
                        return tokens[0].clone(); // "boya"
                    }
                }
            }
            if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); } // "boyu" by default
        }

        // Bile/Bili Exception (2307) - for "bilir", "biliyor" vs "biler"
        if token_id == 2307 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                let next_str = &self.reverse_dict[&ids[i+1]][0];
                if next_str.trim().starts_with('r') || next_str.trim() == "yor" {
                    return tokens[1].clone(); // "bili" + "r" = "bilir"
                }
            }
            return tokens[0].clone(); // Default to "bile"
        }
        
        // Ada/Adı Exception (2218) - Default to "adı" (variant 1)
        if token_id == 2218 {
            if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 let next_str = &self.reverse_dict[&next_id][0];
                 // Use "ada" when followed by 'n' suffixes or 'yı' (for adayı pattern) or 'ma' (adama)
                 // 20017 = suffix yı, 32725 = BPE yı, 20002 = ma/me, 32763 = BPE ma
                 if next_id == 20040 || next_str.starts_with('n') || next_id == 20017 || next_id == 32725 || next_id == 20002 || next_id == 32763 {
                      return tokens[0].clone(); // "ada" for "adanın", "adayı", "adama"
                 }
            }
            // Default to "adı" for most cases
            if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
        }

        // Kap/Kab Exception (336) - favor "kapı" (door) over "kab" (container) context
        if token_id == 336 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                let next_str = &self.reverse_dict[&ids[i+1]][0];
                // If followed by vowel (which causes softening default), check if it looks like possessive plural
                if next_str.trim().starts_with(|c: char| "aeıioöuüAEIİOÖUÜ".contains(c)) {
                    return tokens[0].clone(); // Keep "kap"
                }
            }
            return tokens[0].clone(); // Default "kap"
        }
        
        // Emekli/Emekle Exception (2295) - Default to "emekli" (variant 1)
        if token_id == 2295 {
            if i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 // 20041 = 'yor' - for "emekliyor" use base form
                 if next_id == 20041 {
                      return tokens[0].clone(); // "emekle" + yor = emekliyor
                 }
            }
            // Default to "emekli" 
            if tokens.len() > 1 { return tokens[1].clone(); } else { return tokens[0].clone(); }
        }
        
        // Tutuk/Tutuğ/Tutk Exception (107) - for "tutkun"
        if token_id == 107 {
            if tokens.len() > 2 && i < ids.len() - 1 {
                 let next_str = &self.reverse_dict[&ids[i+1]][0];
                 // Check if next token starts with 'u' (un, unlar, etc.)
                 if next_str.trim().starts_with('u') {
                      return tokens[2].clone(); // "tutk" + "un" = "tutkun"
                 }
            }
            return tokens[0].clone();
        }
        
        // Başla/Başlı Exception (2206) - for "başlıca"
        if token_id == 2206 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                 let next_id = ids[i+1];
                 // 20005 = 'ça/çe' suffix, 20047 = 'ce', 20207 = BPE 'ca'
                 if next_id == 20005 || next_id == 20047 || next_id == 20207 {
                      return tokens[1].clone(); // "başlı" + "ca" = "başlıca"
                 }
            }
            // Continue to existing logic below
        }
        
        // Dip/Dib Exception (2406) - soften to "dib" before vowel suffixes
        if token_id == 2406 {
            if tokens.len() > 1 && i < ids.len() - 1 {
                let next_str = &self.reverse_dict[&ids[i+1]][0];
                if next_str.trim().starts_with(|c: char| "aeıioöuüAEIİOÖUÜ".contains(c)) {
                    return tokens[1].clone(); // "dib" + "inde" = "dibinde"
                }
            }
            return tokens[0].clone(); // "dip" by default
        }
        
        // de (19531) / ye (19968) / başla (2206) narrowing logic
        if token_id == 19531 || token_id == 19968 || token_id == 2206 {
             let mut should_narrow = false;
             
             if i < ids.len() - 1 {
                 let next_token = &self.reverse_dict[&ids[i + 1]][0];
                 // Check for "yor" string match (covers 32621, 20041 etc)
                 if next_token.contains("yor") {
                     should_narrow = true;
                 } else if let Some(suff_forms) = self.reverse_dict.get(&ids[i+1]) {
                     if suff_forms.iter().any(|s| s.starts_with(|c| ALL_VOWELS.contains(c))) {
                          // Only for de/ye, not başla (start vowel usually narrows de/ye->di/yi but başla->başlı?)
                          // Actually 2206 (başla/başlı) only narrows for YOR usually. 
                          // "Başla" + "acak" -> "Başlayacak" (no narrowing)
                          // "Başla" + "yıp" -> "Başlayıp"
                          // So for 2206, ONLY narrow if "yor"
                          if token_id != 2206 {
                              should_narrow = true;
                          }
                     }
                 }
             }
             
             if should_narrow {
                 // For 2206: başla -> başlı (variant 1)
                 if token_id == 2206 {
                      return tokens[1].clone();
                 }
                 
                 let original = &tokens[0];
                 if original.ends_with('e') {
                     let mut s = original.clone();
                     s.pop();
                     s.push('i');
                     return s;
                 } else if original.ends_with('E') {
                     let mut s = original.clone();
                     s.pop();
                     s.push('İ');
                     return s;
                 }
             }
             return tokens[0].clone();
        }
        
        // Range 100-2080: Generic Softening
        if token_id >= 100 && token_id < 2080 {
             // Skip if NO_SOFTENING_ROOTS (already handled) or EXCEPTION_ROOTS
             
             if i < ids.len() - 1 {
                 let next_token = &self.reverse_dict[&ids[i + 1]][0];
                 if self.starts_with_vowel(next_token) {
                     return tokens[1].clone();
                 } else if token_id <= 110 && next_token.trim() == "ı" {
                     if tokens.len() > 2 { return tokens[2].clone(); }
                 }
             }
             return tokens[0].clone();
        }

        // Range 2080-2315: Narrowing (e.g. verbs like demek/yemek other than de/ye)
        if token_id >= 2080 && token_id < 2315 {
             if i < ids.len() - 1 {
                 let next_token = &self.reverse_dict[&ids[i + 1]][0];
                 if next_token.contains("yor") {
                     return tokens[1].clone();
                 }
                 // Python Check: else return variant 0
             }
             return tokens[0].clone();
        }
        
        tokens[0].clone()
    }
    // Capitalize token with proper Turkish I handling
    fn capitalize_token(token: &str) -> String {
        if token.starts_with(' ') {
             // Preserve leading space
             let mut chars = token.chars();
             let _first = chars.next().unwrap(); // ' '
             
             // Find first non-space
             let rest = chars.as_str();
             if rest.is_empty() { return token.to_string(); }
             
             let mut rest_chars = rest.chars();
             if let Some(c) = rest_chars.next() {
                 let cap = match c {
                     'i' => "İ".to_string(),
                     'ı' => "I".to_string(),
                     _ => c.to_uppercase().to_string(),
                 };
                 format!(" {}{}", cap, rest_chars.as_str())
             } else {
                 token.to_string()
             }
        } else {
             let mut chars = token.chars();
             if let Some(c) = chars.next() {
                 let cap = match c {
                     'i' => "İ".to_string(),
                     'ı' => "I".to_string(),
                     _ => c.to_uppercase().to_string(),
                 };
                 format!("{}{}", cap, chars.as_str())
             } else {
                 String::new()
             }
        }
    }

    pub fn decode(&self, ids: Vec<i32>) -> String {
        if ids.is_empty() { return String::new(); }
        
        let mut text_parts: Vec<String> = Vec::with_capacity(ids.len());
        let mut i = 0;
        
        while i < ids.len() {
            let token_id = ids[i];
            
            if token_id == 0 && i < ids.len() - 1 { // uppercase
                // We must process the next token with full logic (softening/vowel drop)
                // before capitalizing it.
                // Determine if next is root or suffix to call correct method.
                let next_id = ids[i + 1];
                let resolved_token = if next_id < 20000 {
                     self.select_correct_root(i + 1, &ids)
                } else if next_id <= 20071 {
                     // Suffix context logic duplication or refactor?
                     // Python select_correct_root handles roots. 
                     // Only roots typically start a word/Sentence.
                     // But if Uppercase is applied to a suffix (unlikely but possible), 
                     // Python only calls select_correct_root in line 436.
                     self.select_correct_root(i + 1, &ids) 
                } else {
                     // BPE or other
                     if let Some(tokens) = self.reverse_dict.get(&next_id) {
                         tokens[0].clone()
                     } else {
                         String::new()
                     }
                };
                
                text_parts.push(Self::capitalize_token(&resolved_token));
                i += 2;
                continue;
            } else if token_id == 1 { // unknown
                text_parts.push("▁u▁".to_string());
            } else if let Some(tokens) = self.reverse_dict.get(&token_id) {
                if tokens.len() > 1 {
                    if token_id >= 20000 && token_id <= 20071 { // suffix
                         // Context construction (looking back up to 3 tokens)
                         let mut vowel_context_str = String::new();
                         let mut found_vowel = false;

                         // 1. Check immediate previous tokens for simple vowel presence
                         let mut j = (text_parts.len() as isize) - 1;
                         let mut tokens_checked = 0;
                         
                         while j >= 0 && tokens_checked < 3 {
                             let prev = &text_parts[j as usize];
                             
                             if !prev.trim().is_empty() {
                                 // Found a non-empty token. Does it have a vowel?
                                 if Self::has_vowel(prev) {
                                     vowel_context_str = prev.clone();
                                     found_vowel = true;
                                     break; // Found it!
                                 }
                                 tokens_checked += 1;
                             }
                             j -= 1;
                         }

                         // 2. If no vowel found in single tokens, look deeper by concatenating (depth 3)
                         if !found_vowel {
                             let mut depth = 0;
                             let mut temp_ctx = String::new();
                             let mut m = (text_parts.len() as isize) - 1;
                             
                             while m >= 0 && depth < 3 {
                                 let prev = &text_parts[m as usize];
                                 temp_ctx = prev.clone() + &temp_ctx; // Prepend
                                 if Self::has_vowel(&temp_ctx) {
                                     vowel_context_str = temp_ctx;
                                     break;
                                 }
                                 m -= 1;
                                 depth += 1;
                             }
                         }
                         
                         text_parts.push(self.select_correct_suffix(i, &ids, &vowel_context_str));
                    } else if token_id < 20000 { // root
                         text_parts.push(self.select_correct_root(i, &ids));
                    } else { // BPE (> 20071) -> Static
                         text_parts.push(tokens[0].clone());
                    }
                } else {
                    text_parts.push(tokens[0].clone());
                }
            } else {
                 text_parts.push("▁".to_string());
            }
            i += 1;
        }
        
        text_parts.join("")
    }
}
