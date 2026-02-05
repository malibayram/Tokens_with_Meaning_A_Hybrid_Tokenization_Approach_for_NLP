use pyo3::prelude::*;
use crate::tokenizer::TurkishTokenizer;

mod decoder;
mod tokenizer;

#[pyclass(name = "TurkishTokenizer")]
struct PyTurkishTokenizer {
    inner: TurkishTokenizer,
}

#[pymethods]
impl PyTurkishTokenizer {
    #[new]
    fn new() -> PyResult<Self> {
        // Embed the JSON files into the binary
        let roots_json = include_str!("resources/kokler.json");
        let ekler_json = include_str!("resources/ekler.json");
        let bpe_json = include_str!("resources/bpe_tokenler.json");
        
        let inner = TurkishTokenizer::from_files(roots_json, ekler_json, bpe_json)
             .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
             
        Ok(PyTurkishTokenizer { inner })
    }
    
    fn encode(&self, text: &str) -> Vec<i32> {
        self.inner.encode(text)
    }
    
    fn decode(&self, ids: Vec<i32>) -> String {
        self.inner.decode(ids)
    }

    fn tokenize_text(&self, text: &str, py: Python<'_>) -> PyResult<Vec<PyObject>> {
        let tokens = self.inner.tokenize_text(text);
        let mut results = Vec::with_capacity(tokens.len());
        
        for token in tokens {
             let dict = pyo3::types::PyDict::new(py);
             dict.set_item("token", token.token)?;
             dict.set_item("id", token.id)?;
             let type_str = match token.token_type {
                 crate::tokenizer::TokenType::ROOT => "ROOT",
                 crate::tokenizer::TokenType::SUFFIX => "SUFFIX",
                 crate::tokenizer::TokenType::BPE => "BPE",
             };
             dict.set_item("type", type_str)?;
             results.push(dict.into());
        }
        Ok(results)
    }
}

#[pymodule]
fn turkish_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTurkishTokenizer>()?;
    Ok(())
}
