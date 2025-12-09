//! Corpus loading for Python to Rust single-shot compilation evaluation.
//!
//! Loads Python examples from the `reprorusted-python-cli` corpus for
//! evaluation against SLM and `SaaS` baselines.
//!
//! Supports two formats:
//! - Directory-based: `example_*` directories with `.py` files
//! - JSONL-based: `transpile_corpus.jsonl` with structured examples

#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_const_for_fn)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during corpus loading
#[derive(Error, Debug)]
pub enum CorpusError {
    #[error("Corpus directory not found: {0}")]
    NotFound(String),

    #[error("No examples found in corpus")]
    Empty,

    #[error("Example missing Python source: {0}")]
    MissingSource(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// A test case for a Python function (from JSONL format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    /// Input arguments
    pub input: Vec<serde_json::Value>,
    /// Expected output
    pub expected: serde_json::Value,
}

/// A JSONL corpus entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonlEntry {
    /// Example ID
    pub id: String,
    /// Python source code
    pub python_code: String,
    /// Test cases
    pub test_cases: Vec<TestCase>,
}

/// A single Python example from the corpus
#[derive(Debug, Clone)]
pub struct PythonExample {
    /// Example name (e.g., `example_abs`)
    pub name: String,
    /// Path to the example directory
    pub path: PathBuf,
    /// Python source code (main file)
    pub source: String,
    /// Test file contents (if present)
    pub tests: Option<String>,
    /// All Python files in the example
    pub files: HashMap<String, String>,
    /// Structured test cases (from JSONL format)
    pub test_cases: Vec<TestCase>,
}

impl PythonExample {
    /// Get the primary Python file name
    #[must_use]
    #[allow(clippy::case_sensitive_file_extension_comparisons)]
    pub fn primary_file(&self) -> String {
        self.files
            .keys()
            .find(|k| !k.starts_with("test_") && k.ends_with(".py"))
            .cloned()
            .unwrap_or_else(|| {
                format!(
                    "{}.py",
                    self.name.strip_prefix("example_").unwrap_or(&self.name)
                )
            })
    }
}

/// Corpus of Python examples for evaluation
#[derive(Debug)]
pub struct Corpus {
    /// Root directory of the corpus
    pub root: PathBuf,
    /// All loaded examples
    pub examples: Vec<PythonExample>,
}

impl Corpus {
    /// Load corpus from directory path or JSONL file
    ///
    /// Supports two formats:
    /// - Directory with `example_*` subdirectories containing `.py` files
    /// - JSONL file with structured examples (e.g., `transpile_corpus.jsonl`)
    ///
    /// # Errors
    ///
    /// Returns an error if the path doesn't exist or contains no examples.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, CorpusError> {
        let root = path.as_ref().to_path_buf();

        if !root.exists() {
            return Err(CorpusError::NotFound(root.display().to_string()));
        }

        // Check if path is a JSONL file or directory containing JSONL
        if root.is_file() && root.extension().is_some_and(|e| e == "jsonl") {
            return Self::load_jsonl(&root);
        }

        // Check for JSONL file in directory
        let jsonl_path = root.join("transpile_corpus.jsonl");
        if jsonl_path.exists() {
            return Self::load_jsonl(&jsonl_path);
        }

        // Fall back to directory-based loading
        Self::load_directory(&root)
    }

    /// Load corpus from JSONL file
    fn load_jsonl(path: &Path) -> Result<Self, CorpusError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut examples = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: JsonlEntry = serde_json::from_str(&line)?;

            let mut files = HashMap::new();
            let filename = format!("{}.py", entry.id);
            files.insert(filename.clone(), entry.python_code.clone());

            examples.push(PythonExample {
                name: entry.id.clone(),
                path: path.to_path_buf(),
                source: entry.python_code,
                tests: None, // Tests are in test_cases instead
                files,
                test_cases: entry.test_cases,
            });
        }

        if examples.is_empty() {
            return Err(CorpusError::Empty);
        }

        // Sort by name for reproducibility
        examples.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Self {
            root: path.parent().unwrap_or(path).to_path_buf(),
            examples,
        })
    }

    /// Load corpus from directory with example_* subdirectories
    fn load_directory(root: &Path) -> Result<Self, CorpusError> {
        let mut examples = Vec::new();

        for entry in std::fs::read_dir(root)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().starts_with("example_") {
                        if let Ok(example) = Self::load_example(&path) {
                            examples.push(example);
                        }
                    }
                }
            }
        }

        if examples.is_empty() {
            return Err(CorpusError::Empty);
        }

        // Sort by name for reproducibility
        examples.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(Self {
            root: root.to_path_buf(),
            examples,
        })
    }

    /// Load a single example from its directory
    fn load_example(path: &Path) -> Result<PythonExample, CorpusError> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        let mut files = HashMap::new();
        let mut source = String::new();
        let mut tests = None;

        // Scan for Python files
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();

            if file_path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("py"))
            {
                let filename = file_path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default();

                let content = std::fs::read_to_string(&file_path)?;

                if filename.starts_with("test_") {
                    tests = Some(content.clone());
                } else if source.is_empty() {
                    source.clone_from(&content);
                }

                files.insert(filename, content);
            }
        }

        if source.is_empty() {
            return Err(CorpusError::MissingSource(name));
        }

        Ok(PythonExample {
            name,
            path: path.to_path_buf(),
            source,
            tests,
            files,
            test_cases: Vec::new(), // Directory-based format doesn't have structured test cases
        })
    }

    /// Get total number of examples
    #[must_use]
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Check if corpus is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Get examples with tests only
    #[must_use]
    pub fn with_tests(&self) -> Vec<&PythonExample> {
        self.examples.iter().filter(|e| e.tests.is_some()).collect()
    }

    /// Get iterator over examples
    pub fn iter(&self) -> impl Iterator<Item = &PythonExample> {
        self.examples.iter()
    }

    /// Sample N random examples
    #[must_use]
    pub fn sample(&self, n: usize) -> Vec<&PythonExample> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..self.examples.len()).collect();
        indices.shuffle(&mut rng);
        indices
            .into_iter()
            .take(n)
            .map(|i| &self.examples[i])
            .collect()
    }

    /// Compute statistics about the corpus
    #[must_use]
    pub fn stats(&self) -> CorpusStats {
        let total_examples = self.examples.len();
        let examples_with_tests = self.examples.iter().filter(|e| e.tests.is_some()).count();
        let total_files: usize = self.examples.iter().map(|e| e.files.len()).sum();
        let total_lines: usize = self
            .examples
            .iter()
            .flat_map(|e| e.files.values())
            .map(|content| content.lines().count())
            .sum();

        CorpusStats {
            total_examples,
            examples_with_tests,
            total_files,
            total_lines,
        }
    }
}

/// Statistics about the corpus
#[derive(Debug, Clone)]
pub struct CorpusStats {
    /// Total number of examples
    pub total_examples: usize,
    /// Examples with test files
    pub examples_with_tests: usize,
    /// Total Python files
    pub total_files: usize,
    /// Total lines of Python code
    pub total_lines: usize,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_corpus() -> TempDir {
        let dir = TempDir::new().unwrap();

        // Create example_abs
        let example_abs = dir.path().join("example_abs");
        std::fs::create_dir(&example_abs).unwrap();
        std::fs::write(
            example_abs.join("abs_tool.py"),
            "def compute_abs(x):\n    return abs(x)\n",
        )
        .unwrap();
        std::fs::write(
            example_abs.join("test_abs_tool.py"),
            "def test_abs():\n    assert compute_abs(-5) == 5\n",
        )
        .unwrap();

        // Create example_hello
        let example_hello = dir.path().join("example_hello");
        std::fs::create_dir(&example_hello).unwrap();
        std::fs::write(
            example_hello.join("hello.py"),
            "def greet():\n    print('Hello')\n",
        )
        .unwrap();

        dir
    }

    #[test]
    fn test_corpus_load() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        assert_eq!(corpus.len(), 2);
        assert!(!corpus.is_empty());
    }

    #[test]
    fn test_corpus_not_found() {
        let result = Corpus::load("/nonexistent/path");
        assert!(matches!(result, Err(CorpusError::NotFound(_))));
    }

    #[test]
    fn test_corpus_empty() {
        let dir = TempDir::new().unwrap();
        let result = Corpus::load(dir.path());
        assert!(matches!(result, Err(CorpusError::Empty)));
    }

    #[test]
    fn test_example_with_tests() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        let with_tests = corpus.with_tests();
        assert_eq!(with_tests.len(), 1);
        assert_eq!(with_tests[0].name, "example_abs");
    }

    #[test]
    fn test_corpus_stats() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();
        let stats = corpus.stats();

        assert_eq!(stats.total_examples, 2);
        assert_eq!(stats.examples_with_tests, 1);
        assert_eq!(stats.total_files, 3); // 2 in abs, 1 in hello
    }

    #[test]
    fn test_example_primary_file() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        let abs_example = corpus.iter().find(|e| e.name == "example_abs").unwrap();
        assert_eq!(abs_example.primary_file(), "abs_tool.py");
    }

    #[test]
    fn test_corpus_sample() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        let sample = corpus.sample(1);
        assert_eq!(sample.len(), 1);

        let sample_all = corpus.sample(10);
        assert_eq!(sample_all.len(), 2); // Only 2 examples exist
    }

    #[test]
    fn test_corpus_iter() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        let names: Vec<_> = corpus.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"example_abs"));
        assert!(names.contains(&"example_hello"));
    }

    #[test]
    fn test_example_files_map() {
        let dir = create_test_corpus();
        let corpus = Corpus::load(dir.path()).unwrap();

        let abs_example = corpus.iter().find(|e| e.name == "example_abs").unwrap();
        assert!(abs_example.files.contains_key("abs_tool.py"));
        assert!(abs_example.files.contains_key("test_abs_tool.py"));
    }
}
