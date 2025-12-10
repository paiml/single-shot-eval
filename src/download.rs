//! `HuggingFace` model download with JIT caching
//!
//! Toyota Way principles:
//! - **Jidoka**: SHA256 checksum verification halts on mismatch
//! - **Poka-yoke**: Reject unsafe pickle files by default
//! - **JIT**: Aggressive cache eviction to eliminate inventory waste
//!
//! ## Example
//!
//! ```rust,ignore
//! use single_shot_eval::download::{DownloadConfig, download_model};
//!
//! let config = DownloadConfig::default();
//! let artifact = download_model("microsoft/phi-2", &config).await?;
//! ```

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};
use thiserror::Error;

/// Download errors
#[derive(Error, Debug)]
pub enum DownloadError {
    /// Repository not found
    #[error("Repository not found: {0}")]
    RepoNotFound(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Checksum mismatch (Jidoka: HALT)
    #[error("JIDOKA HALT: Checksum mismatch for {path}: expected {expected}, got {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    /// Unsafe pickle file (Poka-yoke)
    #[error(
        "POKA-YOKE: Unsafe pickle file rejected: {path}. Use --unsafe-allow-pickle to override"
    )]
    UnsafePickleFile { path: PathBuf },

    /// Unsupported format
    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Download timeout
    #[error("Download timeout after {0:?}")]
    Timeout(Duration),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),
}

/// Model format for safety classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyClass {
    /// Safe: no arbitrary code execution (safetensors, gguf, apr)
    Safe,
    /// Unsafe: can execute arbitrary code (pickle-based formats)
    UnsafePickle,
    /// Unknown format
    Unknown,
}

impl SafetyClass {
    /// Classify format safety from file extension
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "safetensors" | "gguf" | "apr" => Self::Safe,
            "bin" | "pt" | "pth" | "pkl" | "pickle" => Self::UnsafePickle,
            _ => Self::Unknown,
        }
    }

    /// Check if format is safe
    #[must_use]
    pub const fn is_safe(&self) -> bool {
        matches!(self, Self::Safe)
    }
}

/// Download configuration with JIT caching
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    /// Cache directory
    pub cache_dir: PathBuf,
    /// Maximum cache size in bytes (default: 10GB - JIT principle)
    pub max_cache_bytes: u64,
    /// Download timeout
    pub timeout: Duration,
    /// Allow unsafe pickle files (default: false - Poka-yoke)
    pub unsafe_allow_pickle: bool,
    /// Evict after successful test (JIT)
    pub evict_after_pass: bool,
    /// Verify checksums (Jidoka)
    pub verify_checksum: bool,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            cache_dir: dirs::cache_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("batuta")
                .join("models"),
            max_cache_bytes: 10 * 1024 * 1024 * 1024, // 10GB (reduced from 50GB - JIT)
            timeout: Duration::from_secs(600),        // 10 minutes
            unsafe_allow_pickle: false,               // Poka-yoke: reject by default
            evict_after_pass: true,                   // JIT: clear after success
            verify_checksum: true,                    // Jidoka: always verify
        }
    }
}

/// Downloaded model artifact
#[derive(Debug, Clone)]
pub struct ModelArtifact {
    /// Local file path
    pub path: PathBuf,
    /// Original repository ID
    pub repo_id: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA256 checksum
    pub sha256: String,
    /// Safety classification
    pub safety: SafetyClass,
    /// Download timestamp
    pub downloaded_at: SystemTime,
}

impl ModelArtifact {
    /// Get file extension
    #[must_use]
    pub fn extension(&self) -> Option<&str> {
        self.path.extension().and_then(|e| e.to_str())
    }

    /// Check if artifact is safe (no pickle)
    #[must_use]
    pub const fn is_safe(&self) -> bool {
        self.safety.is_safe()
    }
}

/// Cache entry for LRU eviction
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// File path
    pub path: PathBuf,
    /// Size in bytes
    pub size_bytes: u64,
    /// Last access time
    pub last_access: SystemTime,
    /// Pinned for regression testing
    pub pinned: bool,
}

/// JIT cache manager
#[derive(Debug)]
pub struct CacheManager {
    /// Configuration
    config: DownloadConfig,
    /// Cache entries
    entries: HashMap<PathBuf, CacheEntry>,
}

impl CacheManager {
    /// Create new cache manager
    #[must_use]
    pub fn new(config: DownloadConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
        }
    }

    /// Get total cache size
    #[must_use]
    pub fn total_size(&self) -> u64 {
        self.entries.values().map(|e| e.size_bytes).sum()
    }

    /// Check if cache has space for new entry
    #[must_use]
    pub fn has_space(&self, size_bytes: u64) -> bool {
        self.total_size() + size_bytes <= self.config.max_cache_bytes
    }

    /// Add entry to cache
    pub fn add(&mut self, path: PathBuf, size_bytes: u64) {
        let entry = CacheEntry {
            path: path.clone(),
            size_bytes,
            last_access: SystemTime::now(),
            pinned: false,
        };
        self.entries.insert(path, entry);
    }

    /// Evict LRU entries until space is available
    ///
    /// # Errors
    ///
    /// Returns `CacheError` if all entries are pinned and space cannot be freed
    pub fn evict_until_space(&mut self, needed_bytes: u64) -> Result<(), DownloadError> {
        while !self.has_space(needed_bytes) {
            // Find oldest non-pinned entry
            let oldest = self
                .entries
                .iter()
                .filter(|(_, e)| !e.pinned)
                .min_by_key(|(_, e)| e.last_access)
                .map(|(p, _)| p.clone());

            if let Some(path) = oldest {
                self.remove(&path)?;
            } else {
                return Err(DownloadError::CacheError(
                    "Cannot evict: all entries pinned".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Remove entry from cache and disk
    ///
    /// # Errors
    ///
    /// Returns IO error if file deletion fails
    pub fn remove(&mut self, path: &Path) -> Result<(), DownloadError> {
        if let Some(entry) = self.entries.remove(path) {
            if entry.path.exists() {
                fs::remove_file(&entry.path)?;
            }
        }
        Ok(())
    }

    /// Pin entry for regression testing
    pub fn pin(&mut self, path: &Path) {
        if let Some(entry) = self.entries.get_mut(path) {
            entry.pinned = true;
        }
    }

    /// Update last access time
    pub fn touch(&mut self, path: &Path) {
        if let Some(entry) = self.entries.get_mut(path) {
            entry.last_access = SystemTime::now();
        }
    }
}

/// Validate format safety (Poka-yoke gate)
///
/// # Errors
///
/// Returns `UnsafePickleFile` if pickle format and `unsafe_allow_pickle` is false
pub fn validate_format_safety(path: &Path, config: &DownloadConfig) -> Result<(), DownloadError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let safety = SafetyClass::from_extension(ext);

    match safety {
        SafetyClass::Safe => Ok(()),
        SafetyClass::UnsafePickle => {
            if config.unsafe_allow_pickle {
                tracing::warn!(
                    path = %path.display(),
                    "SECURITY WARNING: Loading pickle file with --unsafe-allow-pickle"
                );
                Ok(())
            } else {
                Err(DownloadError::UnsafePickleFile {
                    path: path.to_path_buf(),
                })
            }
        }
        SafetyClass::Unknown => Err(DownloadError::UnsupportedFormat(ext.to_string())),
    }
}

/// Compute SHA256 checksum of file
///
/// # Errors
///
/// Returns IO error if file cannot be read
pub fn compute_sha256(path: &Path) -> Result<String, DownloadError> {
    use sha2::{Digest, Sha256};

    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify SHA256 checksum (Jidoka gate)
///
/// # Errors
///
/// Returns `ChecksumMismatch` if checksums don't match - this HALTS the pipeline
pub fn verify_checksum(path: &Path, expected: &str) -> Result<(), DownloadError> {
    let actual = compute_sha256(path)?;

    if actual != expected {
        tracing::error!(
            path = %path.display(),
            expected = %expected,
            actual = %actual,
            "JIDOKA HALT: Checksum mismatch - pipeline stopped"
        );
        return Err(DownloadError::ChecksumMismatch {
            path: path.to_path_buf(),
            expected: expected.to_string(),
            actual,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ==========================================================================
    // SafetyClass Tests (Poka-yoke)
    // ==========================================================================

    #[test]
    fn test_safety_class_safe_formats() {
        assert_eq!(
            SafetyClass::from_extension("safetensors"),
            SafetyClass::Safe
        );
        assert_eq!(
            SafetyClass::from_extension("SAFETENSORS"),
            SafetyClass::Safe
        );
        assert_eq!(SafetyClass::from_extension("gguf"), SafetyClass::Safe);
        assert_eq!(SafetyClass::from_extension("apr"), SafetyClass::Safe);
    }

    #[test]
    fn test_safety_class_unsafe_pickle() {
        assert_eq!(
            SafetyClass::from_extension("bin"),
            SafetyClass::UnsafePickle
        );
        assert_eq!(SafetyClass::from_extension("pt"), SafetyClass::UnsafePickle);
        assert_eq!(
            SafetyClass::from_extension("pth"),
            SafetyClass::UnsafePickle
        );
        assert_eq!(
            SafetyClass::from_extension("pkl"),
            SafetyClass::UnsafePickle
        );
        assert_eq!(
            SafetyClass::from_extension("pickle"),
            SafetyClass::UnsafePickle
        );
    }

    #[test]
    fn test_safety_class_unknown() {
        assert_eq!(SafetyClass::from_extension("txt"), SafetyClass::Unknown);
        assert_eq!(SafetyClass::from_extension("json"), SafetyClass::Unknown);
        assert_eq!(SafetyClass::from_extension(""), SafetyClass::Unknown);
    }

    #[test]
    fn test_safety_class_is_safe() {
        assert!(SafetyClass::Safe.is_safe());
        assert!(!SafetyClass::UnsafePickle.is_safe());
        assert!(!SafetyClass::Unknown.is_safe());
    }

    // ==========================================================================
    // DownloadConfig Tests
    // ==========================================================================

    #[test]
    fn test_download_config_default() {
        let config = DownloadConfig::default();

        // JIT: 10GB max cache (reduced from 50GB)
        assert_eq!(config.max_cache_bytes, 10 * 1024 * 1024 * 1024);

        // Poka-yoke: reject pickle by default
        assert!(!config.unsafe_allow_pickle);

        // JIT: evict after pass
        assert!(config.evict_after_pass);

        // Jidoka: always verify
        assert!(config.verify_checksum);
    }

    // ==========================================================================
    // Format Safety Validation Tests (Poka-yoke Gate)
    // ==========================================================================

    #[test]
    fn test_validate_format_safety_safe() {
        let config = DownloadConfig::default();

        assert!(validate_format_safety(Path::new("model.safetensors"), &config).is_ok());
        assert!(validate_format_safety(Path::new("model.gguf"), &config).is_ok());
        assert!(validate_format_safety(Path::new("model.apr"), &config).is_ok());
    }

    #[test]
    fn test_validate_format_safety_pickle_rejected() {
        let config = DownloadConfig::default();

        let result = validate_format_safety(Path::new("model.bin"), &config);
        assert!(result.is_err());

        if let Err(DownloadError::UnsafePickleFile { path }) = result {
            assert_eq!(path, PathBuf::from("model.bin"));
        } else {
            panic!("Expected UnsafePickleFile error");
        }
    }

    #[test]
    fn test_validate_format_safety_pickle_allowed() {
        let config = DownloadConfig {
            unsafe_allow_pickle: true,
            ..Default::default()
        };

        assert!(validate_format_safety(Path::new("model.bin"), &config).is_ok());
        assert!(validate_format_safety(Path::new("model.pt"), &config).is_ok());
    }

    #[test]
    fn test_validate_format_safety_unknown_rejected() {
        let config = DownloadConfig::default();

        let result = validate_format_safety(Path::new("model.xyz"), &config);
        assert!(result.is_err());

        if let Err(DownloadError::UnsupportedFormat(ext)) = result {
            assert_eq!(ext, "xyz");
        } else {
            panic!("Expected UnsupportedFormat error");
        }
    }

    // ==========================================================================
    // Checksum Tests (Jidoka Gate)
    // ==========================================================================

    #[test]
    fn test_compute_sha256() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "hello world").expect("Failed to write file");

        let hash = compute_sha256(&file_path).expect("Failed to compute hash");

        // SHA256 of "hello world"
        assert_eq!(
            hash,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_verify_checksum_success() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "hello world").expect("Failed to write file");

        let result = verify_checksum(
            &file_path,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_checksum_mismatch_halts() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let file_path = temp_dir.path().join("test.txt");

        fs::write(&file_path, "hello world").expect("Failed to write file");

        let result = verify_checksum(&file_path, "wrong_checksum");

        assert!(result.is_err());
        if let Err(DownloadError::ChecksumMismatch {
            expected, actual, ..
        }) = result
        {
            assert_eq!(expected, "wrong_checksum");
            assert_eq!(
                actual,
                "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
            );
        } else {
            panic!("Expected ChecksumMismatch error");
        }
    }

    // ==========================================================================
    // CacheManager Tests (JIT)
    // ==========================================================================

    #[test]
    fn test_cache_manager_new() {
        let config = DownloadConfig::default();
        let cache = CacheManager::new(config);

        assert_eq!(cache.total_size(), 0);
    }

    #[test]
    fn test_cache_manager_add() {
        let config = DownloadConfig::default();
        let mut cache = CacheManager::new(config);

        cache.add(PathBuf::from("model1.safetensors"), 1000);
        cache.add(PathBuf::from("model2.safetensors"), 2000);

        assert_eq!(cache.total_size(), 3000);
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn test_cache_manager_has_space() {
        let config = DownloadConfig {
            max_cache_bytes: 5000,
            ..Default::default()
        };
        let mut cache = CacheManager::new(config);

        assert!(cache.has_space(5000));
        assert!(!cache.has_space(5001));

        cache.add(PathBuf::from("model.safetensors"), 3000);

        assert!(cache.has_space(2000));
        assert!(!cache.has_space(2001));
    }

    #[test]
    fn test_cache_manager_evict_lru() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = DownloadConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_cache_bytes: 3000,
            ..Default::default()
        };
        let mut cache = CacheManager::new(config);

        // Create actual files
        let file1 = temp_dir.path().join("model1.safetensors");
        let file2 = temp_dir.path().join("model2.safetensors");
        fs::write(&file1, "content1").expect("write file1");
        fs::write(&file2, "content2").expect("write file2");

        cache.add(file1.clone(), 1500);

        // Sleep to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));

        cache.add(file2.clone(), 1500);

        // Now at 3000, try to add 1000 more
        let result = cache.evict_until_space(1000);
        assert!(result.is_ok());

        // file1 should be evicted (oldest)
        assert!(!file1.exists());
        assert!(file2.exists());
        assert_eq!(cache.entries.len(), 1);
    }

    #[test]
    fn test_cache_manager_pin_prevents_eviction() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let config = DownloadConfig {
            cache_dir: temp_dir.path().to_path_buf(),
            max_cache_bytes: 2000,
            ..Default::default()
        };
        let mut cache = CacheManager::new(config);

        let file1 = temp_dir.path().join("model1.safetensors");
        fs::write(&file1, "content").expect("write file");

        cache.add(file1.clone(), 1500);
        cache.pin(&file1);

        // Try to evict when only pinned entry exists
        let result = cache.evict_until_space(1000);

        assert!(result.is_err());
        if let Err(DownloadError::CacheError(msg)) = result {
            assert!(msg.contains("pinned"));
        } else {
            panic!("Expected CacheError");
        }
    }

    // ==========================================================================
    // ModelArtifact Tests
    // ==========================================================================

    #[test]
    fn test_model_artifact_extension() {
        let artifact = ModelArtifact {
            path: PathBuf::from("model.safetensors"),
            repo_id: "test/model".to_string(),
            size_bytes: 1000,
            sha256: "abc123".to_string(),
            safety: SafetyClass::Safe,
            downloaded_at: SystemTime::now(),
        };

        assert_eq!(artifact.extension(), Some("safetensors"));
        assert!(artifact.is_safe());
    }

    // ==========================================================================
    // Error Display Tests
    // ==========================================================================

    #[test]
    fn test_error_display() {
        let err = DownloadError::ChecksumMismatch {
            path: PathBuf::from("test.bin"),
            expected: "abc".to_string(),
            actual: "xyz".to_string(),
        };
        assert!(err.to_string().contains("JIDOKA HALT"));
        assert!(err.to_string().contains("test.bin"));

        let err = DownloadError::UnsafePickleFile {
            path: PathBuf::from("model.bin"),
        };
        assert!(err.to_string().contains("POKA-YOKE"));
        assert!(err.to_string().contains("--unsafe-allow-pickle"));
    }
}
