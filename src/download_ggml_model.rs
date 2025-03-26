use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use reqwest::Error as ReqwestError;
use tempfile::NamedTempFile;
use zip::ZipArchive;

// Define a configuration struct for flexibility
#[derive(Debug)]
pub struct DownloadConfig {
    pub src: String,
    pub pfx: String,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        DownloadConfig {
            src: "https://ggml.ggerganov.com".to_string(),
            pfx: "models".to_string(),
        }
    }
}

/// Downloads the specified ggml model.
///
/// # Arguments
///
/// * `model` - The name of the model to download (e.g., "tiny", "base").
/// * `config` - Configuration for downloading the model. Defaults to default configuration.
///
/// # Returns
///
/// A Result containing the path to the downloaded file or an error message.
/// 
/// Dyn err result
pub fn download_model(model: &str, config: Option<DownloadConfig>) -> Result<PathBuf, Box<dyn Error>> {
    let config = config.unwrap_or_default();

    // Determine the source URL and prefix based on whether 'tdrz' is in the model name
    let src = if model.contains("tdrz") {
        "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
    } else {
        &config.src
    };

    let pfx = if model.contains("tdrz") {
        "resolve/main/ggml"
    } else {
        &config.pfx
    };

    // Construct the full URL for the model file
    let url = format!("{}/{}.bin", src, model);

    println!("Downloading ggml model {} from '{}'...", model, url);

    // Download the model using reqwest
    let response = reqwest::blocking::get(&url)?;
    if !response.status().is_success() {

    return Err(format!("Failed to download model from '{}'", url).into());
    }

    // Create a temporary file to store the downloaded data
    let temp_file = NamedTempFile::new()?;
    fs::write(temp_file.path(), response.bytes()?)?;
    let temp_path = temp_file.path();
    if temp_path.metadata()?.len() == 0 {
        return Err("Downloaded model file is empty".into());
    }

    Ok(temp_file.into_temp_path().to_path_buf())
}

/// Extracts the zip archive to the specified models path.
///
/// # Arguments
///
/// * `archive_path` - The path to the downloaded zip file.
/// * `models_path` - The directory where the model will be saved.
///
/// # Returns
///
/// A Result containing a boolean indicating success or an error message.
pub fn extract_model(archive_path: &Path, models_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut archive = ZipArchive::new(fs::File::open(archive_path)?)?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let outpath = models_path.join(file.mangled_name());
        if (&*file.name()).ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                fs::create_dir_all(p)?;
            }
            let mut outfile = fs::File::create(&outpath)?;
            std::io::copy(&mut file, &mut outfile)?;
        }
    }

    Ok(())
}

/// Downloads and extracts the specified ggml model to the given models_path.
///
/// # Arguments
///
/// * `model` - The name of the model to download (e.g., "tiny", "base").
/// * `models_path` - The directory where the model will be saved.
/// * `config` - Configuration for downloading the model. Defaults to default configuration.
///
/// # Returns
///
/// A Result containing a boolean indicating success or an error message.
pub fn download_and_extract_model(model: &str, models_path: &Path, config: Option<DownloadConfig>) -> Result<(), Box<dyn std::error::Error>> {
    let downloaded_file = download_model(model, config)?;
    extract_model(&downloaded_file, models_path)?;

    Ok(())
}
