use std::fs;
use std::path::Path;
use reqwest::blocking::get;
use tempfile::NamedTempFile;

fn get_script_path() -> String {
    if let Ok(path) = std::env::current_exe() {
        return path.parent().unwrap().to_str().unwrap().to_string();
    }
    String::from(".")
}

pub fn download_model(model: &str, models_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let src = if model.contains("tdrz") {
        "https://huggingface.co/akashmjn/tinydiarize-whisper.cpp"
    } else {
        "https://ggml.ggerganov.com"
    };

    let pfx = if model.contains("tdrz") {
        "resolve/main/ggml"
    } else {
        "models"
    };

    let url = format!("{}/{}.bin", src, model);

    println!("Downloading ggml model {} from '{}'...", model, url);

    let response = get(&url)?;
    if !response.status().is_success() {
        return Err(format!("Failed to download model: {}", response.status()).into());
    }

    let temp_file = NamedTempFile::new()?;
    fs::write(temp_file.path(), response.bytes()?)?;

    // Extract the zip file
    let archive_path = temp_file.path();
    let mut archive = zip::ZipArchive::new(fs::File::open(archive_path)?)?;
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

    // Remove the temporary zip file
    fs::remove_file(archive_path)?;

    Ok(())
}