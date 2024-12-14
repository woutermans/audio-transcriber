use hound::{SampleFormat, WavReader};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod download_ggml_model;

fn parse_wav_file(path: &Path) -> io::Result<Vec<i16>> {
    let reader = WavReader::open(path).map_err(|e| {
        io::Error::new(
            io::ErrorKind::Other,
            format!("Error opening WAV file: {}", e),
        )
    })?;

    if reader.spec().channels != 1 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected mono audio file",
        ));
    }
    if reader.spec().sample_format != SampleFormat::Int {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected integer sample format",
        ));
    }
    if reader.spec().sample_rate != 16000 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected 16KHz sample rate",
        ));
    }
    if reader.spec().bits_per_sample != 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Expected 16 bits per sample",
        ));
    }

    Ok(reader
        .into_samples::<i16>()
        .map(|x| x.expect("sample"))
        .collect())
}

fn download_ffmpeg() -> Result<(), Box<dyn std::error::Error>> {
    // Check if ffmpeg is already installed
    if Command::new("ffmpeg").output().is_ok() {
        println!(
            "FFmpeg is already installed. Skipping download. If you want to reinstall, delete the FFmpeg binary and run this script again."
        );
        return Ok(());
    }

    if cfg!(target_os = "windows") {
        let url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z";

        println!("Downloading FFmpeg for Windows...");
        let response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
            return Err("Failed to download FFmpeg".into());
        }

        let temp_file = tempfile::NamedTempFile::new()?;
        fs::write(temp_file.path(), &response.bytes()?)?;

        println!("Extracting FFmpeg...");
        sevenz_rust::decompress_file(
            temp_file.path(),
            Path::new("."),
        )?;

        // Remove the temporary zip file
        fs::remove_file(temp_file.path())?;
    }

    Ok(())
}

fn ensure_wav_compatibility(
    input_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    Command::new("ffmpeg")
        .arg("-i")
        .arg(input_path)
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg(output_path)
        .spawn()?
        .wait()?;

    Ok(())
}

fn create_temporary_directory() -> Result<TempDir, Box<dyn std::error::Error>> {
    TempDir::new().map_err(|e| e.into())
}

fn handle_transcription(
    whisper_path: &Path,
    samples: Vec<f32>,
    chunk_size: usize,
    input_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let ctx = WhisperContext::new_with_params(
        &whisper_path.to_string_lossy(),
        WhisperContextParameters {
            flash_attn: true,
            ..Default::default()
        },
    )?;

    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_initial_prompt("experience");

    let sample_batches = samples.chunks(chunk_size).collect::<Vec<_>>();
    let chunk_count = sample_batches.len();

    let out_file_path = format!(
        "{}_timestamps.txt",
        input_path.file_stem().unwrap().to_string_lossy()
    );
    let out_raw_path = format!(
        "{}_raw.txt",
        input_path.file_stem().unwrap().to_string_lossy()
    );

    let mut out_file = fs::File::create(&out_file_path)?;
    let mut out_file_raw = fs::File::create(&out_raw_path)?;

    let pb = indicatif::ProgressBar::new(chunk_count as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(Duration::from_millis(100));

    let mut last_timestamp = 0;
    for samples in sample_batches {
        state
            .full(params.clone(), &samples)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let num_segments = state.full_n_segments()?;
        let mut end_timestamp = 0;
        for i in 0..num_segments {
            let segment = state.full_get_segment_text(i)?;
            let start_timestamp = state.full_get_segment_t0(i)?;
            end_timestamp = state.full_get_segment_t1(i)?;

            out_file.write_all(
                format!(
                    "[{} - {}]: {}\n",
                    start_timestamp + last_timestamp,
                    end_timestamp + last_timestamp,
                    segment
                )
                .as_bytes(),
            )?;
            out_file_raw.write_all(format!("{} ", segment).as_bytes())?;
        }
        last_timestamp = end_timestamp;
        pb.inc(1);
    }

    pb.finish_with_message("Done");

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_wav_file> [model_path]", args[0]);
        return;
    }

    let audio_path = Path::new(&args[1]);
    if !audio_path.exists() {
        eprintln!("Error: Audio file does not exist at {}", &args[1]);
        return;
    }

    let model_path = args
        .get(2)
        .unwrap_or(&"ggml-large-v3-turbo.bin".to_string());
    let whisper_path = Path::new(model_path);

    // Download FFmpeg if not already installed
    match download_ffmpeg() {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to download FFmpeg: {}", e);
            std::process::exit(1);
        }
    }

    let temp_dir = match create_temporary_directory() {
        Ok(dir) => dir,
        Err(e) => {
            eprintln!("Failed to create temporary directory: {}", e);
            std::process::exit(1);
        }
    };

    let output_path = temp_dir.path().join("converted_audio.wav");

    // Ensure WAV file compatibility using FFmpeg
    match ensure_wav_compatibility(audio_path, &output_path) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to ensure WAV compatibility: {}", e);
            std::process::exit(1);
        }
    }

    let original_samples = match parse_wav_file(&output_path) {
        Ok(samples) => samples,
        Err(e) => {
            eprintln!("Failed to parse WAV file: {}", e);
            std::process::exit(1);
        }
    };

    let mut samples = vec![0.0f32; original_samples.len()];
    match whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to convert integer audio samples: {}", e);
            std::process::exit(1);
        }
    };

    const SAMPLE_RATE: usize = 16000;
    const CHUNK_SIZE: usize = 30 * SAMPLE_RATE; // 30 seconds

    // Perform transcription
    match handle_transcription(whisper_path, samples, CHUNK_SIZE, audio_path) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
            std::process::exit(1);
        }
    }

    // Cleanup temporary directory
    match temp_dir.close() {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to clean up temporary directory: {}", e);
            std::process::exit(1);
        }
    };

    println!(
        "Raw output written to {}.",
        &format!(
            "{}_raw.txt",
            audio_path.file_stem().unwrap().to_string_lossy()
        )
    );
    println!(
        "Timestamped output written to {}.",
        &format!(
            "{}_timestamps.txt",
            audio_path.file_stem().unwrap().to_string_lossy()
        )
    );
}
