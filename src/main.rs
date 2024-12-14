use hound::{SampleFormat, WavReader};
use std::io::Write;
use std::process::Command;
use std::time::Duration;
use std::{fs, path::Path};
use tempfile::TempDir;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

mod download_ggml_model;

fn parse_wav_file(path: &Path) -> Vec<i16> {
    let reader = WavReader::open(path).expect("failed to read file");

    if reader.spec().channels != 1 {
        panic!("expected mono audio file");
    }
    if reader.spec().sample_format != SampleFormat::Int {
        panic!("expected integer sample format");
    }
    if reader.spec().sample_rate != 16000 {
        panic!("expected 16KHz sample rate");
    }
    if reader.spec().bits_per_sample != 16 {
        panic!("expected 16 bits per sample");
    }

    reader
        .into_samples::<i16>()
        .map(|x| x.expect("sample"))
        .collect::<Vec<_>>()
}

fn download_ffmpeg() -> Result<(), Box<dyn std::error::Error>> {
    // Check if already installed
    if Command::new("ffmpeg").output().is_ok() {
        println!(
            "ffmpeg is already installed. Skipping download. If you want to reinstall, delete the ffmpeg binary and run this script again."
        );
        return Ok(());
    }

    if cfg!(target_os = "windows") {
        let url = "https://objects.githubusercontent.com/github-production-release-asset-2e65be/292087234/a99db424-f32b-407e-810f-2ecb9ee16873?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20241214%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20241214T000504Z&X-Amz-Expires=300&X-Amz-Signature=f360238ecf53d898f99634a655e1253166d2dd595b7be8d4659297f724292db3&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dffmpeg-master-latest-win64-gpl.zip&response-content-type=application%2Foctet-stream";

        println!("Downloading FFmpeg for Windows...");
        let response = reqwest::blocking::get(url)?;
        if !response.status().is_success() {
            return Err("Failed to download FFmpeg".into());
        }

        let temp_file = tempfile::NamedTempFile::new()?;
        fs::write(temp_file.path(), &response.bytes()?)?;

        println!("Extracting FFmpeg...");
        let unzip_status = Command::new("unzip")
            .arg("-o")
            .arg(temp_file.path())
            .current_dir(".")
            .spawn()?
            .wait()?;

        if !unzip_status.success() {
            return Err("Failed to extract FFmpeg".into());
        }

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

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("first argument should be path to WAV file");
    let audio_path = Path::new(&arg1);
    if !audio_path.exists() {
        panic!("audio file doesn't exist");
    }
    let arg2 = std::env::args()
        .nth(2)
        .unwrap_or("ggml-large-v3-turbo.bin".to_string());
    let whisper_path = Path::new(&arg2);

    // download_ggml_model::download_and_extract_model(&arg2, Path::new("models"), None).unwrap();

    // let whisper_path = Path::new("models").join(&format!("{}.bin", arg2));

    // Create a temporary directory
    let temp_dir = TempDir::new().expect("Failed to create temporary directory");

    // Define the output path for the converted audio
    let output_path = temp_dir.path().join("converted_audio.wav");

    download_ffmpeg().expect("Failed to download ffmpeg. Please install ffmpeg and try again.");

    // Ensure the WAV file meets the requirements using FFmpeg
    ensure_wav_compatibility(audio_path, &output_path).expect("Failed to ensure WAV compatibility");

    let original_samples = parse_wav_file(&output_path);
    let mut samples = vec![0.0f32; original_samples.len()];

    whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)
        .expect("failed to convert samples");

    const SAMPLE_RATE: usize = 16000;
    const N_FRAMES: usize = 30 * SAMPLE_RATE; // 30 seconds
    let sample_batches = samples.chunks(N_FRAMES).into_iter().collect::<Vec<_>>();

    let ctx = WhisperContext::new_with_params(
        &whisper_path.to_string_lossy(),
        WhisperContextParameters {
            flash_attn: true,
            ..Default::default()
        },
    )
    .expect("failed to open model");
    let mut state = ctx.create_state().expect("failed to create key");
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_initial_prompt("experience");
    // params.set_progress_callback_safe(|progress| println!("Progress callback: {}%", progress));

    let st = std::time::Instant::now();

    let out_file_path = format!(
        "{}_timestamps.txt",
        audio_path.file_stem().unwrap().to_string_lossy()
    );
    let out_raw_path = format!(
        "{}_raw.txt",
        audio_path.file_stem().unwrap().to_string_lossy()
    );

    let mut out_file = std::fs::File::create(&out_file_path).unwrap();
    let mut out_file_raw = std::fs::File::create(&out_raw_path).unwrap();

    let mut last_timestamp = 0;
    let chunk_count = sample_batches.len();
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
    for samples in sample_batches {
        state
            .full(params.clone(), &samples)
            .expect("failed to convert samples");

        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");

        let mut e_time = 0;
        for i in 0..num_segments {
            let segment = state
                .full_get_segment_text(i)
                .expect("failed to get segment");
            let start_timestamp = state
                .full_get_segment_t0(i)
                .expect("failed to get start timestamp");
            let end_timestamp = state
                .full_get_segment_t1(i)
                .expect("failed to get end timestamp");
            // println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
            out_file
                .write_all(
                    &format!(
                        "[{} - {}]: {}\n",
                        start_timestamp + last_timestamp,
                        end_timestamp + last_timestamp,
                        segment
                    )
                    .as_bytes(),
                )
                .unwrap();
            out_file_raw
                .write_all(format!("{} ", segment).as_bytes())
                .unwrap();
            e_time = end_timestamp;
        }
        last_timestamp = e_time;
        pb.inc(1);
    }
    pb.finish_with_message("Done");
    let et = std::time::Instant::now();
    println!("took {}ms", (et - st).as_millis());
    println!("processed {} chunks", chunk_count);
    println!("Raw output written to {}.", out_raw_path);
    println!("Timestamped output written to {}.", out_file_path);

    // Cleanup: Remove the temporary directory and its contents
    temp_dir
        .close()
        .expect("Failed to clean up temporary directory");
}
