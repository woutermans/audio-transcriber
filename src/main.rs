use hound::{SampleFormat, WavReader};
use std::fs;
use std::path::Path;
use std::process::Command;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

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

fn main() {
    let arg1 = std::env::args()
        .nth(1)
        .expect("first argument should be path to audio file");
    let audio_path = Path::new(&arg1);
    if !audio_path.exists() {
        panic!("audio file doesn't exist");
    }

    // Check if FFmpeg is installed
    match Command::new("ffmpeg").arg("-version").output() {
        Ok(output) => {
            if output.status.success() {
                println!("FFmpeg is already installed.");
            } else {
                println!("FFmpeg is not installed. Downloading now...");
                download_ffmpeg().expect("Failed to install FFmpeg");
            }
        }
        Err(_) => {
            println!("FFmpeg is not installed. Downloading now...");
            download_ffmpeg().expect("Failed to install FFmpeg");
        }
    }

    let arg2 = std::env::args()
        .nth(2)
        .expect("second argument should be path to Whisper model");
    let whisper_path = Path::new(&arg2);
    if !whisper_path.exists() {
        panic!("whisper file doesn't exist")
    }

    // Convert audio to WAV format using FFmpeg
    println!("Converting audio to WAV...");
    Command::new("ffmpeg")
        .args([
            "-i",
            &audio_path.to_string_lossy(),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "temp_converted.wav",
        ])
        .spawn()
        .expect("Failed to run FFmpeg")
        .wait()
        .expect("FFmpeg conversion failed");

    let original_samples = parse_wav_file(Path::new("temp_converted.wav"));
    fs::remove_file("temp_converted.wav").unwrap();

    let mut samples = vec![0.0f32; original_samples.len()];
    whisper_rs::convert_integer_to_float_audio(&original_samples, &mut samples)
        .expect("failed to convert samples");

    let ctx = WhisperContext::new_with_params(
        &whisper_path.to_string_lossy(),
        WhisperContextParameters::default(),
    )
    .expect("failed to open model");
    let mut state = ctx.create_state().expect("failed to create key");
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_initial_prompt("experience");
    params.set_progress_callback_safe(|progress| println!("Progress callback: {}%", progress));

    let st = std::time::Instant::now();
    state
        .full(params, &samples)
        .expect("failed to convert samples");
    let et = std::time::Instant::now();

    let num_segments = state
        .full_n_segments()
        .expect("failed to get number of segments");
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
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
    println!("took {}ms", (et - st).as_millis());
}
