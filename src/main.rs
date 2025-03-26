use hound::{SampleFormat, WavReader};
use std::error::Error;
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::Command;
use std::time::Duration;
use tempfile::TempDir;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
use clap::Parser;

// If windows: use ./ffmpeg else use ffmpeg
const FFMPEG_PATH: &str = if cfg!(windows) {
    "./ffmpeg.exe"
} else {
    "ffmpeg"
};
const YT_DLP_PATH: &str = if cfg!(windows) {
    "./yt-dlp.exe"
} else {
    "yt-dlp"
};

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
    if Command::new(FFMPEG_PATH).output().is_ok() {
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
        sevenz_rust::decompress_file(temp_file.path(), Path::new("."))?;

        // Find the ffmpeg folder "ffmpeg*"
        let ffmpeg_folder = fs::read_dir(".")?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().ok().map_or(false, |t| t.is_dir()))
            .filter(|entry| entry.file_name().to_str().unwrap_or("").starts_with("ffmpeg"))
            .next();

        let ffmpeg_folder = match ffmpeg_folder {
            Some(folder) => folder,
            None => return Err("FFmpeg folder not found after download".into()),
        };

        // Move the ffmpeg folder to the current directory
        let src = ffmpeg_folder.path().join("bin").join("ffmpeg.exe");
        let dst = Path::new("ffmpeg.exe");

        println!("{} -> {}", src.to_str().unwrap(), dst.to_str().unwrap());

        fs::rename(src, dst)?;

        // Remove the temporary zip file
        fs::remove_file(temp_file.path())?;
        fs::remove_dir_all(ffmpeg_folder.path())?;
    }

    Ok(())
}

fn download_yt_dlp() -> Result<(), Box<dyn Error>> {
    // Check if yt-dlp is already installed
    if Command::new(YT_DLP_PATH).output().is_ok() {
        println!(
            "YT-DLP is already installed. Skipping download. If you want to reinstall, delete the FFmpeg binary and run this script again."
        );
        return Ok(());
    }

    Ok(())
}

fn ensure_wav_compatibility(
    input_path: &Path,
    output_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    Command::new(FFMPEG_PATH)
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

fn create_temporary_directory() -> Result<TempDir, Box<dyn Error>> {
    TempDir::new().map_err(|e| e.into())
}

struct Subtitle {
    seq: u32,
    start_time_cs: u64, // centiseconds
    end_time_cs: u64,   // centiseconds
    text: String,
}

fn cs_to_srt_time(cs: u64) -> String {
    let seconds = cs / 100;
    let milliseconds = (cs % 100) * 10; // Convert centiseconds to milliseconds
    let hours = (seconds / 3600) % 24;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    format!("{:02}:{:02}:{:02},{:03}", hours, minutes, seconds, milliseconds)
}

fn subtitle_to_srt(sub: &Subtitle) -> String {
    let start_str = cs_to_srt_time(sub.start_time_cs);
    let end_str = cs_to_srt_time(sub.end_time_cs);
    format!("{}\n{} --> {}\n{}\n", sub.seq, start_str, end_str, sub.text)
}

fn write_raw_transcript(subtitles: &[Subtitle], input_path: &Path) -> Result<(), Box<dyn Error>> {
    let raw_file_path = format!(
        "{}_raw.txt",
        input_path.file_stem().unwrap().to_string_lossy()
    );
    let mut out_file = fs::File::create(&raw_file_path)?;
    for sub in subtitles {
        out_file.write_all(sub.text.as_bytes())?;
    }
    Ok(())
}

fn handle_transcription(
    whisper_path: &Path,
    samples: Vec<f32>,
    chunk_size: usize,
    input_path: &Path,
    flash_attn: bool,
) -> Result<(), Box<dyn Error>> {
    let ctx = WhisperContext::new_with_params(
        &whisper_path.to_string_lossy(),
        WhisperContextParameters {
            flash_attn,
            ..Default::default()
        },
    )?;

    let mut state = ctx.create_state()?;
    let mut params = FullParams::new(SamplingStrategy::default());
    params.set_initial_prompt("experience");

    let sample_batches = samples.chunks(chunk_size).collect::<Vec<_>>();
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

    let mut subtitles = Vec::new();
    let mut seq_number = 1;
    let mut total_cs = 0;

    for samples in sample_batches {
        state
            .full(params.clone(), &samples)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

        let num_segments = state.full_n_segments()?;
        for i in 0..num_segments {
            let bytes = state.full_get_segment_bytes(i)?;
            let segment = String::from_utf8_lossy(&bytes).to_string();
            let start_timestamp_cs = state.full_get_segment_t0(i)? + total_cs;
            let end_timestamp_cs = state.full_get_segment_t1(i)? + total_cs;

            subtitles.push(Subtitle {
                seq: seq_number,
                start_time_cs: start_timestamp_cs as u64,
                end_time_cs: end_timestamp_cs as u64,
                text: segment,
            });

            seq_number += 1;
        }

        total_cs += (chunk_size as f32 / 16000.0 * 100.0) as i64; // Convert chunk size to centiseconds
        pb.inc(1);
    }

    pb.finish_with_message("Done");

    // Write subtitles to SRT file
    let srt_file_path = format!(
        "{}_timestamps.srt",
        input_path.file_stem().unwrap().to_string_lossy()
    );
    let mut out_file_srt = fs::File::create(&srt_file_path)?;
    for sub in &subtitles {
        out_file_srt.write_all(subtitle_to_srt(sub).as_bytes())?;
    }

    // Write subtitles to _timestamps.txt file
    let timestamps_file_path = format!(
        "{}_timestamps.txt",
        input_path.file_stem().unwrap().to_string_lossy()
    );
    let mut out_file_timestamps = fs::File::create(&timestamps_file_path)?;
    for sub in &subtitles {
        out_file_timestamps.write_all(
            format!(
                "[{} --> {}]: {}\n",
                cs_to_srt_time(sub.start_time_cs),
                cs_to_srt_time(sub.end_time_cs),
                sub.text
            )
            .as_bytes(),
        )?;
    }

    // Write raw transcript to raw.txt file
    match write_raw_transcript(&subtitles, input_path) {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Failed to write raw transcript: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}


// Usage: {} <path_to_wav_file> [model_path]
#[derive(Parser)]
struct Args {
    #[arg(help = "Path to the audio containing file")]
    audio_path: String, // Path to the audio file
    #[arg(help = "Path to the model")]
    model_path: Option<String>, // Path to the model
    #[arg(long, help = "Use flash attention")]
    fa: bool, // Use flash attention
}

fn main() {
    let args = Args::parse();

    // Introduce a temporary binding for the default model path
    let binding = "ggml-large-v3-turbo.bin".to_string();

    let audio_path = Path::new(&args.audio_path);
    if !audio_path.exists() {
        eprintln!("Error: Audio file does not exist at {}", &args.audio_path);
        return;
    }

    // Use the temporary binding in unwrap_or
    let model_path = args.model_path.unwrap_or(binding);
    let whisper_path = Path::new(&model_path);
    if !whisper_path.exists() {
        eprintln!("Model not found at {}", whisper_path.display());
        std::process::exit(1);
    }

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
    match handle_transcription(whisper_path, samples, CHUNK_SIZE, audio_path, args.fa) {
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
        "Timestamped output written to {} and {}.",
        &format!(
            "{}_timestamps.txt",
            audio_path.file_stem().unwrap().to_string_lossy()
        ),
        &format!(
            "{}_timestamps.srt",
            audio_path.file_stem().unwrap().to_string_lossy()
        )
    );
}
