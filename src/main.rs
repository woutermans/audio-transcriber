use hound::{SampleFormat, WavReader};
use std::fs;
use std::path::Path;
use std::process::Command;
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

fn get_audio_duration(audio_path: &Path) -> Result<f64, Box<dyn std::error::Error>> {
    let duration_output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run FFmpeg");

    println!("{}", String::from_utf8_lossy(&duration_output.stdout));

    if !duration_output.status.success() {
        return Err(format!("FFmpeg failed with status: {}", duration_output.status).into());
    }

    let output_str = String::from_utf8(duration_output.stdout)?;

    let duration = output_str
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse duration: {}", e))?;

    Ok(duration)
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

    // Get total duration of the audio
    let total_seconds = get_audio_duration(audio_path).expect("Failed to get audio duration");

    // Calculate number of chunks
    let chunk_duration = 120.0; // two minutes in seconds
    let num_chunks = (total_seconds / chunk_duration).ceil() as usize;

    let mut combined_transcript = String::new();
    let total_time = std::time::Duration::from_millis(0);

    for i in 0..num_chunks {
        let start_time = (i as f64) * chunk_duration;
        let end_time = if (i as f64 + 1.0) * chunk_duration > total_seconds {
            total_seconds
        } else {
            (i as f64 + 1.0) * chunk_duration
        };

        // Create a temporary directory for each chunk
        let temp_dir = TempDir::new().expect("Failed to create temporary directory");

        // Define the output file path for this chunk
        let chunk_path = temp_dir.path().join(format!("chunk_{}.wav", i));

        // Use FFmpeg to extract the chunk
        Command::new("ffmpeg")
            .args([
                "-ss",
                &format!("{:.2}", start_time),
                "-t",
                &format!("{:.2}", end_time - start_time),
                "-i",
                &audio_path.to_string_lossy(),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                chunk_path.to_str().unwrap(),
            ])
            .spawn()
            .expect("Failed to run FFmpeg")
            .wait()
            .expect("FFmpeg extraction failed");

        // Convert the chunk to WAV format
        let converted_chunk_path = temp_dir
            .path()
            .join(format!("temp_converted_chunk_{}.wav", i));
        Command::new("ffmpeg")
            .args([
                "-i",
                &chunk_path.to_string_lossy(),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                converted_chunk_path.to_str().unwrap(),
            ])
            .spawn()
            .expect("Failed to run FFmpeg")
            .wait()
            .expect("FFmpeg conversion failed");

        let original_samples = parse_wav_file(&converted_chunk_path);
        fs::remove_file(&converted_chunk_path).unwrap();

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

        // Record the start time for this chunk
        let st = std::time::Instant::now();

        state
            .full(params, &samples)
            .expect("failed to convert samples");

        // Record the end time after transcription is complete
        let et = std::time::Instant::now();

        // Calculate and print duration for this chunk
        let chunk_time = et - st;
        println!("Chunk {} took {:.2}ms", i, chunk_time.as_millis());

        // Collect all segment texts into a single string for this chunk
        let mut transcript = String::new();
        let num_segments = state
            .full_n_segments()
            .expect("failed to get number of segments");

        for j in 0..num_segments {
            match state.full_get_segment_text(j) {
                Ok(segment_text) => {
                    let segment_t0 = state.full_get_segment_t0(j).unwrap();
                    let segment_t1 = state.full_get_segment_t1(j).unwrap();

                    transcript.push_str(&format!(
                        "[{} - {}]: {}\n",
                        ((start_time + segment_t0 as f64) * 1000.0).round() as i64,
                        ((start_time + segment_t1 as f64) * 1000.0).round() as i64,
                        segment_text
                    ));
                }
                Err(e) => {
                    eprintln!("Error getting segment text at index {}: {:?}", j, e);
                    continue;
                }
            }
        }

        // Print the transcript immediately after processing the chunk
        println!("Chunk {} transcript:\n{}", i, transcript);

        // Add the chunk's transcript to the combined transcript
        combined_transcript.push_str(&transcript);

        // Clean up temporary files for this chunk
    }

    // Write the final transcript to 'output.txt'
    if !combined_transcript.is_empty() {
        let output_path = Path::new("output.txt");
        fs::write(output_path, &combined_transcript).expect("Failed to write to output.txt");
        println!("Transcription written to {}", output_path.display());
    } else {
        println!("No transcription available.");
    }

    // Calculate and print total duration
    println!("Total transcription took {:.2}ms", total_time.as_millis());
}
