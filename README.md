# Audio Transcriber [![License](https://img.shields.io/badge/License-Unlicense-blue.svg)](LICENSE)

A command-line tool for transcribing audio files using the Whisper ASR model with GPU acceleration support.

---

## Features

### Whisper Integration
- Uses the whisper-rs crate ([tazz4843/whisper-rs](https://github.com/tazz4843/whisper-rs)) to provide accurate transcription
- Supports all Whisper model sizes (tiny, base, small, medium, large)
- Flash Attention acceleration for faster inference

### Multi-backend Support
| Backend    | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| **Vulkan** | Cross-platform GPU acceleration via Vulkan API                             |
| **CUDA**   | NVIDIA GPU optimization using CUDA                                         |
| **HIPBLAS**| AMD GPU support through ROCm platform                                      |
| **Metal**  | Optimized performance on Apple Silicon devices                            |

---

## Prerequisites

1. ### Rust Toolchain
   - Install Rust: [rust-lang.org](https://www.rust-lang.org/tools/install)

2. ### FFmpeg
   - The tool will automatically download pre-built binaries if missing
   - Manual installation:
     ```bash
     # macOS (Homebrew)
     brew install ffmpeg
     # Ubuntu
     sudo apt-get install ffmpeg
     ```

3. ### GPU Requirements (Optional for acceleration):
   | Backend    | Minimum Requirement                                  |
   |------------|-----------------------------------------------------|
   | **Vulkan** | Vulkan 1.2 compatible graphics drivers              |
   | **CUDA**   | NVIDIA driver + CUDA Toolkit v11+                   |
   | **HIPBLAS**| ROCm platform (AMD GPUs)                             |
   | **Metal**  | Apple Silicon Mac with macOS Monterey or later      |

---

## Installation

### From Source
```bash
# Clone repository
git clone https://github.com/woutermans/audio-transcriber.git
cd audio-transcriber

# Compile with GPU backend of choice:
cargo build --release --features <backend> 
```

Available features: `vulkan`, `cuda`, `hipblas`, `metal`

### Example Builds
```bash
# Basic CPU mode (no GPU)
cargo build --release 

# NVIDIA GPU acceleration
cargo build --release --features cuda

# Cross-platform GPU support via Vulkan
cargo build --release --features vulkan
```

---

## Usage

### Basic Transcription
```bash
./target/release/audio-transcriber [OPTIONS] <input_path>
```

#### Input Path Options:
- Local audio/video files (WAV, MP3, etc.)
- YouTube URLs supported via embedded yt-dlp integration

#### Common Parameters:
| Flag               | Description                                  |
|--------------------|----------------------------------------------|
| `--model-path`     | Specify custom model path (default: ./ggml-large-v3-turbo.bin) |
| `--fa`   | Enable Flash Attention |

---

### Example Workflow
1. Transcribe local file with GPU acceleration:
```bash
cargo run --release --features vulkan -i audio.mp3 
```

2. Use specific model and enable Flash Attention:
```bash
./target/release/audio-transcriber video.mp4 \
    --model-path ./models/ggml-medium.bin \
    --fa
```

---

### Output Files
For input file `sample_audio.mp3` produces:
- Raw transcript: `sample_audio_raw.txt`
- Timestamped SRT file: `sample_audio_timestamps.srt`
- Formatted timestamps: `sample_audio_timestamps.txt`

---

## Dependencies

| Crate          | Purpose                                |
|----------------|----------------------------------------|
| **whisper-rs** | Core transcription engine              |
| **hound**      | WAV audio parsing                      |
| **reqwest**    | FFmpeg download requests               |
| **clap**       | Command-line argument parsing          |
| **indicatif**  | Progress bar display                   |

---

## Contribution Workflow

1. Fork repository
2. Create new branch:
   ```bash
   git checkout -b feature/my-feature
   ```
3. Make changes & test with:
   ```bash
   cargo clippy    # Code linting
   cargo test      # Unit tests
   ```
4. Submit PR for review

---

## License

This project is unlicensed (Public Domain). See [LICENSE](./LICENSE) file.