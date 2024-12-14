# Audio Transcriber

[![License](https://img.shields.io/badge/License-Unlicense-blue.svg)](LICENSE)

A robust command-line tool for transcribing audio files using the powerful Whisper model, supported by various GPU backends: Vulkan, CUDA, HIPBLAS, and Metal.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
  - [Whisper Integration](#whisper-integration)
  - [Multi-backend Support](#multi-backend-support)
    - [Vulkan](#vulkan)
    - [CUDA](#cuda)
    - [HIPBLAS](#hipblas)
    - [Metal](#metal)
- [Prerequisites](#prerequisites)
  - [Rust Toolchain](#rust-toolchain)
  - [FFmpeg](#ffmpeg)
- [Installation](#installation)
  - [Using Cargo](#using-cargo)
  - [Supported Backends Installation](#supported-backends-installation)
    - [Vulkan](#vulkan)
    - [CUDA](#cuda)
    - [HIPBLAS](#hipblas)
    - [Metal](#metal)
- [Usage](#usage)
  - [Basic Transcription](#basic-transcription)
  - [Handling Multiple Backends](#handling-multiple-backends)
- [Dependencies](#dependencies)
  - [Crate Dependencies](#crate-dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The `audio-transcriber` is designed to transcribe audio files using the advanced Whisper model. It offers flexible support for different GPU backends, allowing you to leverage hardware acceleration with Vulkan, CUDA, HIPBLAS, and Metal. This ensures efficient processing of large audio files while maintaining high accuracy.

## Features

- **Whisper Integration**: Utilizes the [whisper-rs](https://github.com/alyssaq/rust-whisper) library for accurate transcription.
- **Multi-backend Support**:
  - **Vulkan**: Leverages GPU acceleration using the Vulkan API.
  - **CUDA**: Optimized for NVIDIA GPUs with CUDA support.
  - **HIPBLAS**: Utilizes AMD GPUs with HIPBLAS for high-performance linear algebra operations.
  - **Metal**: Supports Apple's Metal API for optimized performance on macOS.

## Prerequisites

Before installing and running the `audio-transcriber`, ensure you have the following prerequisites:

- **Rust Toolchain**:
  - Ensure Rust is installed. You can download it from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install).
  
- **FFmpeg**:
  - The program checks for FFmpeg's presence and downloads it if missing. However, manual installation is possible on specific operating systems.

## Installation

### Using Cargo

The recommended method to install the `audio-transcriber` is via Cargo, Rust's package manager.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/woutermans/audio-transcriber.git
   cd audio-transcriber
   ```

2. **Install Dependencies and Compile with Backend Features**

   To compile with specific backend support (e.g., Vulkan), enable the corresponding feature flag.

   - **For Vulkan**
     ```bash
     cargo build --release --features vulkan
     ```
   
   - **For CUDA**
     ```bash
     cargo build --release --features cuda
     ```
   
   - **For HIPBLAS**
     ```bash
     cargo build --release --features hipblas
     ```
   
   - **For Metal**
     ```bash
     cargo build --release --features metal
     
   **Note**: Some backends may require additional system dependencies. Please refer to the [Cargo.toml](./Cargo.toml) for specific feature dependencies.

3. **Using `cargo install`**

   Alternatively, you can install it globally using Cargo's `install` command with a specified backend:

   ```bash
   cargo install audio-transcriber --features vulkan
   ```

## Supported Backends Installation

The `audio-transcriber` supports the following GPU backends for enhanced performance. Enabling a feature will enable the corresponding backend during compilation.

- **Vulkan**
  - Utilizes GPU acceleration using the Vulkan API.
  - Requires system libraries and drivers compatible with Vulkan.
  
- **CUDA**
  - Optimized for NVIDIA GPUs.
  - Requires CUDA toolkit installation on your system.
  
- **HIPBLAS**
  - Leverages AMD GPUs with HIPBLAS support.
  - Requires ROCm or other compatible GPU drivers.

- **Metal**
  - Supports Apple's Metal API for optimized performance on macOS systems.

**Additional Notes:**

Ensure that your system meets the hardware and software requirements for each backend. Refer to the documentation provided by NVIDIA, AMD, or Apple for installation guides specific to each GPU architecture.

## Usage

1. **Basic Transcription**

   ```bash
   cargo run --release --features vulkan /path/to/audio.wav [model_path]
   ```

   Replace `/path/to/audio.wav` with the path to your WAV file and optionally provide a custom model path (default is `ggml-large-v3-turbo.bin`).

2. **Handling Multiple Backends**

   To switch between backends, enable the desired feature flag during compilation or use environment variables if supported.

## Dependencies

The project relies on several crates for functionality:

- `hound`: For reading WAV files.
- `whisper-rs`: Integration with the Whisper model.
- `reqwest`: Handles HTTP requests for downloading FFmpeg.
- `tempfile` and `zip`: Manage temporary files and compress/decompress archives.
- `indicatif`: Displays progress bars during transcription.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a New Branch**
3. **Make Your Changes**
4. **Run Tests**
   ```bash
   cargo test
   ```
5. **Submit a Pull Request**

Please ensure your code adheres to Rust's best practices and the project's coding standards.

## License

This project is released under the Unlicense - see the [LICENSE](LICENSE) file for details.