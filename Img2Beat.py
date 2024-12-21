# Python Imports
import sys
import os
import math
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
from threading import Lock
import glob
from typing import List, Tuple, Optional
import traceback

# Third-party imports
from PIL import Image
import librosa
import librosa.display
import numpy as np
import scipy.ndimage

# ANSI color codes for colored output
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Constants
MIN_DURATION = 0.15  # Minimum duration for a segment in seconds
MAX_DURATION = 1  # Maximum duration for a segment in seconds
DEBUG_MODE = True  # Set to True for detailed logging

def debug_print(message: str) -> None:
    """Print debug messages if DEBUG_MODE is True."""
    if DEBUG_MODE:
        print(f"{YELLOW}DEBUG: {message}{RESET}")

def get_input_or_arg(index: int, prompt: str) -> str:
    """
    Get input from CLI arguments if available, otherwise prompt the user.
    """
    if len(sys.argv) > index:
        return sys.argv[index]
    return input(prompt).strip()

def make_even(image_path: str) -> None:
    """Ensure image dimensions are even for FFmpeg compatibility."""
    img = Image.open(image_path)
    width, height = img.size
    new_width = math.ceil(width / 2) * 2
    new_height = math.ceil(height / 2) * 2
    img = img.resize((new_width, new_height))
    img.save(image_path)

def get_absolute_path(file_or_folder: str) -> str:
    return os.path.abspath(file_or_folder)

def animated_print(message: str, done_message="done ✅", delay=0.25, repeat=3):
    """
    Displays an animated message with dots, and replaces it with a final message.

    Args:
        message (str): The message to display during animation.
        done_message (str): The final message to display.
        delay (float): Time delay between each dot update.
        repeat (int): Number of dot updates before marking as done.
    """
    for i in range(repeat + 1):  # Add 1 for the clean ending
        dots = "." * (i % 4)  # Cycles through "", ".", "..", "..."
        sys.stdout.write(f"\r{message}{dots} ")  # Overwrite the line
        sys.stdout.flush()  # Ensure immediate output
        time.sleep(delay)
    
    sys.stdout.write(f"\r{message}... {done_message}\n")  # Final message
    sys.stdout.flush()

def spinner_animation(message:str, duration=3):
    """
    Displays a spinner animation while a task runs.
    
    Args:
        message (str): Message to display alongside the spinner.
        duration (float): How long the animation runs in seconds.
    """
    spinner = ["|", "/", "-", "\\"]
    start_time = time.time()
    while time.time() - start_time < duration:
        for char in spinner:
            sys.stdout.write(f"\r{message} {char}")
            sys.stdout.flush()
            time.sleep(0.1)  # Controls spinner speed

    sys.stdout.write(f"\r{message} done ✅\n")
    sys.stdout.flush()

def advanced_beat_detection(y: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """
    Improved beat detection with dynamic thresholding, low-pass filtering, 
    and manual tempo adjustments using PLP (Predominant Local Pulse).
    
    Args:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate.

    Returns:
        beat_times (np.ndarray): Timestamps of detected beats.
        tempo (float): Estimated tempo in BPM.
    """
    # Step 1: Onset Envelope with Dynamic Thresholding
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_env = np.clip(onset_env, 0, None)  # Remove negative values
    onset_env = librosa.util.normalize(onset_env)  # Normalize to range [0, 1]
    
    # Apply dynamic threshold to ignore low-energy noise
    threshold = 0.1  # Adjust as needed
    onset_env[onset_env < threshold] = 0
    
    # Step 2: Low-pass Filtering to Smooth Onset Envelope
    onset_env_smooth = scipy.ndimage.gaussian_filter1d(onset_env, sigma=2)
    
    # Step 3: Dynamic Tempo Adjustments using PLP
    plp = librosa.beat.plp(onset_envelope=onset_env_smooth, sr=sr)
    beat_frames = librosa.util.localmax(plp)  # Detect local peaks
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Step 4: Refine Tempo Detection
    tempo = librosa.feature.tempo(onset_envelope=onset_env_smooth, sr=sr)
    tempo = tempo[0]  # Extract the scalar value from the array
    
    debug_print(f"Estimated tempo: {tempo:.2f} BPM")
    debug_print(f"Detected {len(beat_times)} beats with dynamic smoothing and PLP.")
    
    # if DEBUG_MODE:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 4))
    #     librosa.display.waveshow(y, sr=sr, alpha=0.5)
    #     plt.vlines(beat_times, ymin=-1, ymax=1, color='r', linestyle='--', label='Beats')
    #     plt.title('Waveform with Detected Beats')
    #     plt.legend()
    #     plt.show()
        
    return beat_times, tempo

def calculate_beat_intervals(beats: List[float]) -> List[float]:
    if len(beats) < 2:
        return []
    return [max(beats[i] - beats[i - 1], MIN_DURATION) for i in range(1, len(beats))]


def select_images_for_beats(
    images: List[str], 
    beat_times: np.ndarray, 
    segment_start: float, 
    segment_end: float
) -> Tuple[List[str], List[float]]:
    """
    Dynamically select images and calculate durations based on beat intervals.
    Ensures smooth durations and handles irregular tempo.

    Args:
        images (List[str]): List of image file paths.
        beat_times (np.ndarray): Array of detected beat times.
        segment_start (float): Start time of the segment.
        segment_end (float): End time of the segment.

    Returns:
        Tuple[List[str], List[float]]: Selected images and corresponding durations.
    """
    # Filter beats within segment boundaries
    segment_beats = beat_times[(beat_times >= segment_start) & (beat_times <= segment_end)]

    # Add boundaries explicitly
    segment_beats = np.concatenate([[segment_start], segment_beats, [segment_end]])

    # Calculate beat intervals
    beat_intervals = np.diff(segment_beats)
    beat_intervals = np.clip(beat_intervals, MIN_DURATION, MAX_DURATION)

    debug_print(f"Segment beats: {segment_beats}")
    debug_print(f"Beat intervals: {beat_intervals}")
    debug_print(f"Total segment duration: {segment_end - segment_start:.2f} seconds")

    # Ensure beat intervals add up to the total segment duration
    total_segment_duration = segment_end - segment_start
    scaling_factor = total_segment_duration / np.sum(beat_intervals)
    beat_intervals *= scaling_factor

    if len(beat_intervals) == 0:
        raise ValueError("No valid beat intervals found for the segment.")

    # Cycle through images to match the number of beat intervals
    num_beats = len(beat_intervals)
    selected_images = [images[i % len(images)] for i in range(num_beats)]

    return selected_images, beat_intervals.tolist()


def clean_up_files(file_paths: List[str]) -> None:
    """
    Delete a list of files safely.
    """
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"{RED}Error removing file {file_path}: {e}{RESET}")

def run_ffmpeg_command(command: List[str], description: str):
    """
    Run an FFmpeg command and wait for it to finish.
    """
    spinner_animation(f"Running FFmpeg command: {description}...")
    
    try:
        debug_print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        print(f"{GREEN}Success: {description}{RESET}")
    except subprocess.CalledProcessError as e:
        print(f"{RED}Error during FFmpeg command: {description}{RESET}")
        print(e.stderr.decode())
        raise

def concatenate_segments(segment_paths: List[str], output_path: str) -> None:
    segment_paths = [os.path.abspath(path) for path in segment_paths if os.path.exists(path)]
    with open("file_list.txt", "w") as f:
        for path in segment_paths:
            f.write(f"file '{path}'\n")
    run_ffmpeg_command(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "file_list.txt", "-c", "copy", output_path], "Concatenating segments")

def validate_image_list(image_list_path: str) -> bool:
    """
    Validates the image list file to ensure it conforms to FFmpeg expectations.
    """
    try:
        with open(image_list_path, "r") as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            # Strip whitespace
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Check for 'file' lines
            if line.startswith("file"):
                file_path = line.split(" ", 1)[1].strip("'")
                if not os.path.isfile(file_path):
                    print(f"{RED}Error: Image file does not exist: {file_path}{RESET}")
                    return False
                continue

            # Check for 'duration' lines
            if line.startswith("duration"):
                try:
                    duration = float(line.split(" ")[1])
                    if duration < MIN_DURATION:
                        print(f"{RED}Error: Duration under minimum limit: {duration}{RESET}")
                        return False
                    elif duration > MAX_DURATION:
                        print(f"{RED}Warning: Duration exceeds maximum limit: {duration}{RESET}")
                except ValueError:
                    print(f"{RED}Error: Invalid duration format: {line}{RESET}")
                    return False

        # If no issues, return True
        return True

    except Exception as e:
        print(f"{RED}Error validating image list file {image_list_path}: {e}{RESET}")
        return False

def create_image_list_file(image_paths: List[str], beat_intervals: List[float], output_file: str, segment_duration: float) -> None:
    """
    Create an FFmpeg-compatible image list file with durations based on beat intervals.
    """
    if not beat_intervals or not image_paths:
        raise ValueError("beat_intervals and image_paths must not be empty")
    
    # Ensure the beat intervals sum matches the segment duration
    total_duration = sum(beat_intervals)
    if total_duration > 0:
        scaling_factor = segment_duration / total_duration
        beat_intervals = [max(duration * scaling_factor, MIN_DURATION) for duration in beat_intervals]
    else:
        raise ValueError("Total beat interval duration must be greater than 0")
    
    debug_print(f"Adjusted beat intervals: {beat_intervals}")
    debug_print(f"Total segment duration: {segment_duration:.2f} seconds")
    
    with open(output_file, "w") as f:
        for idx, image_path in enumerate(image_paths):
            duration = beat_intervals[idx % len(beat_intervals)]  # Cycle through scaled beat intervals
            duration = max(duration, MAX_DURATION)
            f.write(f"file '{image_path}'\n")
            f.write(f"duration {duration:.3f}\n")
        
        # Add the last frame to meet FFmpeg requirements
        f.write(f"file '{image_paths[-1]}'\n")

    print(f"Image list file created successfully: {output_file}")

def calculate_beat_intervals(beat_times: np.ndarray) -> List[float]:
    """
    Calculate time intervals (durations) between consecutive beats.
    """
    return [beat_times[i+1] - beat_times[i] for i in range(len(beat_times)-1)]

def process_segment(start: float, end: float, segment_index: int, audio_path: str, images: List[str], beat_times: np.ndarray, output_folder: str, tempo: float) -> Optional[str]:
    """
    Process a segment of audio with beat-synchronized image changes.
    """
    os.makedirs(output_folder, exist_ok=True)

    segment_video_path = os.path.join(output_folder, f"segment_{segment_index}.mp4")
    audio_segment_path = os.path.join(output_folder, f"audio_segment_{segment_index}.mp3")
    image_list_path = os.path.join(output_folder, f"image_list_{segment_index}.txt")

    try:
        # Select images and calculate durations based on beats
        segment_images, durations = select_images_for_beats(
            images,
            beat_times,
            start,
            end
        )

        # Handle fallback if no beats are detected
        if not durations:
            debug_print("Fallback: Using default duration for images.")
            total_segment_duration = end - start
            
            debug_print(f"Total segment duration: {total_segment_duration:.2f} seconds")
            
            default_duration = min(MAX_DURATION, max(MIN_DURATION, 60.0 / tempo))
            num_repeats = int(total_segment_duration / default_duration)
            segment_images = [images[i % len(images)] for i in range(num_repeats)]
            durations = [default_duration] * num_repeats

        # Create the image list file
        create_image_list_file(segment_images, durations, image_list_path, end - start)
        debug_print(f"Image list file created successfully: {image_list_path}")

        # Validate the image list file
        if not validate_image_list(image_list_path):
            print(f"{YELLOW}Warning: Validation failed for image list {image_list_path}.{RESET}")
            return None

        # Create video segment
        ffmpeg_video_cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", image_list_path,
            "-vf", "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", segment_video_path
        ]
        run_ffmpeg_command(ffmpeg_video_cmd, f"Creating video segment {segment_index} from images")

        # Extract audio segment
        ffmpeg_audio_cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end), "-c", "copy", audio_segment_path
        ]
        run_ffmpeg_command(ffmpeg_audio_cmd, f"Extracting audio segment {segment_index}")

        # Merge video and audio
        final_segment_path = os.path.join(output_folder, f"final_segment_{segment_index}.mp4")
        ffmpeg_merge_cmd = [
            "ffmpeg", "-y", "-i", segment_video_path, "-i", audio_segment_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest", final_segment_path
        ]
        run_ffmpeg_command(ffmpeg_merge_cmd, f"Merging video and audio for segment {segment_index}")

        # Clean up temporary files
        clean_up_files([segment_video_path, audio_segment_path, image_list_path])

        return final_segment_path

    except Exception as e:
        print(f"{RED}Error processing segment {segment_index}: {e}{RESET}")
        traceback.print_exc()
        return None
    
def main():
    # Get audio path and images folder from CLI or input
    audio_path = get_absolute_path(get_input_or_arg(1, "Enter the path to the audio file (e.g., 'your_song.mp3'): "))
    image_folder = get_absolute_path(get_input_or_arg(2, "Enter the path to the folder containing images: "))
    segment_folder = get_absolute_path("output_segments")

    # Validate input paths
    if not os.path.isfile(audio_path):
        print(f"{RED}Error: The audio file '{audio_path}' does not exist.{RESET}")
        return

    if not os.path.isdir(image_folder):
        print(f"{RED}Error: The folder '{image_folder}' does not exist.{RESET}")
        return
    
    # Create output folder for segments
    os.makedirs(segment_folder, exist_ok=True)

    start_time = time.time()
    
    # Load audio and detect beats
    animated_print(f"Analyzing audio for beats")
    y, sr = librosa.load(audio_path, sr=None)
    beat_times, tempo = advanced_beat_detection(y, sr)
    
    if len(beat_times) == 0:
        raise ValueError("No beats detected in the audio. Check the beat detection logic or input audio file.")
    
    # Prepare images
    for image in os.listdir(image_folder):
        if image.lower().endswith(('.jpg', '.png', '.jpeg')):
            make_even(os.path.join(image_folder, image))
            
    images = sorted([
        os.path.join(image_folder, f) 
        for f in os.listdir(image_folder) 
        if f.lower().endswith(('png', 'jpg', 'jpeg'))
    ])
    
    if not images:
        print(f"{RED}Error: No image files found in the folder '{image_folder}'.{RESET}")
        return
    
    if len(images) < len(beat_times):
        print("Warning: The number of images is fewer than the detected beats. Some beats may not have corresponding images.")

    print(f"{CYAN}Using {len(images)} images from the folder.{RESET}")
    
    # Segment settings
    segment_duration = 40  # Duration of each segment in seconds
    total_duration = librosa.get_duration(y=y, sr=sr)
    segments = [(i, min(i + segment_duration, total_duration)) for i in range(0, int(total_duration), segment_duration)]\

    # Get available logical processors and adjust thread count
    max_threads = min(4, len(segments))  # Limit to 4 threads for debugging
    print(f"{CYAN}Using {max_threads} threads for parallel processing.{RESET}\n")
    
    # Process segments
    segment_paths = []
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(
                process_segment,
                start,
                end,
                idx,
                audio_path,
                images,
                beat_times,
                segment_folder,
                tempo
            ) : idx
            for idx, (start, end) in enumerate(segments)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result(timeout=300)
                if result:
                    segment_paths.append(result)
                else:
                    print(f"{YELLOW}Warning: Segment {idx} returned an empty result.{RESET}")
            except Exception as e:
                print(f"{RED}Error processing segment {idx}: {e}{RESET}")
                traceback.print_exc()  # Print full error trace for debugging
        
        # Collect segment paths
        segment_paths = [future.result() for future in futures if future.result()]
        
    print(f"\n{CYAN}Processed {len(segment_paths)} segments.{RESET}")

    # Concatenate the segments
    print(f"Concatenating segments...")
    print(f"{CYAN}Segment Paths:{RESET} {segment_paths}")
    concatenate_segments(segment_paths, "final_output.mp4")
    
    print("Cleaning up temporary files...")
    clean_up_files(segment_paths)
    
    print(f"\n{GREEN}Finished in {time.time() - start_time:.2f} seconds.{RESET}")

if __name__ == "__main__":
    main()