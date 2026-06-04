# Simple Live Acoustic Camera with Pre-Filtering
import argparse
import time
from datetime import datetime
import acoular as ac
import numpy as np
import cv2
import sounddevice as sd
from os import path
import os

# Disable Acoular's HDF5 caching for live processing
ac.config.global_caching = 'none'

# -----------------------------------------------------------------------------
# HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
UMA16_DEVICE_INDEX = None  # Line (MCHStreamer Multi-channels), Windows WASAPI
NUM_CHANNELS = 16
SAMPLE_RATE = 48000
CAMERA_INDEX = 1

# Load UMA-16 microphone geometry
micgeofile = path.join(path.split(ac.__file__)[0], 'xml', 'minidsp_uma-16_mirrored.xml')
mg = ac.MicGeom(file=micgeofile)

# -----------------------------------------------------------------------------
# ACOUSTIC GRID & PROCESSING SETTINGS
# -----------------------------------------------------------------------------
TARGET_FREQ = 2000.0   # Target frequency in Hz
BLOCK_SIZE = 2048      # Samples per processing block
BLEND_ALPHA = 0.75     # Video transparency
DEFAULT_GRID_HALF_WIDTH = 0.2
DEFAULT_GRID_Z = 0.3
DEFAULT_GRID_POINTS = 41
DEFAULT_DB_MODE = 'max'
DEFAULT_DB_RANGE = 3.0

def parse_grid_spec(spec):
    """Parse --grid half_width,z,points into RectGrid params."""
    parts = [p.strip() for p in spec.split(',')]
    if len(parts) != 3:
        raise ValueError("--grid must be in format half_width,z,points (example: 0.2,0.3,41)")

    half_width = float(parts[0])
    z = float(parts[1])
    points = int(parts[2])

    if half_width <= 0:
        raise ValueError("grid half_width must be > 0")
    if z <= 0:
        raise ValueError("grid z must be > 0")
    if points < 2:
        raise ValueError("grid points must be >= 2")

    increment = (2.0 * half_width) / (points - 1)
    return half_width, z, points, increment


def parse_db_spec(spec):
    """Parse --db db_max,db_range where db_max is 'max' or float."""
    parts = [p.strip() for p in spec.split(',')]
    if len(parts) != 2:
        raise ValueError("--db must be in format db_max,db_range (example: max,3 or 85,3)")

    db_max_token = parts[0].lower()
    if db_max_token == 'max':
        db_mode = 'max'
        db_fixed_max = None
    else:
        db_mode = 'fixed'
        db_fixed_max = float(parts[0])

    db_range = float(parts[1])
    if db_range <= 0:
        raise ValueError("db_range must be > 0")

    return db_mode, db_fixed_max, db_range


def build_arg_parser():
    default_grid = f"{DEFAULT_GRID_HALF_WIDTH},{DEFAULT_GRID_Z},{DEFAULT_GRID_POINTS}"
    default_db = f"{DEFAULT_DB_MODE},{DEFAULT_DB_RANGE:g}"
    parser = argparse.ArgumentParser(description='Simple live acoustic camera with pre-filtered beamforming')
    parser.add_argument(
        '--grid',
        type=str,
        default=default_grid,
        help='Grid as half_width,z,points. Example: --grid 0.2,0.3,41'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=default_db,
        help="dB normalization as db_max,db_range where db_max is 'max' or a fixed value. Example: --db max,3 or --db 85,3"
    )
    parser.add_argument(
        '--snapshot-threshold',
        type=float,
        default=None,
        help='Optional peak-trigger threshold in dB SPL for saving frames (example: --snapshot-threshold 55)'
    )
    return parser

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_uma16_device():
    """Find and verify the UMA-16 audio device."""
    global UMA16_DEVICE_INDEX

    devices = sd.query_devices()
    
    if UMA16_DEVICE_INDEX is None:
        # Search for a device with NUM_CHANNELS input channels
        print(f"\nSearching for a device with {NUM_CHANNELS} input channels...")
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] >= NUM_CHANNELS:
                print(f"Found suitable device at index {idx}:")
                print(f"  Name: {dev['name']}")
                print(f"  Channels: {dev['max_input_channels']} in")
                print(f"  Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
                UMA16_DEVICE_INDEX = idx
                return True
        print(f"⚠️  No device with {NUM_CHANNELS} input channels found!")
        return False
    else:
        print(f"\nUsing audio device {UMA16_DEVICE_INDEX}:")
        if UMA16_DEVICE_INDEX < len(devices):
            dev = devices[UMA16_DEVICE_INDEX]
            print(f"  Name: {dev['name']}")
            print(f"  Channels: {dev['max_input_channels']} in")
            print(f"  Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
            
            if dev['max_input_channels'] < NUM_CHANNELS:
                print(f"\n⚠️  WARNING: Device only has {dev['max_input_channels']} input channels!")
                return False
            return True
        else:
            print(f"⚠️  Device {UMA16_DEVICE_INDEX} not found!")
            return False

def find_camera():
    """Find and verify the camera."""
    print(f"\nUsing camera {CAMERA_INDEX}")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Resolution: {width}x{height}")
        cap.release()
        return True
    else:
        print(f"⚠️  Camera {CAMERA_INDEX} not found!")
        return False

# -----------------------------------------------------------------------------
# LIVE CAPTURE & PROCESSING LOOP
# -----------------------------------------------------------------------------
def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        grid_half_width, grid_z, grid_points, grid_increment = parse_grid_spec(args.grid)
        db_mode, db_fixed_max, db_range = parse_db_spec(args.db)
    except ValueError as e:
        parser.error(str(e))
        return

    # Rectangular grid from CLI or defaults
    rg = ac.RectGrid(
        x_min=-grid_half_width, x_max=grid_half_width,
        y_min=-grid_half_width, y_max=grid_half_width,
        z=grid_z,
        increment=grid_increment
    )
    grid_x_dim = rg.shape[0]
    grid_y_dim = rg.shape[1]

    snapshot_threshold = args.snapshot_threshold
    snapshot_dir = path.join(path.dirname(__file__), 'snapshots')
    if snapshot_threshold is not None:
        os.makedirs(snapshot_dir, exist_ok=True)

    # Verify audio device and camera
    if not find_uma16_device():
        return
    if not find_camera():
        return
    
    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        return

    print("\nSetting up Acoular processing pipeline...")
    
    # Setup continuous audio source
    audio_source = ac.SoundDeviceSamplesGenerator(
        device=UMA16_DEVICE_INDEX, 
        num_channels=NUM_CHANNELS
    )
    
    # Get actual sample frequency from the audio source
    fs = audio_source.sample_freq
    
    # Apply amplification first
    amped_source = ac.Calib(source=audio_source)
    amped_source.data = np.full(NUM_CHANNELS, 10.0)  # Gain of 10.0 for all channels
    
    # Apply octave band filter at target frequency BEFORE beamforming
    # FiltOctave filters all channels independently
    filtered_source = ac.FiltOctave(source=amped_source, band=TARGET_FREQ)
    
    # Create steering vector (constant across frames)
    st = ac.SteeringVector(grid=rg, mics=mg)
    
    # Create TIME-DOMAIN beamformer on FILTERED signals
    bb = ac.BeamformerTime(source=filtered_source, steer=st)
    
    print("\nStarting live acoustic camera with pre-filtering...")
    print(f"Microphone array: {NUM_CHANNELS} channels at {fs} Hz")
    print(f"Target frequency: {TARGET_FREQ} Hz (octave band)")
    print(f"Grid: {grid_x_dim}x{grid_y_dim} points = {grid_x_dim * grid_y_dim} grid points")
    print(f"Grid params: half_width={grid_half_width}, z={grid_z}, increment={grid_increment:.6f}")
    if db_mode == 'max':
        print(f"dB normalization: db_max=max(frame), db_range={db_range}")
    else:
        print(f"dB normalization: db_max={db_fixed_max}, db_range={db_range}")
    if snapshot_threshold is not None:
        print(f"Snapshot trigger: threshold={snapshot_threshold} dB SPL, max 1 image/second, peak-capture mode")
        print(f"Snapshot directory: {snapshot_dir}")
    print(f"Block size: {BLOCK_SIZE} samples")
    print("Press 'q' to quit\n")
    
    # Create iterator for continuous beamforming on filtered signals
    beamformer_iter = bb.result(num=BLOCK_SIZE)
    
    frame_count = 0
    prev_time = time.perf_counter()
    fps = 0.0
    last_snapshot_time = -1e9
    event_active = False
    event_peak_db = -1e9
    event_peak_frame = None
    event_peak_timestamp = None
    try:
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            
            # Get beamforming result from pre-filtered signals
            try:
                # Get next beamformed block: shape (num_samples, num_grid_points)
                # Input has already been filtered at TARGET_FREQ before beamforming
                block = next(beamformer_iter)
                
                # Compute RMS power for each grid point
                power = np.sqrt(np.mean(block**2, axis=0))
                spatial_map = power.reshape(rg.shape)
                
                power_db = ac.L_p(spatial_map)
                
                # Transpose and flip to fix coordinate system alignment
                heatmap = np.flipud(np.fliplr(power_db.T))
                observed_max_db = float(np.max(heatmap))
                
            except StopIteration:
                print("Audio stream ended")
                break
            except Exception as e:
                print(f"Beamforming error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Normalize for visualization using CLI-configurable db_max and db_range
            if db_mode == 'max':
                max_db = observed_max_db
            else:
                max_db = db_fixed_max
            min_db = max_db - db_range
            heatmap_clipped = np.clip(heatmap, min_db, max_db)
            heatmap_norm = ((heatmap_clipped - min_db) / db_range * 255).astype(np.uint8)
            
            # Resize to match camera frame and apply colormap (bicubic interpolation)
            heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend with camera frame
            blended_frame = cv2.addWeighted(frame, 1 - BLEND_ALPHA, heatmap_color, BLEND_ALPHA, 0)
            
            # Add overlay text
            cv2.putText(blended_frame, f"Live Acoustic Camera: {int(TARGET_FREQ)} Hz", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Max: {max_db:.1f} dB SPL", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Pre-filtered beamforming", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Smoothed framerate estimate.
            now = time.perf_counter()
            dt = now - prev_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps = inst_fps if fps == 0.0 else (0.9 * fps + 0.1 * inst_fps)
            prev_time = now
            cv2.putText(blended_frame, f"FPS: {fps:.1f}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Optional snapshot capture: arm on threshold crossing, track peak,
            # and save at event end (falling below threshold), limited to 1/s.
            if snapshot_threshold is not None:
                above_threshold = observed_max_db >= snapshot_threshold

                if above_threshold:
                    if not event_active:
                        event_active = True
                        event_peak_db = observed_max_db
                        event_peak_frame = blended_frame.copy()
                        event_peak_timestamp = datetime.now()
                    elif observed_max_db >= event_peak_db:
                        event_peak_db = observed_max_db
                        event_peak_frame = blended_frame.copy()
                        event_peak_timestamp = datetime.now()
                elif event_active:
                    # Event finished: save the best frame if outside cooldown.
                    if (time.perf_counter() - last_snapshot_time) >= 1.0 and event_peak_frame is not None:
                        stamp = event_peak_timestamp.strftime('%Y-%m-%d-%H%M%S.%f')[:-3]
                        snapshot_path = path.join(snapshot_dir, f"{stamp}.png")
                        cv2.imwrite(snapshot_path, event_peak_frame)
                        last_snapshot_time = time.perf_counter()
                        print(f"Snapshot saved: {snapshot_path} (peak {event_peak_db:.1f} dB SPL)")

                    event_active = False
                    event_peak_db = -1e9
                    event_peak_frame = None
                    event_peak_timestamp = None
            
            # Display result
            cv2.imshow('Simple Live Acoustic Camera (UMA-16)', blended_frame)
            
            frame_count += 1
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nAcoustic camera stopped after {frame_count} frames")

if __name__ == '__main__':
    main()
