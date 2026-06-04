# 1. IMPORT ACOULAR FIRST (Fixes the Numba / OpenBLAS parallel performance warning)
import acoular as ac
import numpy as np
import cv2
import sounddevice as sd

# Disable Acoular's HDF5 caching for live processing
ac.config.global_caching = 'none'

# -----------------------------------------------------------------------------
# HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
# To find your device, run: python -c "import sounddevice as sd; print(sd.query_devices())"
# Look for the UMA-16 device with 16 input channels
# IMPORTANT: Use WASAPI (device 20), NOT WDM-KS (device 35) - WDM-KS doesn't support blocking API!
UMA16_DEVICE_INDEX = 21  # Line (MCHStreamer Multi-channels), Windows WASAPI
NUM_CHANNELS = 16
SAMPLE_RATE = 48000

# Load UMA-16 microphone geometry from Acoular's built-in XML file
# Uses the official miniDSP UMA-16 microphone positions
mic_geom = ac.MicGeom(file='.venv/Lib/site-packages/acoular/xml/minidsp_uma-16_mirrored.xml')

# Camera to use
CAMERA_INDEX = 1

# -----------------------------------------------------------------------------
# ACOUSTIC GRID & STEERING VECTOR
# -----------------------------------------------------------------------------
# Production-tested settings from config.json
GRID_DISTANCE = 2.0  # Focus distance in meters
GRID_INCREMENT = 0.05  # Grid spacing (coarser = faster processing)
GRID_X_MIN = -1.5
GRID_X_MAX = 1.5
GRID_Y_MIN = -1.5
GRID_Y_MAX = 1.5

grid = ac.RectGrid(
    x_min=GRID_X_MIN, x_max=GRID_X_MAX, 
    y_min=GRID_Y_MIN, y_max=GRID_Y_MAX, 
    z=GRID_DISTANCE, 
    increment=GRID_INCREMENT
)

# Calculate grid dimensions for reshaping
GRID_X_DIM = int((GRID_X_MAX - GRID_X_MIN) / GRID_INCREMENT + 1)
GRID_Y_DIM = int((GRID_Y_MAX - GRID_Y_MIN) / GRID_INCREMENT + 1)

# Was 4000
TARGET_FREQ = 2000.0  # Target frequency in Hz (higher freq = better resolution)
AVERAGING_SAMPLES = 512  # Number of samples to average for stability
BLEND_ALPHA = 0.75  # Video transparency (0.75 from production config)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def find_uma16_device():
    """Find and verify the UMA-16 audio device."""
    devices = sd.query_devices()
    print(f"\nUsing audio device {UMA16_DEVICE_INDEX}:")
    if UMA16_DEVICE_INDEX < len(devices):
        dev = devices[UMA16_DEVICE_INDEX]
        print(f"  Name: {dev['name']}")
        print(f"  Channels: {dev['max_input_channels']} in, {dev['max_output_channels']} out")
        print(f"  Sample rate: {dev['default_samplerate']} Hz")
        print(f"  Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
        
        if dev['max_input_channels'] < NUM_CHANNELS:
            print(f"\n⚠️  WARNING: Device only has {dev['max_input_channels']} input channels, but {NUM_CHANNELS} required!")
            print("\nAvailable devices with 16+ input channels (prefer WASAPI):")
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= NUM_CHANNELS:
                    api_name = sd.query_hostapis(device['hostapi'])['name']
                    print(f"  {i}: {device['name']} ({device['max_input_channels']} in) [{api_name}]")
            return False
        return True
    else:
        print(f"⚠️  Device {UMA16_DEVICE_INDEX} not found!")
        return False

def find_uma16_camera():
    """Enumerate cameras and find the UMA-16 camera."""
    print("\nScanning for cameras...")
    cameras_found = []
    
    # Try first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get camera name (not always available in OpenCV)
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cameras_found.append((i, width, height))
                print(f"  Camera {i}: {width}x{height}")
            cap.release()
    
    if not cameras_found:
        print("⚠️  No cameras found!")
        return None
    
    # If CAMERA_INDEX is set, use it
    if CAMERA_INDEX is not None:
        if any(cam[0] == CAMERA_INDEX for cam in cameras_found):
            print(f"\nUsing specified camera {CAMERA_INDEX}")
            return CAMERA_INDEX
        else:
            print(f"⚠️  Specified camera {CAMERA_INDEX} not found!")
    
    # Otherwise, use the first camera
    selected = cameras_found[0][0]
    print(f"\nUsing camera {selected} (set CAMERA_INDEX to override)")
    return selected

# -----------------------------------------------------------------------------
# LIVE CAPTURE & PROCESSING LOOP
# -----------------------------------------------------------------------------
def main():
    # Verify audio device
    if not find_uma16_device():
        return
    
    # Determine which camera to use
    if CAMERA_INDEX is not None:
        camera_idx = CAMERA_INDEX
        print(f"\nUsing specified camera {camera_idx}")
    else:
        camera_idx = find_uma16_camera()
        if camera_idx is None:
            return
    
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_idx}.")
        return

    print("\nSetting up Acoular processing pipeline...")
    
    # Build the Acoular pipeline (same as process.py)
    # Step 1: Audio source from hardware
    audio_source = ac.SoundDeviceSamplesGenerator(
        device=UMA16_DEVICE_INDEX, 
        num_channels=NUM_CHANNELS
    )
    
    # Step 2: Convert from volts to pascals (miniDSP UMA-16 sensitivity: 0.0016 V/Pa)
    source_mixer = ac.SourceMixer(
        sources=[audio_source], 
        weights=np.array([1/0.0016])
    )
    
    # Step 3: Time-domain beamformer
    # Steering Vector
    steer = ac.SteeringVector(env=ac.Environment(c=343), grid=grid, mics=mic_geom)

    beamformer = ac.BeamformerBase(
        source=source_mixer, 
        steer=steer
    )
    
    # Step 4: Filter to target frequency band (fractional octave)
    frequency_filter = ac.FiltOctave(
        source=beamformer, 
        band=TARGET_FREQ, 
        fraction='Third octave'
    )
    
    # Step 5: Calculate instantaneous power
    power = ac.TimePower(source=frequency_filter)
    
    # Step 6: Time average for stability
    time_average = ac.Average(
        source=power, 
        num_per_average=AVERAGING_SAMPLES
    )
    
    print("\nStarting live acoustic camera...")
    print(f"Microphone array: {NUM_CHANNELS} channels at {audio_source.sample_freq} Hz")
    print(f"Target frequency: {TARGET_FREQ} Hz (Third octave band)")
    print(f"Grid: {GRID_X_DIM}x{GRID_Y_DIM} points ({GRID_X_MIN} to {GRID_X_MAX}m)")
    print(f"Focus distance: {GRID_DISTANCE}m")
    print(f"Grid increment: {GRID_INCREMENT}m")
    print("Press 'q' to quit\n")
    
    # Create generator for beamforming results
    beamforming_gen = time_average.result(num=1)
    
    frame_count = 0
    try:
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            
            # Get next beamforming result from Acoular pipeline
            try:
                acoustic_result = next(beamforming_gen)
            except StopIteration:
                print("Audio stream ended")
                break
            except Exception as e:
                print(f"Beamforming error: {e}")
                continue
            
            # Convert to dB SPL and reshape to 2D grid
            acoustic_map_db = ac.L_p(acoustic_result)
            heatmap = acoustic_map_db.reshape((GRID_X_DIM, GRID_Y_DIM))
            
            # Flip y-axis for correct display orientation
            heatmap = heatmap[:, ::-1]
            
            # Normalize for visualization (0-255)
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Resize to match camera frame and apply colormap
            heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]))
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend with camera frame (using production alpha value)
            blended_frame = cv2.addWeighted(frame, 1 - BLEND_ALPHA, heatmap_color, BLEND_ALPHA, 0)
            
            # Add overlay text with info
            max_db = np.max(acoustic_map_db)
            cv2.putText(blended_frame, f"Acoustic Camera: {int(TARGET_FREQ)} Hz @ {GRID_DISTANCE}m", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Max: {max_db:.1f} dB SPL", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Grid: {GRID_X_DIM}x{GRID_Y_DIM} ({GRID_INCREMENT}m)", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display result
            cv2.imshow('Live Acoustic Camera (UMA-16)', blended_frame)
            
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
