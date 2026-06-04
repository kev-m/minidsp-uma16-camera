# Simple Live Acoustic Camera using BeamformerTime
import acoular as ac
import numpy as np
import cv2
import sounddevice as sd
from os import path
from scipy import signal

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
TARGET_FREQ = 2000.0  # Target frequency in Hz
F_BANDWIDTH = 200     # Bandwidth around target (±100Hz)
BLOCK_SIZE =  2048    # Samples per processing block (increased for better filtering) 2048
BLEND_ALPHA = 0.75    # Video transparency

# Rectangular grid (similar to example but scaled up)
rg = ac.RectGrid(
    x_min=-1.5, x_max=1.5,
    y_min=-1.5, y_max=1.5,
    z=2.0,  # Focus distance in meters
    increment=0.05
)

# Calculate grid dimensions for reshaping
GRID_X_DIM = int((1.5 - (-1.5)) / 0.05 + 1)
GRID_Y_DIM = int((1.5 - (-1.5)) / 0.05 + 1)

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
    # Verify audio device and camera
    if not find_uma16_device():
        return
    # if not find_camera():
    #     return
    
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
    
    # Create steering vector (constant across frames)
    st = ac.SteeringVector(grid=rg, mics=mg)
    
    # Create TIME-DOMAIN beamformer (not frequency domain!)
    bb = ac.BeamformerTime(source=audio_source, steer=st)
    
    # Design bandpass filter around TARGET_FREQ
    f_low = TARGET_FREQ - F_BANDWIDTH / 2
    f_high = TARGET_FREQ + F_BANDWIDTH / 2
    b, a = signal.butter(4, [f_low, f_high], btype='bandpass', fs=fs)
    
    print("\nStarting simple live acoustic camera...")
    print(f"Microphone array: {NUM_CHANNELS} channels at {fs} Hz")
    print(f"Target frequency: {TARGET_FREQ} Hz (±{F_BANDWIDTH/2:.0f} Hz)")
    print(f"Grid: {GRID_X_DIM}x{GRID_Y_DIM} points = {GRID_X_DIM * GRID_Y_DIM} grid points")
    print(f"Block size: {BLOCK_SIZE} samples")
    print("Press 'q' to quit\n")
    
    # Create iterator for continuous block processing
    beamformer_iter = bb.result(num=BLOCK_SIZE)
    
    frame_count = 0
    try:
        while True:
            # Get camera frame
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break
            
            # Get beamforming result (time-domain block)
            try:
                # Get next time-domain block from beamformer
                # block shape: (num_samples, grid_size)
                block = next(beamformer_iter)
                
                # Apply bandpass filter to each grid point's time series
                filtered_block = signal.filtfilt(b, a, block, axis=0)
                
                # Compute RMS power for each grid point (spatial map at TARGET_FREQ)
                power_2khz = np.sqrt(np.mean(filtered_block**2, axis=0))
                spatial_map_2khz = power_2khz.reshape(rg.shape)

                power_db = ac.L_p(spatial_map_2khz)

                # Transpose to match imshow orientation
                heatmap = power_db.T
                
            except StopIteration:
                print("Audio stream ended")
                break
            except Exception as e:
                print(f"Beamforming error: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Normalize for visualization with dynamic range
            max_db = np.max(power_db)
            min_db = max_db - 10.0  # 10 dB dynamic range
            heatmap_clipped = np.clip(heatmap, min_db, max_db)
            heatmap_norm = ((heatmap_clipped - min_db) / 10.0 * 255).astype(np.uint8)
            
            # Resize to match camera frame and apply colormap (bicubic interpolation)
            heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            # Blend with camera frame
            blended_frame = cv2.addWeighted(frame, 1 - BLEND_ALPHA, heatmap_color, BLEND_ALPHA, 0)
            
            # Add overlay text
            cv2.putText(blended_frame, f"Acoustic Camera: {int(TARGET_FREQ)} Hz", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Max: {max_db:.1f} dB", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blended_frame, f"Time-domain beamforming", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
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
