# 1. IMPORT ACOULAR FIRST (Fixes the Numba / OpenBLAS parallel performance warning)
import acoular as ac
import numpy as np
import cv2
import sounddevice as sd
from scipy import signal

# -----------------------------------------------------------------------------
# HARDWARE CONFIGURATION
# -----------------------------------------------------------------------------
# To find your device, run: python -c "import sounddevice as sd; print(sd.query_devices())"
# Look for the UMA-16 device with 16 input channels
# IMPORTANT: Use WASAPI (device 20), NOT WDM-KS (device 35) - WDM-KS doesn't support blocking API!
UMA16_DEVICE_INDEX = 20  # Line (MCHStreamer Multi-channels), Windows WASAPI
NUM_CHANNELS = 16
SAMPLE_RATE = 48000
BLOCK_SIZE = 4096

# Load UMA-16 microphone geometry from Acoular's built-in XML file
# Uses the official miniDSP UMA-16 microphone positions
mic_geom = ac.MicGeom(file='.venv/Lib/site-packages/acoular/xml/minidsp_uma-16_mirrored.xml')

# Try the last camera
CAMERA_INDEX = 2

# -----------------------------------------------------------------------------
# ACOUSTIC GRID & STEERING VECTOR
# -----------------------------------------------------------------------------
GRID_DISTANCE = 1.0  
grid = ac.RectGrid(x_min=-0.5, x_max=0.5, y_min=-0.5, y_max=0.5, z=GRID_DISTANCE, increment=0.03)
steering_vector = ac.SteeringVector(grid=grid, mics=mic_geom)

# Beamformer setup
beamformer = ac.BeamformerBase(freq_data=None, steer=steering_vector)

TARGET_FREQ = 2000.0 # 2.4kHz is the max supported by this geometry
FREQ_RANGE = (TARGET_FREQ - 500, TARGET_FREQ + 500)  # Frequency band to analyze

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

    print("\nStarting live acoustic camera...")
    print(f"Microphone array: {NUM_CHANNELS} channels at {SAMPLE_RATE} Hz")
    print(f"Target frequency: {TARGET_FREQ} Hz")
    print("Press 'q' to quit\n")
    
    with sd.InputStream(device=UMA16_DEVICE_INDEX, channels=NUM_CHANNELS, 
                        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE) as audio_stream:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            audio_data, overflowed = audio_stream.read(BLOCK_SIZE)
            if overflowed:
                print("Audio buffer overflow - skipping frame")
                continue

            # Transpose audio data to shape (channels, samples)
            audio_block = audio_data.T
            
            # Compute FFT for each channel
            freqs = np.fft.rfftfreq(BLOCK_SIZE, 1/SAMPLE_RATE)
            fft_data = np.fft.rfft(audio_block, axis=1)
            
            # Find target frequency index
            freq_idx = np.argmin(np.abs(freqs - TARGET_FREQ))
            
            # Compute Cross-Spectral Matrix at target frequency
            fft_slice = fft_data[:, freq_idx:freq_idx+1]  # Keep 2D
            csm_matrix = np.dot(fft_slice, fft_slice.conj().T) / BLOCK_SIZE
            
            # Create PowerSpectra object for beamformer
            # PowerSpectra expects shape (num_freqs, num_channels, num_channels)
            csm_3d = csm_matrix[np.newaxis, :, :]  # Add frequency dimension
            
            # Create a mock PowerSpectra object
            class MockPowerSpectra:
                def __init__(self, csm, freq):
                    self.csm = csm
                    self.freq = np.array([freq])
                    self.num_channels = csm.shape[1]
                    
            ps = MockPowerSpectra(csm_3d, TARGET_FREQ)
            beamformer.freq_data = ps
            
            # Compute beamformer output at target frequency
            try:
                acoustic_result = beamformer.synthetic(TARGET_FREQ, 1)
                acoustic_map = acoustic_result[0]  # Get first frequency result
            except Exception as e:
                print(f"Beamformer error: {e}")
                continue
            
            # Post-process map matrix
            heatmap = acoustic_map.reshape(grid.shape)
            heatmap_db = 10 * np.log10(np.maximum(heatmap, 1e-12))
            heatmap_norm = cv2.normalize(heatmap_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Render visual overlays
            heatmap_resized = cv2.resize(heatmap_norm, (frame.shape[1], frame.shape[0]))
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            
            alpha = 0.5
            blended_frame = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)
            
            # Add info text
            cv2.putText(blended_frame, f"Acoustic Camera: {int(TARGET_FREQ)} Hz", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            max_db = np.max(heatmap_db)
            cv2.putText(blended_frame, f"Max: {max_db:.1f} dB", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Live Acoustic Camera (UMA-16)', blended_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Acoustic camera stopped")

if __name__ == '__main__':
    main()
