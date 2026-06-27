import streamlit as st
import numpy as np
import librosa
import noisereduce as nr
from tensorflow.keras.models import load_model
import joblib
import sounddevice as sd
import tempfile
import os

# -----------------------------------------------------
# Page Configuration - MUST BE FIRST
# -----------------------------------------------------
st.set_page_config(
    page_title="Baby Cry Reason Detector", 
    page_icon="🎧",
    layout="wide"
)

# -----------------------------------------------------
# Custom CSS to remove link icons from headings - AFTER set_page_config
# -----------------------------------------------------
st.markdown("""
<style>
    /* Remove link icons from all headings */
    .stMarkdown h1 a,
    .stMarkdown h2 a,
    .stMarkdown h3 a,
    .stMarkdown h4 a,
    .stMarkdown h5 a,
    .stMarkdown h6 a {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Remove hover effects on headings */
    .stMarkdown h1:hover,
    .stMarkdown h2:hover,
    .stMarkdown h3:hover,
    .stMarkdown h4:hover,
    .stMarkdown h5:hover,
    .stMarkdown h6:hover {
        text-decoration: none !important;
    }
    
    /* Remove the anchor link styling */
    .stMarkdown h1::before,
    .stMarkdown h2::before,
    .stMarkdown h3::before,
    .stMarkdown h4::before,
    .stMarkdown h5::before,
    .stMarkdown h6::before {
        content: none !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# Load Model and Encoder
# -----------------------------------------------------
@st.cache_resource
def load_all():
    try:
        model = load_model("baby_cry_reason_model.h5")
        le = joblib.load("label_encoder.pkl")
        st.sidebar.success("Model & Encoder loaded successfully!")
        return model, le
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None

# Load model and encoder
model, le = load_all()

# Display loaded classes
if le is not None:
    st.sidebar.write("**Loaded Classes:**", list(le.classes_))

sample_rate = 22050

# -----------------------------------------------------
# SIMPLE Audio Validation Function
# -----------------------------------------------------
def is_baby_cry_audio(audio, sr=22050):
    """
    SIMPLE check if audio contains actual sound (not silence)
    Returns: (is_sound, message)
    """
    if audio is None or len(audio) == 0:
        return False, "Empty audio"
    
    # Basic statistics
    duration = len(audio) / sr
    max_amp = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio**2))
    
    # Show audio statistics
    st.sidebar.write("**Audio Statistics:**")
    st.sidebar.write(f"Duration: {duration:.2f}s")
    st.sidebar.write(f"Max Amplitude: {max_amp:.6f}")
    st.sidebar.write(f"RMS: {rms:.6f}")
    
    # VERY SIMPLE SILENCE DETECTION
    # If audio is practically silent, reject it
    if rms < 0.0005:  # Very low threshold
        return False, f"Audio too quiet (silence detected)"
    
    if max_amp < 0.001:  # Very low amplitude
        return False, f"Very low audio amplitude"
    
    # Check if audio has any variation (not flat line)
    audio_variation = np.std(audio)
    if audio_variation < 0.0001:
        return False, f"No audio variation (flat line)"
    
    # If we passed all silence checks, assume it's valid audio
    # Let the model decide if it's actually a baby cry or not
    return True, "Audio contains sound - proceeding to analysis"

# -----------------------------------------------------
# Prediction Function (Direct prediction without strict validation)
# -----------------------------------------------------
def predict_audio(audio, sr=22050):
    """Process audio and predict reason"""
    
    # Step 1: Simple silence check
    is_sound, sound_message = is_baby_cry_audio(audio, sr)
    
    if not is_sound:
        return "No Sound Detected", 0.0, sound_message
    
    # Step 2: Process audio for prediction
    audio_processed = audio.astype(np.float32)
    
    # Normalize
    if np.max(np.abs(audio_processed)) > 0:
        audio_processed = audio_processed / np.max(np.abs(audio_processed))
    
    # Remove DC offset
    audio_processed = audio_processed - np.mean(audio_processed)
    
    # Light noise reduction only if needed
    rms = np.sqrt(np.mean(audio_processed**2))
    if rms < 0.01:  # Only apply noise reduction if audio is quiet
        try:
            audio_processed = nr.reduce_noise(y=audio_processed, sr=sr, stationary=True, prop_decrease=0.3)
        except Exception:
            pass
    
    # Exact 3 seconds duration (same as training)
    target_length = sr * 3
    if len(audio_processed) > target_length:
        audio_processed = audio_processed[:target_length]
    else:
        pad_length = target_length - len(audio_processed)
        audio_processed = np.pad(audio_processed, (0, pad_length), mode='constant')
    
    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio_processed, 
        sr=sr, 
        n_mfcc=40,
        n_fft=2048, 
        hop_length=512
    )
    
    mfcc = np.mean(mfcc.T, axis=0)
    mfcc = mfcc.reshape(1, -1)
    
    # Predict
    pred = model.predict(mfcc, verbose=0)
    reason = le.classes_[np.argmax(pred)]
    confidence = np.max(pred)
    
    # Show probabilities
    st.sidebar.write("**Prediction Probabilities:**")
    for i, class_name in enumerate(le.classes_):
        st.sidebar.write(f"  {class_name}: {pred[0][i]*100:.1f}%")
    
    # Determine if it's actually a baby cry based on confidence
    if confidence < 0.3:  # Low confidence might mean it's not a baby cry
        return "Unclear Sound", confidence, "Low confidence - may not be baby crying"
    
    return reason, confidence, "Analysis completed"

def record_audio(duration=3, sr=22050):
    """Record audio from microphone"""
    st.info("Recording... Please let the baby cry naturally")
    
    try:
        audio = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        audio = np.squeeze(audio)
        
        # Remove DC offset and normalize
        audio = audio - np.mean(audio)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        st.success("Recording completed!")
        return audio
        
    except Exception as e:
        st.error(f"Recording failed: {e}")
        return None

# -----------------------------------------------------
# Suggestion Function
# -----------------------------------------------------
def get_suggestions(reason, confidence):
    """Get detailed suggestions based on predicted reason"""
    suggestions = []
    
    if reason == "Hungry":
        suggestions = [
            "Check feeding schedule - Baby might be hungry",
            "Last feed time - Note when baby was last fed",
            "Hunger cues - Look for rooting, lip-smacking, hand-to-mouth movements",
            "Offer feeding - Try breastfeeding or formula",
            "Check hydration - Ensure baby is getting enough fluids"
        ]
    elif reason == "Tired":
        suggestions = [
            "Sleep signals - Look for eye-rubbing, yawning, fussiness",
            "Sleep environment - Check room temperature, lighting, noise",
            "Awake time - Babies get overtired easily - check awake duration",
            "Comfort routine - Try swaddling, rocking, or white noise",
            "Gas discomfort - May need burping or gentle tummy massage"
        ]
    elif reason == "Uncomfortable":
        suggestions = [
            "Diaper check - Check for wet or soiled diaper",
            "Temperature - Feel baby's neck - should be warm, not sweaty or cold",
            "Clothing - Check for tight clothing, tags, or rough fabrics",
            "Skin irritation - Look for rashes, redness, or insect bites",
            "Position - Baby might be uncomfortable in current position"
        ]
    elif reason == "Unclear Sound":
        suggestions = [
            "The audio was unclear or may not contain baby crying",
            "Try recording again with clearer audio",
            "Ensure baby is actually crying during recording",
            "Hold microphone closer to the baby",
            "Record in a quieter environment"
        ]
    
    return suggestions

# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("Baby Cry Reason Detector")
st.markdown("""
Detect why your baby is crying from audio recordings.  
""")
st.markdown("---")

if model is None or le is None:
    st.error("Model files not loaded. Please ensure 'baby_cry_reason_model.h5' and 'label_encoder.pkl' are in the same directory.")
    st.stop()

# Test microphone first
st.sidebar.subheader("Microphone Test")
if st.sidebar.button("Test Microphone"):
    with st.sidebar:
        with st.spinner("Testing microphone for 2 seconds..."):
            test_audio = sd.rec(int(2 * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
            sd.wait()
            test_audio = np.squeeze(test_audio)
            test_rms = np.sqrt(np.mean(test_audio**2))
            st.sidebar.write(f"Test RMS: {test_rms:.6f}")
            if test_rms > 0.001:
                st.sidebar.success("Microphone working!")
            else:
                st.sidebar.error("Microphone may not be working - check permissions")

option = st.radio(
    "Choose Input Method:",
    ["Record via Microphone", "Upload Audio File"],
    horizontal=True
)

# Record Section
if option == "Record via Microphone":
    st.subheader("Record Audio")
    
    duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    
    st.write("**Instructions:**")
    st.write("1. Make sure baby is actually crying")
    st.write("2. Hold phone/microphone 10-20 cm from baby")
    st.write("3. Record in a relatively quiet environment")
    st.write("4. Click 'Start Recording' and wait for completion")
    
    if st.button("Start Recording", type="primary", use_container_width=True):
        with st.spinner("Recording in progress..."):
            audio_data = record_audio(duration, sample_rate)
            
        if audio_data is not None:
            with st.spinner("Analyzing audio..."):
                reason, confidence, message = predict_audio(audio_data, sample_rate)
            
            st.markdown("---")
            st.subheader("Analysis Results")
            
            if reason == "No Sound Detected":
                st.error("NO AUDIO DETECTED")
                st.write(f"**Reason:** {message}")
                
                st.info("""
                **Troubleshooting tips:**
                - Check if microphone permissions are granted
                - Test microphone using the 'Test Microphone' button in sidebar
                - Speak during recording to verify microphone is working
                - Ensure baby is crying loudly enough
                - Move closer to the baby (10-20 cm ideal)
                """)
                
            elif reason == "Unclear Sound":
                st.warning("UNCLEAR AUDIO")
                st.write(f"Confidence: {confidence*100:.1f}%")
                st.write("The audio was detected but may not contain clear baby crying sounds.")
                
            else:
                if confidence < 0.6:
                    st.warning(f"LOW CONFIDENCE: {reason}")
                    st.write(f"Confidence: {confidence*100:.1f}%")
                    st.info("Baby cry detected but prediction confidence is low. Try recording again with clearer audio.")
                else:
                    st.success(f"BABY CRY DETECTED")
                    st.success(f"**Predicted Reason: {reason}**")
                    st.write(f"**Confidence: {confidence*100:.1f}%**")
            
            # Show suggestions for all cases
            st.markdown("---")
            st.subheader("Suggestions & Next Steps")
            suggestions = get_suggestions(reason, confidence)
            
            for suggestion in suggestions:
                st.write(f"• {suggestion}")

# Upload Section
elif option == "Upload Audio File":
    st.subheader("Upload Audio File")
    
    uploaded_file = st.file_uploader("Choose WAV file containing baby crying", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("Analyze Uploaded Audio", type="primary", use_container_width=True):
            with st.spinner("Processing audio file..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    audio, sr = librosa.load(tmp_path, sr=sample_rate, mono=True)
                    
                    reason, confidence, message = predict_audio(audio, sr)
                    
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    if reason == "No Sound Detected":
                        st.error("NO AUDIO DETECTED")
                        st.write(f"**Reason:** {message}")
                        
                    elif reason == "Unclear Sound":
                        st.warning("UNCLEAR AUDIO")
                        st.write(f"Confidence: {confidence*100:.1f}%")
                        st.write("The audio file may not contain clear baby crying sounds.")
                        
                    else:
                        if confidence < 0.6:
                            st.warning(f"LOW CONFIDENCE: {reason}")
                            st.write(f"Confidence: {confidence*100:.1f}%")
                            st.info("Baby cry detected but prediction confidence is low.")
                        else:
                            st.success(f"BABY CRY DETECTED")
                            st.success(f"**Predicted Reason: {reason}**")
                            st.write(f"**Confidence: {confidence*100:.1f}%**")
                    
                    # Show suggestions for all cases
                    st.markdown("---")
                    st.subheader("Suggestions & Next Steps")
                    suggestions = get_suggestions(reason, confidence)
                    
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
                    
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")

st.markdown("---")
st.caption("Developed by Eman Fatima")
