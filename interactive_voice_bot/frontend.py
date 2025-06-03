import streamlit as st
import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel
import tempfile
from datetime import datetime

# Import your existing modules
import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

# Initialize session state
if 'ai_assistant' not in st.session_state:
    st.session_state.ai_assistant = AIVoiceAssistant()

if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

def is_silence(data, max_amplitude_threshold=3000):
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(duration=5):
    """Record audio for specified duration."""
    audio = pyaudio.PyAudio()
    
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024, exception_on_overflow=False)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file_path = temp_file.name
        temp_file.close()
        
        with wave.open(temp_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(16000)
            wf.writeframes(b''.join(frames))
        
        return temp_file_path
        
    finally:
        audio.terminate()

def transcribe_audio(model, file_path):
    """Transcribe audio using Whisper model."""
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        without_timestamps=True,
        language="en",
    )
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def initialize_whisper_model():
    """Initialize the Whisper model."""
    if st.session_state.whisper_model is None:
        with st.spinner("Loading Whisper model..."):
            st.session_state.whisper_model = WhisperModel(
                "base.en",
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
                num_workers=2
            )
        st.success("Whisper model loaded!")
        st.rerun()

# Main UI
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 AI Voice Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Model initialization
    if st.button("Initialize Whisper Model"):
        initialize_whisper_model()
    
    if st.session_state.whisper_model is not None:
        st.success("✅ Whisper model ready")
    else:
        st.warning("⚠️ Click to load Whisper model first")
    
    st.markdown("---")
    
    # Recording duration
    duration = st.slider("Recording Duration (seconds)", 3, 10, 5)
    
    # Record button
    if st.button("🎤 Record Audio", disabled=st.session_state.whisper_model is None):
        with st.spinner(f"Recording for {duration} seconds..."):
            try:
                audio_file = record_audio_chunk(duration)
                
                # Check if audio contains speech
                samplerate, data = wavfile.read(audio_file)
                
                if not is_silence(data):
                    with st.spinner("Transcribing..."):
                        transcription = transcribe_audio(st.session_state.whisper_model, audio_file)
                    
                    if transcription.strip():
                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            'role': 'customer',
                            'text': transcription,
                            'timestamp': datetime.now()
                        })
                        
                        # Get AI response
                        with st.spinner("Getting AI response..."):
                            try:
                                ai_response = st.session_state.ai_assistant.interact_with_llm(transcription)
                                if ai_response:
                                    ai_response = ai_response.lstrip()
                                    
                                    # Add AI response to conversation
                                    st.session_state.conversation_history.append({
                                        'role': 'assistant',
                                        'text': ai_response,
                                        'timestamp': datetime.now()
                                    })
                                    
                                    # Play text-to-speech
                                    try:
                                        vs.play_text_to_speech(ai_response)
                                        st.success("Response played!")
                                    except Exception as e:
                                        st.error(f"TTS error: {str(e)}")
                                        
                            except Exception as e:
                                st.error(f"AI error: {str(e)}")
                        
                        st.rerun()
                    else:
                        st.warning("No speech detected in recording")
                else:
                    st.warning("Only silence detected")
                    
                # Clean up
                os.remove(audio_file)
                
            except Exception as e:
                st.error(f"Recording error: {str(e)}")
    
    st.markdown("---")
    
    # Clear conversation
    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.rerun()

# Main content
st.header("Conversation")

# Display conversation
if st.session_state.conversation_history:
    for i, entry in enumerate(st.session_state.conversation_history):
        timestamp_str = entry['timestamp'].strftime("%H:%M:%S")
        
        if entry['role'] == 'customer':
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; color: black;">
                <strong>🗣️ You</strong> <small>({timestamp_str})</small><br><br>
                {entry['text']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; color: black;">
                <strong>🤖 AI Assistant</strong> <small>({timestamp_str})</small><br><br>
                {entry['text']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("👋 Welcome! Initialize the model and click 'Record Audio' to start chatting!")

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    ### Simple Steps:
    
    1. **Initialize**: Click "Initialize Whisper Model" (one-time setup)
    2. **Record**: Click "🎤 Record Audio" and speak clearly
    3. **Wait**: The system will transcribe and generate a response
    4. **Listen**: The AI response will be played automatically
    5. **Repeat**: Keep recording to continue the conversation
    
    ### Tips:
    - Speak clearly during the recording period
    - Adjust recording duration if needed (3-10 seconds)
    - The AI response will play through your speakers
    - Use "Clear Conversation" to start fresh
    """)