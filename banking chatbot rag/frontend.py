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

if 'input_mode' not in st.session_state:
    st.session_state.input_mode = 'voice'  # 'voice' or 'text'

if 'text_input' not in st.session_state:
    st.session_state.text_input = ""

if 'auto_play_tts' not in st.session_state:
    st.session_state.auto_play_tts = True

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

def process_user_input(user_input):
    """Process user input and get AI response."""
    if not user_input.strip():
        return
    
    # Add to conversation history
    st.session_state.conversation_history.append({
        'role': 'customer',
        'text': user_input,
        'timestamp': datetime.now(),
        'input_type': st.session_state.input_mode
    })
    
    # Get AI response
    with st.spinner("Getting AI response..."):
        try:
            ai_response = st.session_state.ai_assistant.interact_with_llm(user_input)
            if ai_response:
                ai_response = ai_response.lstrip()
                
                # Add AI response to conversation
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'text': ai_response,
                    'timestamp': datetime.now(),
                    'input_type': st.session_state.input_mode
                })
                
                # Play text-to-speech if enabled
                if st.session_state.auto_play_tts:
                    try:
                        vs.play_text_to_speech(ai_response)
                        st.success("Response played!")
                    except Exception as e:
                        st.error(f"TTS error: {str(e)}")
                        
        except Exception as e:
            st.error(f"AI error: {str(e)}")
    
    # Clear text input
    st.session_state.text_input = ""
    st.rerun()

# Main UI
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 AI Voice Assistant")
st.markdown("---")

# Input mode selector
col1, col2 = st.columns([3, 1])
with col1:
    st.subheader("Choose Input Mode:")
    input_mode = st.radio(
        "How would you like to communicate?",
        ["🎤 Voice Input", "💬 Text Input"],
        horizontal=True,
        key="input_mode_selector"
    )
    
    if input_mode == "🎤 Voice Input":
        st.session_state.input_mode = 'voice'
    else:
        st.session_state.input_mode = 'text'

with col2:
    st.subheader("TTS Settings:")
    st.session_state.auto_play_tts = st.checkbox("Auto-play responses", value=True)

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
    
    # Voice input controls
    if st.session_state.input_mode == 'voice':
        st.subheader("Voice Controls")
        
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
                            process_user_input(transcription)
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
        st.session_state.ai_assistant.clear_history()
        st.rerun()

# Main content area
if st.session_state.input_mode == 'text':
    # Text input interface
    st.subheader("💬 Text Chat")
    
    # Chat input
    with st.container():
        # Text input form
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Type your message:",
                placeholder="Ask me anything about the airport...",
                height=100,
                key="text_input_area"
            )
            
            col1, col2, col3 = st.columns([1, 1, 3])
            with col1:
                send_button = st.form_submit_button("Send 📤")
            with col2:
                if st.form_submit_button("🔊 Play Last Response"):
                    if st.session_state.conversation_history:
                        last_response = None
                        for entry in reversed(st.session_state.conversation_history):
                            if entry['role'] == 'assistant':
                                last_response = entry['text']
                                break
                        
                        if last_response:
                            try:
                                vs.play_text_to_speech(last_response)
                                st.success("Playing response!")
                            except Exception as e:
                                st.error(f"TTS error: {str(e)}")
                        else:
                            st.warning("No response to play")
            
            if send_button and user_input:
                process_user_input(user_input)

else:
    # Voice input interface
    st.subheader("🎤 Voice Chat")
    
    if st.session_state.whisper_model is None:
        st.warning("⚠️ Please initialize the Whisper model first using the sidebar.")
    else:
        st.info("🎤 Use the 'Record Audio' button in the sidebar to start speaking.")
        
        # Quick record button in main area
        if st.button("🎤 Quick Record (5s)", key="quick_record"):
            with st.spinner("Recording for 5 seconds..."):
                try:
                    audio_file = record_audio_chunk(5)
                    
                    # Check if audio contains speech
                    samplerate, data = wavfile.read(audio_file)
                    
                    if not is_silence(data):
                        with st.spinner("Transcribing..."):
                            transcription = transcribe_audio(st.session_state.whisper_model, audio_file)
                        
                        if transcription.strip():
                            process_user_input(transcription)
                        else:
                            st.warning("No speech detected in recording")
                    else:
                        st.warning("Only silence detected")
                        
                    # Clean up
                    os.remove(audio_file)
                    
                except Exception as e:
                    st.error(f"Recording error: {str(e)}")

st.markdown("---")

# Conversation Display
st.header("💬 Conversation History")

# Display conversation
if st.session_state.conversation_history:
    for i, entry in enumerate(st.session_state.conversation_history):
        timestamp_str = entry['timestamp'].strftime("%H:%M:%S")
        input_type_icon = "🎤" if entry.get('input_type') == 'voice' else "💬"
        
        if entry['role'] == 'customer':
            st.markdown(f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; margin: 10px 0; color: black;">
                <strong>{input_type_icon} You</strong> <small>({timestamp_str})</small><br><br>
                {entry['text']}
            </div>
            """, unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"""
                <div style="background-color: #f3e5f5; padding: 15px; border-radius: 10px; margin: 10px 0; color: black;">
                    <strong>🤖 AI Assistant</strong> <small>({timestamp_str})</small><br><br>
                    {entry['text']}
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🔊", key=f"play_{i}", help="Play this response"):
                    try:
                        vs.play_text_to_speech(entry['text'])
                        st.success("Playing!")
                    except Exception as e:
                        st.error(f"TTS error: {str(e)}")
else:
    st.info("👋 Welcome! Choose your preferred input mode above and start chatting!")

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    ### Input Modes:
    
    **🎤 Voice Input:**
    - Initialize the Whisper model first (one-time setup)
    - Click "Record Audio" and speak clearly
    - Adjust recording duration in the sidebar (3-10 seconds)
    - Your speech will be transcribed automatically
    
    **💬 Text Input:**
    - Type your message in the text area
    - Click "Send" or press Ctrl+Enter
    - No model initialization required
    
    ### Features:
    - **Auto TTS**: Toggle automatic text-to-speech for responses
    - **Manual TTS**: Click 🔊 buttons to replay any response
    - **Input Indicators**: See whether input was via voice 🎤 or text 💬
    - **Clear History**: Start fresh conversations anytime
    
    ### Tips:
    - Both input modes work with the same AI assistant
    - Voice input requires microphone permissions
    - Text input works immediately without setup
    - You can switch between modes anytime during conversation
    - All responses can be played back using TTS
    """)

# Status bar
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.session_state.whisper_model is not None:
        st.success("🎤 Voice Ready")
    else:
        st.warning("🎤 Voice Not Ready")

with col2:
    st.info(f"📝 Mode: {st.session_state.input_mode.title()}")

with col3:
    st.info(f"💬 Messages: {len(st.session_state.conversation_history)}")