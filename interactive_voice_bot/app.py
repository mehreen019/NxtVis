import os
import wave
import pyaudio
import numpy as np
from scipy.io import wavfile
from faster_whisper import WhisperModel

import voice_service as vs
from rag.AIVoiceAssistant import AIVoiceAssistant

DEFAULT_MODEL_SIZE = "base"  
DEFAULT_CHUNK_LENGTH = 5

ai_assistant = AIVoiceAssistant()

def is_silence(data, max_amplitude_threshold=3000):  
    """Check if audio data contains silence."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=DEFAULT_CHUNK_LENGTH):
    frames = []
    for _ in range(0, int(16000 / 1024 * chunk_length)):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Check if the recorded chunk contains silence
    try:
        samplerate, data = wavfile.read(temp_file_path)
        print(f"Sample rate: {samplerate}, Data shape: {data.shape}")
        print("data: " + str(data))
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error while reading audio file: {e}")
        return False

def transcribe_audio(model, file_path):
    segments, info = model.transcribe(
        file_path,
        beam_size=5, 
        without_timestamps=True,
        language="en",
    )
    transcription = ' '.join(segment.text for segment in segments)
    return transcription

def main():

    model = WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=2
    )
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        start=False
    )
    stream.start_stream()
    
    customer_input_transcription = ""

    try:
        while True:
            chunk_file = "temp_audio_chunk.wav"
            
            # Record audio chunk
            print("Listening...")
            if not record_audio_chunk(audio, stream):
                # Transcribe audio
                transcription = transcribe_audio(model, chunk_file)
                os.remove(chunk_file)
                print("Customer:", transcription)
                
                # Process response
                if transcription.strip():
                    customer_input_transcription += "Customer: " + transcription + "\n"
                    output = ai_assistant.interact_with_llm(transcription)
                    if output:
                        output = output.lstrip()
                        vs.play_text_to_speech(output)
                        print("AI Assistant:", output)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()