import collections
import io
import json
import os
import queue
import time
from pathlib import Path
import ai
from constants import *
import numpy as np
import pyaudio
from dotenv import load_dotenv
import faster_whisper
import string
import subprocess
from pyt2s.services import streamlabs

load_dotenv()

CHUNKS = queue.Queue()

def on_audio(data, frame_count, time_info, status):
    CHUNKS.put(data)
    return None, pyaudio.paContinue

if __name__ == "__main__":
    whisper = faster_whisper.WhisperModel("base", device="cpu", compute_type="float32")
    pa =pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=AudioConfig.Rate,
        input=True,
        frames_per_buffer=AudioConfig.Chunk,
        stream_callback=on_audio,
    )

    # Sample the room for the first second to set the speech threshold
    ambient = []
    for _ in range(AudioConfig.CalibrateChunks):
        samples = np.frombuffer(CHUNKS.get(), dtype=np.int16)
        ambient.append(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    threshold = max(AudioConfig.MinThreshold, 3 * float(np.mean(ambient)))

    try:
        print("\n\n")
        utterance = []
        silent_chunks = 0
        pre_roll = collections.deque(maxlen=AudioConfig.PreRollChunks)
        while True:
            try:
                data = CHUNKS.get(timeout=0.05)
                samples = np.frombuffer(data, dtype=np.int16)
                level = np.sqrt(np.mean(samples.astype(np.float32) ** 2))

                # once speaking, only a drop well below the threshold counts as a pause
                if level > threshold or (utterance and level > threshold * AudioConfig.ContinueRatio):
                    if not utterance:
                        utterance.extend(pre_roll)  # keep the start of the first word
                    utterance.append(data)
                    silent_chunks = 0
                    
                elif utterance:
                    utterance.append(data)
                    silent_chunks += 1
                    if silent_chunks >= AudioConfig.EndSilenceChunks:
                        audio = np.frombuffer(b"".join(utterance), dtype=np.int16)
                        audio = audio.astype(np.float32) / 32768.0
                        segments, _ = whisper.transcribe(audio)
                        text = " ".join(s.text.strip() for s in segments).strip()
                        if text:
                            compareText = text.translate(str.maketrans("", "", string.punctuation)).lower()
                            if 'hey mirror' in compareText or 'hey mayor' in compareText \
                             or 'hello mirror' in compareText or 'hello mayor' in compareText:
                                if 'how do i look' in compareText:
                                    print("Taking a picture...")
                                    response = ai.send_message(text, image=True)
                                else:
                                    print("Waiting for response...")
                                    response = ai.send_message(text)
                                    
                                ttsData = streamlabs.requestTTS(response, streamlabs.Voice.Kendra)
                                with open("response.mp3", "wb") as f:
                                    f.write(ttsData)
                                subprocess.run(["mpg123", "response.mp3", ">", "/dev/null"])
                            
                        utterance, silent_chunks = [], 0
                        pre_roll.clear()
                        
                else:
                    pre_roll.append(data)
                    
            except queue.Empty:
                pass  # nothing new from the mic; other work can go here
    
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
