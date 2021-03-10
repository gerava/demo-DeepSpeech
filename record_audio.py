#!/usr/bin/env python3
import pyaudio
import wave
import time
import datetime

chunk = 2048  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 16000  # Record at 16000 samples per second


ts = str(datetime.datetime.now())[:19].replace(":", ".")
ts = ts.replace(" ", "_")
filename = "audios/es_16k/audio{}.wav".format(ts)

p = pyaudio.PyAudio()  # Create an interface to PortAudio

frames = []  # Initialize array to store frames


def callback(input_data, frame_count, time_info, flags):
    global frames    
    frames.append(input_data)

    return input_data, pyaudio.paContinue


print('Recording...')
stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                stream_callback=callback,
                frames_per_buffer=chunk,
                input=True)



input("'------- Press Enter to stop recording -------------'")


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finish recording')



# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

print("Record save as: '{}'".format(filename))