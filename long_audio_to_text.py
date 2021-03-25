import sys
import os
#import logging
import argparse
import subprocess
import shlex
import numpy as np
from modules import wavTranscriber, wavSplit
from timeit import default_timer as timer

print("#"*50)
print('Transcribe long audio files using webRTC VAD or use the streaming interface')


if len(sys.argv) < 2:
    print("Audio filename required as command line argument")
    exit()


# Determines how aggressive filtering out non-speech is. (Interger between 0-3)
aggressive = 3 #int
# Path to the audio file to run (WAV format)
audio_path  = sys.argv[1]

# Path to the model (protocol buffer binary file)
model  = 'models/es/output_graph_es.pbmm'
# Path to the external scorer file
scorer = 'models/es/kenlm_es.scorer'




# Extract filename from the full file path
#path, filename_w_ext = os.path.split(audio_path)
#filename, file_extension = os.path.splitext(filename_w_ext)

#print("path: {}".format(path))
#print("filename: '{}'".format(filename))
#print("ext: '{}'".format(file_extension))





# Load output_graph, alpahbet and scorer
model_retval = wavTranscriber.load_model(model, scorer)




total_inference_time = 0.0

# Run VAD on the input file
segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(audio_path, aggressive)


total_transcription = ""
#logging.debug("Saving Transcript @: %s" % audio_path.rstrip(".wav") + ".txt")
t0 =timer()
segments = list(segments)
print("time convert to list:", timer()-t0)
segments_len = len(segments)



total_voice_audio = 0.0

print("-----")
for i, segment in enumerate(segments):
    # Run deepspeech on the chunk that just completed VAD
    #logging.debug("Processing chunk %002d" % (i,))
    audio = np.frombuffer(segment, dtype=np.int16)
    transcription, inference_time, voice_audio = wavTranscriber.stt(model_retval[0], audio, sample_rate)
    total_inference_time += inference_time
    total_transcription += transcription + " "
    
    total_voice_audio += voice_audio
    print("{:03d}/{:03d}, took: {:3.2f} \ttts: '{}'".format(i+1, segments_len, inference_time, transcription))



print("-----")
print("filename: '{}'".format(audio_path))
print("Audio length: {} s".format(audio_length))
print("Sample rate: {} Hz".format(sample_rate) )
print("-----")
print("Voice audio: {} s".format(total_voice_audio))
print("Segments: {}". format(segments_len))
print("Transcription: '{}'".format(total_transcription))
print("Inference time: {:0.2f} s".format(total_inference_time))


"""
Transcripción   -   Transcription
Transcriptor    -   Transcriber
Transcripciones -   Transcripts
Transcrito      -   Transcribed
Transcribir     -   transcribe

Inferencia      -   Inference
Inferenciador   -   Inferer
"""