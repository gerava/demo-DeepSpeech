import sys
import os
#import logging
import argparse
import subprocess
import shlex
import numpy as np
from modules import wavTranscriber, wavSplit
from timeit import default_timer as timer
import wave

import matplotlib.pyplot as plt
import statistics
import math


print("#"*80)
print('Transcribe long audio files using webRTC VAD or use the streaming interface')
print("#"*80)

if len(sys.argv) < 2:
    print("Audio filename required as command line argument")
    exit()


# Determines how aggressive filtering out non-speech is. (Interger between 0-3)
# 0 being the least aggressive about filtering out non-speech, 3 is the most aggressive.
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
print("time convert to list: {:2.4f}s".format(timer()-t0))
segments_len = len(segments)



total_voice_audio = 0.0
confidences = []
total_words = 0

def print_transcription_meta(transcription_meta):
    print("------------------")
    for tr in transcription_meta.transcripts:
        confidence = tr.confidence
        transcription = ""
        for token in tr.tokens:
            transcription += token.text
        print("{:3.3f}\t{}".format(confidence/len(tr.tokens), transcription))
    print("------------------")

def segment_to_audio_file(path, segment):
    audio = np.frombuffer(segment, dtype=np.int16)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b''.join(audio))
    wf.close()



print("-----")
for i, segment in enumerate(segments):
    # Run deepspeech on the chunk that just completed VAD
    audio = np.frombuffer(segment, dtype=np.int16)
    if len(segments) > 1:
        segment_name = os.path.join(os.path.dirname(audio_path), os.path.splitext(os.path.basename(audio_path))[0] + "_" + "{:03d}-{:03d}.wav".format(i+1, segments_len))
        segment_to_audio_file(segment_name, segment)
    transcription_meta, inference_time, voice_audio = wavTranscriber.sttWithMetadata(model_retval[0], audio, sample_rate, 4)
    #print_transcription_meta(transcription_meta) # Metadata( transcripts=[ CandidateTranscript(confidence, tokens), ...] )
    """
    Approximated confidence value for this transcription. This is roughly the
    sum of the acoustic model logit values for each timestep/character that
    contributed to the creation of this transcription.
    """
    confidence = transcription_meta.transcripts[0].confidence
    tokens = transcription_meta.transcripts[0].tokens # [TokenMetadata(text='e', timestep=343, start_time=6.859999656677246), ...]
    
    if len(tokens) == 0:
        print("no tokens", confidence)
        continue
    
    confidence /= len(tokens) # logits
    transcription = ""
    words = 1
    prev_char = "1"
    for token in tokens:
        transcription += token.text
        if token.text == ' ':
            words += 1
    #transcription += "."
    
    total_inference_time += inference_time
    if confidence > -5.0:
        total_transcription += transcription + " "
        confidences.append(confidence)
    else:
        print("confidence < '-5.0'")
        continue
    
    total_voice_audio += voice_audio
    total_words += words
    print("{:03d}/{:03d}> took: {:3.2f}\tconfidence:{:3.2f}\tw:{:2d}\ttts: '{}'".format(i+1, segments_len, inference_time, confidence, words, transcription))

if len(confidences) == 0:
    print("Audio file does not contain voices")
    exit()


if len(confidences) > 1:
    confidence_mean     = statistics.mean(confidences)
    confidence_mode     = float(statistics.mode([ int(confidence*10) for confidence in confidences ])) / 10.0
    confidence_median   = statistics.median(confidences)
    confidence_min      = min(confidences)
    confidence_max      = max(confidences)
    confidences_sorted  = sorted(confidences)
else:
    confidence_mean     = confidences[0]
    confidence_mode     = confidences[0]
    confidence_median   = confidences[0]
    confidence_min      = confidences[0]
    confidence_max      = confidences[0]
    confidences_sorted  = confidences


probability_mean    = 100*2*1/(1+math.exp(confidence_mean * -1.0))
probability_mode    = 100*2*1/(1+math.exp(confidence_mode * -1.0))
probability_median  = 100*2*1/(1+math.exp(confidence_median * -1.0))
probability_min     = 100*2*1/(1+math.exp(confidence_min * -1.0))
probability_max     = 100*2*1/(1+math.exp(confidence_max * -1.0))



print("-"*20)
print("filename:        '{}'".format(audio_path))
print("Audio length:    {} s".format(audio_length))
print("Sample rate:     {} Hz".format(sample_rate) )
print("-"*20)
print("Voice audio:     {} s".format(total_voice_audio))
print("Segments:        {}". format(segments_len))
print("Transcription:   '{}'".format(total_transcription))
print("Words:           {}".format(total_words))
print("Inference time:  {:0.2f} s".format(total_inference_time))
print("Confidences:     {}".format( ["{:2.2f}".format(conf) for conf in confidences]   ))

print("-"*20)
print("confidence_mean:     {:3.2f}".format(confidence_mean))
print("confidence_mode:     {:3.2f}".format(confidence_mode))
print("confidence_median:   {:3.2f}".format(confidence_median))
print("confidence_min:      {:3.2f}".format(confidence_min))
print("confidence_max:      {:3.2f}".format(confidence_max))

print("-"*20)
print("probability_mean:    {:3.2f}".format(probability_mean))
print("probability_mode:    {:3.2f}".format(probability_mode))
print("probability_median:  {:3.2f}".format(probability_median))
print("probability_min:     {:3.2f}".format(probability_min))
print("probability_max:     {:3.2f}".format(probability_max))
print("-"*20)

fig, axs = plt.subplots(1,2)

axs[0].plot(confidences)
axs[0].plot([confidence_mean]*len(confidences))
axs[0].set_title("confidences")
axs[0].set_ylim(-4.0, 0.0)

axs[1].plot(confidences_sorted)
axs[1].plot([confidence_mean]*len(confidences))
axs[1].set_title("confidences_sorted")
axs[1].set_ylim(-4.0, 0.0)

plt.suptitle(audio_path)
plt.show()