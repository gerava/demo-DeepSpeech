#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import shlex
import subprocess
import sys
import wave
import json


from deepspeech import Model, version
from timeit import default_timer as timer

from os import path, listdir
from os.path import join, dirname, abspath


# Path to the model (protocol buffer binary file)
model  = 'models/es/output_graph_es.pbmm'
# Path to the external scorer file
scorer = 'models/es/kenlm_es.scorer'
# Path to the audio file to run (WAV format)
audios_path  = 'audios/es_16k/'

# Beam width for the CTC decoder
beam_width = None # int
# Language model weight (lm_alpha). If not specified, use default from the scorer package.
lm_alpha = None # float
# Word insertion bonus (lm_beta). If not specified, use default from the scorer package.
lm_beta = None # float,

# Hot-words and their boosts.
hot_words = None  # str


print('Loading model from file {}'.format(model))
model_load_start = timer()
# sphinx-doc: python_ref_model_start
ds = Model(model)
# sphinx-doc: python_ref_model_stop
model_load_end = timer() - model_load_start
print('Loaded model in {:.3}s.'.format(model_load_end))

if beam_width:
    ds.setBeamWidth(beam_width)

desired_sample_rate = ds.sampleRate()

if scorer:
    print('Loading scorer from files {}'.format(scorer))
    scorer_load_start = timer()
    ds.enableExternalScorer(scorer)
    scorer_load_end = timer() - scorer_load_start
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end))

    if lm_alpha and lm_beta:
        print("Set Scorer Alpha and Beta")
        ds.setScorerAlphaBeta(lm_alpha, lm_beta)

if hot_words:
    print('Adding hot-words')
    for word_boost in hot_words.split(','):
        word, boost = word_boost.split(':')
        ds.addHotWord(word,float(boost))



def get_audios_list(audios_path, with_path = False):
        audios = []
        if path.exists(audios_path):
            for mfile in listdir(audios_path):
                if with_path:
                    mfile = join(audios_path, mfile)
                if '.wav' in mfile:
                    audios.append(mfile)
        return audios

audios_list = get_audios_list(audios_path, with_path=True)

if not audios_list:
    print('Path "{}" not contains .wav audio files')
    exit()


print('Running inference.')
total_inference = 0.0
total_audios = 0.0
count = 1
for audio_path in audios_list:

    fin = wave.open(audio_path, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != desired_sample_rate:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate))
        fs_new, audio = convert_samplerate(audio, desired_sample_rate)
    else:
        audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    
    inference_start = timer()
    results = ds.stt(audio)
    inference_end = timer() - inference_start

    total_inference += inference_end
    total_audios += audio_length
    # Inference results
    print('[{}/{}]Took {:0.3f}s for {:0.3f}s audio file "{}", stt: "{}"'.format(count, len(audios_list), inference_end, audio_length, audio_path, results))
    count += 1

print('END')
print('All inferences took {:0.3f}s for {:0.3f}s of audios'.format(total_inference, total_audios))
