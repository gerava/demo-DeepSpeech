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
try:
    from shhlex import quote
except ImportError:
    from pipes import quote

if len(sys.argv) < 2:
    print("Audio filename required as command line argument")
    exit()



# Path to the model (protocol buffer binary file)
model  = 'models/es/output_graph_es.pbmm'
# Path to the external scorer file
scorer = 'models/es/kenlm_es.scorer'
# Path to the audio file to run (WAV format)
audio  = sys.argv[1]

# Beam width for the CTC decoder
beam_width = None # int
# Language model weight (lm_alpha). If not specified, use default from the scorer package.
lm_alpha = None # float
# Word insertion bonus (lm_beta). If not specified, use default from the scorer package.
lm_beta = None # float,

# Hot-words and their boosts.
hot_words = None  # str



def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)




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

fin = wave.open(audio, 'rb')
fs_orig = fin.getframerate()
if fs_orig != desired_sample_rate:
    print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, desired_sample_rate))
    fs_new, audio = convert_samplerate(audio, desired_sample_rate)
else:
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

audio_length = fin.getnframes() * (1/fs_orig)
fin.close()

print('Running inference.', file=sys.stderr)
inference_start = timer()
results = ds.stt(audio)
inference_end = timer() - inference_start
# Inference results
print('Took {:0.3f}s for {:0.3f}s audio file, stt: "{}"'.format(inference_end, audio_length, results))
