from modules.mic_vad_streaming import *

# Stream from microphone to DeepSpeech using VAD


# Set aggressiveness of VAD: an integer between 0 and 3, 
#   0 being the least aggressive about filtering out non-speech, 
#   3 the most aggressive. 
#   Default: 3
vad_aggressiveness = 3 # int
# Disable spinner
nospinner = False
# Save .wav files of utterences to given directory
savewav = None
# Read from .wav file instead of microphone
file = None

# Path to the model (protocol buffer binary file, or entire directory containing all standard-named files for model)
model = 'models/es/output_graph_es.pbmm'
# Path to the external scorer file.
scorer = 'models/es/kenlm_es.scorer'
# Device input index (Int) as listed by pyaudio.PyAudio.get_device_info_by_index(). 
#   If not provided, falls back to PyAudio.get_default_device().
device = None # int
# Input device sample rate. Your device may require 44100.
rate = 16000 # int


if savewav: 
    os.makedirs(savewav, exist_ok=True)



print('Initializing model...')
logging.info("model: %s", model)
model = deepspeech.Model(model)
if scorer:
    logging.info("scorer: %s", scorer)
    model.enableExternalScorer(scorer)

# Start audio with VAD
vad_audio = VADAudio(aggressiveness=vad_aggressiveness,
                        device=device,
                        input_rate=rate,
                        file=file)
print("Listening (ctrl-C to exit)...")
frames = vad_audio.vad_collector()

# Stream from microphone to DeepSpeech using VAD
spinner = None
if not nospinner:
    spinner = Halo(spinner='line')
stream_context = model.createStream()
wav_data = bytearray()
for frame in frames:
    if frame is not None:
        if spinner: spinner.start()
        logging.debug("streaming frame")
        stream_context.feedAudioContent(np.frombuffer(frame, np.int16))
        if savewav: wav_data.extend(frame)
    else:
        if spinner: spinner.stop()
        logging.debug("end utterence")
        if savewav:
            vad_audio.write_wav(os.path.join(savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
            wav_data = bytearray()
        text = stream_context.finishStream()
        print("Recognized: '{}'".format(text))
        stream_context = model.createStream()


