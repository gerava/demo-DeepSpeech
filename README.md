# DeepSpeech from Mozilla

This demo are tested on ubuntu 18.04 and python 3.6

## Install

Installation extracted from [DeepSpeech docs](https://deepspeech.readthedocs.io/en/v0.9.3/?badge=latest)

Install required packages:

    sudo apt install portaudio19-dev
    sudo apt-get install sox libsox-fmt-mp3
    
Create virtualenvironment

    python3.6 -m venv env
    source env/bin/activate

upgrade setuptools and pip

    pip install --upgrade setuptools pip

Some python package are needed, install it from:

    pip install -r requirements.txt


## Get audios and model

**Audios**

Create audio folder and put all audios @ 16Khz in 'spanish':

    mkdir -p audios/es_16k/

If audios have other sample rate, resample it with. If source audio in audios/es/:

    ./audios/recursive-resample.sh audios/es/ audios/es_16k/


Record a sample audio

    python record_audio.py


Play recorded audios:

    play audios/es_16k/audio*


**Model**

create model folder:

mkdir -p models/es/

Download *.scorer and *.pbmm model files from [DeepSpeech-Ployglot-ES](https://drive.google.com/drive/folders/1-3UgQBtzEf8QcH2qc8TJHkUqCBp5BBmO) folder in Google Drive, and paste in **models/es/**

Spanish Model is extracted from [DeepSpeech-Polyglot](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/) project


## Test

**Command line tool**

Run inference on terminal with recorded audio:

    deepspeech --model models/es/output_graph_es.pbmm --scorer models/es/kenlm_es.scorer --audio audios/es_16k/audio2021-03-10_16.42.42.wav


**Python scripts** estracted from [DeepSpeech](https://github.com/mozilla/DeepSpeech) from Mozilla

Run for single audio:

    python audio_to_text.py audios/es_16k/audio2021-03-10_16.42.42.wav


Run for all audios in folder **audios/es_16k**

    python audios_to_text.py


Run for microphone streaming

    python streaming_to_text.py




