# YouTranscribe with Q&A

# Install requirements for macOS:

`brew install ffmpeg whisper`

# Setup local environment

`python -m venv venv`
`source venv/bin/activate`

# Install dependencies

`pip install -r requirements.txt`

# How it works

- Transcribe from YouTube using open-source whisper

`python src/whisper_transcribe.py`

- Transcribe from YouTube using assembly ai (preferred)

`python src/assembly_transcribe.py`

- Clean a certain speaker transcript (if needed)

`python src/clean_speaker_transcript.py`

- Vectorize transcript and ask questions

`python src/chroma.py`
