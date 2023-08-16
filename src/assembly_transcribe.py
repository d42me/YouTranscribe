import assemblyai as aai

aai.settings.api_key = "df401489a47b4996aa7fb3e11797b3f5"

config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()

transcript = transcriber.transcribe("transcribe.mp3", config=config)

print(transcript)

# Store transcript in a file
with open('./transcript.json', 'w') as f:
    f.write(transcript)
