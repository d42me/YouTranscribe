import assemblyai as aai

from constants import ASSEMBLY_API_KEY

aai.settings.api_key = ASSEMBLY_API_KEY

config = aai.TranscriptionConfig(speaker_labels=True, speakers_expected=3)

transcriber = aai.Transcriber()

transcript = transcriber.transcribe("transcribe.mp3", config=config)

# extract all utterances from the response
utterances = transcript.utterances

transcript = ""

# Store transcript in a file
for utterance in utterances:
  speaker = utterance.speaker
  text = utterance.text
  speaker_transcript = f"Speaker {speaker}: {text}\n\n"
  transcript += speaker_transcript

with open("transcript-by-speaker.txt", "w") as f:
        f.write(transcript)
