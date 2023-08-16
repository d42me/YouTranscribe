from pytube import YouTube
import ffmpeg
import whisper

yt_link = "https://www.youtube.com/watch?v=6yQEA18C-XI&t=4454s"

# Download the video
yt_vid_src = YouTube(yt_link).streams[1].download()
print(yt_vid_src)

# Convert to mp3
ffmpeg.input(yt_vid_src).output('transcribe.mp3').run()

# Init whisper model
model = whisper.load_model("base")

# Transcribe to text
transcript = model.transcribe('transcribe.mp3')

# Store transcript in a file
with open('./transcript.txt', 'w') as f:
    f.write(transcript["text"])
