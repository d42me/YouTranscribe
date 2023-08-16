with open("transcript-by-speaker.txt", "r") as f:
    transcript = f.read()

filtered_transcript = ""

# Split the transcript by line and filter out lines with Speaker C
for line in transcript.split("\n\n"):
    if "Speaker C:" not in line:
        filtered_transcript += line + "\n\n"

# Save to transcript.txt
with open("transcript-filtered.txt", "w") as f:
    f.write(filtered_transcript)
