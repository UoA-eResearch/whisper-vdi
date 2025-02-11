import webvtt
vtt = webvtt.read('/home/ubuntu/whisper-vdi/data/paraini.vtt')
transcript = ""

lines = []
for line in vtt:
    # Strip the newlines from the end of the text.
    # Split the string if it has a newline in the middle
    # Add the lines to an array
    lines.extend(line.text.strip().splitlines())

# Remove repeated lines
previous = None
for line in lines:
    if line == previous:
       continue
    transcript += " " + line
    previous = line

print(transcript)

with open('/home/ubuntu/whisper-vdi/data/paraini.txt', 'x') as f:
    f.write(transcript)