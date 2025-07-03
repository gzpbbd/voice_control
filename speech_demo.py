import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.m4a")
print(result["text"])