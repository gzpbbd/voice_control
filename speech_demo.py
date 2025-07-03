import whisper

model = whisper.load_model("tiny")
result = model.transcribe("audio.m4a")
print(result["text"])