import whisper
import glob
import os

# Whisper 모델 로드
model = whisper.load_model("large")  # tiny, base, small, medium, large 중 선택 가능

# 오디오 파일 찾기
audio_files = glob.glob("/home/elicer/Prometheus/Report_Summary/data/*.mp3")

# MP3 파일이 있는지 확인
if len(audio_files) == 0:
    raise FileNotFoundError("MP3 파일이 존재하지 않습니다. 올바른 경로를 확인하세요.")

# MP3 파일 처리
for audio_path in audio_files:
    print(f"Processing: {audio_path}")

    # 텍스트 변환 실행
    transcription = model.transcribe(audio_path, language='ko')

    # 결과 출력
    print(f"Transcription for {audio_path}:\n{transcription['text']}\n")