from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch


# 모델과 토크나이저 로드
model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# 한국어 요약 데이터셋 로드
dataset = load_dataset("daekeun-ml/naver-news-summarization-ko")

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ✅ 저장된 모델 불러오기
print("저장된 모델 로드 중...")
model_path = "./Pegasus_lora/pegasus-lora-korean"
model = PegasusForConditionalGeneration.from_pretrained(model_path)
tokenizer = PegasusTokenizer.from_pretrained(model_path)
model.to(device)
print("저장된 모델 로드 완료!")

# ✅ 테스트 문장 요약
test_text = "삼성전자는 2025년까지 AI 반도체 시장에서 1위를 목표로 하고 있다."
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding="longest").to(device)

# 요약 생성
output = model.generate(**inputs, max_length=50, num_beams=5)
summary = tokenizer.decode(output[0], skip_special_tokens=True)

print("요약 결과:", summary)