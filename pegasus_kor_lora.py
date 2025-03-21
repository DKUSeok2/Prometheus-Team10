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

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,  # 저차원 행렬 차원
    lora_alpha=16,  # 학습률 조정
    lora_dropout=0.1,  # 드롭아웃
    target_modules=["self_attn.q_proj", "self_attn.v_proj"],  # 수정된 target_modules
)

# LoRA 적용
model = get_peft_model(model, lora_config)

# GPU 사용 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("모델 로드 완료!")

# 데이터 전처리 함수
def preprocess_function(examples):
    inputs = tokenizer(examples["document"], max_length=512, truncation=True, padding="max_length")
    targets = tokenizer(examples["summary"], max_length=128, truncation=True, padding="max_length")

    inputs["labels"] = targets["input_ids"]
    return inputs

# 데이터셋 전처리
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./pegasus-lora-korean",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,  # VRAM에 맞게 조정
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    push_to_hub=False,
    fp16=True  # Mixed Precision Training (FP16) 사용
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

print("학습 시작!")
trainer.train()

# ✅ 학습된 모델 저장
trainer.save_model("./Pegasus_lora/pegasus-lora-korean")  # 모델 저장
tokenizer.save_pretrained("./Pegasus_lora/pegasus-lora-korean")  # 토크나이저 저장
print("학습된 모델 저장 완료!")

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