import torch
import pandas as pd
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split

# ✅ 1️⃣ GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 실행 디바이스: {device}")

# ✅ 2️⃣ Train/Validation 데이터 로드
dataset = load_dataset("parquet", data_files={
    "train": "Save/train.parquet",
    "validation": "Save/valid.parquet"
})

# ✅ 3️⃣ Pegasus 모델 및 토크나이저 로드 (QLoRA 적용)
model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ✅ 4️⃣ QLoRA 적용을 위한 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # ✅ 4-bit 양자화 적용
    bnb_4bit_compute_dtype=torch.bfloat16,  # ✅ 4-bit 연산을 위해 float16 사용
    bnb_4bit_use_double_quant=True  # ✅ 이중 양자화 활성화 (메모리 절약)
)

# ✅ 5️⃣ 모델을 4-bit 양자화된 형태로 불러오기
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, quantization_config=bnb_config).to(device)

# ✅ 6️⃣ QLoRA 설정 (LoRA Adapter 적용)
lora_config = LoraConfig(
    r=8,  # LoRA 랭크 (적절한 값: 4~16)
    lora_alpha=32,  # LoRA 학습률 스케일링
    target_modules=["q_proj", "v_proj"],  # LoRA 적용할 레이어 선택
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

# ✅ 7️⃣ QLoRA 모델 생성
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ✅ 학습 가능한 파라미터 확인 (LoRA 적용 여부 체크)

# ✅ 8️⃣ 데이터 전처리 (토큰화)
def preprocess_function(examples):
    model_inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# ✅ 9️⃣ 학습 설정 (체크포인트 저장 반영)
training_args = TrainingArguments(
    output_dir="./results_qlora",  # ✅ QLoRA 체크포인트 저장 경로
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,  # ✅ 최근 2개의 체크포인트만 유지
    per_device_train_batch_size=16,  # ✅ 더 큰 배치 사이즈 가능 (VRAM 절약)
    per_device_eval_batch_size=16,
    learning_rate=5e-4,  # ✅ QLoRA는 일반적인 학습률보다 높게 설정 가능
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=10,
    fp16=True  # ✅ Mixed Precision 활성화
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer
)

# ✅ 10️⃣ QLoRA 모델 학습 시작!
trainer.train()

# ✅ 11️⃣ 학습 완료 후 최종 모델 저장
model.save_pretrained("./final_pegasus_qlora_model")  # ✅ 학습된 모델 저장
tokenizer.save_pretrained("./final_pegasus_qlora_model")  # ✅ 토크나이저 저장

# ✅ 12️⃣ 학습된 모델로 테스트하기
def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    output = model.generate(**inputs, max_length=128, num_beams=5)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 🎯 테스트 문장
test_sentence = "서울은 대한민국의 수도이며, 경제, 정치, 문화의 중심지이다."
summary = generate_summary(test_sentence)
print("요약 결과:", summary)

# ✅ 13️⃣ ROUGE 점수 평가
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure for key, value in result.items()}

# 🎯 Test 데이터 평가
test_dataset = load_dataset("parquet", data_files={"test": "Save/test.parquet"})
tokenized_test = test_dataset.map(preprocess_function, batched=True)
predictions = trainer.predict(tokenized_test["test"])

print("ROUGE 점수:", compute_metrics(predictions))