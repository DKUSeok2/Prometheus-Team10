import json
import torch
import nltk
from collections import Counter
from nltk import ngrams
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, AutoModel

# NLTK 토크나이저 다운로드
nltk.download("punkt")

# 1️⃣ JSON 파일 로드
json_path = "/home/elicer/Prometheus/Report_Summary/Report_data/REPORT-minute-00001-00001.json"  # JSON 파일 경로
with open(json_path, "r", encoding="utf-8") as file:
    data = json.load(file)

# 2️⃣ GPU 사용 여부 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 3️⃣ 모델 및 토크나이저 불러오기 (GPU로 이동)
# model_name = "google/pegasus-large"  # 또는 "google/pegasus-large"
# tokenizer = PegasusTokenizer.from_pretrained(model_name)
#model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

# Load model directly


tokenizer = AutoTokenizer.from_pretrained("gogamza/kobart-base-v2")
model = AutoModel.from_pretrained("gogamza/kobart-base-v2").to(device)

# 4️⃣ 회의록 원문 가져오기
passage = data["Meta(Refine)"]["passage"]

# 5️⃣ PEGASUS 요약 생성 (입력을 GPU로 이동)
inputs = tokenizer(passage, return_tensors="pt", truncation=True, padding="longest", max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}  # GPU로 이동

summary_ids = model.generate(**inputs)
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 6️⃣ 참조 요약 가져오기
reference_summary = data["Annotation"]["summary1"]  # summary1을 비교 대상으로 사용

# 7️⃣ F1 Score 계산 함수 (n-gram 기반)
def f1_score(preds, targets, n=4):
    f1_list = []
    for i in range(1, n+1):
        f1_n = []
        for pred, target in zip(preds, targets):
            epsilon = 1e-12

            # n-gram 카운트
            pred_cnt = Counter(ngrams(pred, i))
            target_cnt = Counter(ngrams(target, i))

            # BLEU@n, ROUGE@n 계산
            pred_all = sum(pred_cnt.values())
            target_all = sum(target_cnt.values())
            matched = 0
            for k, v in pred_cnt.items():
                if k in target_cnt:
                    matched += min(pred_cnt[k], target_cnt[k])
            bleu = matched / (pred_all + epsilon)
            rouge = matched / (target_all + epsilon)

            # F1@n 계산
            f1 = (2 * bleu * rouge) / (bleu + rouge + epsilon)
            f1_n.append(f1)
        f1_list.append(sum(f1_n) / len(f1_n))

    # 최종 F1-score 계산
    f1 = sum(f1_list) / len(f1_list)
    return f1

# 8️⃣ 토큰화 및 F1 Score 계산
gen_tokens = nltk.word_tokenize(generated_summary.lower())
ref_tokens = nltk.word_tokenize(reference_summary.lower())

f1_result = f1_score([gen_tokens], [ref_tokens])

# 9️⃣ 결과 출력
print(f"\n🔹 PEGASUS 요약: {generated_summary}")
print(f"🔹 참조 요약: {reference_summary}")
print(f"\n✅ F1 Score: {f1_result:.4f}") 