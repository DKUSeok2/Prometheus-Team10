import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
from nltk import ngrams
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# ✅ paust/pko-t5-base 모델 및 BPE 기반 토크나이저 로드
model_name = "paust/pko-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def truncate_text(text, max_tokens=512):
    """입력 텍스트를 최대 토큰 길이에 맞게 자르는 함수"""
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        return tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return text

def summarize_text_pko_t5(model, tokenizer, text, max_length=60):
    """pko-T5 모델을 사용하여 한국어 텍스트 요약"""
    text = truncate_text(text)  # 🔥 입력 길이 512 이하로 줄이기
    input_text = "요약: " + text  # 🔥 한국어 프롬프트로 변경
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="longest", max_length=512)

    with torch.no_grad():
        summary_ids = model.generate(
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"], 
            max_length=max_length, 
            num_beams=7,  # 🔥 더 나은 요약 선택
            temperature=0.7,  # 🔥 더 다양한 요약 생성
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def f1_score(preds, targets, n=4):
    """n-gram 기반 F1 Score 계산"""
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

# ✅ 한국어 원본 텍스트
original_text = """위원장 신학용] "좌석을 정돈해 주시기 바랍니다. 성원이 되었으므로 제309회 국회(임시회) 제1차 교육과학기술위원회를 개의하겠습니다. 우선 입법조사관의 보고사항이 있겠습니다." 입법조사관 김대형] "보고사항을 말씀드리겠습니다. (보고사항은 끝에 실음)" 위원장 신학용] "존경하는 선배ㆍ동료 위원님 여러분! 어제 본회의에서 여야를 막론하고 저를 교과위원장으로 선임해 주신 데 대해서 이 자리를 빌려 다시 한번 감사의 말씀드립니다..."""

# 정답 요약 (참조 요약)
reference_summary = "위원장 신 씨는 고질적인 기초과학 기반 부실에 이공계를 홀대하고 있는 우리나라는 교육과 과학기술 발전에 총력을 기울여야 한다고 했다."

# pko-T5를 이용한 요약 생성
predicted_summary = summarize_text_pko_t5(model, tokenizer, original_text)
print("Generated Summary:", predicted_summary)

# 정상적인 요약이 생성되지 않는 경우 경고 메시지
if len(predicted_summary.strip()) == 0 or predicted_summary in [",   .", ".", ""]:
    print("⚠️ 모델이 정상적으로 요약을 생성하지 않았습니다. 입력 길이 및 모델 선택을 확인하세요.")

# F1 Score 계산 (n-gram을 위한 토큰화)
predicted_tokens = word_tokenize(predicted_summary.lower())
reference_tokens = word_tokenize(reference_summary.lower())

# F1 Score 계산
f1 = f1_score([predicted_tokens], [reference_tokens])
print(f"F1 Score: {f1:.4f}")