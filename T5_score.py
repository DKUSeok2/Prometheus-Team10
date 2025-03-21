import re
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter
from nltk import ngrams
# nltk 데이터 다운로드
nltk.download('punkt')

# 토크나이저 및 모델 불러오기
model_name = "eenzeenee/t5-base-korean-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def clean_meeting_text(text):
    """ 발언자 제거 및 텍스트 정리 """
    text = re.sub(r"\[.*?\]\s*", "", text)  # 발언자 제거
    text = re.sub(r"\s+", " ", text).strip()  # 불필요한 공백 정리
    return text

def summarize_text(text, max_length=150):
    """ 텍스트 요약을 수행하는 함수 """
    cleaned_text = clean_meeting_text(text)
    input_ids = tokenizer.encode("summarize: " + cleaned_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def f1_score(preds, targets, n=4):
    """ n-gram 기반 F1-score 계산 함수 """
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
            matched = sum(min(pred_cnt[k], target_cnt[k]) for k in pred_cnt if k in target_cnt)

            bleu = matched / (pred_all + epsilon)
            rouge = matched / (target_all + epsilon)

            # F1@n 계산
            f1 = (2 * bleu * rouge) / (bleu + rouge + epsilon)
            f1_n.append(f1)
        f1_list.append(sum(f1_n) / len(f1_n))

    # 최종 F1-score 계산
    f1 = sum(f1_list) / len(f1_list)
    return f1

# 긴 회의록 예제 (발언자 포함)
meeting_text = """
[위원장 신학용] "좌석을 정돈해 주시기 바랍니다. 성원이 되었으므로 제309회 국회(임시회) 제1차 교육과학기술위원회를 개의하겠습니다. 우선 입법조사관의 보고사항이 있겠습니다."
[입법조사관 김대형] "보고사항을 말씀드리겠습니다. (보고사항은 끝에 실음)"
[위원장 신학용] "존경하는 선배ㆍ동료 위원님 여러분! 어제 본회의에서 여야를 막론하고 저를 교과위원장으로 선임해 주신 데 대해서 이 자리를 빌려 다시 한번 감사의 말씀드립니다.
잘 아시다시피 우리 국민의 최대 관심사는 교육 문제입니다. 그리고 매년 집행되는 국가 예산 중에 가장 많은 비중을 차지하는 것도 바로 우리 교과위 소관 기관들입니다.
하지만 우리나라 교육의 현실은 어떻습니까? 여전히 공교육에 대한 불신이 가득하고 사교육비 부담은 해소될 기미를 보이지 않고 있습니다.
그뿐입니까? 이웃 나라 중국은 유인 우주선 발사에 성공하고 우주정거장 시대를 열어 가고 있습니다. 일본은 역대 노벨상 수상자 18명 중 15명이 기초과학 분야에서 업적을 인정받았다고 합니다.
하지만 우리나라는 고질적인 기초과학 기반 부실에다 이공계 홀대로 우수 인재들은 의학전문대학원이나 로스쿨로 몰리고 있습니다.
세계 경제 불황과 잠재성장률 하락의 위기를 목전에 둔 우리나라로서는 무엇보다도 교육과 과학기술 발전에 총력을 기울여야 할 것입니다.
물론 학문에는 왕도가 없다는 말처럼 교육과 과학기술 발전에 만능열쇠 같은 묘수는 없다고 봅니다.
다만 저는 이처럼 막중한 사명감을 가지고 대한민국의 교육과 과학기술 발전을 위해 착실히 한 걸음씩 나아가는 것이야말로 우리를 선택해 주신 유권자들의 기대에 부응하는 것이리라 생각합니다.
앞으로 우리 상임위의 운영에 있어서는 여야가 서로 존중하며 대화와 협력을 통해 대승적인 견지에서 합의를 도출해내도록 노력하겠습니다.
여러 모로 부족한 점이 있더라도 선배ㆍ동료 위원 여러분께서 적극적으로 위원회 운영에 협조해 주실 것을 간곡히 당부드립니다.
의사일정에 들어가기 전에 위원님들 상호간에 인사를 나누도록 하겠습니다.
"""

# 실제 정답 요약 (reference summary)
reference_summary = "위원장 신 씨는 고질적인 기초과학 기반 부실에 이공계를 홀대하고 있는 우리나라는 교육과 과학기술 발전에 총력을 기울여야 한다고 했다."

# 모델을 사용한 요약 생성
predicted_summary = summarize_text(meeting_text)

# 토큰화하여 F1 Score 계산
predicted_tokens = nltk.word_tokenize(predicted_summary)
reference_tokens = nltk.word_tokenize(reference_summary)

# n-gram 기반 F1 Score 계산
f1 = f1_score([predicted_tokens], [reference_tokens])

# 결과 출력
print("생성된 요약:", predicted_summary)
print(f"F1 Score: {f1:.4f}")