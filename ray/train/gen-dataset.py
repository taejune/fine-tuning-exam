import json

# 간단한 지시어-답변 쌍 데이터셋
data = [
    {"instruction": "서버 상태가 어때?", "output": "현재 모든 시스템이 정상이며, 응답 시간은 20ms입니다."},
    {"instruction": "에러 로그 확인해줘.", "output": "최근 5분간 발생한 에러는 없으며, 트래픽은 안정적입니다."},
    {"instruction": "점검 일정 알려줘.", "output": "다음 정기 점검은 이번 주 금요일 새벽 2시부터 4시까지입니다."},
    {"instruction": "데이터베이스 연결 상태는?", "output": "메인 DB와 레플리카 모두 연결 활성 상태입니다."},
    {"instruction": "현재 접속자 수 알려줘.", "output": "실시간 접속자 수는 1,240명으로 평소보다 10% 증가했습니다."}
]

with open("my_dataset.jsonl", "w", encoding="utf-8") as f:
    for entry in data:
        # Qwen-Instruct 포맷에 맞게 구조화 (사용자-어시스턴트 대화형)
        json_obj = {
            "messages": [
                {"role": "user", "content": entry["instruction"]},
                {"role": "assistant", "content": entry["output"]}
            ]
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

print("데이터셋 생성 완료: my_dataset.jsonl")
