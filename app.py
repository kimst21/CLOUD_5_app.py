import base64
import requests
import sys
import json
import time
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

HA_URL = "http://192.168.10.11:8123" # HA ip
HA_TOKEN = "발행토큰" # insert token
HA_SENSOR_ENTITY = "sensor.vision_bridge_result"
HA_SAFETY_ENTITY = "input_text.vision_safety_status"

app = FastAPI()
client = OpenAI()

print("=" * 50, file=sys.stderr)
print("Vision Bridge - GPT 직접 판단 모드", file=sys.stderr)
print("=" * 50, file=sys.stderr)

class AnalyzeRequest(BaseModel):
    image_url: str
    context: str | None = None

def fetch_image_base64(url: str) -> str:
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    return base64.b64encode(r.content).decode("utf-8")

def update_ha_state(entity_id: str, state: str, attributes: dict = {}):
    url = f"{HA_URL}/api/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {"state": state, "attributes": attributes}
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    r.raise_for_status()
    print(f"HA 업데이트: {entity_id} = {state}", file=sys.stderr)

def delayed_ready(seconds: int):
    """백그라운드에서 대기 후 READY 복귀"""
    time.sleep(seconds)
    try:
        update_ha_state(HA_SAFETY_ENTITY, "READY")
        print("READY 복귀 완료", file=sys.stderr)
    except Exception as e:
        print(f"READY 복귀 실패: {e}", file=sys.stderr)

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    print("Vision 분석 시작", file=sys.stderr)
    raw = ""

    try:
        # 1️⃣ 이미지 로드
        image_b64 = fetch_image_base64(req.image_url)

        # 2️⃣ GPT 판단 + 설명 요청
        prompt = (
            "이 영상을 분석하세요.\n\n"
            "응답 형식 (JSON만 출력):\n"
            "{\n"
            '  "status": "DANGER 또는 SAFE",\n'
            '  "summary": "40자 이내 한국어 한 문장 설명"\n'
            "}\n\n"
            "DANGER 조건: 사람이 명확히 보일 때\n"
            "SAFE 조건: 사람이 없거나 불분명할 때\n"
            "JSON만 출력하세요. 다른 텍스트 금지."
        )

        # 3️⃣ OpenAI Vision 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.1,
        )

        # 4️⃣ JSON 파싱
        raw = response.choices[0].message.content.strip()
        print(f"GPT 응답 원본: [{raw}]", file=sys.stderr)

        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

        status = result.get("status", "SAFE").upper()
        summary = result.get("summary", "분석 결과 없음")

        if status not in ["SAFE", "DANGER"]:
            status = "SAFE"

        if len(summary) > 60:
            summary = summary[:58] + ".."

        print(f"판정: {status} / 설명: {summary} ({len(summary)}자)", file=sys.stderr)

        # 5️⃣ HA 센서 업데이트
        update_ha_state(
            HA_SENSOR_ENTITY,
            summary[:240],
            {"full_summary": summary}
        )

        # 6️⃣ THINKING → SAFE/DANGER (자동화 트리거)
        update_ha_state(HA_SAFETY_ENTITY, "THINKING")
        time.sleep(0.5)
        update_ha_state(HA_SAFETY_ENTITY, status)

        # 7️⃣ ✅ 백그라운드에서 15초 후 READY 복귀 (블로킹 없음)
        t = threading.Thread(target=delayed_ready, args=(15,), daemon=True)
        t.start()

        # 8️⃣ 즉시 응답 반환 (HA REST 타임아웃 방지)
        return {"status": status, "summary": summary}

    except json.JSONDecodeError as e:
        print(f"JSON 파싱 실패: {e} / 원본: {raw}", file=sys.stderr)
        update_ha_state(HA_SAFETY_ENTITY, "THINKING")
        time.sleep(0.5)
        update_ha_state(HA_SAFETY_ENTITY, "SAFE")
        t = threading.Thread(target=delayed_ready, args=(15,), daemon=True)
        t.start()
        raise HTTPException(status_code=500, detail=f"JSON parse error: {e}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        try:
            update_ha_state(HA_SAFETY_ENTITY, "READY")
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))
# ==========================================
# TTS 완료 후 정리 (별도 스레드)
# ==========================================
def cleanup_after_tts(delay_sec: int = 12):
    """TTS 재생 완료 후 LED 끄기 + READY 복귀
    - 별도 스레드 실행 → /analyze 응답 블로킹 없음
    - daemon=True → 메인 프로세스 종료 시 자동 종료
    """
    time.sleep(delay_sec)
    try:
        call_ha_service("light", "turn_off", {"entity_id": LED_ENTITY})
        update_ha_sensor(STATUS_ENTITY, "READY")
        print("[CLEANUP] LED 끄기 + READY 완료", file=sys.stderr)
    except Exception as e:
        print(f"[CLEANUP ERROR] {e}", file=sys.stderr)

# ==========================================
# 메인 분석 엔드포인트
# ==========================================
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        t0 = time.time()
        print("=" * 50, file=sys.stderr)
        print(f"[ANALYZE] Started: {time.strftime('%H:%M:%S')}", file=sys.stderr)

        # ----------------------------------------
        # 1단계: 이미지 취득
        # ----------------------------------------
        image_b64 = fetch_image_base64(req.image_url)
        print(f"[FETCH] 완료: {time.time()-t0:.1f}초 / {len(image_b64)} bytes", file=sys.stderr)

        # ----------------------------------------
        # 2단계: GPT-4o Vision 영상 분석
        # ----------------------------------------
        t1 = time.time()
        prompt = """이 보안 카메라 영상을 분석하세요.

다음을 반드시 확인하세요:
1. 사람이 있는가? (전체 또는 일부)
2. 얼굴, 머리, 손, 팔, 다리, 몸통이 보이는가?
3. 옷이나 신발이 보이는가?
4. 사람의 그림자나 실루엣이 보이는가?

결과를 한국어로 사실만 간단히 2문장으로 설명하세요.
사람이 있으면 반드시 사람, 얼굴, 손 등의 단어를 포함하세요.
사람이 없으면 사람 없음을 명시하세요."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high"
                            }
                        },
                    ],
                }
            ],
            max_tokens=150,
            temperature=0.0,
        )

        vision_summary = response.choices[0].message.content.strip()
        print(f"[VISION] 완료: {time.time()-t1:.1f}초 / {vision_summary}", file=sys.stderr)

        # ----------------------------------------
        # 3단계: LLM 위협 판단
        # ----------------------------------------
        t2 = time.time()
        llm_result = llm_reasoning(vision_summary)
        print(f"[LLM] 완료: {time.time()-t2:.1f}초 / {llm_result}", file=sys.stderr)

        # ----------------------------------------
        # 4단계: HA 센서 업데이트
        # ----------------------------------------
        t3 = time.time()
        update_ha_sensor(HA_SENSOR_ENTITY, vision_summary)
        update_ha_sensor(HA_THREAT_LEVEL, str(llm_result['threat_level']))
        update_ha_sensor(HA_REASONING, llm_result['reasoning'])
        update_ha_sensor(HA_DOOR_POSITION, str(llm_result['door_position']))
        print(f"[SENSOR] 완료: {time.time()-t3:.1f}초", file=sys.stderr)

        # ----------------------------------------
        # 5단계: DANGER/SAFE 즉시 처리
        # ----------------------------------------
        t4 = time.time()
        if llm_result['threat_level'] >= 1:
            print("[ACTION] DANGER 처리 시작", file=sys.stderr)

            update_ha_sensor(STATUS_ENTITY, "DANGER")

            call_ha_service("tts", "google_translate_say", {
                "entity_id": MEDIA_PLAYERS,
                "message": f"경고! 경고! 사람이 감지되어 문을 잠급니다. {vision_summary}",
                "language": "ko",
                "cache": False
            })

            call_ha_service("light", "turn_on", {
                "entity_id": LED_ENTITY,
                "brightness": 255,
                "rgb_color": [255, 0, 0]
            })

            call_ha_service("number", "set_value", {
                "entity_id": SERVO_ENTITY,
                "value": 180
            })

        else:
            print("[ACTION] SAFE 처리 시작", file=sys.stderr)

            update_ha_sensor(STATUS_ENTITY, "SAFE")

            call_ha_service("tts", "google_translate_say", {
                "entity_id": MEDIA_PLAYERS,
                "message": f"안전하므로 문을 엽니다. {vision_summary}",
                "language": "ko",
                "cache": False
            })

            call_ha_service("light", "turn_on", {
                "entity_id": LED_ENTITY,
                "brightness": 255,
                "rgb_color": [0, 255, 0]
            })

            call_ha_service("number", "set_value", {
                "entity_id": SERVO_ENTITY,
                "value": 0
            })

        print(f"[HA] 완료: {time.time()-t4:.1f}초", file=sys.stderr)
