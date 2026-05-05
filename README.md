# VoiceKin Speaker Verification Server

VoiceKin 1차 서버는 가족 사칭형 AI 보이스피싱 방지를 위한 **AI 모델 기반 화자 인증(Speaker Verification) API**입니다. 두 음성 파일을 업로드하면 SpeechBrain의 pretrained ECAPA-TDNN 모델로 speaker embedding을 추출하고 cosine similarity를 계산해 같은 화자인지 판별합니다.

## 기술 스택

- Python
- FastAPI
- SpeechBrain
- PyTorch / Torchaudio
- Pretrained model: `speechbrain/spkrec-ecapa-voxceleb`

## 프로젝트 구조

```text
app/
  main.py
  api/routes/voice.py
  core/config.py
  schemas/voice.py
  services/speaker_service.py
  utils/audio.py
requirements.txt
README.md
.gitignore
```

## 설치 방법

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

mp3, m4a 디코딩은 실행 환경의 torchaudio/FFmpeg 지원 상태에 영향을 받습니다. 이 서버는 먼저 torchaudio로 디코딩을 시도하고, 실패하면 로컬 FFmpeg 명령어로 16kHz mono wav 변환을 다시 시도합니다. macOS에서 m4a 또는 일부 mp3가 디코딩되지 않으면 FFmpeg 설치가 필요합니다.

```bash
brew install ffmpeg
```

## 실행 방법

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

처음 `/api/v1/voice/compare` 요청이 들어오면 SpeechBrain이 Hugging Face에서 pretrained model 파일을 `pretrained_models/spkrec-ecapa-voxceleb` 경로로 자동 다운로드합니다. 첫 요청은 다운로드 때문에 시간이 오래 걸릴 수 있습니다.

## API

### Health Check

```bash
curl http://localhost:8000/health
```

응답:

```json
{"status":"ok"}
```

### Voice Compare

```bash
curl -X POST "http://localhost:8000/api/v1/voice/compare" \
  -F "audio_file_1=@/path/to/family_voice.wav" \
  -F "audio_file_2=@/path/to/call_voice.wav"
```

응답 예시:

```json
{
  "similarity": 0.83,
  "threshold": 0.75,
  "is_same_speaker": true,
  "message": "same_speaker",
  "model_name": "speechbrain/spkrec-ecapa-voxceleb"
}
```

Swagger UI로도 테스트할 수 있습니다.

```text
http://localhost:8000/docs
```

## 설정 변경

설정은 `app/core/config.py`에 모아두었고 환경변수로 덮어쓸 수 있습니다. 환경변수 prefix는 `VOICEKIN_`입니다.

```bash
VOICEKIN_SPEAKER_THRESHOLD=0.72 uvicorn app.main:app --reload
VOICEKIN_MAX_UPLOAD_SIZE_MB=50 uvicorn app.main:app --reload
VOICEKIN_MIN_AUDIO_SECONDS=1.5 uvicorn app.main:app --reload
VOICEKIN_DEVICE=cpu uvicorn app.main:app --reload
```

주요 설정:

- `speaker_threshold`: 같은 화자로 판단할 cosine similarity 기준값입니다. 기본값은 `0.75`입니다.
- `max_upload_size_mb`: 업로드 파일당 최대 크기입니다. 기본값은 `25MB`입니다.
- `min_audio_seconds`: 너무 짧은 음성을 거부하기 위한 최소 길이입니다. 기본값은 `1.0초`입니다.
- `target_sample_rate`: 모델 입력용 샘플레이트입니다. 기본값은 `16000Hz`입니다.
- `device`: 기본값은 `cpu`입니다. CUDA 서버에서는 `cuda` 또는 `auto`로 바꿀 수 있습니다.

## 모델 설명

이 서버는 SpeechBrain의 `speechbrain/spkrec-ecapa-voxceleb` pretrained 모델을 사용합니다. 이 모델은 VoxCeleb 데이터셋으로 학습된 ECAPA-TDNN 기반 speaker verification 모델입니다.

처리 흐름:

1. 업로드된 `wav`, `mp3`, `m4a` 파일을 임시 파일로 저장합니다.
2. torchaudio로 디코딩합니다. torchaudio가 m4a/mp3를 읽지 못하면 FFmpeg CLI fallback으로 wav 변환을 시도합니다.
3. mono 채널로 변환하고 16kHz로 resampling합니다.
4. 표준 wav 임시 파일로 저장합니다.
5. SpeechBrain `EncoderClassifier.encode_batch()`로 speaker embedding을 추출합니다.
6. 두 embedding 사이의 cosine similarity를 계산합니다.
7. `similarity >= threshold`이면 `same_speaker`, 아니면 `different_speaker`를 반환합니다.

SpeechBrain 모델 카드에 따르면 해당 모델은 speaker embedding 추출과 cosine similarity 기반 speaker verification에 사용할 수 있으며, 입력 음성은 16kHz single-channel 기준으로 학습되었습니다.

참고:

- https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- https://speechbrain.readthedocs.io/en/stable/API/speechbrain.inference.classifiers.html

## 예외 처리

서버는 다음 상황을 API 오류로 반환합니다.

- 파일 누락 또는 빈 파일: `400`
- 지원하지 않는 확장자: `415`
- 업로드 파일 크기 초과: `413`
- 음성 디코딩 실패: `422`
- 너무 짧은 음성: `400`
- embedding 추출 또는 모델 오류: `500`

요청 처리 중 생성한 원본 임시 파일과 변환 wav 파일은 `finally` 블록에서 정리됩니다.

## 확장 방향

현재는 speaker verification만 구현되어 있습니다. 이후 기능은 현재 구조에 맞춰 확장할 수 있습니다.

- 등록된 가족 voiceprint 저장: speaker embedding을 DB에 저장하고 재사용
- 가족 화이트리스트 비교: 여러 가족 embedding과 call embedding을 일괄 비교
- anti-spoofing / 딥페이크 탐지: `app/services/anti_spoofing_service.py`를 추가하고 AASIST 같은 모델을 별도 서비스 클래스로 구현
- Android 앱 연동: multipart 업로드 API를 앱에서 호출

이 구조에서는 FastAPI 라우터가 모델 세부 구현을 직접 알지 않고 `SpeakerVerificationService`만 호출합니다. 따라서 anti-spoofing 모델을 추가할 때도 API 레이어에서 `SpeakerVerificationService`와 `AntiSpoofingService`를 조합하는 방식으로 확장할 수 있습니다.
