# **Development Plan: Media-to-Text Converter (CLI & GUI)**

## 개요

이 개발 계획서는 '음성/영상 파일 변환기' 요구사항 명세서를 기반으로, **1단계: 핵심 CLI 애플리케이션 개발**과 **2단계: macOS GUI 애플리케이션 확장**으로 나누어 구체적인 실행 단계를 정의합니다. 개발은 '필수 기능'을 우선적으로 구현하여 최소 기능 제품(MVP)을 빠르게 구축하고, '선택/고급 기능'은 필요에 따라 추후에 확장하는 점진적인 접근 방식을 따릅니다.

---

## **Part 1: 핵심 CLI 애플리케이션 개발**

CLI 버전은 전체 애플리케이션의 백엔드 로직과 핵심 기능을 담당합니다.

### **1.1. 필수 개발 범위 (MVP: Minimum Viable Product)**

안정적인 기본 기능을 갖춘 CLI 도구를 우선적으로 개발합니다.

#### Step 1: 프로젝트 기본 구조 및 환경 설정
- **Action:**
    - 디렉토리 구조 생성: `media_converter/`, `tests/`
    - 메인 스크립트 파일 생성: `media_converter/main.py`
    - **핵심 의존성 파일 `requirements_core.txt` 생성:**
        - `faster-whisper`: 핵심 음성 인식 라이브러리
        - `pdfplumber`: PDF 텍스트 추출
        - `python-docx`: DOCX 텍스트 추출
        - `pytest`: 테스트 프레임워크
    - Python 가상 환경 초기화 및 의존성 설치
- **Unit Test:**
    - `pytest`가 올바르게 설정되었는지 확인하는 기본 테스트 (`test_initial.py`) 작성

#### Step 2: 기본 음성 추출(Transcription) 기능 구현
- **Action:**
    - `main.py`에 `transcribe_audio(file_path, model_name, language)` 함수 구현
    - **`faster-whisper`** 라이브러리를 사용하여 오디오 파일에서 텍스트를 추출
- **Unit Test:**
    - 샘플 오디오 파일(`sample_en.mp3`)을 사용하여 `transcribe_audio()` 호출 후, 결과 텍스트가 예상과 일치하는지 검증 (`test_transcription.py`)

#### Step 3: 기본 CLI 옵션 및 파일 출력 구현
- **Action:**
    - Python의 `argparse`를 사용하여 필수 CLI 인자 및 옵션 구현
        - 필수 인자: `input_file`
        - 선택 옵션: `--input_type`, `--model`, `--language`, `--output_dir`
    - 처리 결과를 입력 파일명 기반의 `.txt` 파일로 지정된 경로에 저장하는 로직 구현
- **Unit Test:**
    - CLI 호출을 시뮬레이션하여 올바른 함수가 정확한 인자와 함께 호출되는지 검증 (`test_cli.py`)
    - 출력 파일이 정확한 위치에 올바른 내용으로 생성되는지 검증하고, 테스트 후 생성된 파일 정리

---

### **1.2. 선택적/고급 기능 개발 범위**

기본 기능이 안정화된 후, 사용성을 높이는 고급 기능들을 추가로 개발합니다.

#### Step 4: 다국어 번역 기능 추가
- **Action:**
    - `requirements_full.txt`에 의존성 추가: `transformers`, `torch`, `sentencepiece`
    - `--translate_to <lang_code>` CLI 옵션 추가
    - `transformers` 라이브러리와 **`facebook/nllb-200-distilled-600M`** 모델을 사용하여 번역 기능 함수 `translate_text(...)` 구현
- **Unit Test:**
    - 번역 기능 호출 시, 외부 라이브러리를 모킹(mocking)하여 올바른 파라미터로 함수가 호출되는지 검증

#### Step 5: 텍스트/PDF 직접 처리 및 요약 기능 확장
- **Action:**
    - `--summarize` CLI 옵션 추가
    - 입력 파일 유형에 따라 처리 로직 분기:
        - **오디오/비디오:** `transcribe_audio` 함수에서 반환된 타임스탬프(`segments`) 정보를 활용하여 시간대별 요약 기능 구현.
        - **텍스트/PDF:** 음성인식 단계를 건너뛰고, 파일에서 직접 텍스트를 읽어와 섹션별/문단별 요약을 수행하는 `summarize_text(...)` 함수 구현.
    - `transformers`의 요약 모델(예: `facebook/bart-large-cnn`)을 사용하여 요약 기능 구현
- **Unit Test:**
    - 샘플 텍스트 및 PDF 파일로 요약 기능 검증
    - 요약 기능 호출 시, 외부 라이브러리를 모킹하여 요약 파일이 정상적으로 생성되는지 검증

#### Step 6: 자막(SRT) 생성 및 비디오 지원 추가
- **Action:**
    - `--subtitle` CLI 옵션 추가 및 SRT 파일 포맷 생성 함수 구현
    - 비디오 파일(`mp4`, `mov` 등) 입력을 처리하기 위해 **`ffmpeg`**을 `subprocess`로 호출하여 오디오를 임시 파일로 추출하는 로직 추가
- **Unit Test:**
    - 샘플 비디오 파일로 비디오 처리 기능 검증
    - SRT 생성 기능 호출 시, 타임스탬프 형식이 올바른지 검증

#### Step 7: 배치 처리 기능 추가
- **Action:**
    - `input_file` 인자가 디렉토리일 경우, 내부의 모든 미디어 파일을 순차적으로 처리하는 로직 추가
    - `tqdm` 라이브러리를 `requirements_full.txt`에 추가하고, 처리 진행 상황을 프로그레스 바로 시각화
- **Unit Test:**
    - 여러 개의 샘플 파일이 담긴 디렉토리를 입력했을 때, 정확한 수의 출력 파일이 생성되는지 검증

---

## **Part 2: macOS GUI 애플리케이션 개발**

CLI의 모든 기능을 감싸는 직관적인 macOS GUI를 개발합니다.

### **2.1. 필수 개발 범위 (기본 GUI)**

사용자가 핵심 기능을 마우스 클릭만으로 사용할 수 있는 기본 인터페이스를 구축합니다.

#### Step 8: GUI 기술 스택 선정 및 기본 창 구현
- **Action:**
    - GUI 기술 스택으로 **PySide6** 선정 (빠른 개발 속도와 네이티브에 가까운 위젯 스타일 제공)
    - `requirements_gui.txt` 파일에 `PySide6` 추가
    - 파일 목록(ListWidget), 옵션 설정 영역, 시작 버튼을 포함하는 메인 윈도우(`QMainWindow`) 설계 및 구현

#### Step 9: 핵심 기능 UI 연동
- **Action:**
    - '파일/폴더 추가' 버튼과 파일 목록 기능 구현
    - '모델', '언어' 선택을 위한 드롭다운 메뉴(ComboBox) 구현
    - '출력 경로 변경' 버튼 기능 구현
    - [변환 시작] 버튼 클릭 시, UI에서 설정된 값들을 바탕으로 Part 1에서 개발한 핵심 `transcribe_audio` 함수를 호출하도록 연동

### **2.2. 선택적/고급 기능 개발 범위**

사용자 경험(UX)을 극대화하기 위한 고급 GUI 기능을 추가합니다.

#### Step 10: 고급 기능 UI 연동 및 비동기 처리
- **Action:**
    - '번역', '요약', '자막' 등 선택적 기능들을 위한 체크박스(CheckBox) UI 추가 및 로직 연동
    - 입력 파일 유형에 따라 UI 옵션 활성화/비활성화 로직 추가 (예: 텍스트/PDF 파일 선택 시 '자막 생성' 체크박스 비활성화)
    - **`QThread`**를 사용하여 파일 처리 로직을 백그라운드 스레드에서 실행. 이를 통해 처리 중 UI가 멈추는 현상(freezing)을 방지 (비동기 처리)

#### Step 11: 사용자 경험(UX) 개선
- **Action:**
    - 파일 처리 진행 상황을 실시간으로 표시하는 프로그레스 바(ProgressBar) 및 상태 메시지 라벨(Label) 구현
    - 파일 목록에 **드래그 앤 드롭(Drag and Drop)** 기능 추가
    - 작업 완료 후 "Finder에서 보기"와 같은 편의 기능 링크 추가

---

## **Part 3: 최종화 및 배포**

#### Step 12: 코드 리팩토링 및 문서화
- **Action:**
    - 전체 코드베이스의 가독성 및 모듈성 향상을 위한 리팩토링 수행
    - 모든 함수와 클래스에 Docstring 및 타입 힌트 추가
    - CLI와 GUI 사용법을 상세히 설명하는 `README.md` 최종 업데이트

#### Step 13: 최종 테스트 및 릴리즈
- **Action:**
    - 다양한 실제 미디어 파일로 통합 테스트 수행
    - `git tag`를 사용하여 버전 릴리즈 (예: `v1.0-cli`, `v1.1-gui`)