# **개발 체크리스트: Media-to-Text Converter**

이 문서는 `requirement.md`와 `development_plan.md`를 기반으로 프로젝트의 각 개발 단계를 점검하기 위한 체크리스트입니다.

---

## **Part 1: 핵심 CLI 애플리케이션 개발**

### **Step 1: 프로젝트 기본 구조 및 환경 설정**
- [ ] `media_converter/` 및 `tests/` 디렉토리 생성
- [ ] `media_converter/main.py` 파일 생성
- [ ] `requirements_core.txt` 파일 생성 및 `faster-whisper`, `pytest` 명시
- [ ] Python 가상 환경 설정 및 의존성 설치 확인
- [ ] `tests/test_initial.py`에 `pytest` 기본 설정 확인용 테스트 작성 및 통과

### **Step 2: 기본 음성 추출(Transcription) 기능 구현**
- [ ] `transcribe_audio(file_path, model_name, language)` 함수 구현
- [ ] `faster-whisper` 라이브러리를 사용하여 텍스트 추출 기능 동작 확인
- [ ] 샘플 오디오 파일(`sample_en.mp3`)로 기능 테스트 및 결과 텍스트 검증

### **Step 3: 기본 CLI 옵션 및 파일 출력 구현**
- [ ] `argparse`를 이용한 CLI 인터페이스 구현
- [ ] 필수 인자 `input_file` (파일/디렉토리 경로) 정상 동작 확인
- [ ] 선택 옵션 `--model` 동작 및 기본값(`base`) 설정 확인
- [ ] 선택 옵션 `--language` 동작 및 자동 감지 기능 확인
- [ ] 선택 옵션 `--output_dir` 동작 및 기본값(입력 파일 위치) 설정 확인
- [ ] 결과가 `.txt` 파일로 올바른 경로에 정확한 내용으로 저장되는지 확인

### **Step 4: 다국어 번역 기능 추가**
- [ ] `requirements_full.txt`에 `transformers`, `torch`, `sentencepiece` 추가
- [ ] `--translate_to <lang_code>` CLI 옵션 추가
- [ ] 번역 기능(`translate_text`) 구현 및 NLLB 모델 연동 확인
- [ ] 번역된 텍스트 파일(예: `..._translated_ko.txt`)이 추가로 생성되는지 확인

### **Step 5: 시간대별 요약 기능 추가**
- [ ] `--summarize` CLI 옵션 추가
- [ ] `segments` 정보를 활용한 시간대별 요약 기능(`summarize_transcript`) 구현
- [ ] 요약 모델(BART) 연동 및 요약 파일 생성 확인

### **Step 6: 자막(SRT) 생성 및 비디오 지원 추가**
- [ ] `--subtitle` CLI 옵션 추가
- [ ] 타임스탬프를 포함한 `.srt` 자막 파일 생성 기능 확인
- [ ] `ffmpeg`을 사용하여 비디오 파일(`mp4`, `mov` 등)에서 오디오 트랙을 추출하는 로직 구현
- [ ] 샘플 비디오 파일 입력 시 정상 처리되는지 확인

### **Step 7: 배치 처리 기능 추가**
- [ ] `input_file`에 디렉토리 지정 시, 내부 미디어 파일들을 순차적으로 처리하는 기능 확인
- [ ] `tqdm` 라이브러리 추가 및 파일 처리 시 프로그레스 바 표시 확인

---

## **Part 2: macOS GUI 애플리케이션 개발**

### **Step 8: GUI 기술 스택 선정 및 기본 창 구현**
- [ ] `requirements_gui.txt`에 `PySide6` 추가
- [ ] 메인 윈도우(`QMainWindow`) 생성
- [ ] UI 요소 배치: 파일 목록, 설정 패널, 시작 버튼

### **Step 9: 핵심 기능 UI 연동**
- [ ] '파일/폴더 추가' 버튼 및 드래그 앤 드롭으로 파일 목록 추가 기능 구현
- [ ] '모델', '언어' 선택 드롭다운 메뉴 기능 구현 및 CLI 로직 연동
- [ ] '출력 경로 변경' 버튼 기능 구현
- [ ] '변환 시작' 버튼 클릭 시, 설정된 값으로 CLI 핵심 기능이 호출되는지 확인

### **Step 10: 고급 기능 UI 연동 및 비동기 처리**
- [ ] '번역', '요약', '자막' 기능 체크박스 UI 추가 및 로직 연동
- [ ] `QThread`를 사용하여 파일 처리 중 UI가 멈추지 않는지 (비동기) 확인

### **Step 11: 사용자 경험(UX) 개선**
- [ ] 처리 진행률을 표시하는 프로그레스 바 및 상태 메시지 라벨 구현
- [ ] 작업 완료 후 'Finder에서 보기' 링크 기능 구현
- [ ] 다크 모드 등 macOS 시스템 설정과 자연스럽게 연동되는지 확인

---

## **Part 3: 최종화 및 배포**

### **Step 12: 코드 리팩토링 및 문서화**
- [ ] 전체 코드 가독성 및 모듈성 검토
- [ ] 모든 함수/클래스에 Docstring 및 타입 힌트 추가 완료
- [ ] `README.md`에 CLI 및 GUI 사용법 상세히 업데이트

### **Step 13: 최종 테스트 및 릴리즈**
- [ ] 다양한 종류의 실제 미디어 파일(오디오, 비디오, 여러 언어)로 통합 테스트 수행
- [ ] `git tag`를 사용하여 최종 버전 릴리즈 (예: `v1.0`)
