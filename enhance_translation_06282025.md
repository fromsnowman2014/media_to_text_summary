# 번역 성능 개선 계획 (Translation Enhancement Plan)

## 문제 상황 (Problem Statement)
현재 NLLB-200-distilled-600M 모델을 사용한 번역 시스템에서 모델 로딩 오류가 발생하여 번역이 제대로 수행되지 않고 있습니다. 
또한, 긴 텍스트를 한 번에 번역하려는 시도가 모델의 최적 입력 크기를 초과하여 성능 저하를 초래할 수 있습니다.

## 개선 방안 (Enhancement Plan)

### 1. NLLB-distilled-600M 모델의 최적 입력 크기 확인 (Identify Optimal Input Size)
- NLLB-distilled-600M 모델은 본래 문서보다는 문장 단위 번역을 위해 설계되었습니다.
- 토큰 제한: 512 토큰 (약 1,000-1,500자 정도)
- 최적 성능을 위해 문단 단위 (500-800자)로 나누는 것이 권장됩니다.
- 모델이 입력을 처리하는 방식을 고려할 때, 자연스러운 문단 분리를 유지하면서 텍스트를 분할하는 것이 좋습니다.

### 2. 텍스트 분할 전략 (Text Splitting Strategy)
- 트랜스크립트 텍스트를 자연스러운 문단 경계에서 최대한 분할합니다.
- 가능하다면 문장 경계를 우선적으로 사용합니다.
- 각 청크(chunk)는 대략 400-500 토큰 (500-800자) 정도로 제한합니다.
- 청크 간 약간의 중첩(50-100자)을 통해 문맥 연속성을 유지합니다.

### 3. 반복 번역 구현 (Implement Iterative Translation)
```python
# 개선된 번역 알고리즘의 의사코드 (Pseudocode for enhanced translation)
def translate_large_text(text, src_lang, target_lang):
    # 1. 텍스트를 적절한 크기의 청크로 분할
    chunks = split_text_into_chunks(text, max_tokens=400)
    
    # 2. 각 청크를 개별적으로 번역
    translated_chunks = []
    for chunk in chunks:
        try:
            translated_chunk = translator.translate(chunk, src_lang, target_lang)
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            else:
                # 번역 실패 시 원본 텍스트 유지 또는 오류 표시
                translated_chunks.append(f"[번역 오류] {chunk}")
        except Exception as e:
            logging.error(f"청크 번역 중 오류 발생: {e}")
            translated_chunks.append(f"[번역 오류] {chunk}")
    
    # 3. 번역된 청크 병합
    return merge_translated_chunks(translated_chunks)
```

### 4. 청크 분할 함수 구현 (Implement Chunk Splitting)
```python
def split_text_into_chunks(text, max_tokens=400):
    # 문단 단위로 먼저 분할 시도
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    for paragraph in paragraphs:
        # 대략적인 토큰 수 추정 (실제 구현에서는 토크나이저 사용 권장)
        paragraph_tokens = len(paragraph.split()) 
        
        # 현재 청크에 현재 문단을 추가했을 때 최대 토큰 수를 초과하는 경우
        if current_token_count + paragraph_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
            current_token_count = paragraph_tokens
        # 현재 문단 자체가 최대 토큰 수를 초과하는 경우 문장 단위로 분할
        elif paragraph_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_tokens = len(sentence.split())
                if current_token_count + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_token_count = sentence_tokens
                else:
                    current_chunk += " " + sentence
                    current_token_count += sentence_tokens
        # 현재 청크에 현재 문단을 추가해도 최대 토큰 수를 초과하지 않는 경우
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            current_token_count += paragraph_tokens
    
    # 마지막 청크 추가
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### 5. 번역된 청크 병합 (Merge Translated Chunks)
```python
def merge_translated_chunks(translated_chunks):
    # 단순히 청크들을 연결
    # 필요에 따라 더 복잡한 병합 로직 구현 가능
    return "\n\n".join(translated_chunks)
```

### 6. 메인 번역 함수 개선 (Enhance Main Translation Function)
`translator.py` 파일의 `translate` 메소드를 수정하여 대용량 텍스트 처리를 지원하도록 합니다:

```python
def translate(self, text: str, src_lang: str, target_lang: str):
    """
    Translate text from source language to target language.
    
    Args:
        text: The text to translate
        src_lang: Source language code (e.g., 'en', 'ko')
        target_lang: Target language code (e.g., 'en', 'ko')
        
    Returns:
        Translated text or None if translation failed
    """
    if not self.is_language_supported(src_lang) or not self.is_language_supported(target_lang):
        logging.error(f"Translation from '{src_lang}' to '{target_lang}' is not supported.")
        return None

    # 텍스트 길이 확인 - 긴 텍스트는 청크로 분할하여 처리
    if len(text.split()) > 500:  # 대략적인 기준, 정확한 토큰 수는 토크나이저로 계산해야 함
        logging.info(f"Long text detected. Splitting into chunks for translation...")
        return translate_large_text(text, src_lang, target_lang)
        
    # 기존 번역 로직 (짧은 텍스트의 경우)
    try:
        model, tokenizer = self.load_model()
        
        src_code = NLLB_LANGUAGE_CODES[src_lang]
        target_code = NLLB_LANGUAGE_CODES[target_lang]

        inputs = tokenizer(text, return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_code])
        
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        logging.error(f"An error occurred during translation: {e}")
        return None
```

## Phase 2
### 7. Optional - 만약 모델 로딩에 문제가 있다면 오류 해결 (Fix Model Loading Issues)
현재 발생 중인 모델 로딩 오류를 해결하기 위한 추가 개선사항:

1. 메모리 관리 개선:
   - 모델 로딩 시 메모리 사용량 모니터링
   - 필요시 더 작은 모델 버전 사용 고려 (예: distilled-350M)

2. 의존성 버전 확인:
   - torch와 transformers 버전 충돌 가능성 확인
   - 호환되는 버전으로 업데이트

3. 오프라인 모델 저장 및 로딩:
   - 모델을 로컬에 저장하여 안정적으로 로딩

4. 대체 모델 옵션:
   - NLLB-200-distilled-600M에서 모델 로딩 문제가 지속될 경우 대체 모델 검토

## 구현 계획 (Implementation Plan)
1. 텍스트 분할 및 병합 유틸리티 함수 개발
2. 번역기 성능 테스트 및 최적화 (토큰 제한, 청크 크기 등)
3. 오류 처리 및 로깅 강화
4. 메모리 사용량 최적화

## 예상 결과 (Expected Results)
- 안정적인 번역 처리: 모델 로딩 오류 해결
- 향상된 번역 품질: 최적 크기의 입력을 통한 번역 품질 향상
- 대용량 텍스트 처리 지원: 길이 제한 없이 모든 트랜스크립트 처리 가능
- 사용자 경험 개선: 단일 파일로 원활한 번역 결과 제공

## 후속 개선 가능성 (Future Improvements)
- 문맥 일관성 보장을 위한 청크 간 문맥 전달 메커니즘
- 병렬 처리를 통한 대규모 번역 작업 속도 향상
- 도메인 특화 사후 처리 단계 추가
