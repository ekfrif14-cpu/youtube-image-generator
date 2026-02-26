"""
유틸리티 함수 모듈
Streamlit에 의존하지 않는 순수 함수들
"""

import csv
import io
import json
import re
import zipfile

from PIL import Image


def safe_filename(text: str, max_len: int = 40) -> str:
    """한글·영문·숫자·공백만 남기고 파일명에 안전한 문자열로 변환"""
    cleaned = re.sub(r"[^\w가-힣\s-]", "", text)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:max_len] if cleaned else "untitled"


def mood_badge_html(mood: str) -> str:
    css_class = {
        "positive": "mood-positive",
        "negative": "mood-negative",
        "neutral": "mood-neutral",
        "dramatic": "mood-dramatic",
    }.get(mood, "mood-neutral")
    return f'<span class="mood-tag {css_class}">{mood}</span>'


def build_zip(images_dict: dict, prompts: list) -> bytes:
    """images_dict: { 'ratio_label': [(index, section_title, pil_image), ...] }"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for ratio_label, items in images_dict.items():
            folder = ratio_label.replace(":", "x")
            for idx, title, img in items:
                fname = f"{idx:02d}_{safe_filename(title)}.png"
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"{folder}/{fname}", img_buf.getvalue())
        zf.writestr("prompts.json", json.dumps(prompts, ensure_ascii=False, indent=2))
    buf.seek(0)
    return buf.getvalue()


def build_csv(prompts: list) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["번호", "섹션 제목", "분위기", "이미지 설명", "원고 구간"])
    for p in prompts:
        writer.writerow([
            p.get("index", ""),
            p.get("section_title", ""),
            p.get("mood", ""),
            p.get("description", ""),
            p.get("script_segment", ""),
        ])
    # utf-8-sig 인코딩이 자동으로 BOM(EF BB BF) 추가
    return buf.getvalue().encode("utf-8-sig")


def parse_json_from_text(text: str) -> list:
    """Gemini 응답 텍스트에서 JSON 배열을 추출"""
    # ```json ... ``` 블록 추출 시도
    match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text)
    if match:
        return json.loads(match.group(1))
    # 순수 JSON 배열 추출 시도
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        return json.loads(match.group(0))
    raise ValueError("JSON 배열을 찾을 수 없습니다.")
