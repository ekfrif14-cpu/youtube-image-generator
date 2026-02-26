"""
🎬 유튜브 이미지 일괄 생성기
유튜브 주식 투자 채널용 영상 이미지를 Gemini API로 일괄 생성하는 Streamlit 웹앱
"""

import io
import json
import time
from datetime import datetime

import streamlit as st
from PIL import Image

from utils import (
    safe_filename,
    mood_badge_html,
    build_zip,
    build_csv,
    parse_json_from_text,
)

# ──────────────────────────────────────────────
# 상수 & 스타일 프리셋
# ──────────────────────────────────────────────

STYLE_PRESETS = {
    "📈 주식 투자 (기본)": (
        "Cinematic cartoon illustration style with bold outlines and vivid colors. "
        "Expressive webtoon/comic art with dramatic lighting, rich environmental storytelling, "
        "and exaggerated character emotions. Financial/stock market theme with real-world settings "
        "like Wall Street, trading floors, city streets, offices. "
        "Bold visual props: giant LED tickers, oversized chart arrows, glowing screens. "
        "Each scene should feel like a frame from an animated movie."
    ),
    "💰 수익/성장 (긍정적)": (
        "Bright, celebratory cartoon illustration with bold outlines. "
        "Warm golden sunlight, confetti, fireworks, green arrows shooting upward. "
        "Character showing extreme joy — jumping, fist pumping, dancing. "
        "Rich real-world backgrounds: rooftop party, sunny park, luxury penthouse. "
        "Vivid green and gold color palette with cinematic composition."
    ),
    "⚠️ 리스크/하락 (경고)": (
        "Dark, dramatic cartoon illustration with bold outlines and cinematic tension. "
        "Stormy skies, rain, lightning, cracked ground, crumbling buildings. "
        "Giant red arrows crashing down, broken LED tickers showing losses. "
        "Character showing panic — on knees, hands on head, running away. "
        "Deep red and dark grey palette with dramatic spotlight lighting."
    ),
    "📊 분석/데이터 (전문적)": (
        "Detailed cartoon illustration with bold outlines in a tech/analytical setting. "
        "Character surrounded by holographic displays, multiple floating screens, data streams. "
        "Futuristic control room, high-tech office, or sci-fi command center aesthetic. "
        "Cool teal and blue tones with glowing neon data elements. "
        "Character actively interacting with data — swiping screens, pointing at charts."
    ),
    "🏦 경제 뉴스 (정보)": (
        "News broadcast cartoon illustration with bold outlines. "
        "Breaking news studio, giant world map, currency symbols floating in air. "
        "Newspaper headlines flying around, TV screens showing market data. "
        "Character as news anchor or reporter in real-world settings: parliament, central bank, stock exchange floor. "
        "Professional blue and white palette with dramatic broadcast lighting."
    ),
    "🎯 노후 준비/연금 (따뜻한)": (
        "Warm, heartfelt cartoon illustration with bold outlines and soft lighting. "
        "Cozy real-world settings: peaceful garden, comfortable living room, sunset beach. "
        "Symbols of security: growing money tree, protective umbrella over family, golden nest egg. "
        "Character looking hopeful and content — relaxing, smiling warmly, looking at a bright future. "
        "Warm earth tones with golden hour lighting and gentle atmosphere."
    ),
    "✏️ 직접 입력": "",
}

TEXT_MODEL = "gemini-2.5-flash"
IMAGE_MODEL_STANDARD = "gemini-2.5-flash-image"
IMAGE_MODEL_PRO = "gemini-3-pro-image-preview"

ASPECT_RATIO_SIZES = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
}

CHARACTER_REFERENCE_INSTRUCTION = """IMPORTANT CHARACTER REFERENCE RULES:
1. Use ONLY the character's face, hair style, body shape, clothing style, and overall visual identity from the reference image.
2. COMPLETELY IGNORE the character's pose, hand position, and any objects they are holding in the reference image.
3. Give the character a DRAMATIC, EXPRESSIVE new pose that fits the scene — exaggerated emotions and body language like a cartoon/webtoon character.
4. Keep the same bold-outline cartoon/illustration art style as the reference.
5. The character must be INSIDE a rich, detailed real-world environment (not floating on a plain background)."""

CUSTOM_CSS = """
<style>
.prompt-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.prompt-card h4 { color: #e94560; margin-bottom: 0.5rem; }
.prompt-card .mood-tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
}
.mood-positive { background: #064e3b; color: #6ee7b7; }
.mood-negative { background: #7f1d1d; color: #fca5a5; }
.mood-neutral  { background: #1e3a5f; color: #93c5fd; }
.mood-dramatic { background: #4a1d96; color: #c4b5fd; }
.image-caption {
    text-align: center;
    font-size: 0.85rem;
    color: #aaa;
    margin-top: 0.3rem;
}
</style>
"""


# ──────────────────────────────────────────────
# Gemini API 함수
# ──────────────────────────────────────────────


def get_genai_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def generate_prompts(
    client,
    script: str,
    num_images: int,
    style_prompt: str,
    has_character: bool,
) -> list:
    """원고를 분석하여 장면별 이미지 프롬프트 JSON을 생성"""
    from google.genai import types

    character_instruction = ""
    if has_character:
        character_instruction = """
CHARACTER RULES:
- EVERY image must feature the character from the reference image.
- Include in every prompt: "the character from the reference image (same face, hair, outfit style only)"
- Character must show EXAGGERATED, DRAMATIC emotions and poses like a webtoon/cartoon character:
  * Bull market / good news: jumping with joy, fist pump in the air, dancing, riding a rocket, surfing on a green arrow
  * Bear market / bad news: on knees with hands on head in despair, running away in panic, hiding under desk, getting rained on
  * Analysis / explanation: dramatically pointing at giant floating screens, surrounded by holographic data, detective with magnifying glass
  * Warning / risk: sweating nervously, biting nails, standing at edge of cliff, holding cracking ice
  * Intro / confident: heroic pose with cape blowing, standing on mountain top, arms crossed with city skyline behind
- The character must be INSIDE a real environment (street, office, rooftop, etc.), NOT standing on a plain/flat background.
"""

    system_prompt = f"""You are a professional image prompt designer for a Korean YouTube stock investment channel called "행복한 노후 준비" (Happy Retirement Preparation).

Your task: Analyze the given Korean script and create exactly {num_images} image generation prompts.

CRITICAL RULES:
1. Divide the script from beginning to end into exactly {num_images} segments. NO overlap, NO gaps — cover the ENTIRE script.
2. Split at natural breakpoints (paragraphs, topic changes, sentence endings).
3. Each prompt must be in ENGLISH for optimal Gemini image generation.
4. General style tone: {style_prompt}
5. Tag each scene's mood: positive / negative / neutral / dramatic
6. Include the EXACT Korean script text for each segment in "script_segment".
{character_instruction}
7. Keep section_title in Korean, concise (under 15 chars).

ART STYLE — MANDATORY FOR ALL IMAGES:
- Cinematic cartoon/webtoon illustration with BOLD BLACK OUTLINES and vivid saturated colors.
- Like a high-quality animated movie frame or Korean webtoon panel.
- Rich, detailed REAL-WORLD environments (NOT plain/flat/abstract backgrounds).
- Dramatic cinematic lighting with depth, shadows, and atmosphere (rain, fog, sunbeams, neon glow, etc.).
- Bold oversized visual props: giant LED tickers, huge arrows, oversized coins, dramatic weather effects.
- Every prompt MUST start with: "Cinematic cartoon illustration with bold outlines, "
- CRITICAL: NEVER include Korean, Chinese, or Japanese text in the image prompts. Use ONLY English text/numbers or visual symbols (arrows, icons, charts) instead of written text.

VISUAL VARIETY — EXTREMELY IMPORTANT:
- Each image MUST have a COMPLETELY DIFFERENT environment, composition, and mood.
- NEVER use plain dark backgrounds. Always use REAL-WORLD LOCATIONS with depth and detail.
- Locations: Wall Street with skyscrapers, rainy city street, sunny rooftop, cozy living room, high-tech trading floor, factory/semiconductor lab, peaceful park, stormy ocean, mountain peak, neon-lit city at night, etc.
- Lighting: stormy with lightning, warm golden sunlight, dramatic spotlight, neon glow, foggy morning, sunset, etc.
- Camera angles: wide establishing shot, dynamic low angle, dramatic close-up, bird's eye view, etc.
- The environment must MATCH the script content. Semiconductor topic → tech lab. Market crash → stormy destruction. Growth → sunny bright setting.

OUTPUT FORMAT — return ONLY a JSON array, no other text:
[
  {{
    "index": 1,
    "section_title": "섹션 제목",
    "prompt": "Detailed English image prompt with specific unique background, setting, lighting, and composition",
    "description": "이미지 설명 (한국어)",
    "mood": "positive",
    "script_segment": "해당 원고 텍스트 전체"
  }}
]
"""

    response = client.models.generate_content(
        model=TEXT_MODEL,
        contents=[f"다음 원고를 분석해서 이미지 프롬프트를 생성해주세요:\n\n{script}"],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        ),
    )
    return parse_json_from_text(response.text)


def _extract_image_from_response(response) -> Image.Image:
    """API 응답에서 이미지를 추출. 실패 시 RuntimeError 발생"""
    if not response.candidates:
        block_reason = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"응답 없음 (candidates 비어있음). prompt_feedback={block_reason}")

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    if not hasattr(candidate, "content") or candidate.content is None:
        raise RuntimeError(f"content가 없습니다. finish_reason={finish_reason}")

    for part in candidate.content.parts:
        if part.inline_data and part.inline_data.mime_type.startswith("image/"):
            return Image.open(io.BytesIO(part.inline_data.data))

    text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
    raise RuntimeError(
        f"이미지가 응답에 포함되지 않음. finish_reason={finish_reason}, "
        f"text={text_parts[0][:200] if text_parts else '(없음)'}"
    )


def generate_single_image(
    client,
    prompt_text: str,
    style_prompt: str,
    aspect_ratio: str,
    character_image: Image.Image | None,
    use_pro: bool,
    max_retries: int = 3,
    status_callback=None,
) -> Image.Image:
    """단일 이미지 생성. 429 에러 시 자동 재시도. 성공 시 PIL Image 반환"""
    from google.genai import types

    model = IMAGE_MODEL_PRO if use_pro else IMAGE_MODEL_STANDARD

    # 최종 프롬프트 조립
    full_prompt_parts = []
    if character_image:
        full_prompt_parts.append(CHARACTER_REFERENCE_INSTRUCTION)
    full_prompt_parts.append(
        "ART STYLE: Cinematic cartoon/webtoon illustration with bold black outlines, "
        "vivid saturated colors, rich detailed real-world environment backgrounds, "
        "dramatic cinematic lighting, and exaggerated expressive character emotions. "
        "Like a high-quality animated movie frame. "
        "CRITICAL: Do NOT include any Korean, Chinese, or Japanese text/characters in the image. "
        "Use only English text or numbers if text is needed, or use no text at all. "
        "Use visual symbols (arrows, icons, charts) instead of written text whenever possible."
    )
    full_prompt_parts.append(f"Style direction: {style_prompt}")
    full_prompt_parts.append(f"Scene: {prompt_text}")
    full_prompt = "\n\n".join(full_prompt_parts)

    # contents 구성
    contents = []
    if character_image:
        img_buf = io.BytesIO()
        character_image.save(img_buf, format="PNG")
        contents.append(
            types.Part.from_bytes(data=img_buf.getvalue(), mime_type="image/png")
        )
    contents.append(types.Part.from_text(text=full_prompt))

    # aspect_ratio를 API 파라미터로 직접 설정
    config = types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        temperature=1,
        top_p=0.95,
        top_k=40,
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
    )

    # 재시도 로직 (429 RESOURCE_EXHAUSTED 대응)
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config,
            )
            img = _extract_image_from_response(response)
            # 정확한 해상도로 리사이즈 (16:9 → 1920x1080)
            target = ASPECT_RATIO_SIZES.get(aspect_ratio)
            if target and img.size != target:
                img = img.resize(target, Image.LANCZOS)
            return img
        except Exception as e:
            last_error = e
            err_str = str(e)
            # 429 Rate Limit → 대기 후 재시도
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = 10 * (attempt + 1)  # 10초, 20초, 30초
                if status_callback:
                    status_callback(
                        f"⏳ API 할당량 초과 — {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})..."
                    )
                time.sleep(wait)
                continue
            # 그 외 에러는 즉시 raise
            raise

    raise RuntimeError(f"최대 재시도({max_retries}회) 초과. 마지막 에러: {last_error}")


# ──────────────────────────────────────────────
# Streamlit 앱
# ──────────────────────────────────────────────


def init_session_state():
    defaults = {
        "prompts": None,
        "generated_images": {},  # { ratio_label: [(idx, title, PIL.Image), ...] }
        "generation_done": False,
        "failed_indices": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 설정")

        api_key = st.text_input(
            "Gemini API 키",
            type="password",
            help="Google AI Studio에서 발급받은 API 키를 입력하세요.",
        )

        st.divider()

        num_images = st.slider(
            "이미지 생성 수", min_value=4, max_value=40, value=30, step=1,
        )

        st.markdown("**화면 비율**: 16:9 (1920×1080)")

        style_key = st.selectbox("스타일 프리셋", list(STYLE_PRESETS.keys()))
        if style_key == "✏️ 직접 입력":
            style_prompt = st.text_area(
                "커스텀 스타일 프롬프트",
                placeholder="이미지 스타일을 영어로 설명하세요...",
            )
        else:
            style_prompt = STYLE_PRESETS[style_key]

        st.divider()

        use_pro = st.toggle(
            "Pro 모델 사용",
            value=False,
            help="캐릭터 일관성이 더 좋지만 속도가 느리고 비용이 높습니다.",
        )

        default_delay = 5 if use_pro else 3
        api_delay = st.slider(
            "API 호출 딜레이 (초)",
            min_value=3, max_value=15, value=default_delay, step=1,
            help="Rate Limit 보호를 위한 호출 간 대기 시간",
        )

        st.divider()
        if api_key and st.button("🔌 API 연결 테스트", use_container_width=True):
            with st.spinner("연결 확인 중..."):
                try:
                    client = get_genai_client(api_key)
                    # 간단한 텍스트 호출로 키 유효성 확인
                    from google.genai import types
                    resp = client.models.generate_content(
                        model=TEXT_MODEL,
                        contents="Say OK",
                        config=types.GenerateContentConfig(
                            max_output_tokens=10,
                        ),
                    )
                    st.success(f"✅ API 연결 성공! 텍스트 모델: `{TEXT_MODEL}`")
                except Exception as e:
                    st.error(f"❌ API 연결 실패: {e}")

        return {
            "api_key": api_key,
            "num_images": num_images,
            "selected_ratios": ["16:9"],
            "style_prompt": style_prompt,
            "use_pro": use_pro,
            "api_delay": api_delay,
        }


def render_step1():
    """Step 1: 캐릭터 참조 이미지 업로드"""
    st.subheader("Step 1. 캐릭터 참조 이미지")
    st.caption(
        "모든 이미지에 등장할 AI 캐릭터의 참조 이미지를 업로드하세요. "
        "얼굴, 헤어스타일, 체형, 옷 스타일만 참조되며 포즈나 소지품은 무시됩니다."
    )

    uploaded = st.file_uploader(
        "캐릭터 이미지 (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="character_upload",
    )

    character_image = None
    if uploaded:
        character_image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(character_image, caption="캐릭터 참조 이미지", width=200)
        with col2:
            st.info(
                "✅ 참조 이미지가 설정되었습니다.\n\n"
                "**참조 항목**: 얼굴, 헤어, 체형, 옷 스타일\n\n"
                "**무시 항목**: 포즈, 손 위치, 소지품"
            )

    return character_image


def render_step2(config: dict):
    """Step 2~3: 원고 입력 및 프롬프트 생성"""
    st.subheader("Step 2. 원고 입력")

    input_method = st.radio(
        "입력 방식",
        ["텍스트 직접 입력", "TXT 파일 업로드", "기존 프롬프트 JSON 업로드"],
        horizontal=True,
    )

    script = ""
    loaded_prompts = None

    if input_method == "텍스트 직접 입력":
        script = st.text_area(
            "원고 붙여넣기",
            height=300,
            placeholder="유튜브 영상 원고를 여기에 붙여넣으세요...",
        )
    elif input_method == "TXT 파일 업로드":
        txt_file = st.file_uploader("TXT 파일", type=["txt"], key="txt_upload")
        if txt_file:
            script = txt_file.read().decode("utf-8")
            st.text_area("업로드된 원고", value=script, height=200, disabled=True)
    else:
        json_file = st.file_uploader("프롬프트 JSON 파일", type=["json"], key="json_upload")
        if json_file:
            loaded_prompts = json.loads(json_file.read().decode("utf-8"))
            st.success(f"✅ {len(loaded_prompts)}개의 프롬프트를 불러왔습니다.")

    return script, loaded_prompts


def render_step3(config: dict, script: str, loaded_prompts: list | None, character_image):
    """Step 3: AI 프롬프트 자동 생성"""
    st.subheader("Step 3. AI 프롬프트 생성")

    # 이미 생성된 프롬프트가 있으면 표시
    if st.session_state.prompts:
        _display_prompts(st.session_state.prompts)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 프롬프트 다시 생성", use_container_width=True):
                st.session_state.prompts = None
                st.session_state.generated_images = {}
                st.session_state.generation_done = False
                st.rerun()
        with col2:
            st.download_button(
                "💾 프롬프트 JSON 다운로드",
                data=json.dumps(st.session_state.prompts, ensure_ascii=False, indent=2),
                file_name="prompts.json",
                mime="application/json",
                use_container_width=True,
            )
        return

    # JSON 직접 업로드한 경우
    if loaded_prompts:
        st.session_state.prompts = loaded_prompts
        _display_prompts(loaded_prompts)
        return

    # 원고가 없으면 안내
    if not script.strip():
        st.info("👆 Step 2에서 원고를 입력하면 프롬프트를 생성할 수 있습니다.")
        return

    if not config["api_key"]:
        st.warning("⚠️ 사이드바에서 Gemini API 키를 입력하세요.")
        return

    st.markdown(f"원고 길이: **{len(script)}자** / 생성할 이미지: **{config['num_images']}장**")

    if st.button("🤖 프롬프트 자동 생성", type="primary", use_container_width=True):
        with st.spinner("Gemini가 원고를 분석하고 프롬프트를 생성 중입니다..."):
            try:
                client = get_genai_client(config["api_key"])
                prompts = generate_prompts(
                    client=client,
                    script=script,
                    num_images=config["num_images"],
                    style_prompt=config["style_prompt"],
                    has_character=character_image is not None,
                )
                st.session_state.prompts = prompts
                st.rerun()
            except Exception as e:
                st.error(f"프롬프트 생성 실패: {e}")


def _display_prompts(prompts: list):
    """프롬프트 목록을 카드 형태로 표시"""
    st.success(f"✅ {len(prompts)}개의 프롬프트가 준비되었습니다.")

    for p in prompts:
        mood = p.get("mood", "neutral")
        badge = mood_badge_html(mood)
        with st.expander(f"#{p['index']} — {p['section_title']} {mood}", expanded=False):
            st.markdown(f"**분위기**: {badge}", unsafe_allow_html=True)
            st.markdown(f"**이미지 설명**: {p.get('description', '')}")
            st.code(p["prompt"], language=None)
            if p.get("script_segment"):
                st.markdown("**해당 원고 구간:**")
                st.markdown(
                    f"<div style='background:#1e293b; padding:10px; border-radius:8px; "
                    f"font-size:0.9rem; line-height:1.6;'>{p['script_segment']}</div>",
                    unsafe_allow_html=True,
                )


def render_step4(config: dict, character_image):
    """Step 4: 이미지 일괄 생성"""
    st.subheader("Step 4. 이미지 일괄 생성")

    prompts = st.session_state.prompts
    if not prompts:
        st.info("👆 Step 3에서 프롬프트를 먼저 생성하세요.")
        return

    if not config["selected_ratios"]:
        st.warning("⚠️ 사이드바에서 화면 비율을 최소 1개 선택하세요.")
        return

    if not config["api_key"]:
        st.warning("⚠️ 사이드바에서 Gemini API 키를 입력하세요.")
        return

    # 이미 생성 완료된 경우
    if st.session_state.generation_done and st.session_state.generated_images:
        _show_generation_summary()
        if st.button("🔄 이미지 다시 생성", use_container_width=True):
            st.session_state.generated_images = {}
            st.session_state.generation_done = False
            st.session_state.failed_indices = []
            st.rerun()
        return

    total_images = len(prompts) * len(config["selected_ratios"])
    estimated_time = total_images * config["api_delay"]
    st.markdown(
        f"생성 예정: **{len(prompts)}장** × **{len(config['selected_ratios'])}비율** = "
        f"**{total_images}장** (예상 소요: ~{estimated_time // 60}분 {estimated_time % 60}초)"
    )

    model_label = "Pro" if config["use_pro"] else "Standard"
    st.caption(f"모델: `{IMAGE_MODEL_PRO if config['use_pro'] else IMAGE_MODEL_STANDARD}` ({model_label})")

    if st.button("🚀 이미지 생성 시작", type="primary", use_container_width=True):
        _run_image_generation(config, prompts, character_image)


def _run_image_generation(config: dict, prompts: list, character_image):
    """이미지 일괄 생성 실행"""
    client = get_genai_client(config["api_key"])
    selected_ratios = config["selected_ratios"]
    total = len(prompts) * len(selected_ratios)

    progress_bar = st.progress(0, text="이미지 생성 준비 중...")
    status_area = st.empty()
    preview_area = st.empty()

    results = {r: [] for r in selected_ratios}
    failed = []
    success_count = 0
    current = 0

    for p in prompts:
        for ratio in selected_ratios:
            current += 1
            progress_bar.progress(
                current / total,
                text=f"생성 중... ({current}/{total}) — #{p['index']} {p['section_title']} [{ratio}]",
            )
            status_area.info(
                f"🎨 #{p['index']} — {p['section_title']} ({ratio}) 생성 중..."
            )

            try:
                img = generate_single_image(
                    client=client,
                    prompt_text=p["prompt"],
                    style_prompt=config["style_prompt"],
                    aspect_ratio=ratio,
                    character_image=character_image,
                    use_pro=config["use_pro"],
                    status_callback=lambda msg: status_area.warning(msg),
                )
                results[ratio].append((p["index"], p["section_title"], img))
                success_count += 1
                preview_area.image(
                    img,
                    caption=f"#{p['index']} {p['section_title']} ({ratio})",
                    width=400,
                )
            except Exception as e:
                err_msg = str(e)
                failed.append((p["index"], ratio, err_msg))
                status_area.error(f"❌ #{p['index']} ({ratio}) 실패: {err_msg[:150]}")

            # Rate limit 보호 — 마지막 이미지가 아니면 딜레이
            if current < total:
                time.sleep(config["api_delay"])

    progress_bar.progress(1.0, text="✅ 이미지 생성 완료!")
    status_area.empty()
    preview_area.empty()

    st.session_state.generated_images = results
    st.session_state.generation_done = True
    st.session_state.failed_indices = failed

    _show_generation_summary()


def _show_generation_summary():
    results = st.session_state.generated_images
    failed = st.session_state.failed_indices

    total_success = sum(len(v) for v in results.values())
    col1, col2 = st.columns(2)
    col1.metric("✅ 성공", f"{total_success}장")
    col2.metric("❌ 실패", f"{len(failed)}건")

    if failed:
        with st.expander("실패 목록"):
            for idx, ratio, err in failed:
                st.markdown(f"- **#{idx}** ({ratio}): {err}")


def render_step5(config: dict):
    """Step 5: 결과 표시 및 다운로드"""
    st.subheader("Step 5. 결과 및 다운로드")

    if not st.session_state.generation_done or not st.session_state.generated_images:
        st.info("👆 Step 4에서 이미지를 먼저 생성하세요.")
        return

    results = st.session_state.generated_images
    prompts = st.session_state.prompts

    # ── 이미지 갤러리 ──
    ratio_tabs = list(results.keys())
    if len(ratio_tabs) > 1:
        tabs = st.tabs(ratio_tabs)
    else:
        tabs = [st.container()]

    for tab, ratio in zip(tabs, ratio_tabs):
        with tab:
            items = results[ratio]
            if not items:
                st.warning(f"{ratio} 비율 이미지가 없습니다.")
                continue

            ncols = 4 if ratio == "16:9" else 5
            cols = st.columns(ncols)
            for i, (idx, title, img) in enumerate(items):
                with cols[i % ncols]:
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f"<p class='image-caption'>#{idx} {title}</p>",
                        unsafe_allow_html=True,
                    )

    # ── 원고-이미지 매핑 시트 ──
    st.markdown("---")
    st.markdown("### 📋 원고-이미지 매핑 시트")
    st.caption("Vrew 편집 시 참고용 — 각 이미지가 원고의 어느 부분에 해당하는지 확인하세요.")

    for p in prompts:
        with st.expander(f"#{p['index']} — {p['section_title']} ({p.get('mood', '')})"):
            # 해당 이미지 표시
            for ratio, items in results.items():
                for idx, title, img in items:
                    if idx == p["index"]:
                        st.image(img, caption=f"{ratio}", width=300)
            st.markdown(f"**설명**: {p.get('description', '')}")
            st.markdown(f"**분위기**: {p.get('mood', '')}")
            st.markdown("**원고 구간:**")
            st.markdown(f"> {p.get('script_segment', '(없음)')}")

    # ── 다운로드 버튼 ──
    st.markdown("---")
    st.markdown("### 📥 다운로드")

    col1, col2, col3 = st.columns(3)

    with col1:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_data = build_zip(results, prompts)
        st.download_button(
            "📦 ZIP 다운로드",
            data=zip_data,
            file_name=f"youtube_images_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with col2:
        st.download_button(
            "📄 프롬프트 JSON",
            data=json.dumps(prompts, ensure_ascii=False, indent=2),
            file_name=f"prompts_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        csv_data = build_csv(prompts)
        st.download_button(
            "📊 매핑 시트 CSV",
            data=csv_data,
            file_name=f"mapping_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────────


def main():
    st.set_page_config(
        page_title="유튜브 이미지 일괄 생성기",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.title("🎬 유튜브 이미지 일괄 생성기")
    st.caption("유튜브 주식 투자 채널 영상용 이미지를 Gemini API로 일괄 생성합니다.")

    # 사이드바
    config = render_sidebar()

    # Step 1
    character_image = render_step1()

    st.markdown("---")

    # Step 2
    script, loaded_prompts = render_step2(config)

    st.markdown("---")

    # Step 3
    render_step3(config, script, loaded_prompts, character_image)

    st.markdown("---")

    # Step 4
    render_step4(config, character_image)

    st.markdown("---")

    # Step 5
    render_step5(config)


if __name__ == "__main__":
    main()
