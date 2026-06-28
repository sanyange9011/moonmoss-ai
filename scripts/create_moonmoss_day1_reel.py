import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ============================================
# 설정
# ============================================

OUTPUT_VIDEO = "moonmoss_day1_reel.mp4"
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 60
VIDEO_DURATION = 20  # 초

# 이미지 경로 설정 (당신의 이미지 경로로 수정)
IMAGE_PATHS = {
    "image1": "/path/to/image1_blank_notebook.jpg",  # 이미지1: 빈 노트
    "image2": "/path/to/image2_pattern_drawing.jpg",  # 이미지2: 패턴 드로잉
    "image3": "/path/to/image3_wardrobe.jpg",  # 이미지3: 옷장 슬랙스
    "image4_5": "/path/to/image4_5_notebook_with_hand.jpg",  # 이미지4/5: 손 + 노트
}

# BGM 경로 (제공된 미니멀 암비언트 음악 파일)
BGM_PATH = "/path/to/minimal_ambient_music.mp3"  # 약 20초 음악

# ============================================
# 폰트 설정 (자막이 한글이므로 한글 글리프를 지원하는 폰트가 필요함)
# ============================================

_KOREAN_FONT_CANDIDATES = [
    os.environ.get("MOONMOSS_FONT_PATH"),
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansKR-Bold.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]
_LATIN_FALLBACK_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _find_font_path():
    for path in _KOREAN_FONT_CANDIDATES:
        if path and os.path.exists(path):
            return path
    if os.path.exists(_LATIN_FALLBACK_FONT):
        print(
            "⚠️  한글을 지원하는 폰트를 찾지 못해 DejaVu Sans로 대체합니다. "
            "한글 자막이 깨져 보일 수 있습니다. MOONMOSS_FONT_PATH 환경변수로 "
            "한글 폰트(예: NanumGothic, Noto Sans CJK KR) 경로를 지정하세요."
        )
        return _LATIN_FALLBACK_FONT
    return None


_FONT_PATH = _find_font_path()
_FONT_CACHE = {}


def get_font(font_size):
    """폰트를 크기별로 캐싱해서 매 프레임 디스크에서 다시 읽지 않도록 함"""
    if font_size not in _FONT_CACHE:
        if _FONT_PATH:
            _FONT_CACHE[font_size] = ImageFont.truetype(_FONT_PATH, font_size)
        else:
            try:
                _FONT_CACHE[font_size] = ImageFont.load_default(size=font_size)
            except TypeError:
                _FONT_CACHE[font_size] = ImageFont.load_default()
    return _FONT_CACHE[font_size]


# ============================================
# 함수 정의
# ============================================

def load_image(path, width, height):
    """이미지 로드 및 리사이징"""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")

    # 이미지를 세로형(1080x1920)으로 맞춤
    h, w = img.shape[:2]
    aspect_ratio = w / h
    target_aspect = width / height

    if aspect_ratio > target_aspect:
        new_w = int(height * aspect_ratio)
        new_h = height
    else:
        new_w = width
        new_h = int(width / aspect_ratio)

    img = cv2.resize(img, (new_w, new_h))

    # 중앙 크롭
    start_x = (new_w - width) // 2
    start_y = (new_h - height) // 2
    img = img[start_y:start_y+height, start_x:start_x+width]

    return img

def create_black_frame(width, height):
    """검은 프레임 생성"""
    return np.zeros((height, width, 3), dtype=np.uint8)

def apply_zoom(frame, zoom_factor, zoom_type="in"):
    """줌인/줌아웃 효과 적용"""
    h, w = frame.shape[:2]

    if zoom_type == "in":
        # 줌인: 중앙에서 확대
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)

        frame = cv2.resize(frame, (new_w, new_h))

        # 센터 크롭
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        frame = frame[start_y:start_y+h, start_x:start_x+w]

    elif zoom_type == "out":
        # 줌아웃: 축소 후 원본 크기의 검은 배경 중앙에 배치
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        resized = cv2.resize(frame, (new_w, new_h))

        new_frame = np.zeros((h, w, 3), dtype=frame.dtype)
        start_x = (w - new_w) // 2
        start_y = (h - new_h) // 2
        new_frame[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        frame = new_frame

    return frame

def apply_pan(frame, pan_direction="right", pan_amount=0.1):
    """팬 효과 적용 (좌우 이동)"""
    h, w = frame.shape[:2]
    shift = int(w * pan_amount)

    if pan_direction == "right":
        # 좌에서 우로 팬
        new_frame = np.zeros_like(frame)
        new_frame[:, :w-shift] = frame[:, shift:]
    else:
        # 우에서 좌로 팬
        new_frame = np.zeros_like(frame)
        new_frame[:, shift:] = frame[:, :w-shift]

    return new_frame

def add_subtitle(frame, text, font_size=45, opacity=1.0):
    """자막 추가 (알파 합성을 사용해 opacity가 실제로 반영됨)"""
    if opacity <= 0:
        return frame

    base = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = get_font(font_size)

    lines = text.split('\n')
    line_height = font_size + 10
    alpha = max(0, min(255, int(255 * opacity)))

    # 텍스트 위치 (상단 중앙, 여러 줄은 가장 긴 줄 기준으로 블록 정렬)
    text_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    x = (base.width - text_width) // 2
    y = 60

    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255, alpha))
        y += line_height

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composed), cv2.COLOR_RGB2BGR)

def fade_in(frame, progress):
    """페이드인 효과"""
    return (frame.astype(float) * progress).astype(np.uint8)

def fade_out(frame, progress):
    """페이드아웃 효과"""
    return (frame.astype(float) * (1 - progress)).astype(np.uint8)

# ============================================
# 메인 영상 생성
# ============================================

def create_video():
    """메인 비디오 생성 함수"""

    # 비디오 라이터 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))

    if not out.isOpened():
        print("❌ 비디오 라이터를 열 수 없습니다.")
        return

    total_frames = int(VIDEO_DURATION * FPS)

    # 이미지 로드
    print("📸 이미지 로드 중...")
    images = {}
    for key, path in IMAGE_PATHS.items():
        if os.path.exists(path):
            images[key] = load_image(path, VIDEO_WIDTH, VIDEO_HEIGHT)
            print(f"✅ {key} 로드됨")
        else:
            print(f"⚠️  {key} 파일을 찾을 수 없습니다: {path}")

    print(f"\n🎬 {total_frames}개 프레임 생성 중... (총 {VIDEO_DURATION}초)")

    # 프레임별 생성
    for frame_num in range(total_frames):
        current_time = frame_num / FPS
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)

        # ============================================
        # Scene 1: 오프닝 타이틀 (0.0~1.2초)
        # ============================================
        if 0.0 <= current_time < 1.2:
            frame = create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 자막 페이드인 (0.0~0.3초)
            fade_progress = min(1.0, current_time / 0.3)
            frame = add_subtitle(frame, "DAY-1\n우당탕탕 문모스 제작일지", font_size=50, opacity=fade_progress)

        # ============================================
        # Scene 2: 나무 책상 + 노트 (1.2~4.0초)
        # ============================================
        elif 1.2 <= current_time < 4.0:
            frame = images["image4_5"].copy() if "image4_5" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌인 효과 (2.0~3.0초)
            if 2.0 <= current_time < 3.0:
                zoom_progress = (current_time - 2.0) / 1.0
                zoom_factor = 1.0 + (0.02 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="in")

            # 자막 추가
            if 1.5 <= current_time < 3.5:
                fade = 1.0 if current_time < 3.2 else max(0.0, (3.5 - current_time) / 0.3)
                frame = add_subtitle(frame, "이 바지는", font_size=45, opacity=fade)

        # ============================================
        # Scene 3: 노트 메모 클로즈업 (4.0~6.8초)
        # ============================================
        elif 4.0 <= current_time < 6.8:
            frame = images["image4_5"].copy() if "image4_5" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌인 효과 (4.0~5.0초)
            if 4.0 <= current_time < 5.0:
                zoom_progress = (current_time - 4.0) / 1.0
                zoom_factor = 1.0 + (0.03 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="in")

            # 자막 추가: 첫 줄 등장 -> 두 줄 합쳐서 노출 -> 둘째 줄만 남아 페이드아웃
            text1 = "예쁜 바지를 만들려고"
            text2 = "시작한 게 아닙니다."

            if 4.2 <= current_time < 4.8:
                frame = add_subtitle(frame, text1, font_size=42, opacity=1.0)
            elif 4.8 <= current_time < 5.7:
                frame = add_subtitle(frame, text1 + "\n" + text2, font_size=42, opacity=1.0)
            elif 5.7 <= current_time < 6.3:
                fade = 1.0 if current_time < 6.0 else max(0.0, (6.3 - current_time) / 0.3)
                frame = add_subtitle(frame, text2, font_size=42, opacity=fade)

        # ============================================
        # Scene 4: 옷장 슬랙스 (6.8~9.0초)
        # ============================================
        elif 6.8 <= current_time < 9.0:
            frame = images["image3"].copy() if "image3" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 팬 효과 (6.8~8.2초) - 좌에서 우로
            if 6.8 <= current_time < 8.2:
                pan_progress = (current_time - 6.8) / 1.4
                pan_amount = pan_progress * 0.15  # 최대 15% 팬
                frame = apply_pan(frame, pan_direction="right", pan_amount=pan_amount)

            # 자막 추가: 첫 줄 등장 -> 두 줄 합쳐서 노출 -> 둘째 줄만 남아 페이드아웃
            text1 = "살이 조금만 빠져도."
            text2 = "조금만 쪄도."

            if 6.9 <= current_time < 7.4:
                frame = add_subtitle(frame, text1, font_size=45, opacity=1.0)
            elif 7.4 <= current_time < 8.1:
                frame = add_subtitle(frame, text1 + "\n" + text2, font_size=45, opacity=1.0)
            elif 8.1 <= current_time < 8.7:
                fade = max(0.0, (8.7 - current_time) / 0.6)
                frame = add_subtitle(frame, text2, font_size=45, opacity=fade)

        # ============================================
        # Scene 5: 빈 노트 (9.0~10.5초)
        # ============================================
        elif 9.0 <= current_time < 10.5:
            frame = images["image1"].copy() if "image1" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌아웃 효과 (9.0~9.8초)
            if 9.0 <= current_time < 9.8:
                zoom_progress = (current_time - 9.0) / 0.8
                zoom_factor = 1.0 + (0.02 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="out")

            # 자막 추가
            text = "가장 먼저\n못 입게 되는 게\n슬랙스였습니다."

            if 9.2 <= current_time < 10.4:
                fade = 1.0 if current_time < 10.1 else max(0.0, (10.4 - current_time) / 0.3)
                frame = add_subtitle(frame, text, font_size=44, opacity=fade)

        # ============================================
        # Scene 6: 패턴 드로잉 클로즈업 (10.5~12.5초)
        # ============================================
        elif 10.5 <= current_time < 12.5:
            frame = images["image2"].copy() if "image2" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌인 효과 (10.5~11.5초)
            if 10.5 <= current_time < 11.5:
                zoom_progress = (current_time - 10.5) / 1.0
                zoom_factor = 1.0 + (0.04 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="in")

            # 자막 추가
            if 10.7 <= current_time < 12.2:
                fade = 1.0 if current_time < 11.9 else max(0.0, (12.2 - current_time) / 0.3)
                frame = add_subtitle(frame, "그래서 생각했습니다.", font_size=43, opacity=fade)

        # ============================================
        # Scene 7: 완성된 패턴 페이지 (12.5~13.8초)
        # ============================================
        elif 12.5 <= current_time < 13.8:
            frame = images["image2"].copy() if "image2" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌인 효과 (12.5~13.2초)
            if 12.5 <= current_time < 13.2:
                zoom_progress = (current_time - 12.5) / 0.7
                zoom_factor = 1.0 + (0.03 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="in")

            # 자막 추가
            text = "몸이 바뀌어도\n오래 입을 수 있는 바지는\n없을까?"

            if 12.6 <= current_time < 13.6:
                fade = 1.0 if current_time < 13.3 else max(0.0, (13.6 - current_time) / 0.3)
                frame = add_subtitle(frame, text, font_size=42, opacity=fade)

        # ============================================
        # Scene 8: 프로젝트 타이틀 (13.8~14.8초)
        # ============================================
        elif 13.8 <= current_time < 14.8:
            frame = create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 자막 페이드인 (13.8~14.1초)
            fade_progress = min(1.0, (current_time - 13.8) / 0.3)
            text = "PROJECT\nALL BANDING SLACKS"
            frame = add_subtitle(frame, text, font_size=55, opacity=fade_progress)

        # ============================================
        # Scene 9: 노트 덮기 (14.8~15.3초)
        # ============================================
        elif 14.8 <= current_time < 15.3:
            frame = images["image4_5"].copy() if "image4_5" in images else create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 줌인 효과 (14.8~15.0초)
            if 14.8 <= current_time < 15.0:
                zoom_progress = (current_time - 14.8) / 0.2
                zoom_factor = 1.0 + (0.03 * zoom_progress)
                frame = apply_zoom(frame, zoom_factor, zoom_type="in")

            # 자막 추가
            fade = 1.0 if current_time < 15.0 else max(0.0, (15.3 - current_time) / 0.3)
            frame = add_subtitle(frame, "첫 번째 샘플 제작 시작.", font_size=45, opacity=fade)

        # ============================================
        # Scene 10: 엔딩 (15.3~20.0초)
        # ============================================
        elif 15.3 <= current_time < 20.0:
            frame = create_black_frame(VIDEO_WIDTH, VIDEO_HEIGHT)

            # 자막 추가 (매우 흐릿하게)
            frame = add_subtitle(frame, "다음 편을 기다려주세요.", font_size=35, opacity=0.15)

        # 프레임 쓰기
        out.write(frame)

        # 진행 상황 출력
        if (frame_num + 1) % 60 == 0:
            print(f"  [{frame_num + 1}/{total_frames}] {(frame_num + 1) / total_frames * 100:.1f}% 완료")

    out.release()
    print(f"\n✅ 영상 생성 완료: {OUTPUT_VIDEO}")

# ============================================
# 오디오 추가 (선택사항)
# ============================================

def add_audio_to_video(video_path, audio_path, output_path):
    """오디오를 비디오에 추가"""
    try:
        import subprocess
        command = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-y",
            output_path
        ]
        subprocess.run(command, check=True)
        print(f"✅ 오디오 추가 완료: {output_path}")
    except Exception as e:
        print(f"⚠️  오디오 추가 실패: {e}")

# ============================================
# 실행
# ============================================

if __name__ == "__main__":
    print("🎬 문모스 DAY-1 릴스 생성 시작...\n")

    # 비디오 생성
    create_video()

    # 오디오 추가 (옵션)
    if os.path.exists(BGM_PATH):
        print("\n🎵 오디오 추가 중...")
        add_audio_to_video(OUTPUT_VIDEO, BGM_PATH, "moonmoss_day1_reel_with_audio.mp4")
    else:
        print(f"\n⚠️  BGM 파일을 찾을 수 없습니다: {BGM_PATH}")
        print("   (FFmpeg가 설치되어 있으면 별도로 오디오를 추가할 수 있습니다)")

    print("\n✨ 영상 생성 완료!")
    print(f"   출력: {OUTPUT_VIDEO}")
