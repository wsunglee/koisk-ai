import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import cv2
from deepface import DeepFace
import numpy as np
import time
import queue
import os
import traceback

# --- Streamlit 페이지 설정 (가장 먼저 호출되어야 합니다) ---
st.set_page_config(layout="wide", page_title="나이 맞춤형 스마트 키오스크")

# --- 필요한 디렉토리 생성 (DeepFace 모델 캐시 등) ---
home = os.path.expanduser("~")
deepface_home = os.path.join(home, ".deepface")
models_path = os.path.join(deepface_home, "weights")
if not os.path.exists(models_path):
    os.makedirs(models_path, exist_ok=True)

# --- 나이 추정 결과 처리를 위한 큐 ---
age_result_queue = queue.Queue(maxsize=1)

# --- UI 상태 관리를 위한 Session State 초기화 ---
if 'ui_scale' not in st.session_state:
    st.session_state.ui_scale = 'normal'
if 'last_detected_age' not in st.session_state:
    st.session_state.last_detected_age = None
if 'cart' not in st.session_state:
    st.session_state.cart = []
# Keep track of the active tab if needed, although st.tabs handles it visually
# if 'active_tab' not in st.session_state:
#     st.session_state.active_tab = "메인" # Example: start on main tab

# --- 나이 추정 기준 설정 ---
AGE_THRESHOLD = 20  # !!! 테스트 용도: 20세 초과 시 UI 확대 기준 !!!

# --- DeepFace 모델 로딩 (앱 시작 시 캐시) ---
@st.cache_resource(ttl=3600)
def load_deepface_model():
    """DeepFace 모델 로딩을 시도합니다."""
    try:
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_img, actions=['age'], enforce_detection=False, detector_backend='opencv', silent=True)
        return True
    except Exception as e:
        print(f"--- Error pre-loading DeepFace: {e} ---")
        return False

model_loaded = load_deepface_model()

# --- WebRTC 비디오 처리 클래스 ---
class AgeEstimator(VideoTransformerBase):
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 1.5
        self.last_detected_age_in_thread = None

    def _get_face_box(self, img: np.ndarray):
        try:
            face_objs = DeepFace.extract_faces(img, detector_backend='opencv', enforce_detection=False, align=False)
            if not face_objs: return None
            largest_face_area = 0
            largest_face_box = None
            for face_obj in face_objs:
                if 'confidence' in face_obj and face_obj['confidence'] > 0.7:
                    box = face_obj['facial_area']
                    area = box['w'] * box['h']
                    if area > largest_face_area:
                        largest_face_area = area
                        largest_face_box = (box['x'], box['y'], box['w'], box['h'])
            return largest_face_box
        except Exception as e:
            # print(f"--- Error in _get_face_box: {e} ---")
            return None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()

        if current_time - self.last_detection_time > self.detection_interval:
            self.last_detection_time = current_time
            try:
                 results = DeepFace.analyze(img_path=img, actions=['age'], enforce_detection=False, detector_backend='opencv', silent=True)
                 current_frame_age = None
                 if isinstance(results, list) and len(results) > 0 and 'age' in results[0]:
                     current_frame_age = round(results[0]['age'])

                 self.last_detected_age_in_thread = current_frame_age
                 if age_result_queue.full():
                     try: age_result_queue.get_nowait()
                     except queue.Empty: pass
                 age_result_queue.put(self.last_detected_age_in_thread)

            except Exception as e:
                 # print(f"--- Error during DeepFace analysis in transform: {e} ---")
                 self.last_detected_age_in_thread = None
                 if age_result_queue.full():
                     try: age_result_queue.get_nowait()
                     except queue.Empty: pass
                 age_result_queue.put(None)

        # --- 화면에 결과 오버레이 ---
        display_age_on_video = self.last_detected_age_in_thread if self.last_detected_age_in_thread is not None else 'N/A'
        display_text_on_video = f"Age: {display_age_on_video}"

        face_box = self._get_face_box(img)
        if face_box:
           (x, y, wf, hf) = face_box
           cv2.rectangle(img, (x, y), (x + wf, y + hf), (0, 255, 0), 2)
           text_y = y - 10 if y - 10 > 10 else y + hf + 20
           cv2.putText(img, display_text_on_video, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return img

# --- Streamlit UI 메인 로직 ---

# === 나이 추정 결과 처리 및 UI 상태 업데이트 ===
try:
    new_age_from_queue = age_result_queue.get_nowait()
    if new_age_from_queue != st.session_state.last_detected_age:
         # print(f"--- UI State: Updating age from {st.session_state.last_detected_age} to {new_age_from_queue} ---")
         st.session_state.last_detected_age = new_age_from_queue
         st.rerun()
except queue.Empty:
    pass
except Exception as e:
    print(f"--- Error processing queue in main thread: {e} ---")
    traceback.print_exc()
    pass

# === UI 스케일 결정 로직 ===
current_ui_scale = st.session_state.ui_scale
determined_ui_scale = 'normal'
if st.session_state.last_detected_age is not None and isinstance(st.session_state.last_detected_age, (int, float)) and st.session_state.last_detected_age > AGE_THRESHOLD:
    determined_ui_scale = 'large'

if current_ui_scale != determined_ui_scale:
     # print(f"--- UI State: Updating scale from {st.session_state.ui_scale} to {determined_ui_scale} ---")
     st.session_state.ui_scale = determined_ui_scale
     # st.rerun() # 나이 변경 시 이미 rerun되므로 불필요


# --- 동적 CSS 생성 및 주입 ---
font_size_large = "24px"
font_size_normal = "18px"
button_padding_large = "20px 40px"
button_padding_normal = "10px 20px"
header_font_size_large = "40px"
header_font_size_normal = "32px"
spacing_large = "20px"
spacing_normal = "10px"

applied_font_size = font_size_large if st.session_state.ui_scale == 'large' else font_size_normal
applied_button_padding = button_padding_large if st.session_state.ui_scale == 'large' else button_padding_normal
applied_header_font_size = header_font_size_large if st.session_state.ui_scale == 'large' else header_font_size_normal
applied_spacing = spacing_large if st.session_state.ui_scale == 'large' else spacing_normal

primary_color = "#FF4B4B"
secondary_color = "#F0F2F6"
background_color = "#FFFFFF"
card_background_color = "#F0F2F6"
text_color = "#31333F"
button_color = "#007BFF"
button_hover_color = "#0056b3"
menu_item_border_color = "#CCCCCC"
menu_item_background_color = "#FFFFFF"
cart_background_color = "#e9ecef"

st.markdown(f"""
<style>
    /* 전체 페이지 스타일 */
    .stApp {{
        background-color: {background_color};
        color: {text_color};
        font-family: 'Noto Sans KR', sans-serif;
        padding: {applied_spacing};
    }}

    /* 기본 텍스트, 라벨 등 폰트 크기 */
    .stMarkdown p, .stText, .stWrite, label, .stTextInput label,
    .stTextArea label, .stSelectbox label, .stRadio label, .stCheckbox label {{
        font-size: {applied_font_size} !important;
        line-height: 1.6;
    }}

    /* 헤더 크기 */
    h1, h2, h3 {{
       font-size: {applied_header_font_size} !important;
       color: {text_color};
       margin-bottom: {applied_spacing};
       padding-top: {applied_spacing};
    }}

    /* 버튼 스타일 */
    /* 일반 버튼 (메뉴 항목 버튼과 구분될 수 있도록) */
    .stButton>button:not([key^="select_"]):not([key="clear_cart"]):not([key="place_order"]):not([key="misc_staff"]):not([key="misc_input"]) {{ /* 키오스크 메인 액션 버튼 제외 */
         /* 예시 스타일 - 필요에 따라 조정 */
        background-color: #6C757D; /* 회색 버튼 */
        color: white;
        padding: {applied_button_padding};
        font-size: {applied_font_size};
        width: 100%;
        margin-bottom: {applied_spacing};
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
     .stButton>button:not([key^="select_"]):not([key="clear_cart"]):not([key="place_order"]):not([key="misc_staff"]):hover {{
         background-color: #5A6268;
     }}

    /* 메뉴 항목 내 버튼 스타일 */
     .stButton>button[key^="select_"] {{
        background-color: transparent;
        color: {button_color};
        border: 1px solid {button_color};
        padding: 10px 20px;
        font-size: {applied_font_size};
        width: 100%;
        margin-top: {applied_spacing};
        border-radius: 8px;
        transition: background-color 0.3s ease, color 0.3s ease;
     }}
     .stButton>button[key^="select_"]:hover {{
        background-color: {button_color};
        color: white;
     }}

    /* 장바구니 비우기 버튼 스타일 */
     .stButton>button[key="clear_cart"] {{
        background-color: #DC3545; /* 빨간색 */
        color: white;
        padding: {applied_button_padding};
        font-size: {applied_font_size};
        width: 100%;
        margin-bottom: {applied_spacing};
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
     }}
      .stButton>button[key="clear_cart"]:hover {{
          background-color: #C82333; /* 진한 빨간색 */
      }}

    /* 주문하기 버튼 스타일 */
    .stButton>button[key="place_order"] {{
        background-color: {primary_color}; /* 강조색 (빨강) */
        color: white;
        padding: {applied_button_padding};
        font-size: {applied_font_size};
        width: 100%;
        margin-bottom: {applied_spacing};
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
     .stButton>button[key="place_order"]:hover {{
         background-color: #E0A800; /* 약간 다른 강조색 호버 */
     }}

    /* 직원 호출 버튼 스타일 (기타 섹션) */
    .stButton>button[key="misc_staff"] {{ /* key 추가 필요 */
        background-color: #17A2B8; /* 파란색 */
        color: white;
        padding: {applied_button_padding};
        font-size: {applied_font_size};
        width: 100%;
        margin-bottom: {applied_spacing};
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }}
     .stButton>button[key="misc_staff"]:hover {{
         background-color: #138496;
     }}

    /* Input 필드 스타일 */
    .stTextInput input, .stTextArea textarea, .stSelectbox div input {{
        font-size: {applied_font_size} !important;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid {secondary_color};
        margin-bottom: {applied_spacing};
    }}
     .stTextInput>label, .stTextArea>label {{
         margin-bottom: 5px !important;
         display: block;
     }}


    /* 메트릭 스타일 */
    div[data-testid="stMetric"] {{
        background-color: {card_background_color};
        padding: {applied_spacing};
        border-radius: 8px;
        margin-bottom: {applied_spacing};
        border: 1px solid {secondary_color};
    }}
    div[data-testid="stMetricValue"] {{
        font-size: {applied_header_font_size} !important;
        color: {primary_color};
        font-weight: bold;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: {applied_font_size} !important;
        color: {text_color};
    }}

    /* Expander 스타일 */
     div[data-testid="stExpander"] {{
        background-color: {card_background_color};
        padding: 0px {applied_spacing} {applied_spacing} {applied_spacing};
        border-radius: 8px;
        margin-bottom: {applied_spacing};
        border: 1px solid {secondary_color};
     }}
    div[data-testid="stExpander"] div[role="button"] {{
        background-color: {secondary_color};
        padding: {applied_spacing};
        margin: -{applied_spacing} -{applied_spacing} {applied_spacing} -{applied_spacing};
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
    }}
     div[data-testid="stExpander"] div[role="button"] p {{
        font-size: {applied_font_size} !important;
        font-weight: bold;
     }}

    /* Divider 스타일 */
    .stDivider {{
        margin-top: {applied_spacing};
        margin-bottom: {applied_spacing};
    }}

    /* 컬럼 간 간격 */
    .stColumns {{
        gap: {applied_spacing};
    }}

    /* 메뉴 항목 컨테이너 스타일 */
    .menu-item-container {{
        border: 1px solid {menu_item_border_color};
        border-radius: 8px;
        padding: {applied_spacing};
        margin-bottom: {applied_spacing};
        background-color: {menu_item_background_color};
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        box-sizing: border-box;
    }}

    /* 메뉴 항목 이미지 컨테이너 (stImage) */
    .menu-item-container .stImage img {{
        max-width: 100%;
        height: auto;
        border-radius: 4px;
        margin-bottom: {applied_spacing};
        object-fit: cover;
    }}
     .menu-item-container .stImage {{
         margin-bottom: {applied_spacing};
     }}


    /* 메뉴 항목 텍스트 스타일 */
    .menu-item-name {{
        font-size: {applied_font_size} !important;
        font-weight: bold;
        margin-bottom: 5px;
    }}
    .menu-item-price {{
        font-size: calc({applied_font_size} * 0.9) !important;
        color: {primary_color};
        margin-bottom: {applied_spacing};
    }}

    /* 장바구니 컨테이너 스타일 */
    .cart-container {{
        background-color: {cart_background_color};
        padding: {applied_spacing};
        border-radius: 8px;
        margin-bottom: {applied_spacing};
        min-height: 300px;
        max-height: 600px; /* Adjust max height as needed */
        overflow-y: auto; /* Add scroll if content exceeds max height */
        border: 1px solid {secondary_color};
    }}

    /* 장바구니 항목 스타일 */
    .cart-item {{
        border-bottom: 1px dashed {menu_item_border_color};
        padding-bottom: 5px;
        margin-bottom: 5px;
        font-size: {applied_font_size} !important;
        /* Ensure items are in a row or block */
        display: flex; /* Use flex for layout */
        justify-content: space-between; /* Distribute space between name/quantity and price */
        align-items: center; /* Vertically align items */
        flex-wrap: wrap; /* Allow wrapping if necessary */
    }}
     .cart-item:last-child {{
         border-bottom: none;
         margin-bottom: 0;
         padding-bottom: 0;
     }}
    .cart-item .item-info {{ /* Container for name and quantity */
         flex-grow: 1; /* Allow item info to take available space */
         text-align: left;
         margin-right: 10px; /* Space between info and price */
    }}
    .cart-item .item-name {{ /* Item name within item-info */
        font-weight: bold;
        display: block; /* Name and quantity on separate lines or inline? Let's try inline-block for flexibility */
        display: inline-block;
        margin-right: 5px;
    }}
     /* Style for the quantity number itself */
     .cart-item .item-quantity-display {{
         font-weight: normal; /* Quantity number style */
         display: inline-block;
     }}
     .cart-item .item-price {{ /* Item price */
         color: {primary_color};
         font-weight: bold;
         text-align: right; /* Align price to the right within the item */
         white-space: nowrap; /* Prevent price from wrapping */
     }}

</style>
""", unsafe_allow_html=True)

# --- Streamlit UI 메인 레이아웃 ---

st.title("☕ 스마트 키오스크")

st.markdown("카메라를 보고 나이를 인식하면 화면 크기가 자동으로 조절됩니다.")


# --- UI 컬럼 레이아웃 시작 ---
# 카메라 피드와 장바구니 영역
col1, col2 = st.columns([2, 1]) # 2:1 비율 유지

with col1:
    st.header("카메라 화면")
    # 모델 로드 실패 시 경고 표시
    if not model_loaded:
        st.warning("AI 모델 로딩에 실패했습니다. 얼굴 인식 기능이 작동하지 않을 수 있습니다. 터미널/콘솔 로그를 확인해주세요.")
    else:
        ctx = webrtc_streamer(key="age-detection",
                              video_processor_factory=AgeEstimator,
                              rtc_configuration=RTCConfiguration(
                                  {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                              ),
                              media_stream_constraints={"video": {"height": 360}, "audio": False}, # 비디오 높이 조정
                              async_processing=True
                             )

        if ctx.video_processor:
            pass # 설명 문구는 위에 별도로 표시
        else:
            st.info("카메라를 시작해주세요.")

with col2:
    # --- 장바구니 섹션 ---
    st.header("장바구니")

    # 장바구니 내용을 담을 컨테이너
    st.markdown('<div class="cart-container">', unsafe_allow_html=True)

    # 장바구니 내용 표시
    if not st.session_state.cart:
        st.write("장바구니가 비어있습니다.") # 장바구니 비어있을 때 메시지
    else:
        total_price = 0
        # 장바구니 항목 순회하며 표시 (수량 포함)
        for item in st.session_state.cart:
            item_name = item["name"]
            item_price_str = item["price"]
            item_quantity = item["quantity"]

            # 가격 문자열에서 숫자만 추출
            try:
                price_value = int(item_price_str.replace(",", "").replace("원", "").strip())
                total_price += price_value * item_quantity
            except ValueError:
                # 가격 변환 오류 시 무시하거나 로깅
                print(f"Warning: Could not parse price '{item_price_str}' for item '{item.get('name')}'")
                pass # 오류가 나더라도 앱이 멈추지 않도록 함

            # 장바구니 항목 표시 형식: <div class="cart-item"> <span class="item-info"><span class="item-name">이름</span> * <span class="item-quantity-display">수량</span></span> <span class="item-price">가격</span> </div>
            # CSS 클래스를 사용하여 이름, 수량, 가격 부분에 스타일 적용 및 레이아웃 조정
            st.markdown(f'<div class="cart-item"><span class="item-info"><span class="item-name">{item_name}</span> * <span class="item-quantity-display">{item_quantity}</span></span> <span class="item-price">{item_price_str}</span></div>', unsafe_allow_html=True)

        st.markdown("---", unsafe_allow_html=True) # 구분선
        st.markdown(f'<div style="font-size: {applied_font_size}; font-weight: bold; text-align: right;">총 금액: {total_price:,}원</div>', unsafe_allow_html=True) # 총 금액 표시 (천단위 콤마)

    st.markdown('</div>', unsafe_allow_html=True) # 장바구니 컨테이너 닫기

    # 장바구니 관련 버튼 (예: 주문하기, 비우기)
    st.markdown("") # 장바구니 아래 약간의 간격
    col_cart_btns = st.columns(2)
    with col_cart_btns[0]:
         if st.button("장바구니 비우기", key="clear_cart"):
             st.session_state.cart = [] # 장바구니 비우기
             st.rerun() # UI 업데이트를 위해 재실행
    with col_cart_btns[1]:
         if st.button("주문하기", key="place_order", type="primary"): # 주문 버튼 강조
              if st.session_state.cart:
                 # TODO: 실제 주문 처리 로직 구현 (예: 데이터베이스 저장, 결제 모듈 연동 등)
                 st.success("주문이 접수되었습니다! (실제 주문 처리 로직 필요)")
                 # 주문 처리 후 장바구니 비우기
                 st.session_state.cart = []
                 st.rerun() # UI 업데이트
              else:
                 st.warning("장바구니가 비어있습니다.")


    # 나이/스케일 정보는 장바구니 아래에 유지
    st.markdown("---") # 구분선 추가
    st.header("정보") # 정보 헤더 다시 추가
    st.write(f"**화면 확대 기준:** {AGE_THRESHOLD}세 초과")
    st.metric("예상 나이", value=f"{st.session_state.last_detected_age if st.session_state.last_detected_age is not None else '확인 중...'}")
    st.metric("현재 UI 크기", value=st.session_state.ui_scale.capitalize())


st.divider() # 구분선

# --- 메뉴 선택 섹션 ---
st.header("메뉴 선택")

# 예시 메뉴 항목 데이터 (이미지 파일 경로, 이름, 가격)
# images 폴더 안에 '아메리카노.jpg', '카페라떼.jpg', '샌드위치.jpg',
# '조각케이크.jpg', '아이스티.jpg', '치킨베이컨랩.jpg' 파일이 있다고 가정합니다.
menu_items = [
    {"name": "아메리카노", "image_path": "images/아메리카노.jpg", "price": "4,500원"},
    {"name": "카페 라떼", "image_path": "images/카페라떼.jpg", "price": "5,000원"},
    {"name": "샌드위치", "image_path": "images/샌드위치.jpg", "price": "6,500원"},
    {"name": "조각 케이크", "image_path": "images/조각 케이크.jpg", "price": "5,500원"},
    {"name": "아이스티", "image_path": "images/아이스티.jpg", "price": "4,000원"},
    {"name": "치킨 베이컨 랩", "image_path": "images/치킨 베이컨 랩.jpg", "price": "7,500원"}, # 예시 가격
]


# 메뉴 항목을 컬럼으로 배열하여 그리드 형태로 표시
items_per_row = 3
cols = st.columns(items_per_row)

for index, item in enumerate(menu_items):
    col = cols[index % items_per_row]
    with col:
        st.markdown('<div class="menu-item-container">', unsafe_allow_html=True)

        if os.path.exists(item["image_path"]):
             st.image(item["image_path"], use_column_width=True, output_format="auto", # 캡션 제거 (텍스트를 별도 표시)
                      clamp=False, channels="RGB")
        else:
             st.warning(f"이미지 파일을 찾을 수 없습니다: {item['image_path']}")
             st.markdown(f'<div style="height: 150px; background-color: #eee; display: flex; align-items: center; justify-content: center; border-radius: 4px; margin-bottom: {applied_spacing};">이미지 없음</div>', unsafe_allow_html=True)


        # 이름과 가격을 별도 텍스트로 표시 (CSS 클래스 활용)
        st.markdown(f'<p class="menu-item-name">{item["name"]}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="menu-item-price">{item["price"]}</p>', unsafe_allow_html=True)


        # 메뉴 항목 선택 버튼
        # 버튼 클릭 시 장바구니에 항목 추가
        if st.button(f"선택하기", key=f"select_{item['name']}", help=f"{item['name']}을(를) 선택합니다."):
            # 클릭된 메뉴 항목 정보를 가져옵니다.
            selected_item_name = item["name"]
            selected_item_price = item["price"]

            # 장바구니에 동일한 항목이 있는지 확인합니다.
            found_in_cart = False
            for cart_item in st.session_state.cart:
                if cart_item["name"] == selected_item_name:
                    # 이미 있으면 수량만 증가
                    cart_item["quantity"] += 1
                    found_in_cart = True
                    # print(f"Increased quantity for {selected_item_name}. New quantity: {cart_item['quantity']}") # Debug
                    break

            # 장바구니에 없으면 새로 추가 (수량 1로 시작)
            if not found_in_cart:
                st.session_state.cart.append({
                    "name": selected_item_name,
                    "price": selected_item_price,
                    "quantity": 1
                })
                # print(f"Added new item to cart: {selected_item_name}") # Debug

            st.toast(f"'{selected_item_name}'를 장바구니에 담았습니다.")
            st.rerun() # 장바구니 내용 업데이트를 위해 재실행


        st.markdown('</div>', unsafe_allow_html=True)

# --- 기타 버튼 섹션 ---
st.markdown("---")

st.header("기타")

with st.container():
    col_misc = st.columns(2)
    with col_misc[0]:
         # 직원 호출 버튼 (일반 버튼 스타일 적용)
         # key 추가하여 CSS 타겟팅 가능하게 함
         if st.button("👩‍💼 직원 호출", key="misc_staff"):
             st.toast("직원 호출 버튼을 누르셨습니다.")
             # TODO: 직원 호출 알림 로직 구현
    with col_misc[1]:
         # 추가 요청 사항 입력 필드 (key 추가)
         st.text_input("추가 요청 사항 입력", placeholder="예) 얼음 많이 주세요, 포장해 주세요 등", key="misc_input")


# --- 개발 정보 섹션 ---
with st.expander("개발 정보 (참고용)"):
    st.write(f"마지막 감지된 나이 (내부값): {st.session_state.last_detected_age}")
    st.write(f"UI 스케일 상태: {st.session_state.ui_scale}")
    try:
        st.write(f"탐지 간격: {AgeEstimator().detection_interval} 초 (근사치)")
    except Exception:
        st.write("탐지 간격 정보를 가져올 수 없습니다.")
    st.write(f"DeepFace 모델 로드 상태: {'성공' if model_loaded else '실패/오류'}")