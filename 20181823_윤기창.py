import cv2
import numpy as np

# 초기 변수 설정
current_mask_index = 0
drawing = False  # 마우스 드래그 상태
mask_names = ['Blurring', 'Sharpening', 'Prewitt', 'Sobel', 'Laplacian']  # 마스크 이름
mask_kernels = [
    np.ones((3, 3), np.float32) / 9,  # Blurring
    np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32),  # Sharpening
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32),  # Prewitt
    np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32),  # Sobel
    np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32),  # Laplacian
]

# 마우스 이벤트 콜백 함수
def mouse_event(event, x, y, flags, param):
    global drawing, cursor_combined, img, touched_area

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 마우스 드래그 중에 회선을 적용함
            apply_convolution(x, y)
        else:
            cursor_temp = img.copy()
            cv2.circle(cursor_temp, (x, y), 30, (0, 0, 255), -1)
            cursor_combined = cv2.addWeighted(img, 0.7, cursor_temp, 0.3, 0)
            if not drawing:  # 마우스 드래그 중이 아닐 때만 업데이트
                cv2.imshow('Touch Effect', cursor_combined)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# 회선을 적용하는 함수
def apply_convolution(x, y):
    global img, current_mask_index, touched_area

    # 회선 마스크 선택
    mask = mask_kernels[current_mask_index]

    # 회선을 적용할 원 영역 설정
    center = (x, y)
    radius = 30

    # 원 영역 내부에 포함되는 픽셀 좌표 생성
    y_coords, x_coords = np.ogrid[:img.shape[0], :img.shape[1]]
    mask_radius = (x_coords - center[0]) ** 2 + (y_coords - center[1]) ** 2 <= radius ** 2

    # 이미 터치된 영역인지 확인
    touched_mask_radius = np.logical_and(mask_radius, touched_area[:, :, 0] == 255)

    # 중복 적용을 방지하기 위해 이미 터치된 영역은 회선을 적용하지 않음
    mask_radius = np.logical_and(mask_radius, ~touched_mask_radius)

    #magnitude로 표현
    img_convolved = cv2.filter2D(img, -1, mask)
    img_magnitude = np.abs(img_convolved)
    img[mask_radius] = img_magnitude[mask_radius]

    # 터치된 영역 표시
    touched_area[mask_radius] = (255, 255, 255)

# 입력 영상 로드
img = cv2.imread('input.jpg')
img = cv2.resize(img, (1920, 1080))

# 윈도우 생성 및 이벤트 콜백 함수 등록
cv2.namedWindow('Touch Effect')
cv2.setMouseCallback('Touch Effect', mouse_event)

cursor_img = img.copy()
cursor_combined = img.copy()  # 초기 커서 이미지
touched_area = np.zeros_like(img)

while True:
    cv2.imshow('Touch Effect', cursor_combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('1'):
        current_mask_index = 0
    elif key == ord('2'):
        current_mask_index = 1
    elif key == ord('3'):
        current_mask_index = 2
    elif key == ord('4'):
        current_mask_index = 3
    elif key == ord('5'):
        current_mask_index = 4
    elif key == ord('q'):
        break

    # 마스크 이름 업데이트
    cv2.putText(cursor_combined, f'{current_mask_index + 1}. {mask_names[current_mask_index]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# 터치 효과가 적용된 영상 저장
cv2.imwrite('20181823_1.jpg', img)

# 회선이 적용된 영역만 0으로 초기화하여 touched_area 생성
touched_mask = np.where(np.any(touched_area != [0, 0, 0], axis=-1, keepdims=True), [255, 255, 255], [0, 0, 0])

# ouched_area를 단일 채널 이진 마스크로 변환
touched_mask_gray = cv2.cvtColor(touched_area, cv2.COLOR_BGR2GRAY)
ret, touched_mask_binary = cv2.threshold(touched_mask_gray, 1, 255, cv2.THRESH_BINARY)

# mg 크기에 맞게 touched_mask_binary 크기 조정
touched_mask_binary = cv2.resize(touched_mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

# bitwise_and 연산 적용
result_img = cv2.bitwise_and(img, img, mask=touched_mask_binary)

# touched_mask_binary 반전
inverted_mask = cv2.bitwise_not(touched_mask_binary)

# bitwise_or 연산 적용
result_img = cv2.bitwise_or(result_img, img, mask=inverted_mask)

# 터치 효과 적용 영역의 화소 값은 모두 0으로 저장된 이미지 저장
cv2.imwrite('20181823_2.jpg', result_img)
cv2.destroyAllWindows()