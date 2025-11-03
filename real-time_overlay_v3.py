import cv2
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image

# ----------------- RUNTIME CONFIG -----------------
print("=== Overlay Editor Setup ===")
BASE_PATH = input("Enter base image path: ").strip('"').strip()
CAR_PATH = input("Enter car image path: ").strip('"').strip()
FONT_PATH = input("Enter .ttf font file path: ").strip('"').strip()

# --------------------------------------------------
# Car & text defaults
INIT_DISPLAY_W = 800
INIT_DISPLAY_H = 400
RESIZE_STEP = 5

TEXT_DEFAULT = "MH03BWW8493"
TEXT_FONT_SIZE = 54
TEXT_RESIZE_STEP = 1

WINDOW_NAME = "Overlay Editor"
# --------------------------------------------------


def load_images():
    base = cv2.imread(BASE_PATH, cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"Base image not found at {BASE_PATH}")

    car = cv2.imread(CAR_PATH, cv2.IMREAD_UNCHANGED)
    if car is None:
        raise FileNotFoundError(f"Car image not found at {CAR_PATH}")
    return base, car


def overlay_alpha(bg, overlay_rgba, x, y):
    bg_h, bg_w = bg.shape[:2]
    ol_h, ol_w = overlay_rgba.shape[:2]

    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return bg

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + ol_w, bg_w)
    y2 = min(y + ol_h, bg_h)

    ol_x1 = x1 - x
    ol_y1 = y1 - y
    ol_x2 = ol_x1 + (x2 - x1)
    ol_y2 = ol_y1 + (y2 - y1)

    bg_region = bg[y1:y2, x1:x2].astype(float)
    ol_region = overlay_rgba[ol_y1:ol_y2, ol_x1:ol_x2].astype(float)

    if ol_region.shape[2] == 4:
        alpha = ol_region[:, :, 3:] / 255.0
        ol_rgb = ol_region[:, :, :3]
    else:
        alpha = np.ones((ol_region.shape[0], ol_region.shape[1], 1), dtype=float)
        ol_rgb = ol_region

    comp = alpha * ol_rgb + (1 - alpha) * bg_region
    bg[y1:y2, x1:x2] = comp.astype(np.uint8)
    return bg


# ----------------- Main interactive logic -----------------
base_img, car_img = load_images()
base_h, base_w = base_img.shape[:2]

if car_img.shape[2] == 3:
    alpha_channel = np.ones((car_img.shape[0], car_img.shape[1], 1), dtype=car_img.dtype) * 255
    car_img = np.concatenate([car_img, alpha_channel], axis=2)

current_w = INIT_DISPLAY_W
current_h = INIT_DISPLAY_H


def make_resized_car(car_rgba, width):
    width = max(1, int(width))
    height = max(1, int(round(width / 2.0)))
    resized = cv2.resize(car_rgba, (width, height), interpolation=cv2.INTER_AREA)
    return resized


car_display = make_resized_car(car_img, current_w)
x_offset = (base_w - car_display.shape[1]) // 2
y_offset = (base_h - car_display.shape[0]) // 2

# ---- Text state ----
text_pos = (100, 100)
text_font_size = TEXT_FONT_SIZE
text_dragging = False
text_drag_start = (0, 0)
text_offset_start = (0, 0)

# ---- Car dragging ----
dragging = False
drag_start_mouse = (0, 0)
drag_start_offset = (x_offset, y_offset)

# ---- Marker ----
text_coord = None


def mouse_cb(event, x, y, flags, param):
    global dragging, drag_start_mouse, drag_start_offset, x_offset, y_offset
    global text_dragging, text_drag_start, text_offset_start, text_pos, text_coord

    if event == cv2.EVENT_LBUTTONDOWN:
        cx1, cy1 = x_offset, y_offset
        cx2, cy2 = x_offset + car_display.shape[1], y_offset + car_display.shape[0]

        font = ImageFont.truetype(FONT_PATH, text_font_size)
        mask = font.getmask(TEXT_DEFAULT)
        tw, th = mask.size
        tx, ty = text_pos

        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            dragging = True
            drag_start_mouse = (x, y)
            drag_start_offset = (x_offset, y_offset)
        elif tx <= x <= tx + tw and ty <= y <= ty + th:
            text_dragging = True
            text_drag_start = (x, y)
            text_offset_start = text_pos

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx = x - drag_start_mouse[0]
            dy = y - drag_start_mouse[1]
            x_offset = int(drag_start_offset[0] + dx)
            y_offset = int(drag_start_offset[1] + dy)
        elif text_dragging:
            dx = x - text_drag_start[0]
            dy = y - text_drag_start[1]
            text_pos = (int(text_offset_start[0] + dx), int(text_offset_start[1] + dy))

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        text_dragging = False

    elif event == cv2.EVENT_RBUTTONDOWN:
        text_coord = (x, y)
        x_norm = x / base_w
        y_norm = - (y / base_h)
        print(f"Text marker -> Abs: ({x}, {y}), Norm: ({x_norm:.4f}, {y_norm:.4f})")


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback(WINDOW_NAME, mouse_cb)

instructions = [
    "Left-click + drag: move car or text",
    "'+' / '-' : resize car",
    "'[' / ']' : resize text",
    "Right-click: mark text coordinate (prints)",
    "'s' : print final coordinates & sizes",
    "'q' : quit"
]


def draw_text_overlay(image_bgr):
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(FONT_PATH, text_font_size)
    draw.text(text_pos, TEXT_DEFAULT, font=font, fill=(0, 0, 0, 255))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_overlay_preview():
    display = base_img.copy()
    overlay_alpha(display, car_display, x_offset, y_offset)
    display = draw_text_overlay(display)

    # Draw boxes
    x1, y1 = x_offset, y_offset
    x2, y2 = x_offset + car_display.shape[1], y_offset + car_display.shape[0]
    cv2.rectangle(display, (x1, y1), (x2, y2), (200, 200, 200), 1)

    y0 = 20
    for instr in instructions:
        cv2.putText(display, instr, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        y0 += 18

    if text_coord is not None:
        cv2.drawMarker(display, (text_coord[0], text_coord[1]), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

    return display


print("\nOverlay editor started. Drag, resize, and align. Press 's' to print info or 'q' to quit.\n")

while True:
    preview = draw_overlay_preview()
    cv2.imshow(WINDOW_NAME, preview)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        abs_x, abs_y = int(x_offset), int(y_offset)
        x_norm = abs_x / base_w
        y_norm = - (abs_y / base_h)
        cur_w = car_display.shape[1]
        cur_h = car_display.shape[0]

        tx, ty = text_pos
        t_x_norm = tx / base_w
        t_y_norm = - (ty / base_h)

        print("=== FINAL OVERLAY INFO ===")
        print(f"Car size: {cur_w}x{cur_h}")
        print(f"Car position -> Abs: ({abs_x}, {abs_y}), Norm: ({x_norm:.4f}, {y_norm:.4f})")
        print(f"Text position -> Abs: ({tx}, {ty}), Norm: ({t_x_norm:.4f}, {t_y_norm:.4f})")
        print(f"Text font size: {text_font_size}")
        if text_coord is not None:
            txc, tyc = text_coord
            txc_norm = txc / base_w
            tyc_norm = - (tyc / base_h)
            print(f"Text marker -> Abs: ({txc}, {tyc}), Norm: ({txc_norm:.4f}, {tyc_norm:.4f})")
        print("==========================")

    elif key in (ord('+'), ord('=')):
        current_w = min(base_w, car_display.shape[1] + RESIZE_STEP)
        car_display = make_resized_car(car_img, current_w)
    elif key == ord('-'):
        current_w = max(10, car_display.shape[1] - RESIZE_STEP)
        car_display = make_resized_car(car_img, current_w)
    elif key == ord('['):
        text_font_size = max(5, text_font_size - TEXT_RESIZE_STEP)
    elif key == ord(']'):
        text_font_size += TEXT_RESIZE_STEP

cv2.destroyAllWindows()
