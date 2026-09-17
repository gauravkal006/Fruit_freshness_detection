import os
import sys
import io
import base64
import warnings
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response, send_from_directory

# Filter warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(BASE_DIR / "static"), template_folder=str(BASE_DIR / "templates"))

@app.route("/models/<path:filename>")
def serve_models(filename):
    return send_from_directory(BASE_DIR / "models", filename)

# Display mapping for detected produce classes
CLASS_DISPLAY_MAP = {
    "apple": "Fresh Apple",
    "banana": "Fresh Banana",
    "orange": "Fresh Orange",
    "rottenapples": "Rotten Apple",
    "rottenbanana": "Rotten Banana",
    "rottenoranges": "Rotten Orange"
}

model = None

def get_model():
    """
    Lazy loads the best fine-tuned 15-epoch YOLOv11 classification model from local models/ directory.
    Falls back to system cache or default weights if local file is unavailable.
    """
    global model
    if model is None:
        local_model_path = BASE_DIR / "models" / "best.pt"
        
        # User cache fallbacks
        c_cache_dir = Path(r"C:\Users\HP\.cache\ultralytics")
        c_runs_dir = c_cache_dir / "runs"
        c_weights_dir = c_cache_dir / "weights"
        best_weights_15ep = c_runs_dir / "classify" / "yolo11n_spoilage_15ep" / "weights" / "best.pt"
        
        if local_model_path.exists():
            print(f"⚡ Loading fine-tuned YOLOv11 model weights from: {local_model_path}")
            from ultralytics import YOLO
            model = YOLO(str(local_model_path))
        elif best_weights_15ep.exists():
            print(f"⚡ Loading fine-tuned YOLOv11 model weights from user cache: {best_weights_15ep}")
            from ultralytics import YOLO
            model = YOLO(str(best_weights_15ep))
        else:
            print("📦 Loading default YOLOv11 classification model")
            from ultralytics import YOLO
            model = YOLO("yolo11n-cls.pt")
            
    return model

def detect_and_draw_rot_spots(cv_img, produce_bbox, is_rotten):
    """
    Locates rot and decay spots on produce items and highlights them with bounding markers and location tags.
    Determines spatial quadrant (e.g., Upper-Left, Center, Lower-Right) of localized decay spots.
    """
    x1, y1, x2, y2 = produce_bbox
    crop_h = y2 - y1
    crop_w = x2 - x1
    
    if crop_h <= 10 or crop_w <= 10:
        return cv_img, []
        
    produce_crop = cv_img[y1:y2, x1:x2]
    lab_crop = cv2.cvtColor(produce_crop, cv2.COLOR_BGR2LAB)
    l_channel = lab_crop[:, :, 0]
    
    mean_l = np.mean(l_channel)
    std_l = np.std(l_channel)
    rot_thresh_val = max(30, int(mean_l - 1.2 * std_l))
    
    _, rot_mask = cv2.threshold(l_channel, rot_thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rot_mask = cv2.morphologyEx(rot_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(rot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rot_spots = []
    
    if is_rotten and contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for idx, c in enumerate(contours[:3]):
            spot_area = cv2.contourArea(c)
            if spot_area > (crop_h * crop_w * 0.008):
                sx, sy, sw, sh = cv2.boundingRect(c)
                
                gx1 = x1 + sx
                gy1 = y1 + sy
                gx2 = gx1 + sw
                gy2 = gy1 + sh
                
                cx = (gx1 + gx2) // 2
                cy = (gy1 + gy2) // 2
                
                fx_center = (x1 + x2) / 2.0
                fy_center = (y1 + y2) / 2.0
                
                rel_x = "Right" if cx > fx_center else "Left"
                rel_y = "Lower" if cy > fy_center else "Upper"
                
                if abs(cx - fx_center) < crop_w * 0.15 and abs(cy - fy_center) < crop_h * 0.15:
                    quadrant = "Center Region"
                else:
                    quadrant = f"{rel_y}-{rel_x} Region"
                    
                spot_info = {
                    "spot_id": idx + 1,
                    "location": quadrant,
                    "x": cx,
                    "y": cy,
                    "width": sw,
                    "height": sh
                }
                rot_spots.append(spot_info)
                
                # Draw rot spot bounding rectangle
                cv2.rectangle(cv_img, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                cv2.circle(cv_img, (cx, cy), 4, (0, 0, 255), -1)
                
                spot_label = f"ROT #{idx+1}: {quadrant}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_img, spot_label, (gx1, max(15, gy1 - 5)), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
                
    return cv_img, rot_spots

def draw_yolo_bounding_box_fixed(cv_img, bbox, display_label, top1_conf, is_rotten):
    """Draws YOLO bounding box and accent corners around detected fruit produce"""
    x1, y1, x2, y2 = bbox
    box_color = (38, 38, 220) if is_rotten else (74, 160, 22)
    thickness = 3
    
    cv2.rectangle(cv_img, (x1, y1), (x2, y2), box_color, thickness)
    
    corner_len = min(25, int((x2 - x1) * 0.15))
    cv2.line(cv_img, (x1, y1), (x1 + corner_len, y1), box_color, thickness + 2)
    cv2.line(cv_img, (x1, y1), (x1, y1 + corner_len), box_color, thickness + 2)
    cv2.line(cv_img, (x2, y1), (x2 - corner_len, y1), box_color, thickness + 2)
    cv2.line(cv_img, (x2, y1), (x2, y1 + corner_len), box_color, thickness + 2)
    cv2.line(cv_img, (x1, y2), (x1 + corner_len, y2), box_color, thickness + 2)
    cv2.line(cv_img, (x1, y2), (x1, y2 - corner_len), box_color, thickness + 2)
    cv2.line(cv_img, (x2, y2), (x2 - corner_len, y2), box_color, thickness + 2)
    cv2.line(cv_img, (x2, y2), (x2, y2 - corner_len), box_color, thickness + 2)

    label_text = f"YOLOv11: {display_label} ({top1_conf*100:.1f}%)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    
    label_y1 = max(0, y1 - text_h - 10)
    cv2.rectangle(cv_img, (x1, label_y1), (x1 + text_w + 12, y1), box_color, -1)
    cv2.putText(cv_img, label_text, (x1 + 6, y1 - 6), font, font_scale, (255, 255, 255), font_thickness)
    
    return cv_img

def extract_produce_region(pil_img):
    """Isolates fruit from background clutter using HSV color chroma thresholding."""
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w, _ = cv_img.shape
    
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    _, sat_mask = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)
    _, val_mask = cv2.threshold(val, 20, 245, cv2.THRESH_BINARY)
    produce_mask = cv2.bitwise_and(sat_mask, val_mask)
    
    contours, _ = cv2.findContours(produce_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > (h * w * 0.02):
            x, y, bw, bh = cv2.boundingRect(c)
            pad = 12
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)
            
            cropped_cv = cv_img[y1:y2, x1:x2]
            if cropped_cv.shape[0] > 10 and cropped_cv.shape[1] > 10:
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
                return cropped_pil, (x1, y1, x2, y2), True
                
    x1, y1, x2, y2 = int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)
    cropped_cv = cv_img[y1:y2, x1:x2]
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
    return cropped_pil, (x1, y1, x2, y2), True

def predict_with_multi_angle_tta(yolo_model, cropped_pil):
    """Evaluates produce item from 4 cardinal angles (0°, 90°, 180°, 270°) and horizontal flip."""
    variants = [
        cropped_pil,
        cropped_pil.rotate(90, expand=True),
        cropped_pil.rotate(180, expand=True),
        cropped_pil.rotate(270, expand=True),
        cropped_pil.transpose(Image.FLIP_LEFT_RIGHT),
    ]
    
    all_raw_probs = []
    for var in variants:
        res = yolo_model.predict(var, device="cpu", verbose=False)[0]
        all_raw_probs.append(res.probs.data.cpu().numpy())
        
    avg_probs = np.mean(all_raw_probs, axis=0)
    top1_idx = int(np.argmax(avg_probs))
    top1_conf = float(avg_probs[top1_idx])
    
    return top1_idx, top1_conf, avg_probs

def compute_hierarchical_produce_analysis(yolo_model, avg_probs):
    """
    Stage 1: Identify Fruit Type (Apple, Banana, Orange)
    Stage 2: Quantify Freshness % vs Spoilage %
    """
    prob_dict = {}
    for idx, conf in enumerate(avg_probs):
        cls_name = yolo_model.names[idx].lower()
        prob_dict[cls_name] = float(conf)
        
    apple_fresh = prob_dict.get("apple", 0.0)
    apple_rotten = prob_dict.get("rottenapples", 0.0)
    banana_fresh = prob_dict.get("banana", 0.0)
    banana_rotten = prob_dict.get("rottenbanana", 0.0)
    orange_fresh = prob_dict.get("orange", 0.0)
    orange_rotten = prob_dict.get("rottenoranges", 0.0)
    
    family_scores = {
        "Apple": apple_fresh + apple_rotten,
        "Banana": banana_fresh + banana_rotten,
        "Orange": orange_fresh + orange_rotten
    }
    
    detected_fruit_type = max(family_scores, key=family_scores.get)
    type_confidence = family_scores[detected_fruit_type]
    
    if detected_fruit_type == "Apple":
        f_prob, r_prob = apple_fresh, apple_rotten
    elif detected_fruit_type == "Banana":
        f_prob, r_prob = banana_fresh, banana_rotten
    else:
        f_prob, r_prob = orange_fresh, orange_rotten
        
    total_type_prob = f_prob + r_prob
    if total_type_prob > 0:
        freshness_pct = (f_prob / total_type_prob) * 100
        spoilage_pct = (r_prob / total_type_prob) * 100
    else:
        freshness_pct, spoilage_pct = 50.0, 50.0
        
    is_rotten = spoilage_pct > freshness_pct
    display_label = f"Fresh {detected_fruit_type}" if not is_rotten else f"Rotten {detected_fruit_type}"
    
    return {
        "fruit_type": detected_fruit_type,
        "type_confidence": round(type_confidence * 100, 1),
        "freshness_pct": round(freshness_pct, 1),
        "spoilage_pct": round(spoilage_pct, 1),
        "is_rotten": is_rotten,
        "display_label": display_label,
        "family_scores": [
            {"fruit": f, "confidence": round(s * 100, 1)}
            for f, s in sorted(family_scores.items(), key=lambda x: x[1], reverse=True)
        ]
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/detect_image", methods=["POST"])
def process_prediction():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
            
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        cropped_pil, (x1, y1, x2, y2), is_valid_fruit = extract_produce_region(pil_img)
        
        yolo_model = get_model()
        top1_idx, top1_conf, avg_probs = predict_with_multi_angle_tta(yolo_model, cropped_pil)
        analysis = compute_hierarchical_produce_analysis(yolo_model, avg_probs)
        
        display_label = analysis["display_label"]
        is_rotten = analysis["is_rotten"]
        
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        annotated_cv = draw_yolo_bounding_box_fixed(cv_img, (x1, y1, x2, y2), display_label, analysis["type_confidence"] / 100.0, is_rotten)
        annotated_cv, rot_spots = detect_and_draw_rot_spots(annotated_cv, (x1, y1, x2, y2), is_rotten)
        
        _, buffer = cv2.imencode(".jpg", annotated_cv)
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        
        status = f"{analysis['fruit_type']} - {'Rotten / Spoiled' if is_rotten else 'Fresh'}"
        status_color = "rotten" if is_rotten else "fresh"
        
        if is_rotten:
            spot_loc_text = f" Decay detected at: {', '.join([s['location'] for s in rot_spots])}." if rot_spots else ""
            advice = f"Unsafe for consumption. This {analysis['fruit_type']} is {analysis['spoilage_pct']}% decayed/spoiled.{spot_loc_text}"
        else:
            advice = f"Safe for consumption. This {analysis['fruit_type']} is {analysis['freshness_pct']}% fresh."
            
        all_probs = []
        for idx, conf in enumerate(avg_probs):
            cls_name = yolo_model.names[idx]
            disp = CLASS_DISPLAY_MAP.get(cls_name.lower(), cls_name)
            all_probs.append({"class": disp, "confidence": round(float(conf) * 100, 1)})
            
        all_probs.sort(key=lambda x: x["confidence"], reverse=True)
        
        return jsonify({
            "success": True,
            "fruit_type": analysis["fruit_type"],
            "type_confidence": analysis["type_confidence"],
            "freshness_pct": analysis["freshness_pct"],
            "spoilage_pct": analysis["spoilage_pct"],
            "display_label": display_label,
            "confidence": analysis["type_confidence"],
            "status": status,
            "status_color": status_color,
            "advice": advice,
            "rot_spots": rot_spots,
            "image_b64": img_b64,
            "family_scores": analysis["family_scores"],
            "all_probabilities": all_probs
        })
        
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "project": "Banana, Apple and Orange Freshness Prediction",
        "classes": [
            {"name": "Fresh Apple", "samples": 3215},
            {"name": "Fresh Banana", "samples": 3360},
            {"name": "Fresh Orange", "samples": 1854},
            {"name": "Rotten Apple", "samples": 4236},
            {"name": "Rotten Banana", "samples": 3832},
            {"name": "Rotten Orange", "samples": 1998}
        ],
        "model_name": "YOLOv11 15-Epoch Fine-Tuned Produce Classifier",
        "epochs": 15
    })

def generate_webcam_frames(cam_index=0):
    camera = cv2.VideoCapture(cam_index)
    if not camera.isOpened():
        for alt_idx in [1, 2, 0]:
            if alt_idx != cam_index:
                camera = cv2.VideoCapture(alt_idx)
                if camera.isOpened():
                    break
                    
    if not camera.isOpened():
        return
    
    yolo_model = get_model()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cropped_pil, (x1, y1, x2, y2), _ = extract_produce_region(pil_frame)
        top1_idx, top1_conf, _ = predict_with_multi_angle_tta(yolo_model, cropped_pil)
        
        raw_class_name = yolo_model.names[top1_idx]
        display_label = CLASS_DISPLAY_MAP.get(raw_class_name.lower(), raw_class_name)
        is_rotten = "rotten" in raw_class_name.lower()
        
        annotated_frame = draw_yolo_bounding_box_fixed(frame, (x1, y1, x2, y2), display_label, top1_conf, is_rotten)
        annotated_frame, _ = detect_and_draw_rot_spots(annotated_frame, (x1, y1, x2, y2), is_rotten)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    camera.release()

@app.route("/video_feed")
def video_feed():
    cam_index = request.args.get("cam_index", default=0, type=int)
    return Response(generate_webcam_frames(cam_index), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("🚀 Starting Banana, Apple and Orange Freshness Prediction Web App on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
