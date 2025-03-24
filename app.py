from flask import Flask, render_template, url_for, request, jsonify
import os, json
from PIL import Image
import numpy as np

app = Flask(__name__)

# 預設目錄設定（以 app.static_folder 為基底）
DEFAULT_PATHS = {
    "original": os.path.join(app.static_folder, "original"),
    "predict": os.path.join(app.static_folder, "predict"),
    "ground_truth": os.path.join(app.static_folder, "ground_truth"),
    "draw_predict": os.path.join(app.static_folder, "draw_predict")
}

def load_paths():
    # 若 savepath.json 不存在，則自動產生
    if not os.path.exists("savepath.json"):
        with open("savepath.json", "w", encoding="utf8") as f:
            json.dump(DEFAULT_PATHS, f, ensure_ascii=False, indent=2)
        return DEFAULT_PATHS

    # 若存在則嘗試讀取並驗證
    try:
        with open("savepath.json", "r", encoding="utf8") as f:
            paths = json.load(f)
        # 檢查所有路徑是否存在
        valid = True
        for key, path in paths.items():
            if not os.path.isdir(path):
                valid = False
                break
        if valid:
            return paths
    except Exception as e:
        pass

    # 若檔案內容無效，則覆蓋為預設路徑
    with open("savepath.json", "w", encoding="utf8") as f:
        json.dump(DEFAULT_PATHS, f, ensure_ascii=False, indent=2)
    return DEFAULT_PATHS

# 讀取路徑設定
paths = load_paths()
ORIG_DIR = paths["original"]
PRED_DIR = paths["predict"]
GT_DIR = paths["ground_truth"]
DRAW_DIR = paths["draw_predict"]

def get_filenames_without_ext(directory):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    return set(
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_extensions
    )

def get_common_filenames():
    orig_names = get_filenames_without_ext(ORIG_DIR)
    pred_names = get_filenames_without_ext(PRED_DIR)
    gt_names   = get_filenames_without_ext(GT_DIR)
    draw_names = get_filenames_without_ext(DRAW_DIR)
    return sorted(list(orig_names & pred_names & gt_names & draw_names))

def find_file(directory, base_name):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}
    for f in os.listdir(directory):
        if os.path.splitext(f)[0] == base_name and os.path.splitext(f)[1].lower() in valid_extensions:
            return f
    return None

def get_url(file_full):
    """
    若檔案在 app.static_folder 底下，則傳回以相對路徑生成的 URL，
    否則傳回完整路徑（需另外處理非 static 下的檔案服務）。
    """
    if file_full:
        static_folder_abs = os.path.abspath(app.static_folder)
        file_full_abs = os.path.abspath(file_full)
        if file_full_abs.startswith(static_folder_abs):
            rel = os.path.relpath(file_full_abs, start=static_folder_abs)
            rel = rel.replace(os.path.sep, '/')
            return url_for('static', filename=rel)
    return file_full

@app.route('/')
def index():
    orig_names = get_filenames_without_ext(ORIG_DIR)
    pred_names = get_filenames_without_ext(PRED_DIR)
    count_orig = len(orig_names)
    count_pred = len(pred_names)
    pred_ratio = count_pred / count_orig if count_orig > 0 else 0

    # 計算 original 中存在但 predict 缺少的檔案
    missing_in_predict = sorted(list(orig_names - pred_names))
    
    # 檢查 Predict 中沒有檢測到圖（前景像素為 0）的檔案
    no_detection_files = []
    threshold = 128
    for name in pred_names:
        pred_file = find_file(PRED_DIR, name)
        if not pred_file:
            continue
        pred_path = os.path.join(PRED_DIR, pred_file)
        pred_img = Image.open(pred_path).convert('L')
        pred_array = np.array(pred_img) > threshold
        if pred_array.sum() == 0:
            no_detection_files.append(name)
    
    # 計算整體漏檢率與過殺率
    orig_names_for_metric = get_filenames_without_ext(ORIG_DIR)
    gt_names = get_filenames_without_ext(GT_DIR)
    all_metric_names = sorted(list(orig_names_for_metric & gt_names))
    
    total_target = 0
    total_no_target = 0
    missed_count = 0
    overkill_count = 0
    missed_files = []
    overkill_files = []
    
    for name in all_metric_names:
        gt_file = find_file(GT_DIR, name)
        if not gt_file:
            continue
        gt_path = os.path.join(GT_DIR, gt_file)
        gt_img = Image.open(gt_path).convert('L')
        gt_array = np.array(gt_img) > threshold

        pred_file = find_file(PRED_DIR, name)
        if pred_file:
            pred_path = os.path.join(PRED_DIR, pred_file)
            pred_img = Image.open(pred_path).convert('L')
            pred_array = np.array(pred_img) > threshold
        else:
            pred_array = np.zeros_like(np.array(gt_img))
        
        if gt_array.sum() > 0:
            total_target += 1
            if pred_array.sum() == 0:
                missed_count += 1
                missed_files.append(name)
        else:
            total_no_target += 1
            if pred_array.sum() > 0:
                overkill_count += 1
                overkill_files.append(name)
    
    overall_missed_rate = missed_count / total_target if total_target > 0 else 0
    overall_overkill_rate = overkill_count / total_no_target if total_no_target > 0 else 0

    # 針對所有共同檔名產生檔案列表與指標計算
    common_names = get_common_filenames()
    files = []
    total_iou = 0
    total_dice = 0
    for name in common_names:
        orig_file = find_file(ORIG_DIR, name)
        pred_file = find_file(PRED_DIR, name)
        gt_file   = find_file(GT_DIR, name)
        draw_file = find_file(DRAW_DIR, name)
        
        orig_full = os.path.join(ORIG_DIR, orig_file) if orig_file else None
        pred_full = os.path.join(PRED_DIR, pred_file) if pred_file else None
        gt_full   = os.path.join(GT_DIR, gt_file) if gt_file else None
        draw_full = os.path.join(DRAW_DIR, draw_file) if draw_file else None
        
        orig_url = get_url(orig_full)
        pred_url = get_url(pred_full)
        gt_url   = get_url(gt_full)
        draw_url = get_url(draw_full)
        
        pred_img = Image.open(os.path.join(PRED_DIR, pred_file)).convert('L')
        gt_img = Image.open(os.path.join(GT_DIR, gt_file)).convert('L')
        
        pred_array = np.array(pred_img) > threshold
        gt_array = np.array(gt_img) > threshold
        
        intersection = np.logical_and(pred_array, gt_array).sum()
        union = np.logical_or(pred_array, gt_array).sum()
        iou = intersection / union if union != 0 else 0
        
        pred_area = pred_array.sum()
        gt_area = gt_array.sum()
        dice = (2 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) != 0 else 0
        
        error_type = ""
        if gt_array.sum() > 0 and pred_array.sum() == 0:
            error_type = "漏檢"
        elif gt_array.sum() == 0 and pred_array.sum() > 0:
            error_type = "過殺"
        
        total_iou += iou
        total_dice += dice
        
        files.append({
            'filename': name,
            'orig_url': orig_url,
            'pred_url': pred_url,
            'gt_url': gt_url,
            'draw_url': draw_url,
            'iou': iou,
            'dice': dice,
            'error_type': error_type
        })
    
    count = len(files)
    avg_iou = total_iou / count if count > 0 else 0
    avg_dice = total_dice / count if count > 0 else 0

    return render_template('index.html',
                           files=files,
                           avg_iou=avg_iou,
                           avg_dice=avg_dice,
                           count_pred=count_pred,
                           count_orig=count_orig,
                           pred_ratio=pred_ratio,
                           no_detection_files=no_detection_files,
                           missing_in_predict=missing_in_predict,
                           overall_missed_rate=overall_missed_rate,
                           overall_overkill_rate=overall_overkill_rate,
                           missed_files=missed_files,
                           overkill_files=overkill_files,
                           original_path=ORIG_DIR,
                           predict_path=PRED_DIR,
                           ground_truth_path=GT_DIR,
                           draw_predict_path=DRAW_DIR)

@app.route('/update_paths', methods=['POST'])
def update_paths():
    new_paths = request.get_json()
    required_keys = ["original", "predict", "ground_truth", "draw_predict"]
    # 檢查是否提供所有必要欄位，若路徑不存在則嘗試建立
    for key in required_keys:
        if key not in new_paths:
            return jsonify({"success": False, "message": f"缺少 {key} 欄位"}), 400
        if not os.path.isdir(new_paths[key]):
            try:
                os.makedirs(new_paths[key], exist_ok=True)
            except Exception as e:
                return jsonify({"success": False, "message": f"建立 {key} 路徑失敗"}), 400
    try:
        with open("savepath.json", "w", encoding="utf8") as f:
            json.dump(new_paths, f, ensure_ascii=False, indent=2)
        # 更新全域變數
        global ORIG_DIR, PRED_DIR, GT_DIR, DRAW_DIR, paths
        paths = new_paths
        ORIG_DIR = paths["original"]
        PRED_DIR = paths["predict"]
        GT_DIR = paths["ground_truth"]
        DRAW_DIR = paths["draw_predict"]
        return jsonify({"success": True, "message": "路徑更新成功"})
    except Exception as e:
        return jsonify({"success": False, "message": "更新路徑失敗"}), 500

if __name__ == '__main__':
    app.run(debug=True)
