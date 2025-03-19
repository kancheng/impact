from flask import Flask, render_template, url_for
import os
from PIL import Image
import numpy as np

app = Flask(__name__)

# 設定四個目錄的路徑 (依需求修改)
ORIG_DIR = os.path.join(app.static_folder, 'original')
PRED_DIR = os.path.join(app.static_folder, 'predict')
GT_DIR   = os.path.join(app.static_folder, 'ground_truth')
DRAW_DIR = os.path.join(app.static_folder, 'draw_predict')

def get_filenames_without_ext(directory):
    # 只保留有效的圖像檔案
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
    for f in os.listdir(directory):
        # 只回傳有效的圖像檔案
        if os.path.splitext(f)[0] == base_name and os.path.splitext(f)[1].lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif'}:
            return f
    return None

@app.route('/')
def index():
    # 取得 original 與 predict 的檔案名稱
    orig_names = get_filenames_without_ext(ORIG_DIR)
    pred_names = get_filenames_without_ext(PRED_DIR)
    count_orig = len(orig_names)
    count_pred = len(pred_names)
    pred_ratio = count_pred / count_orig if count_orig > 0 else 0

    # 計算 original 中存在但 predict 缺少的檔案
    missing_in_predict = sorted(list(orig_names - pred_names))
    
    # 檢查 Predict 中沒有檢測到圖（前景像素為 0）的檔案
    no_detection_files = []
    for name in pred_names:
        pred_file = find_file(PRED_DIR, name)
        if pred_file is None:
            continue
        pred_path = os.path.join(PRED_DIR, pred_file)
        pred_img = Image.open(pred_path).convert('L')
        threshold = 128
        pred_array = np.array(pred_img) > threshold
        if pred_array.sum() == 0:
            no_detection_files.append(name)
    
    # 針對所有共同檔名產生檔案列表與指標計算 (僅取四個資料夾皆存在的檔案)
    common_names = get_common_filenames()
    files = []
    total_iou = 0
    total_dice = 0
    for name in common_names:
        orig_file = find_file(ORIG_DIR, name)
        pred_file = find_file(PRED_DIR, name)
        gt_file   = find_file(GT_DIR, name)
        draw_file = find_file(DRAW_DIR, name)
        
        orig_url = url_for('static', filename=f'original/{orig_file}')
        pred_url = url_for('static', filename=f'predict/{pred_file}')
        gt_url   = url_for('static', filename=f'ground_truth/{gt_file}')
        draw_url = url_for('static', filename=f'draw_predict/{draw_file}')
        
        pred_img = Image.open(os.path.join(PRED_DIR, pred_file)).convert('L')
        gt_img = Image.open(os.path.join(GT_DIR, gt_file)).convert('L')
        
        threshold = 128
        pred_array = np.array(pred_img) > threshold
        gt_array = np.array(gt_img) > threshold
        
        intersection = np.logical_and(pred_array, gt_array).sum()
        union = np.logical_or(pred_array, gt_array).sum()
        iou = intersection / union if union != 0 else 0
        
        pred_area = pred_array.sum()
        gt_area = gt_array.sum()
        dice = (2 * intersection) / (pred_area + gt_area) if (pred_area + gt_area) != 0 else 0
        
        total_iou += iou
        total_dice += dice
        
        files.append({
            'filename': name,
            'orig_url': orig_url,
            'pred_url': pred_url,
            'gt_url': gt_url,
            'draw_url': draw_url,
            'iou': iou,
            'dice': dice
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
                           missing_in_predict=missing_in_predict)

if __name__ == '__main__':
    app.run(debug=True)
