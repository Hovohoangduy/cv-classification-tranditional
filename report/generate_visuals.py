"""Build report pipeline and prediction galleries from saved CIFAR-10 checkpoints.

Run from the repository root: python -m report.generate_visuals
Training artifacts and the full-test comparison metrics are never overwritten.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', 'outputs/.matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import torch
from deep_models.models import create_model

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'report/assets/figures'
RESULTS = ROOT / 'report/assets/results'
NAMES = ('alexnet', 'vgg11', 'resnet18', 'vit_tiny')
DISPLAY = ('AlexNet (GroupNorm)', 'VGG11-BN', 'ResNet18', 'ViT tiny')
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set(xlim=(0, 12), ylim=(-0.4, 10.8))
    ax.axis('off')
    nodes = {}

    def box(key, x, y, width, height, content, color):
        nodes[key] = (x, y, width, height)
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                                   boxstyle='round,pad=0.07,rounding_size=0.12',
                                   facecolor=color, edgecolor='#334155', linewidth=1.2))
        ax.text(x, y, content, ha='center', va='center', fontsize=12, linespacing=1.3)

    def arrow(start, end, dashed=False):
        x, y, w, h = nodes[start]; xx, yy, ww, hh = nodes[end]
        ax.add_patch(FancyArrowPatch((x, y-h/2-0.06), (xx, yy+hh/2+0.06),
                                    arrowstyle='-|>', mutation_scale=14,
                                    color='#475569', linewidth=1.3,
                                    linestyle='--' if dashed else '-'))

    box('data', 6, 10.15, 5.3, 0.7, 'CIFAR-10: 60.000 ảnh RGB 32 × 32 / 10 lớp', '#e2e8f0')
    box('split', 3.15, 8.85, 5.0, 0.9, '50.000 ảnh train chính thức → chia phân tầng (seed 42)\n45.000 train: học trọng số / 5.000 val: chọn checkpoint', '#dbeafe')
    box('test', 9.15, 8.85, 4.6, 0.9, '10.000 ảnh test chính thức\nGiữ riêng cho đánh giá cuối', '#fef3c7')
    box('stats', 6, 7.45, 9.4, 0.9, 'Tiền xử lý: chỉ học thống kê chuẩn hóa từ tập train\nGiữ nguyên thống kê khi xử lý validation và test', '#e2e8f0')
    box('classic', 3.15, 5.95, 4.9, 1.0, 'NHÁNH TRUYỀN THỐNG\nPixel (3.072) / HOG (1.764) / SIFT phẳng (8.192)\nChuẩn hóa từng chiều đặc trưng', '#e0f2fe')
    box('deep', 8.85, 5.95, 4.9, 1.0, 'NHÁNH HỌC SÂU\nẢnh RGB chuẩn hóa theo kênh, không augmentation\nAlexNet / VGG11-BN / ResNet18 / ViT tiny', '#dcfce7')
    box('classictrain', 3.15, 4.4, 4.9, 0.9, 'Huấn luyện trên 45.000 ảnh\nLinear SVM: 200 epoch / Random Forest: 200 cây', '#e0f2fe')
    box('deeptrain', 8.85, 4.4, 4.9, 0.9, 'Huấn luyện trên 45.000 ảnh: 200 epoch\nCross-entropy → backpropagation → AdamW', '#dcfce7')
    box('classicmodel', 3.15, 3.0, 4.9, 0.8, 'Cố định bộ chuẩn hóa và mô hình baseline\nDùng cấu hình đã chạy, không chọn bằng test', '#e0f2fe')
    box('checkpoint', 8.85, 3.0, 4.9, 0.8, 'Đánh giá validation sau mỗi epoch\nLưu checkpoint có validation accuracy cao nhất', '#dcfce7')
    box('eval', 6, 1.6, 9.4, 0.9, 'Suy luận trên test bằng mô hình đã cố định\nHọc sâu: model.eval() + inference_mode(); baseline: predict()', '#fef3c7')
    box('result', 6, 0.15, 9.4, 0.9, 'Kết quả: accuracy / macro-F1 / confusion matrix\nBảng so sánh 10 cấu hình + biểu đồ học + ảnh dự đoán minh họa', '#e2e8f0')
    for a,b in [('data','split'),('data','test'),('split','stats'),('stats','classic'),
                ('stats','deep'),('classic','classictrain'),('deep','deeptrain'),
                ('classictrain','classicmodel'),('deeptrain','checkpoint'),
                ('classicmodel','eval'),('checkpoint','eval'),('eval','result')]:
        arrow(a,b)
    # Explicit held-out route, separate from all training nodes.
    ax.plot([11.5,11.8,11.8,11.2], [8.85,8.85,1.6,1.6], linestyle='--', color='#b45309', linewidth=1.5)
    ax.add_patch(FancyArrowPatch((11.2,1.6),(10.8,1.6),arrowstyle='-|>',mutation_scale=14,color='#b45309'))
    ax.text(11.66,5.5,'Test chỉ đi vào đánh giá',rotation=90,ha='center',va='center',fontsize=9,color='#92400e',backgroundcolor='white')
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES/'pipeline_overview.png',dpi=240)
    plt.close(fig)


def gallery(images, labels, ids, predictions, scores, title, destination, columns=5):
    fig, axes = plt.subplots(2, columns, figsize=(10.5, 5.3))
    for ax, idx in zip(axes.flat, ids):
        prediction = int(predictions[idx]); target = int(labels[idx])
        ax.imshow(images[idx], interpolation='nearest')
        color = '#166534' if prediction == target else '#b91c1c'
        ax.set_title(f'Test #{idx} | True: {CLASSES[target]}',fontsize=9)
        ax.set_xlabel(f'Pred: {CLASSES[prediction]}\np(top-1): {scores[idx]:.1%}',color=color,fontsize=9,labelpad=3)
        ax.set(xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_color(color);spine.set_linewidth(2)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout(rect=(0,0.02,1,0.93),h_pad=3.4,w_pad=0.8)
    fig.savefig(destination,dpi=220)
    plt.close(fig)


def predictions() -> None:
    torch.set_num_threads(4)
    with (ROOT/'data/cifar-10-batches-py/test_batch').open('rb') as file:
        batch=pickle.load(file,encoding='bytes')
    images=batch[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    labels=np.asarray(batch[b'labels'])
    count=500
    shared=[int(np.flatnonzero(labels[:count]==label)[0]) for label in range(10)]
    manifest={'selection': 'First test image of each class in official file order; same ten images for all four networks.',
              'error_selection': 'First ten AlexNet errors among official test indices 0..499, in ascending index order.',
              'candidate_test_indices': [0,count-1], 'inference_device':'cpu', 'class_names':CLASSES,
              'models':{}}
    for name,title in zip(NAMES,DISPLAY):
        path=ROOT/'outputs/deep_models_200_gn'/f'{name}.pt'
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        metrics=json.loads(path.with_name(f'{name}_metrics.json').read_text())
        assert checkpoint['epoch']==metrics['best_epoch'] and checkpoint['model_name']==name
        assert tuple(checkpoint['class_names'])==CLASSES
        model=create_model(name);model.load_state_dict(checkpoint['model'],strict=True);model.eval()
        assert sum(p.numel() for p in model.parameters())==metrics['parameters']
        mean=torch.tensor(checkpoint['mean']).view(1,3,1,1)
        std=torch.tensor(checkpoint['std']).view(1,3,1,1)
        assert np.allclose(checkpoint['mean'],metrics['normalization']['mean'])
        assert np.allclose(checkpoint['std'],metrics['normalization']['std'])
        probs=[]
        with torch.inference_mode():
            for first in range(0,count,64):
                inputs=torch.from_numpy(np.ascontiguousarray(images[first:min(count,first+64)])).permute(0,3,1,2).float()/255
                probs.append(model((inputs-mean)/std).softmax(1).numpy())
        probs=np.concatenate(probs);pred=probs.argmax(1);score=probs.max(1)
        np.savez_compressed(RESULTS/f'{name}_predict_examples.npz',test_indices=np.arange(count),
                            y_true=labels[:count],y_pred=pred,probabilities=probs)
        gallery(images,labels,shared,pred,score,title,FIGURES/f'{name}_predictions.png')
        errors=np.flatnonzero(pred!=labels[:count])[:10].tolist() if name=='alexnet' else []
        if errors:
            assert len(errors)==10
            gallery(images,labels,errors,pred,score,'AlexNet: first 10 errors in test #0–499',FIGURES/'alexnet_prediction_errors.png')
        def record(idx):
            return {'test_index':idx,'actual':CLASSES[int(labels[idx])], 'predicted':CLASSES[int(pred[idx])],
                    'top1_probability':float(score[idx]), 'correct':bool(pred[idx]==labels[idx])}
        manifest['models'][name]={'checkpoint':str(path.relative_to(ROOT)), 'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'epoch':checkpoint['epoch'],'shared_examples':[record(idx) for idx in shared],
                                  'error_examples':[record(idx) for idx in errors]}
        print(f'{name}: rendered examples, checkpoint epoch {checkpoint["epoch"]}',flush=True)
    (RESULTS/'prediction_examples.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    FIGURES.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    pipeline();predictions()
