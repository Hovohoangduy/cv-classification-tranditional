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
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set(xlim=(-0.1, 10.4), ylim=(0.35, 6.85))
    ax.axis('off')
    nodes = {}

    def box(key, x, y, width, height, content, color):
        nodes[key] = (x, y, width, height)
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                                   boxstyle='round,pad=0.06,rounding_size=0.10',
                                   facecolor=color, edgecolor='#334155', linewidth=1.2))
        ax.text(x, y, content, ha='center', va='center', fontsize=13, linespacing=1.35)

    def arrow(start, end):
        x, y, w, h = nodes[start]; xx, yy, ww, hh = nodes[end]
        ax.add_patch(FancyArrowPatch((x, y-h/2-0.07), (xx, yy+hh/2+0.07),
                                    arrowstyle='-|>', mutation_scale=14,
                                    color='#475569', linewidth=1.3))

    box('data', 4.4, 6.15, 7.5, 0.85,
        'CIFAR-10\nHuấn luyện: 45.000 ảnh  |  Kiểm định: 5.000 ảnh', '#e2e8f0')
    box('test', 9.0, 4.75, 2.1, 0.85, 'Kiểm tra\n10.000 ảnh', '#fef3c7')
    box('normalize', 4.4, 4.75, 6.4, 0.85,
        'Chuẩn hóa dữ liệu đầu vào', '#e2e8f0')
    box('classic', 2.3, 3.0, 3.9, 1.15,
        'Truyền thống\nPixel / HOG / SIFT\nSVM / Random Forest', '#e0f2fe')
    box('deep', 6.5, 3.0, 3.9, 1.15,
        'Học sâu\nAlexNet / VGG / ResNet / ViT\nChọn mô hình trên tập kiểm định', '#dcfce7')
    box('result', 4.4, 1.15, 8.3, 0.95,
        'Đánh giá trên tập kiểm tra\nĐộ chính xác  •  Macro-F1  •  Ma trận nhầm lẫn', '#fef3c7')
    for start, end in [('data', 'normalize'), ('normalize', 'classic'),
                       ('normalize', 'deep'), ('classic', 'result'), ('deep', 'result')]:
        arrow(start, end)
    # Route held-out test directly to evaluation.
    ax.add_patch(FancyArrowPatch((8.18,6.15),(9.0,5.25),arrowstyle='-|>',
                                mutation_scale=14,color='#b45309',linewidth=1.3))
    ax.plot([9.0,9.0],[4.25,1.15],linestyle='--',color='#b45309',linewidth=1.3)
    ax.add_patch(FancyArrowPatch((9.0,1.15),(8.62,1.15),arrowstyle='-|>',
                                mutation_scale=14,color='#b45309',linewidth=1.3,linestyle='--'))
    fig.tight_layout(pad=0.2)
    fig.savefig(FIGURES/'pipeline_overview.png',dpi=240,bbox_inches='tight',pad_inches=0.06)
    plt.close(fig)


def representative_indices(labels, predictions, losses):
    """Select the middle loss-ranked example in each correctness group."""
    selected = []
    for mask in (predictions == labels, predictions != labels):
        group = np.flatnonzero(mask)
        if not len(group):
            raise ValueError('Representative gallery requires both correct and incorrect predictions')
        ordered = group[np.lexsort((group, losses[group]))]
        selected.append(int(ordered[len(ordered) // 2]))
    return selected


def gallery(images, labels, ids, predictions, scores, title, destination):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.5))
    for ax, idx in zip(axes.flat, ids):
        prediction = int(predictions[idx]); target = int(labels[idx])
        ax.imshow(images[idx], interpolation='nearest')
        color = '#166534' if prediction == target else '#b91c1c'
        state = 'Dự đoán đúng' if prediction == target else 'Dự đoán sai'
        ax.set_title(f'{state} | Test #{idx}\nNhãn thật: {CLASSES[target]}',color=color,fontsize=10)
        ax.set_xlabel(f'Dự đoán: {CLASSES[prediction]}\nSoftmax: {scores[idx]:.1%}',color=color,fontsize=10,labelpad=3)
        ax.set(xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_color(color);spine.set_linewidth(2)
    fig.suptitle(title, fontsize=12, fontweight='bold')
    fig.tight_layout(rect=(0,0,1,0.90),w_pad=1.5)
    fig.savefig(destination,dpi=220,bbox_inches='tight',pad_inches=0.08)
    plt.close(fig)


def predictions() -> None:
    torch.set_num_threads(4)
    with (ROOT/'data/cifar-10-batches-py/test_batch').open('rb') as file:
        batch=pickle.load(file,encoding='bytes')
    images=batch[b'data'].reshape(-1,3,32,32).transpose(0,2,3,1)
    labels=np.asarray(batch[b'labels'])
    count=len(labels)
    manifest={'selection': 'Per model: one correct and one incorrect illustrative example. Sort each correctness group by ascending cross-entropy, breaking ties by ascending test index, and select the middle example (position len(group)//2).',
              'ranking': 'Stable float64 cross-entropy from inference logits using log1p for near-zero loss; no class balancing or manual selection.',
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
        losses=[]
        with torch.inference_mode():
            for first in range(0,count,64):
                inputs=torch.from_numpy(np.ascontiguousarray(images[first:min(count,first+64)])).permute(0,3,1,2).float()/255
                logits=model((inputs-mean)/std).double()
                log_probs=logits.log_softmax(1)
                probs.append(log_probs.exp().numpy())
                targets=torch.from_numpy(labels[first:min(count,first+64)])
                # Ordinary log_softmax can round tiny correct-example losses to
                # zero even in float64. log1p preserves their ranking.
                maximum, top_class=logits.max(1)
                alternatives=(logits-maximum[:,None]).exp()
                alternatives[torch.arange(len(targets)),top_class]=0
                loss=(maximum-logits[torch.arange(len(targets)),targets]
                      +torch.log1p(alternatives.sum(1)))
                losses.append(loss.numpy())
                if first % 2048 == 0:
                    print(f'{name}: inference {first}/{count}',flush=True)
        probs=np.concatenate(probs);pred=probs.argmax(1);score=probs.max(1)
        losses=np.concatenate(losses)
        selected=representative_indices(labels,pred,losses)
        assert np.isfinite(losses).all()
        np.savez_compressed(RESULTS/f'{name}_predict_examples.npz',test_indices=np.arange(count),
                            y_true=labels,y_pred=pred,probabilities=probs,per_image_loss=losses)
        gallery(images,labels,selected,pred,score,title,FIGURES/f'{name}_predictions.png')
        def record(idx):
            return {'test_index':idx,'actual':CLASSES[int(labels[idx])], 'predicted':CLASSES[int(pred[idx])],
                    'top1_probability':float(score[idx]), 'true_label_probability':float(probs[idx,labels[idx]]),
                    'cross_entropy':float(losses[idx]), 'correct':bool(pred[idx]==labels[idx])}
        matrix=np.zeros((10,10),dtype=int)
        np.add.at(matrix,(labels,pred),1)
        manifest['models'][name]={'checkpoint':str(path.relative_to(ROOT)), 'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                  'epoch':checkpoint['epoch'],'correct_examples':[record(selected[0])],
                                  'error_examples':[record(selected[1])],
                                  'cpu_test_accuracy':float(np.mean(pred==labels)),
                                  'cpu_matrix_matches_recorded_test':bool(np.array_equal(matrix,metrics['test']['confusion_matrix']))}
        print(f'{name}: rendered examples, checkpoint epoch {checkpoint["epoch"]}',flush=True)
    (RESULTS/'prediction_examples.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':
    FIGURES.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    from report.generate_architectures import architectures
    from report.generate_dataset_samples import dataset_samples
    architectures()
    dataset_samples()
    pipeline();predictions()
