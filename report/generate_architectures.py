"""Draw vector architecture diagrams for the CIFAR-sized implemented models.

Run from the repository root: python -m report.generate_architectures
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', 'outputs/.matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES = Path(__file__).resolve().parent / 'assets/figures'
INK = '#24364b'
BLUE = '#e6f0fa'
GREEN = '#e5f3e9'
YELLOW = '#fff0cd'


def node(ax, x, y, label, color=BLUE, width=1.7, height=0.95, size=11):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                              boxstyle='round,pad=0.025,rounding_size=0.07',
                              facecolor=color, edgecolor=INK, linewidth=1.1))
    ax.text(x, y, label, ha='center', va='center', fontsize=size,
            color=INK, linespacing=1.35)


def arrow(ax, start, end, color=INK, style='-'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=12, linewidth=1.2,
                                color=color, linestyle=style))


def canvas(title, height=4.5):
    fig, ax = plt.subplots(figsize=(12, height))
    ax.set(xlim=(-0.1, 12.1), ylim=(0, height))
    ax.axis('off')
    ax.text(6, height-0.28, title, ha='center', va='center', fontsize=16,
            fontweight='bold', color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES/f'{name}_architecture.pdf', bbox_inches='tight', pad_inches=0.08)
    fig.savefig(FIGURES/f'{name}_architecture.png', dpi=170, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)


def feature_flow(ax, labels):
    xs = [1, 3, 5, 7, 9, 11]
    for i, (x, label) in enumerate(zip(xs, labels)):
        node(ax, x, 3.25, label, YELLOW if i == 0 else BLUE)
        if i:
            arrow(ax, (xs[i-1]+0.89,3.25), (x-0.89,3.25))
    # The classifier continues from the right-most feature stage, right to left.
    arrow(ax, (11,2.72), (11,2.08))


def classifier_flow(ax, labels):
    xs = [10.1, 6, 1.9]
    for i, (x, label) in enumerate(zip(xs, labels)):
        node(ax, x, 1.5, label, GREEN, width=3.35, height=1.0, size=12)
        if i:
            arrow(ax, (xs[i-1]-1.74,1.5), (x+1.74,1.5))


def alexnet():
    fig, ax = canvas('AlexNet CIFAR-10 với GroupNorm')
    feature_flow(ax, [
        'Ảnh RGB\n3 × 32 × 32',
        'Conv 5 × 5, 64\nGN + ReLU + MP\n64 × 16 × 16',
        'Conv 3 × 3, 192\nGN + ReLU + MP\n192 × 8 × 8',
        'Conv 3 × 3, 384\nGN + ReLU\n384 × 8 × 8',
        'Conv 3 × 3, 256\nGN + ReLU\n256 × 8 × 8',
        'Conv 3 × 3, 256\nGN + ReLU + MP\n256 × 4 × 4',
    ])
    classifier_flow(ax, [
        'Adaptive AvgPool 2 × 2\n256 × 2 × 2 → Flatten\n1.024 đặc trưng',
        'Dropout 0,5\nLinear 1.024 → 512\nReLU',
        'Dropout 0,5\nLinear 512 → 10\n10 logits',
    ])
    ax.text(6,0.35,'GN: GroupNorm, 8 nhóm  •  MP: MaxPool 2 × 2, stride 2  •  Conv: stride 1, padding giữ kích thước',
            ha='center',fontsize=11,color=INK)
    save(fig,'alexnet')


def vgg():
    fig, ax = canvas('VGG11-BN CIFAR-10 với số kênh giảm')
    feature_flow(ax, [
        'Ảnh RGB\n3 × 32 × 32',
        'Stage 1\nConv 32 + MP\n32 × 16 × 16',
        'Stage 2\nConv 64 + MP\n64 × 8 × 8',
        'Stage 3\nConv 128 × 2 + MP\n128 × 4 × 4',
        'Stage 4\nConv 256 × 2 + MP\n256 × 2 × 2',
        'Stage 5\nConv 256 × 2 + MP\n256 × 1 × 1',
    ])
    classifier_flow(ax, ['Flatten\n256 đặc trưng','Linear 256 → 10\nKhông dropout','10 logits'])
    ax.text(6,0.35,'Mỗi Conv: kernel 3 × 3, stride 1, padding 1 → BatchNorm → ReLU  •  MP: MaxPool 2 × 2, stride 2',
            ha='center',fontsize=11,color=INK)
    save(fig,'vgg11')


def resnet():
    fig, ax = canvas('ResNet18 CIFAR-10 với base width 32',height=5.8)
    feature_flow(ax, [
        'Ảnh RGB\n3 × 32 × 32',
        'Stem: Conv 3 × 3\nBN + ReLU, s = 1\n32 × 32 × 32',
        'Stage 1: 2 block\n32 kênh, s = 1\n32 × 32 × 32',
        'Stage 2: 2 block\n64 kênh, s = 2\n64 × 16 × 16',
        'Stage 3: 2 block\n128 kênh, s = 2\n128 × 8 × 8',
        'Stage 4: 2 block\n256 kênh, s = 2\n256 × 4 × 4',
    ])
    classifier_flow(ax, ['Global AvgPool\n256 × 1 × 1 → Flatten\n256 đặc trưng','Linear 256 → 10','10 logits'])
    # Keep the block detail above the main sequence so the residual bypass is explicit.
    ax.text(0.1,4.82,'Một BasicBlock:',fontsize=11,color=INK,fontweight='bold')
    small=[(2.5,'x'),(4.3,'Conv 3 × 3\nBN → ReLU'),(6.5,'Conv 3 × 3\nBN'),(8.3,'+'),(10.4,'ReLU → y')]
    for x,label in small:
        node(ax,x,4.6,label,GREEN if label=='+' else BLUE,width=1.55 if label not in ('x','+') else 0.5,height=0.7,size=10)
    for a,b in zip(small,small[1:]):
        left_width=0.25 if a[1] in ('x','+') else 0.775
        right_width=0.25 if b[1] in ('x','+') else 0.775
        arrow(ax,(a[0]+left_width+0.03,4.6),(b[0]-right_width-0.03,4.6))
    ax.plot([2.5,2.5,8.3],[4.98,5.18,5.18],color='#9a6414',linewidth=1.2)
    arrow(ax,(8.3,5.18),(8.3,4.98),color='#9a6414')
    ax.text(5.65,5.24,'S(x): identity hoặc Conv 1 × 1 + BN',ha='center',fontsize=10,color='#9a6414')
    ax.text(6,0.35,'s: stride của Conv đầu ở block đầu mỗi stage; các block còn lại dùng s = 1  •  BN: BatchNorm',
            ha='center',fontsize=11,color=INK)
    save(fig,'resnet18')


def vit():
    fig, ax = canvas('ViT tiny CIFAR-10: patch 4, embedding 192, 6 block',height=5.6)
    labels=[
        'Ảnh RGB\n3 × 32 × 32',
        'Patch embedding\nConv 4 × 4, s = 4\n192 × 8 × 8',
        'Flatten + đổi trục\n64 patch token\n64 × 192',
        '+ token [CLS]\n+ vị trí học được\n65 × 192',
        'Encoder × 6\n3 attention head\n65 × 192',
        'LN trên [CLS]\nLinear 192 → 10\n10 logits',
    ]
    xs=[1,3,5,7,9,11]
    for i,(x,label) in enumerate(zip(xs,labels)):
        node(ax,x,4.25,label,YELLOW if i==0 else GREEN if i==5 else BLUE,size=11)
        if i: arrow(ax,(xs[i-1]+0.89,4.25),(x-0.89,4.25))
    ax.add_patch(FancyBboxPatch((0.15,0.55),11.7,2.4,boxstyle='round,pad=0.02',
                              facecolor='#f7f9fc',edgecolor='#8392a5',linestyle='--'))
    ax.text(6,2.68,'Chi tiết một encoder block (pre-norm; shape giữ nguyên 65 × 192)',
            ha='center',fontsize=12,color=INK,fontweight='bold')
    arrow(ax,(9,3.74),(9,3.01),color='#8392a5',style='--')
    steps=[(1.2,'LayerNorm'),(3.1,'MSA\n3 head × 64'),(4.7,'+'),
           (6.1,'LayerNorm'),(8.25,'MLP + GELU\n192 → 768 → 192'),(10.45,'+')]
    widths=[1.6,1.7,0.55,1.6,2.2,0.55]
    for (x,label),w in zip(steps,widths):
        node(ax,x,1.45,label,GREEN if label=='+' else BLUE,width=w,height=0.8,size=11)
    for i in range(len(steps)-1):
        arrow(ax,(steps[i][0]+widths[i]/2+0.03,1.45),(steps[i+1][0]-widths[i+1]/2-0.03,1.45))
    arrow(ax,(0.25,1.45),(0.38,1.45))
    for start,end in ((0.27,4.7),(5.12,10.45)):
        ax.plot([start,start,end],[1.45,2.13,2.13],color='#9a6414',linewidth=1.2)
        arrow(ax,(end,2.13),(end,1.9),color='#9a6414')
    ax.text(2.5,2.21,'Nhánh phần dư',ha='center',fontsize=10,color='#9a6414')
    ax.text(8.0,2.21,'Nhánh phần dư',ha='center',fontsize=10,color='#9a6414')
    arrow(ax,(10.76,1.45),(11.6,1.45))
    ax.text(6,0.24,'LN: LayerNorm  •  MSA: multi-head self-attention  •  Không dropout trong encoder',
            ha='center',fontsize=11,color=INK)
    save(fig,'vit_tiny')


def architectures():
    alexnet(); vgg(); resnet(); vit()


if __name__=='__main__':
    architectures()
