import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Canvas ──────────────────────────────────────────────
W, H = 700, 960
CX = W / 2

fig, ax = plt.subplots(figsize=(7.0, 9.6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis('off')
fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.005)

# ── Colors ──────────────────────────────────────────────
TEXT    = '#e6edf3'
DIM     = '#6e7681'
SUB     = '#848d97'
BOX_BG  = '#151b23'
BORDER  = '#2d333b'
GREEN   = '#3fb950'
CYAN    = '#58a6ff'
PURPLE  = '#bc8cff'
GOLD    = '#d29922'
ORANGE  = '#f0883e'
PILL_BG = '#1c2128'


# ── Helpers ─────────────────────────────────────────────
def rbox(x, y, w, h, ec=BORDER, fc=BOX_BG, lw=1.5):
    p = FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={min(10, h/3)}",
        fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)

def txt(x, y, s, c=TEXT, sz=11, w='bold', ha='center'):
    ax.text(x, y, s, ha=ha, va='center',
            fontsize=sz, color=c, fontweight=w, fontfamily='monospace', zorder=3)

def pill(cx, cy, s, color, sz=7.5):
    tw = max(len(s) * 6.5 + 16, 42)
    th = 22
    rbox(cx - tw/2, cy - th/2, tw, th, ec=color, fc=PILL_BG, lw=1.1)
    txt(cx, cy, s, c=color, sz=sz)

def arrow(x1, y1, x2, y2, c=DIM):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle='->', color=c, lw=1.3), zorder=1)

def line(x1, y1, x2, y2, c=DIM, lw=1.3):
    ax.plot([x1, x2], [y1, y2], color=c, lw=lw, zorder=1)


# ================================================================
#  TITLE
# ================================================================
txt(CX, 32, 'StereoRCnet  SELD', sz=19)
txt(CX, 56, 'Stereo  \u00b7  multiACCDOA', c=SUB, sz=10, w='normal')


# ================================================================
#  AUDIO INPUT
# ================================================================
y = 85;  bw = 420;  bh = 55
rbox(CX - bw/2, y, bw, bh)
txt(CX, y + 19, 'Audio Input', sz=12)
txt(CX, y + 39, 'B \u00d7 2 \u00d7 251 \u00d7 64', c=DIM, sz=9, w='normal')
y += bh
arrow(CX, y + 3, CX, y + 25)


# ================================================================
#  MCSANET ENCODER  (section header)
# ================================================================
y += 35
txt(CX, y, 'M C S A N E T   E N C O D E R', c=GREEN, sz=9)
txt(CX, y + 17, 'shared weights  \u00b7  freq-only pooling', c=DIM, sz=7.5, w='normal')
y += 30

# ── split lines ────────────────────────────────────────
gap   = 30
bw_c  = 260                         # column width
lx0   = CX - gap/2 - bw_c           # left box x
rx0   = CX + gap/2                   # right box x
lcx   = lx0 + bw_c / 2              # left center
rcx   = rx0 + bw_c / 2              # right center

y_box = y + 20
line(CX, y, lcx, y_box - 3, c=GREEN)
line(CX, y, rcx, y_box - 3, c=GREEN)


# ================================================================
#  TWO MCSANet COLUMNS
# ================================================================
box_h = 195
rbox(lx0, y_box, bw_c, box_h, ec=GREEN)
rbox(rx0, y_box, bw_c, box_h, ec=GREEN)

# channel labels
txt(lcx, y_box + 18, 'Left Channel',  c=CYAN, sz=9)
txt(rcx, y_box + 18, 'Right Channel', c=CYAN, sz=9)

# block rows
blocks = [
    ('Block 1\u219264',   'pool 1\u00d74'),
    ('Block 64\u2192128', 'pool 1\u00d74'),
    ('Block 128\u2192256','pool 1\u00d72'),
    ('Block 256\u2192256','     \u2014'),
]
by = y_box + 46
for label_t, pool_t in blocks:
    for cx_col, x0_col in [(lcx, lx0), (rcx, rx0)]:
        txt(x0_col + 14,          by, label_t, c=TEXT, sz=8, w='normal', ha='left')
        txt(x0_col + bw_c - 14,   by, pool_t,  c=DIM, sz=7.5, w='normal', ha='right')
    by += 30

# MCSA annotation
txt(lcx, by + 10, '+ MCSA attention', c=GREEN, sz=7.5, w='normal')
txt(rcx, by + 10, '+ MCSA attention', c=GREEN, sz=7.5, w='normal')


# ================================================================
#  MERGE  \u2192  AFF FUSION
# ================================================================
y = y_box + box_h
line(lcx, y + 3, CX, y + 28, c=PURPLE)
line(rcx, y + 3, CX, y + 28, c=PURPLE)

y += 34
bw = 430;  bh = 50
rbox(CX - bw/2, y, bw, bh, ec=PURPLE)
txt(CX, y + 17, 'Attentional Feature Fusion', c=PURPLE, sz=10)
txt(CX, y + 36, 'w \u00b7 Left + (1\u2212w) \u00b7 Right', c=DIM, sz=8, w='normal')
y += bh

# dimension
y += 12
txt(CX, y, 'B \u00d7 251 \u00d7 512', c=DIM, sz=9, w='normal')
arrow(CX, y + 8, CX, y + 28)


# ================================================================
#  LINEAR EMBEDDING
# ================================================================
y += 34
bw = 420;  bh = 42
rbox(CX - bw/2, y, bw, bh, ec=CYAN)
txt(CX, y + 14, 'Linear  512 \u2192 256', c=CYAN, sz=11)
txt(CX, y + 32, 'embedding projection', c=DIM, sz=8, w='normal')
y += bh
arrow(CX, y + 3, CX, y + 22)


# ================================================================
#  SINUSOIDAL PE
# ================================================================
y += 28
bw = 420;  bh = 38
rbox(CX - bw/2, y, bw, bh, ec=PURPLE)
txt(CX, y + 19, 'Sinusoidal Pos. Encoding', c=PURPLE, sz=10)
y += bh

# dimension
y += 10
txt(CX, y, 'B \u00d7 251 \u00d7 256', c=DIM, sz=9, w='normal')
arrow(CX, y + 8, CX, y + 28)


# ================================================================
#  CONFORMER \u00d74
# ================================================================
y += 38
txt(CX, y, 'C O N F O R M E R   \u00d7 4', c=GOLD, sz=9)
y += 18
bw = 530;  bh = 40
rbox(CX - bw/2, y, bw, bh, ec=GOLD)

pills_data = ['FF\u00bd', 'MHSA-8h', 'DWConv-31', 'FF\u00bd', 'LN']
pw = bw / (len(pills_data) + 0.4)
for i, pt in enumerate(pills_data):
    px = CX - bw/2 + pw * (i + 0.7)
    pill(px, y + bh/2, pt, GOLD, sz=7)
y += bh
arrow(CX, y + 5, CX, y + 25)


# ================================================================
#  T-POOL
# ================================================================
y += 32
bw = 420;  bh = 38
rbox(CX - bw/2, y, bw, bh, ec=GREEN)
txt(CX, y + 19, 'T-Pool  251 \u2192 50', c=GREEN, sz=11)
y += bh
arrow(CX, y + 3, CX, y + 22)


# ================================================================
#  OUTPUT HEAD
# ================================================================
y += 28
bw = 470;  bh = 55
rbox(CX - bw/2, y, bw, bh, ec=ORANGE, fc='#1c1510')
txt(CX, y + 18, 'Output Head \u2192 Tanh / ReLU', c=ORANGE, sz=11)
txt(CX, y + 38, 'B \u00d7 50 \u00d7 117', c=DIM, sz=9, w='normal')


# ── Save ────────────────────────────────────────────────
out = '/home/harry/datasets/DCASE2025_SELD_dataset/StereoRCnet_architecture.png'
plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Saved to {out}")
