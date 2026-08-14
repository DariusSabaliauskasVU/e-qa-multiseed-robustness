import csv, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Serif','pdf.fonttype':42,'ps.fonttype':42})

# ---- data: F1 leaderboard ----
S=[13,42,73,101,123,202,333,777,1337,2024]
f1={}
for r in csv.reader(open('results_clean.csv')):
    if r[0]=='config': continue
    f1.setdefault(r[0],[]).append(float(r[5]))
mean={k:sum(v)/len(v) for k,v in f1.items()}
LAB={'neobert':'NeoBERT','modernbert':'ModernBERT','spanbert':'SpanBERT','bert':'BERT',
     'distilbert':'DistilBERT','partial_ft':'Partial FT','lstm_l3':'LSTM-3L','lstm_l1':'LSTM-1L',
     'lstm_l2':'LSTM-2L','linear':'BERT+FF'}
TIER={'neobert':2,'modernbert':2,'spanbert':2,'bert':2,'distilbert':2,'partial_ft':1,
      'lstm_l3':0,'lstm_l1':0,'lstm_l2':0,'linear':0}
order=sorted(mean, key=lambda k:mean[k])   # ascending for barh
COL={2:'#2E4C73',1:'#6A8CBF',0:'#B8CCE4'}
EDGE='#22384f'

fig=plt.figure(figsize=(11.6,4.64))   # 2.5:1 aspect ratio
fig.patch.set_facecolor('white')

# ============ TOP: pipeline flow ============
axf=fig.add_axes([0.0,0.66,1.0,0.34]); axf.set_xlim(0,100); axf.set_ylim(0,26); axf.axis('off')
stages=[
 ('Inputs','SQuAD v2 (answerable)\n+ RE-120 test set\n(120 Q–A pairs)'),
 ('10 configurations','Encoder adaptation:\nfrozen → partial → full FT\n× decoder type / LSTM depth'),
 ('Multi-seed protocol','10 random seeds each\nearly stopping\nEM / F1 / ROUGE-L'),
 ('Statistical analysis','Friedman + Holm-Wilcoxon\npaired d_z + r_rb + TOST\n+ Mann-Whitney sensitivity'),
]
n=len(stages); bw=20; gap=(100-n*bw)/(n+1)
for i,(t,b) in enumerate(stages):
    x=gap+i*(bw+gap)
    box=FancyBboxPatch((x,4),bw,18,boxstyle='round,pad=0.4,rounding_size=1.4',
        linewidth=1.2,edgecolor=EDGE,facecolor='#8FB0D9',alpha=0.93)
    axf.add_patch(box)
    axf.text(x+bw/2,19.2,t,ha='center',va='top',fontsize=10.5,fontweight='bold',color='#12212f')
    axf.text(x+bw/2,13.6,b,ha='center',va='top',fontsize=8.1,color='#1c1c1c')
    if i<n-1:
        axf.add_patch(FancyArrowPatch((x+bw,13),(x+bw+gap,13),arrowstyle='-|>',
            mutation_scale=15,linewidth=1.6,color=EDGE))

# ============ BOTTOM-LEFT: F1 leaderboard bars ============
axb=fig.add_axes([0.055,0.10,0.46,0.50])
labels=[LAB[k] for k in order]; vals=[mean[k] for k in order]; cols=[COL[TIER[k]] for k in order]
y=range(len(order))
axb.barh(list(y),vals,color=cols,edgecolor=EDGE,linewidth=0.6,height=0.72)
for yi,v in zip(y,vals):
    axb.text(v+0.006,yi,f'{v:.3f}',va='center',fontsize=7.4,color='#222')
axb.set_yticks(list(y)); axb.set_yticklabels(labels,fontsize=8)
axb.set_xlim(0,0.88); axb.set_xlabel('Mean F1 (10 seeds)',fontsize=8.5)
axb.tick_params(length=2,labelsize=7.8)
for sp in ['top','right']: axb.spines[sp].set_visible(False)
axb.set_title('Answer-span detection on RE-120',fontsize=9.5,fontweight='bold',loc='left',color='#12212f')
# tier legend in empty bottom-right of bar chart
for i,(lab,c) in enumerate([('full FT',COL[2]),('partial FT',COL[1]),('frozen',COL[0])]):
    xx=0.60+i*0.095
    axb.add_patch(plt.Rectangle((xx,0.9),0.022,0.5,facecolor=c,edgecolor=EDGE,lw=0.5,clip_on=False))
    axb.text(xx+0.03,1.15,lab,fontsize=7.0,va='center',color='#333')

# ============ BOTTOM-RIGHT: key findings ============
axk=fig.add_axes([0.55,0.10,0.42,0.50]); axk.set_xlim(0,10); axk.set_ylim(0,10); axk.axis('off')
kbox=FancyBboxPatch((0.1,0.2),9.8,9.6,boxstyle='round,pad=0.2,rounding_size=0.3',
    linewidth=1.1,edgecolor=EDGE,facecolor='#F2F6FB')
axk.add_patch(kbox)
axk.text(0.7,9.0,'Key findings',fontsize=10.5,fontweight='bold',color='#12212f')
pts=[
 'Encoder adaptation is the largest observed factor:\nfrozen ≈ 0.54 vs. fine-tuned 0.70–0.77 F1.',
 'Modern encoders (NeoBERT, ModernBERT) lead,\nbut are statistically indistinguishable.',
 'LSTM depth, and a feed-forward vs. LSTM head,\nare equivalent (TOST, ±0.05 F1).',
 'Evaluation-protocol choices can materially alter\narchitectural conclusions.',
]
yy=8.1
for p in pts:
    axk.text(0.55,yy,'•',fontsize=11,color='#2E4C73',va='top')
    axk.text(1.1,yy,p,fontsize=8.2,va='top',color='#1c1c1c')
    yy-=2.0

fig.savefig('Graphical_Abstract.pdf'); fig.savefig('Graphical_Abstract.png',dpi=300)
plt.close(fig)
from PIL import Image
print('Graphical_Abstract size px (WxH):', Image.open('Graphical_Abstract.png').size)
