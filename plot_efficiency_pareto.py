import csv, math, matplotlib
from pathlib import Path
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})

S=[13,42,73,101,123,202,333,777,1337,2024]
f1=csv_tm={}
F1={}; TM={}
for r in csv.reader(open(Path(__file__).with_name('results_clean.csv'), newline='', encoding='utf-8')):
    if r[0]=='config': continue
    F1.setdefault(r[0],[]).append(float(r[5])); TM.setdefault(r[0],[]).append(float(r[3]))
mean=lambda d,c: sum(d[c])/len(d[c])
LAB={'neobert':'NeoBERT','modernbert':'ModernBERT','spanbert':'SpanBERT','bert':'BERT',
     'distilbert':'DistilBERT','partial_ft':'Partial FT','lstm_l3':'LSTM 3L','lstm_l1':'LSTM 1L',
     'lstm_l2':'LSTM 2L','linear':'Frozen+FF'}
E2E=['neobert','modernbert','spanbert','bert','distilbert','partial_ft']   # end-to-end
CACHED=['lstm_l3','lstm_l1','lstm_l2','linear']

fig,ax=plt.subplots(figsize=(7.6,7.6/1.5))
# end-to-end points + Pareto frontier
def pts(keys): return [(mean(TM,c),mean(F1,c),c) for c in keys]
e2e=pts(E2E); cac=pts(CACHED)
ax.scatter([t for t,f,c in e2e],[f for t,f,c in e2e],s=55,c='#2E4C73',edgecolor='k',lw=0.5,zorder=3,label='End-to-end fine-tuned')
ax.scatter([t for t,f,c in cac],[f for t,f,c in cac],s=55,c='#B8CCE4',edgecolor='k',lw=0.5,zorder=3,marker='s',label='Cached-representation head (time excludes one-time cache)')
# Pareto frontier over end-to-end (maximise F1, minimise time)
fr=sorted(e2e,key=lambda x:x[0]); front=[]
best=-1
for t,f,c in fr:
    if f>best: front.append((t,f)); best=f
ax.plot([t for t,f in front],[f for t,f in front],'--',color='#2E4C73',lw=1.2,zorder=2,label='Pareto frontier (end-to-end)')
# labels
for t,f,c in e2e+cac:
    dy=0.006 if c not in ('partial_ft','distilbert') else -0.012
    ax.annotate(LAB[c],(t,f),textcoords='offset points',xytext=(5,4 if dy>0 else -10),fontsize=7.5)
ax.set_xlabel('Wall-clock training time per seed (min)')
ax.set_ylabel('Mean F1 (10 seeds)')
ax.grid(True,linestyle=':',linewidth=0.6,color='#CCC'); ax.set_axisbelow(True)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.legend(loc='lower right',fontsize=7.2,frameon=False)
ax.set_ylim(0.50,0.80); ax.set_xlim(0,48)
fig.tight_layout()
fig.savefig('Figure_3.pdf'); fig.savefig('Figure_3.png',dpi=400); plt.close(fig)
from PIL import Image
print('Figure_3 Pareto:', Image.open('Figure_3.png').size)
