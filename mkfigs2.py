import csv, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.linewidth':0.8,
    'pdf.fonttype':42,'ps.fonttype':42})

SEEDS=[13,42,73,101,123,202,333,777,1337,2024]
COLS={'EM':4,'F1':5,'ROUGE-L':6}
data={}
for r in csv.reader(open('results_clean.csv')):
    if r[0]=='config': continue
    data.setdefault(r[0],{})[int(r[1])]={'EM':float(r[4]),'F1':float(r[5]),'ROUGE-L':float(r[6])}

LABEL={'neobert':'NeoBERT','modernbert':'ModernBERT','spanbert':'SpanBERT','bert':'BERT',
  'distilbert':'DistilBERT','partial_ft':'BERT Partial FT','lstm_l3':'BERT+LSTM 3L\n(frozen)',
  'lstm_l1':'BERT+LSTM 1L\n(frozen)','lstm_l2':'BERT+LSTM 2L\n(frozen)','linear':'BERT+FF\n(frozen)'}

def series(metric):
    out=[]
    for c in data:
        vals=[data[c][s][metric] for s in SEEDS]
        out.append((c, sum(vals)/len(vals), vals))
    out.sort(key=lambda x:-x[1])
    return out

FILL='#8FB0D9'; EDGE='#2E4C73'; MED='#C44E52'
def boxfig(metric, ylabel, fname, aspect):
    d=series(metric)
    labels=[LABEL[c] for c,_,_ in d]; vals=[v for _,_,v in d]
    w=7.9; h=w/aspect
    fig,ax=plt.subplots(figsize=(w,h))
    bp=ax.boxplot(vals, patch_artist=True, widths=0.6, showmeans=True,
        medianprops=dict(color=MED,linewidth=1.4),
        meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor=EDGE,markersize=4),
        flierprops=dict(marker='o',markersize=3,markerfacecolor='#999',markeredgecolor='none',alpha=0.6),
        whiskerprops=dict(color=EDGE,linewidth=0.9),capprops=dict(color=EDGE,linewidth=0.9))
    for b in bp['boxes']: b.set(facecolor=FILL,edgecolor=EDGE,linewidth=0.9,alpha=0.9)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1,len(labels)+1)); ax.set_xticklabels(labels,rotation=32,ha='right',fontsize=7)
    ax.yaxis.grid(True,linestyle=':',linewidth=0.6,color='#CCC'); ax.set_axisbelow(True)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.tick_params(length=3)
    fig.tight_layout()
    fig.savefig(fname+'.pdf'); fig.savefig(fname+'.png',dpi=400); plt.close(fig)
    print('wrote',fname,'order:',[l.replace(chr(10),' ') for l in labels])

boxfig('F1','F1 score','Figure_2', 2.80)
# Figure 3 in the paper is the accuracy/time trade-off (plot_efficiency_pareto.py);
# the ROUGE-L box plot is supplementary, so it is written under a separate name.
boxfig('ROUGE-L','ROUGE-L score','Figure_S_ROUGE-L', 2.66)
