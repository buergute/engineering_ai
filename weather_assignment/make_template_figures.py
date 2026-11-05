"""Evidence plots sized for the four-slide course template."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tmp/template2/assets'
OUT.mkdir(parents=True,exist_ok=True)
TEAL='#287f7c'; ORANGE='#ef8b2c'; NAVY='#19344b'; MAGENTA='#b34e87'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':NAVY,'xtick.color':NAVY,'ytick.color':NAVY,'axes.edgecolor':'#9ca9b3','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

raw=pd.read_csv(ROOT/'data/CNC_IEM_2024-2026.csv',na_values=['M'],parse_dates=['valid'])
test=pd.read_csv(ROOT/'output/test_predictions.csv',parse_dates=['origin','target_time'])
forecast=pd.read_csv(ROOT/'output/forecast_2026-10-10_CNC.csv',parse_dates=['target_time'])
report=json.loads((ROOT/'output/model_report.json').read_text())

def save(fig,name):
    fig.savefig(OUT/name,dpi=220,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)

# 1. Wind distribution, observed training period.
fig,ax=plt.subplots(figsize=(6.2,3.85))
v=raw.sknt.dropna()
ax.hist(v,bins=np.arange(-.5,42,2),color=TEAL,edgecolor='white',linewidth=.35)
ax.axvline(v.mean(),color=ORANGE,lw=2,label=f'Mean {v.mean():.2f} kt')
ax.axvline(v.median(),color=NAVY,lw=1.4,ls='--',label=f'Median {v.median():.1f} kt')
ax.set(xlabel='Observed wind speed (knots)',ylabel='Count',title='CNC wind speed distribution')
ax.legend(frameon=False,fontsize=8)
fig.tight_layout();save(fig,'wind_distribution.png')

# 2. Simple relationships from weather observations.
fig,axs=plt.subplots(1,2,figsize=(6.5,3.85),sharey=True)
sample=raw[['sknt','tmpf','alti']].dropna().sample(n=4500,random_state=42)
for ax,x,label in [(axs[0],'tmpf','Temperature (F)'),(axs[1],'alti','Altimeter (inHg)')]:
    ax.hexbin(sample[x],sample.sknt,gridsize=24,cmap='YlGnBu',mincnt=1)
    rho=sample[[x,'sknt']].corr().iloc[0,1]
    ax.set(xlabel=label,title=f'Pearson r = {rho:.2f}')
axs[0].set_ylabel('Wind speed (knots)')
fig.suptitle('Weather variables and wind speed',fontsize=11)
fig.tight_layout();save(fig,'wind_relationships.png')

# 3. Model structure diagram requested by template.
fig,ax=plt.subplots(figsize=(5.3,4.2));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
from matplotlib.patches import FancyBboxPatch
blocks=[(0.4,5.6,2.5,2.3,'Observed data\nwind, T, dewpoint, P'),
        (3.65,5.6,2.55,2.3,'Lag and calendar\n18 predictors'),
        (6.95,5.6,2.55,2.3,'Gradient boosting\n180 trees, 15 leaves'),
        (3.65,1.1,2.55,2.0,'72 direct outputs\n20 min, knots')]
for x,y,w,h,t in blocks:
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12',facecolor='#e8f2f1',edgecolor=TEAL,linewidth=1.5)
    ax.add_patch(patch);ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9,color=NAVY)
for a,b in [((2.9,6.75),(3.65,6.75)),((6.2,6.75),(6.95,6.75)),((8.2,5.6),(5.5,3.1))]:
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.8,color=ORANGE))
fig.tight_layout();save(fig,'model_diagram.png')

# 4. Model and baseline test comparison.
fig,ax=plt.subplots(figsize=(11,4.2))
names=['72h lag baseline','Gradient boosting']
mae=[2.9349602900,2.1938731087];rmse=[3.7876752415,2.8403467401]
x=np.arange(2);w=.28
b1=ax.bar(x-w/2,mae,width=w,color=TEAL,label='MAE')
b2=ax.bar(x+w/2,rmse,width=w,color=ORANGE,label='RMSE')
for bars in [b1,b2]:
    for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.05,f'{b.get_height():.2f}',ha='center',fontsize=10)
ax.set_xticks(x,names);ax.set_ylim(0,4.3);ax.set_ylabel('Error (knots)')
ax.set_title('Held-out test period: 16 Sep to 6 Oct 2026 origins')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
fig.tight_layout();save(fig,'model_comparison.png')

# 5. Test fit on an illustrative complete day.
day=test[test.target_time.dt.date==pd.Timestamp('2026-10-02').date()].copy()
fig,ax=plt.subplots(figsize=(6.9,4.2))
ax.plot(day.target_time.dt.hour+day.target_time.dt.minute/60,day.y,color=NAVY,lw=1.6,label='Observed')
ax.plot(day.target_time.dt.hour+day.target_time.dt.minute/60,day.predicted_knots,color=ORANGE,lw=2,label='Predicted')
ax.set(xlabel='UTC hour on 2 Oct',ylabel='Wind (knots)',title='Held-out test fit')
ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,4));ax.grid(alpha=.2);ax.legend(frameon=False)
fig.tight_layout();save(fig,'test_fit.png')

# 6. Validation hyperparameter comparison, used instead of unsupported epoch curves.
fig,ax=plt.subplots(figsize=(6.1,4.2))
vals=report['validation_results']
names=['HGB\n15 / .05','HGB\n31 / .05','HGB\n15 / .10','RF\n100 / 12']
y=[v['RMSE'] for v in vals]
bars=ax.bar(np.arange(4),y,color=[TEAL,ORANGE,ORANGE,'#8299a5'],width=.58)
for b,v in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,v+.05,f'{v:.2f}',ha='center',fontsize=9)
ax.set_xticks(np.arange(4),names);ax.set_ylim(0,4);ax.set_ylabel('RMSE (knots)')
ax.set_title('Validation parameter sensitivity');ax.grid(axis='y',alpha=.18)
fig.tight_layout();save(fig,'parameter_sensitivity.png')

# 6b. Target-day forecast, matching the template's middle result panel.
fig,ax=plt.subplots(figsize=(6.1,4.2))
t=forecast.target_time.dt.hour+forecast.target_time.dt.minute/60
ax.plot(t,forecast.forecast_knots,color=ORANGE,lw=2.2)
ax.set(xlabel='UTC hour on 10 Oct',ylabel='Forecast wind (knots)',title='72 target time slots')
ax.set_xlim(0,24);ax.set_xticks(np.arange(0,25,4));ax.set_ylim(0,12);ax.grid(alpha=.2)
fig.tight_layout();save(fig,'forecast_curve.png')

# 7. Full test residual distribution.
fig,ax=plt.subplots(figsize=(6.1,4.2))
err=test.predicted_knots-test.y
ax.hist(err,bins=np.arange(-10,10.1,2),color=TEAL,edgecolor='white',linewidth=.5)
ax.axvline(err.mean(),color=ORANGE,lw=2,label=f'Bias {err.mean():+.2f} kt')
ax.set(xlabel='Predicted - observed (knots)',ylabel='Count',title='Held-out residuals')
ax.legend(frameon=False,fontsize=8)
fig.tight_layout();save(fig,'residuals.png')

print('saved',len(list(OUT.glob('*.png'))),'figures')
