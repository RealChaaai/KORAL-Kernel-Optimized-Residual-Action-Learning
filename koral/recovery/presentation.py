"""Recorded-state demonstrations and charts generated from audited result files."""
from pathlib import Path
import json
import numpy as np
from .visualization import render_gif
from ..robot import chain


def read_demo(path):
    path=Path(path);record=json.loads((path/'metrics.json').read_text())
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:
        record['trajectory']={k:z[k].copy() for k in z.files}
    record['controller']='BC' if path.name=='bc' else 'KORAL'
    return record


def plot_benchmarks(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);summary=json.loads((root/'results/reference/summary.json').read_text())
    out=root/'docs/figures';out.mkdir(parents=True,exist_ok=True)
    for bank,methods,names in [
        ('reference',['bc','random','uncertainty','gain','no_sequence','koral'],
         ['BC','Random','Disagreement','Gain FT','No temporal','KORAL']),
        ('confirmation',['bc','local_action','no_recovery','koral'],
         ['BC','Local-action blend','Initial-memory residual','KORAL'])]:
        fig,ax=plt.subplots(figsize=(10.0,4.9));x=np.arange(len(methods));width=.36
        for j,(split,label) in enumerate([('test','Normal'),('stress','Low friction')]):
            groups=[summary[bank][split]['methods'][m] for m in methods]
            heights=[100*r['success_rate'] for r in groups]
            bars=ax.bar(x+(j-.5)*width,heights,width,label=label,alpha=.8)
            for bar,r in zip(bars,groups):
                ax.annotate(f"{r['successes']}/{r['episodes']}",
                    (bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=9)
        ax.set(xticks=x,xticklabels=names,ylim=(0,111),ylabel='Autonomous task success (%)',
               title=f'KORAL | {bank.capitalize()} benchmark | 3 training seeds')
        ax.legend(loc='lower left',ncol=2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        fig.tight_layout();fig.savefig(out/f'{bank}.png',dpi=170);plt.close(fig)
    methods=['koral','no_recovery','no_affine','no_support','local_action']
    names=['KORAL','Initial memory','Constant residual','No support gate','Absolute-action fit']
    fig,ax=plt.subplots(figsize=(10,4.9));x=np.arange(len(methods));width=.36
    for j,(split,label) in enumerate([('test','Normal'),('stress','Low friction')]):
        rates=[100*summary['reference'][split]['methods'][m]['success_rate'] for m in methods]
        ax.bar(x+(j-.5)*width,rates,width,label=label,alpha=.8)
    ax.set(xticks=x,xticklabels=names,ylim=(0,105),ylabel='Autonomous task success (%)',title='KORAL | Component ablations')
    ax.legend(loc='lower left',ncol=2);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(out/'ablations.png',dpi=170);plt.close(fig)


def build_media(root):
    from PIL import Image,ImageSequence
    root=Path(root);d=root/'docs/demo'
    records={m:read_demo(d/m) for m in ('bc','koral')}
    for m,record in records.items():render_gif(record,d/m/'demo.gif',fps=12)
    opened=[];frames={}
    for m in records:
        im=Image.open(d/m/'demo.gif');opened.append(im)
        frames[m]=[f.convert('RGB').copy() for f in ImageSequence.Iterator(im)]
    combined=[];n=max(len(f) for f in frames.values())
    for i in range(n):
        left=frames['bc'][min(i,len(frames['bc'])-1)];right=frames['koral'][min(i,len(frames['koral'])-1)]
        image=Image.new('RGB',(left.width+right.width,max(left.height,right.height)))
        image.paste(left,(0,0));image.paste(right,(left.width,0))
        combined.append(image.convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
    combined[0].save(d/'comparison.gif',save_all=True,append_images=combined[1:],duration=83,loop=0)
    combined[-1].convert('RGB').save(d/'comparison.png')
    for im in opened:im.close()
    write_html(records,d/'replay.html')
    plot_benchmarks(root)
    return {m:{'success':r['success'],'steps':r['steps'],'seed':r['seed']} for m,r in records.items()}


def write_html(records,out):
    data={}
    for m,r in records.items():
        tr=r['trajectory'];frames=[]
        for i,q in enumerate(tr['q']):
            frames.append({'arm':chain(q)[3].tolist(),'ee':tr['ee'][i].tolist(),
                'object':tr['object'][i].tolist(),'opening':float(tr['opening'][i]),'time':float(tr['time'][i])})
        data[m]={'frames':frames,'goal':r['goal'],'seed':r['seed'],'success':r['success'],
                 'steps':r['steps'],'table':float(tr['ee'][0,2]-.12)}
    payload=json.dumps(data,separators=(',',':'),allow_nan=False).replace('</','<\\/')
    template='''<!doctype html><html lang="en"><meta charset="utf-8"><title>KORAL | Recorded-state replay</title>
<style>body{font-family:system-ui,sans-serif;max-width:1440px;margin:30px auto;padding:0 20px}h1{margin-bottom:4px}.row{display:flex;gap:16px}.panel{flex:1;min-width:0;border:1px solid;border-radius:12px;padding:14px}canvas{width:100%;aspect-ratio:1.6;cursor:grab}input{width:65%;vertical-align:middle}button{padding:8px 22px;margin-right:14px}small{opacity:.7}footer{margin-top:20px;line-height:1.5}@media(max-width:800px){.row{display:block}}</style>
<h1>KORAL</h1><p>Support-gated local residual learning · matched-state replay</p>
<div class="row"><section class="panel"><h2>Behavior cloning</h2><canvas id="bc" width="900" height="560"></canvas><p id="bcinfo"></p></section>
<section class="panel"><h2>KORAL</h2><canvas id="koral" width="900" height="560"></canvas><p id="koralinfo"></p></section></div>
<p><button id="play">Play</button><input type="range" id="time" min="0" step="1" value="0"><span id="clock"></span></p>
<footer>Drag a canvas to rotate both views. Recorded joint and object states are rendered without external libraries. Completed episodes hold their last recorded frame.<br><small>This illustration uses development seed __SEED__ and the low-friction condition. It is separate from the reference and confirmation tables.</small></footer>
<script>const data=__DATA__;let index=0,yaw=2.2,pitch=.5,playing=false;const slider=document.getElementById('time');
const max=Math.max(...Object.values(data).map(r=>r.frames.length))-1;slider.max=max;
function point(p){let x=p[0],y=p[1]-.3,z=p[2]-.34;let X=Math.cos(yaw)*x-Math.sin(yaw)*y,Y=Math.sin(yaw)*x+Math.cos(yaw)*y;return [450+X*650,280-(Math.cos(pitch)*z-Math.sin(pitch)*Y)*650]}
function draw(m){const r=data[m],i=Math.min(index,r.frames.length-1),f=r.frames[i],c=document.getElementById(m).getContext('2d');c.clearRect(0,0,900,560);
function line(ps,w=2,alpha=1){c.beginPath();ps.forEach((p,j)=>{let a=point(p);j?c.lineTo(...a):c.moveTo(...a)});c.lineWidth=w;c.globalAlpha=alpha;c.stroke();c.globalAlpha=1}
line([[-.3,.05,r.table],[.25,.05,r.table],[.25,.6,r.table],[-.3,.6,r.table],[-.3,.05,r.table]],1,.4);
line(r.frames.slice(0,i+1).map(f=>f.object),1,.5);line(f.arm,7);
f.arm.forEach(p=>{let a=point(p);c.beginPath();c.arc(...a,5,0,Math.PI*2);c.stroke()});
for(const sign of [-1,1]){const p=[f.ee[0],f.ee[1]+sign*(f.opening+.009),f.ee[2]];line([[p[0]-.024,p[1],p[2]],[p[0]+.024,p[1],p[2]]],9)}
let a=point(f.object);c.beginPath();c.arc(...a,11,0,Math.PI*2);c.stroke();let g=point(r.goal);c.beginPath();c.moveTo(g[0]-9,g[1]-9);c.lineTo(g[0]+9,g[1]+9);c.moveTo(g[0]+9,g[1]-9);c.lineTo(g[0]-9,g[1]+9);c.stroke();c.font='15px system-ui';c.fillText('goal',g[0]+13,g[1]);
let status=i==r.frames.length-1?(r.success?'Task completed':'Time limit reached'):'Executing';document.getElementById(m+'info').textContent=`${status} | ${f.time.toFixed(2)} s | step ${i}/${r.steps}`}
function redraw(){for(const m in data)draw(m);slider.value=index;document.getElementById('clock').textContent=(index*.04).toFixed(2)+' s'}
slider.oninput=()=>{index=+slider.value;redraw()};document.getElementById('play').onclick=()=>{playing=!playing;if(index>=max)index=0;document.getElementById('play').textContent=playing?'Pause':'Play'};
for(const m in data){const canvas=document.getElementById(m);let prev=null;canvas.onpointerdown=e=>{prev=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)};canvas.onpointerup=()=>prev=null;canvas.onpointermove=e=>{if(prev){yaw+=(e.clientX-prev[0])*.008;pitch=Math.max(-1.2,Math.min(1.2,pitch+(e.clientY-prev[1])*.008));prev=[e.clientX,e.clientY];redraw()}}}
setInterval(()=>{if(playing){index=Math.min(max,index+2);redraw();if(index==max){playing=false;document.getElementById('play').textContent='Play'}}},80);redraw();</script></html>'''
    Path(out).write_text(template.replace('__DATA__',payload).replace('__SEED__',str(records['koral']['seed'])),encoding='utf-8')
