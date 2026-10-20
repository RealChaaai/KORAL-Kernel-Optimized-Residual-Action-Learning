"""Render recorded states; never synthesize unexecuted robot motion."""
from pathlib import Path
import json
import numpy as np
from ..robot import chain

def render_gif(record,out,fps=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    tr=record['trajectory'];frames=[]
    fig=plt.figure(figsize=(7.6,5.6));ax=fig.add_subplot(111,projection='3d')
    total=len(tr['ee']);stride=max(1,round(1/(.04*fps)))
    radius=.017;table=float(tr['ee'][0,2]-.12);goal=np.asarray(record['goal'])
    for i in list(range(0,total,stride))+([total-1] if (total-1)%stride else []):
        ax.clear();points=chain(tr['q'][i])[3]
        ax.plot(*points.T,'o-',linewidth=4,markersize=5,label='7R arm')
        p=tr['ee'][i];w=tr['opening'][i]
        for sign in (-1,1):
            pad=p+np.array([0.,sign*(w+.009),0.])
            ax.plot([pad[0]-.024,pad[0]+.024],[pad[1],pad[1]],[pad[2],pad[2]],linewidth=7)
        ax.scatter(*tr['object'][i],s=160,marker='o',label='Dynamic object')
        ax.scatter(*goal,s=160,marker='x',label='Placement goal')
        obj=tr['object'][:i+1];ax.plot(*obj.T,linewidth=1.5,alpha=.7)
        x0,y0=tr['ee'][0,:2]
        rect=np.array([[x0-.16,y0-.15,table],[x0+.20,y0-.15,table],
              [x0+.20,y0+.15,table],[x0-.16,y0+.15,table],[x0-.16,y0-.15,table]])
        ax.plot(*rect.T,linewidth=1,alpha=.5)
        ax.set(xlim=(-.35,.30),ylim=(-.02,.70),zlim=(0,.8),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]')
        ax.view_init(elev=25,azim=135);ax.set_box_aspect((.65,.72,.8));ax.legend(loc='upper left',fontsize=8)
        status = ('completed' if record['success'] else 'time limit') if i==total-1 else 'executing'
        ax.set_title(f"{record.get('controller','policy')} | {status}\nt = {float(tr['time'][i]):.2f} s",fontsize=12)
        fig.tight_layout();fig.canvas.draw();rgba=np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba[:,:,:3]).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0)
    frames[-1].convert('RGB').save(out.with_suffix('.png'));plt.close(fig)
    return out

def plot_results(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=Path(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((results/'episodes.json').read_text());cfg=json.loads((results/'config.json').read_text())
    methods=['bc',*cfg['methods']]
    names={'bc':'BC','random':'Random','uncertainty':'Uncertainty','gain':'Gain-guided','no_sequence':'No temporal loss'}
    for split in ('test','stress'):
        rates=np.array([[np.mean([r['success'] for r in rows if r['split']==split and r['method']==m and r['training_seed']==s])
                        for s in cfg['train_seeds']] for m in methods])*100
        fig,ax=plt.subplots(figsize=(8,4.7));x=np.arange(len(methods))
        ax.bar(x,rates.mean(1),alpha=.6)
        for j,s in enumerate(cfg['train_seeds']):ax.scatter(x+(j-(len(cfg['train_seeds'])-1)/2)*.09,rates[:,j],s=32,label=f'Seed {s}')
        ax.set(xticks=x,xticklabels=[names.get(m,m) for m in methods],ylim=(0,105),ylabel='Autonomous task success [%]',
               title=f"KORAL · {cfg['backend']} · {split}")
        ax.legend(loc='lower left',ncol=len(cfg['train_seeds']),fontsize=8)
        fig.tight_layout();fig.savefig(out/f'{split}.png',dpi=160);plt.close(fig)
