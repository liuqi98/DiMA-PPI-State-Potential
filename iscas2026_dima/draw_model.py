"""DiMA-PPI network schematic, traced to ResidueGraphModel implementation."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7,'mathtext.fontset':'dejavusans','pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
INK='#263746';BLUE='#3978AD';TEAL='#298780';GOLD='#C68A43';GRAY='#91A1AE';LINE='#C6D1DB';PURPLE='#8270A5'
fig,ax=plt.subplots(figsize=(7.16,3.65));fig.subplots_adjust(left=.005,right=.995,top=.99,bottom=.01)
ax.set(xlim=(0,25),ylim=(0,12.7));ax.axis('off')
def text(x,y,s,size=7,color=INK,ha='center',weight='normal'):
 ax.text(x,y,s,ha=ha,va='center',fontsize=size,color=color,fontweight=weight,zorder=10,linespacing=1.10)
def box(x,y,w,h,s='',fc='white',ec=LINE,size=7):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=0.09',fc=fc,ec=ec,lw=.65,zorder=2))
 if s:text(x+w/2,y+h/2,s,size)
def arrow(a,b,c=INK,lw=.7):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=6,color=c,lw=lw,zorder=6))
def route(points,c=INK):
 ax.plot(*zip(*points),c=c,lw=.7,zorder=5);arrow(points[-2],points[-1],c)
def op(x,y,s,col=TEAL,r=.23):
 ax.add_patch(Circle((x,y),r,fc='white',ec=col,lw=.85,zorder=7));text(x,y,s,8,col)
def vector(x,y,n=6,col=BLUE,w=.22,h=.12):
 for i in range(n):ax.add_patch(Rectangle((x,y+i*(h+.025)),w,h,fc=col,alpha=.25+.65*(i+1)/n,ec='white',lw=.25,zorder=7))
def neurons(x,ys,col):
 for y in ys:ax.add_patch(Circle((x,y),.105,fc='white',ec=col,lw=.85,zorder=8))
def connect(x1,ys1,x2,ys2,col):
 for y1 in ys1:
  for y2 in ys2:ax.plot([x1,x2],[y1,y2],c=col,lw=.28,alpha=.45,zorder=4)
# Network occupies almost the whole figure. Only architectural labels appear inside it.
text(.12,12.40,r'a  Shared state network $E_\theta$',8,ha='left',weight='bold')
for x,w,title in [(0.03,7.1,'Gated feature fusion'),(7.43,7.05,'Graph message passing'),(14.78,4.1,'Dual readout'),(19.19,5.77,'Scalar head')]:
 box(x,4.12,w,7.83,fc='#FBFCFD');text(x+w/2,11.62,title,7.1,weight='bold')
# Three input streams, explicit learned projections and gate connections.
for y,label,dim,col in [(10.30,'ESM-2','3840',BLUE),(8.75,'ProteinMPNN','128',TEAL),(6.05,'Scalars','4',GRAY)]:
 text(.3,y+.53,label,6.5,ha='left');vector(.34,y-.59,col=col)
 text(.78,y-.2,dim,6.1,ha='left')
 box(2.0,y-.53,1.34,.68,'Proj.' if label!='Scalars' else 'Linear',fc='white',ec=col,size=6.4)
 arrow((1.56,y-.19),(1.96,y-.19),col)
 text(3.60,y-.19,'96',6.1,col)
text(2.9,4.78,r'Proj. = Linear → LN → SiLU',6.0)
text(2.65,10.83,r'$[e;e;0]$',6.3,BLUE)
# q travels over the gate, t splits to the gate and product, scalar is injected after projection.
route([(3.87,10.11),(6.33,10.11),(6.33,7.76)],BLUE)
route([(3.87,8.56),(5.38,8.56),(5.38,8.00)],TEAL)
box(3.95,6.73,1.05,.62,r'FC + $\sigma$',fc='#EEF5F3',ec=TEAL,size=6.3)
text(4.47,6.35,r'$g_i$',7,TEAL)
# Gate input contains projected sequence, projected structure, raw four scalars.
route([(3.87,10.11),(3.87,7.19),(3.92,7.19)],BLUE)
route([(4.23,8.56),(4.23,7.39)],TEAL)
route([(1.68,5.86),(1.68,7.00),(3.92,7.00)],GRAY)
arrow((5.03,7.04),(5.38,7.04),TEAL)
route([(5.38,7.04),(5.38,7.48)],TEAL)
op(5.38,7.73,'×');arrow((5.64,7.73),(6.07,7.73),TEAL)
op(6.33,7.73,'+')
route([(3.87,5.86),(6.33,5.86),(6.33,7.47)],GRAY)
box(5.65,4.53,1.34,.65,'LN',fc='#E9F4F1',ec=TEAL,size=7)
route([(6.60,7.73),(7.00,7.73),(7.00,5.44),(6.32,5.44),(6.32,5.21)],TEAL)
# The output tensor is directed to the expanded graph block.
route([(6.99,4.86),(7.23,4.86),(7.23,10.40),(7.66,10.40)],TEAL)
text(7.24,11.09,r'$H^{(0)}$',7,TEAL)
# TransformerConv, with four parallel heads visualized as stacked attention graphs.
box(7.72,7.40,4.88,3.35,fc='#F0F6FA',ec='#A9C3D6')
text(10.16,10.43,'TransformerConv',7.2,weight='bold')
text(10.16,10.02,'4 heads · mean + root',6.2)
# Four offset planes show one attention neighborhood per head.
for k in reversed(range(4)):
 x=8.01+k*.45;y=7.95+k*.13
 box(x,y,2.55,1.45,fc='white',ec='#B7CDDD')
 center=(x+1.32,y+.67)
 for j,(dx,dy) in enumerate([(.32,.28),(.38,1.15),(2.10,1.12),(2.14,.27)]):
  ax.plot([x+dx,center[0]],[y+dy,center[1]],c=BLUE,lw=[.35,1.0,.6,1.35][j],alpha=.8,zorder=4)
  ax.add_patch(Circle((x+dx,y+dy),.075,fc=BLUE,ec='white',lw=.3,zorder=5))
 ax.add_patch(Circle(center,.13,fc=GOLD,ec='white',lw=.4,zorder=6))
text(10.16,7.69,r'$\alpha_{ij}\,(Vh_j+We_{ij})$',6.4)
# Edge features enter attention independently of node features.
text(13.58,10.65,'RBF',6.5,TEAL);text(13.58,10.27,'16',6.1,TEAL)
vector(13.43,9.13,n=5,col=TEAL,w=.26)
arrow((13.36,9.45),(12.65,9.45),TEAL)
# Explicit residual path around TransformerConv and SiLU.
box(9.10,6.31,2.1,.61,'SiLU',ec=TEAL,size=7)
arrow((10.16,7.36),(10.16,6.96),TEAL)
op(10.16,5.69,'+')
arrow((10.16,6.28),(10.16,5.95),TEAL)
route([(7.70,10.40),(7.58,10.40),(7.58,5.69),(9.90,5.69)],GRAY)
box(9.10,4.55,2.1,.62,'LN',ec=TEAL,size=7)
arrow((10.16,5.43),(10.16,5.21),TEAL)
text(12.69,5.55,'×3',11,TEAL,weight='bold');text(12.69,4.43,'96 channels',6.3)
# Masked attention readout is shared across site/context node subsets.
route([(11.24,4.86),(14.61,4.86),(14.61,9.45),(15.00,9.45)],TEAL)
text(14.56,10.87,r'$H^{(3)}$',7,TEAL)
ys=[10.45,9.98,9.51,9.04,8.57]
neurons(15.20,ys,BLUE)
ax.add_patch(Circle((15.2,ys[2]),.105,fc=GOLD,ec=GOLD,lw=.8,zorder=9))
for i,y in enumerate(ys):
 target=(16.57,10.00) if i==2 else (16.57,8.62)
 arrow((15.33,y),target,GOLD if i==2 else BLUE,lw=.55)
op(16.80,10.00,r'$\Sigma$',GOLD);op(16.80,8.62,r'$\Sigma$',BLUE)
text(16.75,10.60,'Site',6.6,GOLD);text(16.75,8.03,'Context',6.6,BLUE)
for y,col in [(10,GOLD),(8.62,BLUE)]:
 arrow((17.06,y),(17.65,y),col);vector(17.80,y-.43,col=col);text(18.38,y,'96',6.1,col)
text(16.83,6.93,'Masked softmax',6.6)
text(16.83,6.48,r'$\alpha_i\propto\exp(w^T h_i)$',6.8)
# Descriptor branch below the encoder; it joins only after graph pooling.
box(.03,2.67,18.85,1.05,fc='#FCF6EE',ec='#D7B993')
text(.3,3.20,'Atomic descriptor',7,GOLD,ha='left');text(3.65,3.20,'141',6.7,GOLD)
for x,w,label in [(4.38,1.23,'LN'),(6.22,2.06,'Linear'),(8.88,1.53,'SiLU')]:
 box(x,2.90,w,.58,label,ec='#D7B993',size=6.5)
arrow((3.98,3.20),(4.34,3.20),GOLD);arrow((5.65,3.20),(6.18,3.20),GOLD);arrow((8.32,3.20),(8.84,3.20),GOLD)
arrow((10.45,3.20),(11.04,3.20),GOLD)
vector(11.13,2.88,n=4,col=GOLD);text(11.85,3.20,'96',6.6,GOLD)
route([(12.28,3.20),(19.53,3.20),(19.53,7.65),(20.02,7.65)],GOLD)
# Concatenated vectors and an actual layered MLP drawing.
route([(18.61,10.00),(19.54,10.00),(19.54,9.02),(20.02,9.02)],GOLD)
route([(18.61,8.62),(19.76,8.62),(19.76,8.29),(20.02,8.29)],BLUE)
for y,col in [(8.75,GOLD),(8.02,BLUE),(7.29,GOLD)]:vector(20.07,y,n=4,col=col,w=.26,h=.13)
text(20.20,10.57,'Concat.',6.6);text(20.20,6.80,'288',6.5)
y1=[9.10,8.62,8.14,7.66,7.18];y2=[8.87,8.25,7.63]
connect(20.60,y1,22.00,y2,TEAL);neurons(20.60,y1,TEAL);neurons(22.00,y2,TEAL)
connect(22.0,y2,23.63,[8.25],TEAL);neurons(23.63,[8.25],TEAL)
text(22.00,9.62,'96',6.6);text(23.63,9.04,'1',6.6)
text(22.15,6.45,'SiLU + dropout',6.5)
text(22.15,5.33,r'$E_\theta(S_a^c)$',9,TEAL)
arrow((23.63,8.10),(23.63,5.34),TEAL);arrow((23.63,5.34),(23.01,5.34),TEAL)
# Compact four-state wrapper: graphic signs, no repeated explanatory prose.
text(.12,2.12,'b  Four-state prediction',7.7,ha='left',weight='bold')
for x,s in [(5.00,r'$S_W^b$'),(7.15,r'$S_W^f$'),(12.10,r'$S_M^b$'),(14.25,r'$S_M^f$')]:
 box(x,.97,1.55,.68,r'$E_\theta$',fc='#E9F4F1',ec=TEAL,size=7.4);text(x+.77,1.99,s,7.5);arrow((x+.77,1.78),(x+.77,1.68),TEAL)
text(6.88,1.30,'−',10);text(13.98,1.30,'−',10)
text(6.85,.55,r'$\Phi_\theta(W)$',7.5,BLUE);text(13.95,.55,r'$\Phi_\theta(M)$',7.5,GOLD)
text(1.78,1.14,'Shared weights',7,TEAL)
text(20.53,1.35,r'$\widehat y_{W\to M}=\Phi_\theta(M)-\Phi_\theta(W)$',8)
text(20.53,.55,'Canonical scores reused across edges',6.5,GRAY)
arrow((16.13,1.32),(17.06,1.32),TEAL)
for ext in ['pdf','svg','png']:fig.savefig(HERE/f'fig_model.{ext}',dpi=450)
plt.close(fig)
print('Rendered expanded network schematic.')
