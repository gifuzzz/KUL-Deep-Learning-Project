"""FlameBox Python interface: scenario generation and direct C++ memory bridge.

Python initializes procedural rooms, openings, and starting fields. C++ alone
advances physics, agents, victims, pathfinding and dynamic field state.
There is deliberately no Python implementation of the simulation solver.
"""
from __future__ import annotations
import ctypes as C
import os
import heapq
import math
import os
import shutil
import sys
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

_DLL_DIRECTORY_HANDLES = []  # Keep Windows DLL search registrations alive.
_DLL_SEARCH_READY = False

def register_native_dll_dirs():
    """Locate MSYS2 compiler runtime, for Python 3.8+ safe DLL loading.

    This is called in *each* Python process (including Windows spawn workers).
    It does not alter Windows security policies or require manual PATH edits.
    """
    global _DLL_SEARCH_READY
    if sys.platform != 'win32' or _DLL_SEARCH_READY:
        return
    candidates = []
    for variable in ('FLAMEBOX_MSYS2_BIN',):
        if os.environ.get(variable):
            candidates.append(Path(os.environ[variable]))
    if os.environ.get('CXX'):
        value = shutil.which(os.environ['CXX']) or os.environ['CXX']
        candidates.append(Path(value).expanduser().parent)
    for name in ('g++', 'g++.exe'):
        compiler = shutil.which(name)
        if compiler:
            candidates.append(Path(compiler).parent)
    candidates.extend((Path(r'C:\msys64\ucrt64\bin'), Path(r'C:\msys64\mingw64\bin')))
    # Native CUDA plugin depends on cudart DLLs outside the MSYS2 runtime.
    for var in ('CUDA_PATH','CUDA_HOME'):
        if os.environ.get(var):candidates.append(Path(os.environ[var])/'bin')
    toolkit=Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'NVIDIA GPU Computing Toolkit'/'CUDA'
    if toolkit.is_dir():candidates.extend(sorted(toolkit.glob('v*/bin'),reverse=True))
    seen = set()
    for directory in candidates:
        key = str(directory).lower()
        if key in seen or not directory.is_dir():
            continue
        seen.add(key)
        _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(directory)))
    _DLL_SEARCH_READY = True

DEFAULTS = {
 'initial_oxygen':.82, 'fire_threshold':.10, 'fire_spread':.035,
 'fire_draft_boost':13.0, 'fire_core_flicker':.20, 'fire_flicker_memory':.82, 'smoke_turbulence':.35,
 'fire_growth':.029, 'fire_decay':.018, 'fuel_burn':.006,
 'oxygen_consumption':.009, 'smoke_generation':.026, 'damage_rate':.012,
 'oxygen_diffusion':.038, 'smoke_diffusion':.055,
 'smoke_draft_boost':3.8, 'smoke_oxygen_drift':.022,
 'draft_oxygen_rate':.16, 'draft_smoke_clearance':2.3, 'smoke_settling':.0012, 'initial_blocked_percent':12., 'opening_open_percent':100.,
 'draft_width':.65, 'draft_midpoint_factor':.32, 'draft_falloff':.9,
 'debris_event_rate':.00075, 'debris_length_mean':.8,
 'debris_length_spread':.33, 'debris_thickness_mean':.25,
 'debris_thickness_spread':.3, 'debris_severity_scale':.50, 'debris_opening_bias':3.0,
 'water_strength':1.0, 'water_radius':1.35,
 'wood_ratio':50.0, 'fuel_to_oxygen':.50, 'concrete_flammability':.90, 'wood_flammability':1.15,
 'smoke_flammable_yield':.14, 'smoke_starvation_boost':.95, 'smoke_fire_escape':.7,
 'water_cooling':.55, 'water_base_evaporation':.0015, 'water_fire_evaporation':.035, 'water_heat_radius':1.2,
 'flashback_min_mass':.28, 'flashback_min_oxygen':.13, 'flashback_delay':3.,
 'flashback_strength':.82, 'flashback_min_flammability':.12, 'flashback_min_smoke':.17,
}
META = {
 # Calibrated ranges around the DEFAULT value.  Browser places DEFAULT at slider midpoint.
 'smoke_turbulence':('Light-smoke turbulence',0,.8,.01),
 'initial_oxygen':('Initial oxygen',.62,1.,.005),
 'fire_threshold':('Active-fire threshold',.06,.18,.005),
 'fire_spread':('Fire spread',.025,.11,.001),
 'fire_draft_boost':('Fire spread along drafts',3.0,23.,.1),
 'fire_core_flicker':('Hot-fire turbulence',.07,.36,.005),
 'fire_flicker_memory':('Flame turbulence persistence',.58,.95,.01),
 'fire_growth':('Active fire growth',.015,.075,.001),
 'fire_decay':('Fire decay',.005,.05,.001),
 'fuel_burn':('Combustion rate factor',.002,.014,.00025),
 'oxygen_consumption':('Fire oxygen use',.003,.022,.0005),
 'smoke_generation':('Smoke production',.010,.055,.001),
 'damage_rate':('Damage accumulation',.004,.022,.0005),
 'oxygen_diffusion':('Oxygen diffusion',.018,.075,.001),
 'smoke_diffusion':('Smoke diffusion',.025,.090,.001),
 'smoke_draft_boost':('Smoke draft transport boost',1.0,7.,.05),
 'smoke_oxygen_drift':('Smoke down-oxygen drift',.005,.055,.001),
 'draft_oxygen_rate':('Draft oxygen exchange',.035,.38,.005),
 'draft_smoke_clearance':('Draft smoke removal',.45,5.,.025),
 'smoke_settling':('Background smoke settling',.0002,.005,.0001),
 'draft_width':('Draft width (physical units)',.3,1.15,.025),
 'draft_midpoint_factor':('Draft center strength',.12,.68,.01),
 'draft_falloff':('Draft lateral falloff',.5,1.5,.025),
 'debris_event_rate':('Fire-generated debris frequency',0,.003,.000025),
 'debris_length_mean':('Debris mean length (physical)',.45,1.65,.025),
 'debris_length_spread':('Debris length variability',.1,.57,.025),
 'debris_thickness_mean':('Debris mean thickness (physical)',.13,.53,.01),
 'debris_thickness_spread':('Debris thickness variability',.1,.57,.025),
 'debris_severity_scale':('Debris fire-size scaling',.10,1.1,.025),
 'debris_opening_bias':('Debris near openings',0.,7.5,.05),
 'initial_blocked_percent':('Initially blocked openings (%)',0,80.,1),
 'opening_open_percent':('Open doors and windows at start (%)',0,100,1),
 'wood_ratio':('Wood flooring (%)',0,100,1),
 'fuel_to_oxygen':('Fuel burned / oxygen used',.2,.8,.025),
 'concrete_flammability':('Concrete burning coefficient',.35,1.5,.025),
 'wood_flammability':('Wood burning coefficient',.65,1.9,.025),
 'smoke_flammable_yield':('Smoke flammability at good oxygen',.02,.5,.01),
 'smoke_starvation_boost':('Smoke flammability under oxygen starvation',.1,2.4,.05),
 'smoke_fire_escape':('Smoke displacement from flames',.1,1.7,.025),
 'water_cooling':('Water cooling per tick',.05,1.25,.025),
 'water_base_evaporation':('Water baseline evaporation',.0002,.012,.0001),
 'water_fire_evaporation':('Water heat evaporation',.002,.12,.002),
 'water_heat_radius':('Water fire sensing radius',.5,2.0,.05),
 'flashback_min_mass':('Flashback smoke mass (physical units)',.10,1.4,.025),
 'flashback_min_oxygen':('Flashback minimum oxygen',.07,.32,.005),
 'flashback_delay':('Flashback ignition delay (ticks)',1,8,1),
 'flashback_strength':('Flashback ignition intensity',.5,1.,.025),
 'flashback_min_flammability':('Flashback smoke flammability threshold',.03,.40,.005),
 'flashback_min_smoke':('Flashback smoke concentration threshold',.08,.5,.01),
 'water_radius':('Water spray range (physical units)',.35,2.5,.025),
}
SAFE_PARAMETER_BOUNDS = {'smoke_turbulence':(0,1.5),'initial_oxygen': (0.4, 1), 'fire_threshold': (0.04, 0.3), 'fire_spread': (0, 0.12), 'fire_draft_boost': (0, 23.0), 'fire_core_flicker': (0, 0.45), 'fire_flicker_memory': (0.2, 0.97), 'fire_growth': (0, 0.08), 'fire_decay': (0, 0.12), 'fuel_burn': (0, 0.025), 'oxygen_consumption': (0, 0.035), 'smoke_generation': (0, 0.09), 'damage_rate': (0, 0.04), 'oxygen_diffusion': (0, 0.09), 'smoke_diffusion': (0, 0.09), 'smoke_draft_boost': (0, 7.0), 'smoke_oxygen_drift': (0, 0.075), 'draft_oxygen_rate': (0, 0.38), 'draft_smoke_clearance': (0, 5), 'smoke_settling': (0, 0.015), 'draft_width': (0.25, 1.5), 'draft_midpoint_factor': (0.08, 0.85), 'draft_falloff': (0.4, 1.8), 'debris_event_rate': (0, 0.035), 'debris_length_mean': (0.35, 2.3), 'debris_length_spread': (0.05, 0.7), 'debris_thickness_mean': (0.12, 0.9), 'debris_thickness_spread': (0.05, 0.7), 'debris_severity_scale': (0, 2), 'debris_opening_bias': (0, 7.5), 'initial_blocked_percent': (0, 100), 'opening_open_percent':(0,100), 'water_strength': (0.1, 1), 'water_radius': (0, 3),
 'wood_ratio':(0,100),'fuel_to_oxygen':(.05,.95),'concrete_flammability':(.1,1.9),'wood_flammability':(.1,2.0),
 'smoke_flammable_yield':(0,1),'smoke_starvation_boost':(0,5),'smoke_fire_escape':(0,2),
 'water_cooling':(0,2.0),'water_base_evaporation':(0,.05),'water_fire_evaporation':(0,.6),'water_heat_radius':(.25,3),
 'flashback_min_mass':(.01,5),'flashback_min_oxygen':(.01,.9),'flashback_delay':(1,30),
 'flashback_strength':(.1,1),'flashback_min_flammability':(.01,1),'flashback_min_smoke':(.01,1)}
SLIDER_MODES={'smoke_turbulence':1.3,
 'fire_spread':2,'fire_draft_boost':1.25,'fire_core_flicker':1.5,'fire_growth':1.6,'fuel_burn':1.6,'oxygen_consumption':1.5,
 'smoke_generation':1.5,'damage_rate':1.8,'smoke_oxygen_drift':2,
 'draft_oxygen_rate':1.4,'draft_smoke_clearance':1.3,'smoke_settling':2.4,
 'debris_event_rate':2.4,'debris_length_spread':1.5,
 'debris_thickness_spread':1.5,'debris_opening_bias':1.6,'initial_blocked_percent':1.4,
 'water_base_evaporation':2.2,'water_fire_evaporation':1.5,'smoke_fire_escape':1.4,
 'flashback_min_mass':1.4,'concrete_flammability':1.3,'smoke_flammable_yield':1.2,
}
DESCRIPTIONS={'smoke_turbulence':'Conservative temporal smoke fluctuations; lower-density smoke is more mobile. Zero disables turbulence.',
 'initial_oxygen':'Initial oxygen fraction at the beginning of a scenario. Reset or Generate to apply.',
 'fire_threshold':'Only cells above this fire level can propagate ignition to neighboring cells.',
 'fire_spread':'Base heat transfer from burning cells across their neighboring grid edges.',
 'fire_draft_boost':'Extra fire propagation along open draft paths. Both neighboring cells must lie in the path; its orientation favors spread parallel to the port-to-port segment. Closing windows or doors updates this dynamically.',
 'fire_core_flicker':'Strength of smoothly varying spatial and temporal fluctuations in established high-intensity flames. Affects the displayed hot core and mildly modulates its growth, not sub-threshold ignition.',
 'fire_flicker_memory':'Temporal persistence of the flame noise (higher means slower, smoother changes); the noise is spatially interpolated at a fixed physical scale.',
 'fire_growth':'Growth of active flames while sufficient oxygen and fuel are available.',
 'fire_decay':'How rapidly flame intensity decays naturally, including below-threshold embers.',
 'fuel_burn':'Scales combustion intensity and the simultaneous oxygen/fuel consumption rate; their ratio is set separately.',
 'oxygen_consumption':'Oxygen consumed each tick by active fire at the burning location.',
 'smoke_generation':'Amount of smoke emitted per tick by active burning cells.',
 'damage_rate':'Rate at which burning causes structural damage that can eventually shed debris.',
 'oxygen_diffusion':'Conservative high-to-low oxygen diffusion through open passages.',
 'smoke_diffusion':'Conservative smoke mixing between neighboring cells through permeable boundaries.',
 'smoke_draft_boost':'Additional transport of smoke along draft-source regions.',
 'smoke_oxygen_drift':'Smoke moves down the gradient of oxygen in addition to regular diffusion.',
 'draft_oxygen_rate':'Fresh-air exchange rate toward outdoor oxygen in blue draft-source cells.',
 'draft_smoke_clearance':'Rapid smoke removal on oxygen-generating draft cells, proportional to their source intensity.',
 'smoke_settling':'Small background smoke decay everywhere, even with all openings blocked.',
 'draft_width':'Physical thickness of the smooth draft-source paths between openings.',
 'draft_midpoint_factor':'Draft-source strength in the middle of each port-to-port connection.',
 'draft_falloff':'Steepness of the draft intensity gradient away from its centerline.',
 'debris_event_rate':'Frequency of new debris events driven by active fire area and accumulated damage; the default is deliberately low.',
 'debris_length_mean':'Median physical length of a debris line before fire-intensity scaling.',
 'debris_length_spread':'Log-normal variation of debris length around its median.',
 'debris_thickness_mean':'Median physical width of debris before fire-intensity scaling.',
 'debris_thickness_spread':'Log-normal variation of debris thickness around its median.',
 'debris_severity_scale':'Multiplier by which intense fires create larger fragments, not more frequent fragments.',
 'debris_opening_bias':'Weights where fire-triggered debris lands. Higher settings favor cells alongside doors and windows without directly multiplying the total debris event rate.',
 'initial_blocked_percent':'Percentage of openings covered by one-cell-thick starting debris. Resets and new buildings apply this value.',
 'opening_open_percent':'Percentage of doors, windows and outside exits initially open. Defaults to 100%; closed openings can still be opened manually. Apply by Reset or Generate.',
 'water_strength':'Suppression strength at the center of a water application.',
 'water_radius':'Physical range of the eight-direction triangular Orb spray, or the radial rescuer water action. Walls, closed openings and debris block water.',
 'wood_ratio':'Fraction of rooms assigned wood floor; re-generate/reset to apply.',
 'fuel_to_oxygen':'Solid fuel consumed for each unit of oxygen used by fire, always below 1 at its allowed range.',
 'concrete_flammability':'Concrete floor growth/spread factor. Both materials carry comparable initial fuel so neither becomes completely nonflammable.',
 'wood_flammability':'Wood floor growth/spread factor; faster fires and greater oxygen demand.',
 'smoke_flammable_yield':'Flammability fraction of newly emitted smoke when oxygen is plentiful.',
 'smoke_starvation_boost':'Flammability fraction of newly emitted smoke rises when oxygen is depleted.',
 'smoke_fire_escape':'Conservatively push smoke from high-fire cells to adjacent lower-fire cells.',
 'water_cooling':'Thermal quenching rate per simulation tick; wet cells also resist neighbor reignition. Applied during combustion substeps.',
 'water_base_evaporation':'Baseline disappearance rate of deposited liquid water.',
 'water_fire_evaporation':'Additional evaporation proportional to nearby fire intensity.',
 'water_heat_radius':'Physical distance used to average nearby fire for evaporation.',
 'flashback_min_mass':'Minimum total flammable smoke across the room to arm a delayed flashback.',
 'flashback_min_oxygen':'Minimum mean oxygen over the selected smoke cells after opening a door/window.',
 'flashback_delay':'Time between opening and scheduled rapid ignition; evaluated only on opening events.',
 'flashback_strength':'Immediate fire intensity on cells participating in an opening-triggered flashback.',
 'flashback_min_flammability':'Minimum smoke flammability required for a cell to be included in flashback.',
 'flashback_min_smoke':'Minimum smoke concentration required for a flashback candidate cell.',
}

class GeometryEnvironment:
 def __init__(self, size=8,resolution=2,seed=12,victims=7,ignitions=2,agents=4,params=None):
  if not 4<=int(size)<=64 or not 2<=int(resolution)<=8:raise ValueError('size 4..64; resolution 2..8')
  if int(agents)<0:raise ValueError('agents must be non-negative')
  self.size=int(size);self.resolution=int(resolution);self.seed=int(seed)
  self.nvictims=int(victims);self.nignitions=int(ignitions);self.nagents=int(agents)
  self.params=dict(DEFAULTS)
  if params:self.set_params(params)
  self.reset()
 def set_params(self,values):
  for k,v in values.items():
   if k not in META and k!='water_strength':raise ValueError('unknown parameter '+k)
   try:v=float(v)
   except (ValueError,TypeError):raise ValueError('invalid parameter '+k)
   lo,hi=SAFE_PARAMETER_BOUNDS[k]
   if not math.isfinite(v) or not lo<=v<=hi:raise ValueError(k+' out of range')
   self.params[k]=v
  if hasattr(self,'zone') and any(k.startswith('draft_') for k in values):self.rebuild_draft()
 def _layout(self):
  """Dimension-first rectilinear building: crossing/branching corridor segments,
  then randomized guillotine subdivision of the remaining room rectangles.
  All geometry is sampled in physical integer units before rasterization.
  """
  rng=self.rng;n=self.size
  W=int(round(11+6.0*math.sqrt(n)))+int(rng.integers(-2,3))
  H=int(round(7+4.8*math.sqrt(n)))+int(rng.integers(-2,3))
  def bands(total,kind,extra):
   w=int(rng.integers(2,4))
   mid=int(rng.integers(int(.40*total),int(.60*total)+1))
   corridors=[(mid,min(total-4,mid+w))]
   if extra:
    lo=5;hi=total-5
    candidates=[j for j in range(lo,hi-w) if abs(j-mid)>=7]
    if candidates:
     j=int(rng.choice(candidates));corridors.append((j,j+w))
   corridors.sort()
   out=[];prev=0
   for a,b in corridors:
    if a>prev:out.append((prev,a,False))
    out.append((a,b,True));prev=b
   if prev<total:out.append((prev,total,False))
   return out
  # Three seeded topology families: intersecting corridors, offset wings,
  # and their rotated counterpart. Each corridor segment is a rectangle.
  style=(self.seed+1)%3
  if style==0:
   xb=bands(W,'vertical',n>=16)
   yb=bands(H,'horizontal',n>=12)
   corridors=[];roomrects=[]
   for y0,y1,ycorr in yb:
    for x0,x1,xcorr in xb:
     rect=(y0,x0,y1-y0,x1-x0)
     (corridors if ycorr or xcorr else roomrects).append(rect)
  elif style==1:
   # Dogleg: two horizontally offset wings connected by one vertical spine.
   x=int(rng.integers(int(.39*W),int(.56*W)+1));cw=2
   ya=int(rng.integers(max(5,int(.28*H)),max(6,int(.43*H))))
   yb=int(rng.integers(max(ya+5,int(.55*H)),min(H-4,max(ya+6,int(.75*H))+1)))
   roomrects=[(0,0,ya,x),(ya+2,0,H-ya-2,x),
              (0,x+cw,yb,W-x-cw),(yb+2,x+cw,H-yb-2,W-x-cw)]
   corridors=[(0,x,H,cw),(ya,0,2,x),(yb,x+cw,2,W-x-cw)]
  else:
   # Rotated dogleg: a main horizontal spine and staggered vertical wings.
   y=int(rng.integers(int(.40*H),int(.58*H)+1));ch=2
   xa=int(rng.integers(max(5,int(.28*W)),max(6,int(.43*W))))
   xb=int(rng.integers(max(xa+5,int(.56*W)),min(W-4,max(xa+6,int(.75*W))+1)))
   roomrects=[(0,0,y,xa),(0,xa+2,y,W-xa-2),
              (y+ch,0,H-y-ch,xb),(y+ch,xb+2,H-y-ch,W-xb-2)]
   corridors=[(y,0,ch,W),(0,xa,y,2),(y+ch,xb,H-y-ch,2)]
  if any(h<3 or w<3 for _,_,h,w in roomrects):
   raise RuntimeError('room partition too small')
  # Optional one-sided hallway branches prevent a rigid rectangular grid.
  # A branch divides one suitable quadrant into two rooms and a connecting hall.
  branches=0 if n<7 else (1 if n<13 else 2)
  for _ in range(branches):
   viable=[i for i,(_,_,h,w) in enumerate(roomrects) if h>=9 and w>=7]
   if not viable:break
   index=int(rng.choice(viable));y,x,h,w=roomrects.pop(index)
   top_h=int(rng.integers(3,h-4))
   roomrects.extend([(y,x,top_h,w),(y+top_h+2,x,h-top_h-2,w)])
   corridors.append((y+top_h,x,2,w))
  if len(roomrects)>n:raise RuntimeError('geometry has too many base rooms')
  # Split whichever leaves provide good size balance, but retain variation.
  while len(roomrects)<n:
   candidates=[]
   for idx,(y,x,h,w) in enumerate(roomrects):
    axes=[]
    if w>=6:axes.append('x')
    if h>=6:axes.append('y')
    if axes:candidates.append((idx,axes,max(w,h)*math.sqrt(w*h)))
   if not candidates:raise RuntimeError('could not partition building into requested rooms')
   # Always split a large room before creating tiny ones elsewhere.
   idx,axes,_=max(candidates,key=lambda z:z[2]*(.9+.2*rng.random()))
   y,x,h,w=roomrects.pop(idx)
   ax=str(rng.choice(axes)) if len(axes)==1 else ('x' if rng.random()<w/(w+h) else 'y')
   if ax=='x':
    cut=int(rng.integers(3,w-2));children=[(y,x,h,cut),(y,x+cut,h,w-cut)]
   else:
    cut=int(rng.integers(3,h-2));children=[(y,x,cut,w),(y+cut,x,h-cut,w)]
   roomrects.extend(children)
  roomrects.sort(key=lambda rect:(rect[0],rect[1]));corridors.sort(key=lambda rect:(rect[0],rect[1]))
  return H,W,roomrects,corridors
 def reset(self):
  rng=self.rng=np.random.default_rng(self.seed)
  self.flame_rng=np.random.default_rng(self.seed ^ 0x5DEECE66D)
  q=self.resolution
  Hbase,Wbase,roomrects,corridorrects=self._layout()
  self.physical_width=Wbase;self.physical_height=Hbase
  self.h=Hbase*q;self.w=Wbase*q;H,W=self.h,self.w
  self.zone=np.zeros((H,W),np.int16)
  # Room IDs 1..size, corridor IDs 0 and size+1 onward.
  self.rooms=[];self.corridors=[];self.corridor_id=0
  self.geometry_rooms=[None]*(self.size+len(corridorrects))
  def place(rect,z,kind):
   top,left,h,w=rect
   item={'id':z,'name':('ROOM' if kind=='room' else 'CORRIDOR')+f' {z:02d}',
         'type':kind,'top':top*q,'left':left*q,'height':h*q,'width':w*q}
   self.geometry_rooms[z]=item
   self.zone[item['top']:item['top']+item['height'],item['left']:item['left']+item['width']]=z
   (self.rooms if kind=='room' else self.corridors).append(item)
  for i,rect in enumerate(roomrects):place(rect,i+1,'room')
  for i,rect in enumerate(corridorrects):place(rect,0 if i==0 else self.size+i,'corridor')
  self.openings=[];self._next_id=0
  def opening(kind,orient,at,start,length,zone_a,zone_b=None,opened=True):
   obj={'id':self._next_id,'kind':kind,'orient':orient,'at':int(at),'start':int(start),'length':int(length),
        'zone_a':int(zone_a),'zone_b':None if zone_b is None else int(zone_b),'open':bool(opened)}
   self._next_id+=1;self.openings.append(obj)
  # Detect every geometrically shared boundary, including parallel and side doors.
  seams=[]
  rects=self.geometry_rooms
  for i,a in enumerate(rects):
   for b in rects[i+1:]:
    ar=a['top']+a['height'];ac=a['left']+a['width'];br=b['top']+b['height'];bc=b['left']+b['width']
    if ac==b['left'] or bc==a['left']:
     lo=max(a['top'],b['top']);hi=min(ar,br)
     if hi-lo>=q:seams.append(('V',ac if ac==b['left'] else bc,lo,hi,a['id'],b['id']))
    if ar==b['top'] or br==a['top']:
     lo=max(a['left'],b['left']);hi=min(ac,bc)
     if hi-lo>=q:seams.append(('H',ar if ar==b['top'] else br,lo,hi,a['id'],b['id']))
  # A spanning tree guarantees a connected movement topology when doors open;
  # extra doors create loops and alternative routes, not a fixed central hall.
  parent=list(range(len(rects)))
  def root(a):
   while parent[a]!=a:
    parent[a]=parent[parent[a]];a=parent[a]
   return a
  def unite(a,b):
   ra,rb=root(a),root(b)
   if ra==rb:return False
   parent[ra]=rb;return True
  perm=list(rng.permutation(len(seams)))
  perm.sort(key=lambda idx:0 if any(rects[z]['type']=='corridor' for z in seams[idx][4:]) else 1)
  selected=set()
  for i in perm:
   _,_,_,_,a,b=seams[i]
   if unite(a,b):selected.add(i)
  if len({root(i) for i in range(len(rects))})!=1:raise RuntimeError('disconnected floor-plan topology')
  for i,(_,_,lo,hi,a,b) in enumerate(seams):
   if i not in selected and rng.random()>.31:continue
   orient,at,lo,hi,a,b=seams[i]
   span=(hi-lo)//q
   width=2 if span>=6 and rng.random()<.20 else 1
   margin=1 if span>=width+2 else 0
   start_unit=int(rng.integers(lo//q+margin,hi//q-width-margin+1))
   opening('door',orient,at,start_unit*q,width*q,a,b,opened=bool(rng.random()*100<self.params['opening_open_percent']))
  # Exterior windows: several windowless interiors guaranteed by layout.
  outside=[]
  for a in self.geometry_rooms:
   for orient,at,start,nseg in (
    ('H',0,a['left'],a['width']) if a['top']==0 else (None,0,0,0),
    ('H',H,a['left'],a['width']) if a['top']+a['height']==H else (None,0,0,0),
    ('V',0,a['top'],a['height']) if a['left']==0 else (None,0,0,0),
    ('V',W,a['top'],a['height']) if a['left']+a['width']==W else (None,0,0,0),
   ):
    if orient is not None and nseg>=2*q:outside.append((orient,at,start,nseg,a['id']))
  rng.shuffle(outside)
  # Choose exterior doors FIRST and reserve their wall spans.
  corridor_candidates=[t for t in outside if self.geometry_rooms[t[4]]['type']=='corridor']
  exit_candidates=corridor_candidates if len(corridor_candidates)>=2 else outside
  chosen=[]
  for item in exit_candidates:
   if len(chosen)>=2:break
   if item[4] not in [x[4] for x in chosen] or not chosen:chosen.append(item)
  for orient,at,start,span,z in chosen:
   units=span//q;width=2 if units>=4 else 1
   pos=start//q+(units-width)//2
   opening('exit',orient,at,pos*q,width*q,z,opened=bool(rng.random()*100<self.params['opening_open_percent']))
  if not chosen:raise RuntimeError('no exterior exits')
  reserved=[(o['orient'],o['at'],o['start'],o['start']+o['length']) for o in self.openings if o['kind']=='exit']
  windowless_id=int(rng.choice([r['id'] for r in self.rooms]))
  for orient,at,start,nseg,z in outside:
   if z==windowless_id or rng.random()>.68:continue
   units=nseg//q;width=min(units-1,2 if units>=6 and rng.random()<.6 else 1)
   legal=[]
   for begin in range(start//q,start//q+units-width+1):
    lo=begin*q;hi=lo+width*q
    if any(ori==orient and edge==at and max(lo,a)<min(hi,b) for ori,edge,a,b in reserved):continue
    legal.append(begin)
   if not legal:continue
   begin=int(rng.choice(legal))
   opening('window',orient,at,begin*q,width*q,z,opened=bool(rng.random()*100<self.params['opening_open_percent']))
  for op in self.openings:op['blocked']=False
  # Walls occupy cells rather than being represented only by blocked faces.
  # Opening spans remain floor cells so an open door or window connects spaces.
  self.wall=np.zeros((H,W),dtype=bool)
  def opening_face(orient,at,position):
   return any(op['orient']==orient and op['at']==at and
              op['start']<=position<op['start']+op['length']
              and op['kind'] in ('door','window','exit') for op in self.openings)
  for r in range(H):
   for c in range(W-1):
    if self.zone[r,c]>=0 and self.zone[r,c+1]>=0 and self.zone[r,c]!=self.zone[r,c+1]:
     if not opening_face('V',c+1,r):self.wall[r,c]=True
  for r in range(H-1):
   for c in range(W):
    if self.zone[r,c]>=0 and self.zone[r+1,c]>=0 and self.zone[r,c]!=self.zone[r+1,c]:
     if not opening_face('H',r+1,c):self.wall[r,c]=True
  for r in range(H):
   if not opening_face('V',0,r):self.wall[r,0]=True
   if not opening_face('V',W,r):self.wall[r,W-1]=True
  for c in range(W):
   if not opening_face('H',0,c):self.wall[0,c]=True
   if not opening_face('H',H,c):self.wall[H-1,c]=True
  self.zone[self.wall]=-1
  self.east=np.zeros((H,W-1),dtype=np.float32)
  self.south=np.zeros((H-1,W),dtype=np.float32)
  first=self.corridors[0]
  self.corridor=[first['top'],first['top']+first['height']]
  self._opening_proximity=self._compute_opening_proximity()
  self._apply_openings()
  self.oxygen=np.full((H,W),self.params['initial_oxygen'],dtype=np.float32)
  self.oxygen[self.wall]=0
  self.smoke=np.zeros((H,W),dtype=np.float32)
  # Smooth noise is sampled on a coarse physical lattice, then bilinearly
  # interpolated onto the simulation grid. It does not vary in scale with q.
  step=max(1.,1.45*q)
  ry=np.arange(H,dtype=np.float32)/step;cx=np.arange(W,dtype=np.float32)/step
  self._noise_ri=np.floor(ry).astype(np.int32);self._noise_ci=np.floor(cx).astype(np.int32)
  self._noise_rf=(ry-self._noise_ri)[:,None];self._noise_cf=(cx-self._noise_ci)[None,:]
  self._noise_shape=(int(self._noise_ri[-1])+2,int(self._noise_ci[-1])+2)
  self.flame_noise=np.zeros((H,W),dtype=np.float32)
  self.fire=np.zeros((H,W),dtype=np.float32)
  mat_rng=np.random.default_rng(self.seed ^ 0x5EEDCAFE)
  self.material=np.zeros((H,W),dtype=np.uint8)
  wood_count=round(len(self.rooms)*self.params['wood_ratio']/100.)
  wood_ids=set(int(z) for z in mat_rng.choice([rm['id'] for rm in self.rooms],size=wood_count,replace=False))
  for z in wood_ids:self.material[self.zone==z]=1
  self.fuel=np.where(self.material==1,1.,.85).astype(np.float32)
  self.fuel[self.wall]=0
  self.smoke_flammability=np.zeros((H,W),np.float32)
  self.smoke_flammable_mass=np.zeros((H,W),np.float32)
  self.water_level=np.zeros((H,W),np.float32)

  self.damage=np.zeros((H,W),dtype=np.float32)
  self.debris=np.zeros((H,W),dtype=np.uint8)
  self.debris_lines=[];self.ticks=0
  self._spawn_initial_blockages()
  # Ignition is constrained to rooms, not arbitrary exterior positions.
  for _ in range(self.nignitions):
   room=self.rooms[int(rng.integers(len(self.rooms)))]
   # Same physical ignition position for a given seed across resolutions.
   ry=int(rng.integers(room['top']//q+1,max(room['top']//q+2,(room['top']+room['height'])//q-1)))
   cx=int(rng.integers(room['left']//q+1,max(room['left']//q+2,(room['left']+room['width'])//q-1)))
   # Choose the center of the SAME physical position at every resolution.
   # (r+0.5)/q is the coordinate of a numerical cell center.
   r=(ry+.5)*q-.5;c=(cx+.5)*q-.5
   intensity=float(rng.uniform(.58,.88))
   radius=.52*q;rad=int(math.ceil(radius+1))
   r0=max(0,int(math.floor(r-rad)));r1=min(H,int(math.ceil(r+rad+1)))
   c0=max(0,int(math.floor(c-rad)));c1=min(W,int(math.ceil(c+rad+1)))
   yy,xx=np.ogrid[r0:r1,c0:c1];mask=((yy-r)**2+(xx-c)**2<=radius**2)&(self.zone[r0:r1,c0:c1]==room['id'])
   self.fire[r0:r1,c0:c1][mask]=intensity
   # No unsolicited ignition-time debris. Initial blocked openings are configured separately.
  room_mask=np.isin(self.zone,[r['id'] for r in self.rooms])
  viable=np.argwhere(room_mask&(self.debris==0)&(self.fire<.05))
  self.victims=[]
  if self.nvictims>len(viable):raise ValueError('too many victims')
  choices=rng.choice(len(viable),size=self.nvictims,replace=False)
  for k,idx in enumerate(choices):
   r,c=map(int,viable[int(idx)])
   self.victims.append({'id':k,'r':r,'c':c,'category':'ABC'[k%3],'health':1.})
  self.agents=[]
  exit0=next(o for o in self.openings if o['kind']=='exit')
  spawn=self._cells_for_opening(exit0,exit0['zone_a'])
  available=[];seen={spawn};todo=deque([spawn])
  while todo and len(available)<self.nagents:
   rr,cc=todo.popleft()
   if not self.debris[rr,cc] and self.fire[rr,cc]<.2 and all(abs(rr-y)+abs(cc-x)>=max(2,q) for y,x in available):available.append((rr,cc))
   for nxt in ((rr-1,cc),(rr+1,cc),(rr,cc-1),(rr,cc+1)):
    ny,nx=nxt
    if 0<=ny<H and 0<=nx<W and nxt not in seen and self.zone[nxt]==exit0['zone_a']:
     seen.add(nxt);todo.append(nxt)
  if len(available)<self.nagents:
   raise ValueError(f'not enough safe spawn positions for {self.nagents} agents')
  for k in range(self.nagents):
   rr,cc=available[k]
   self.agents.append({'id':k,'r':rr,'c':cc})
  self.rebuild_draft()
  self._purge_debris_fields()
  return self
 def _portal_cells(self,op,z,depth=1):
  """All cells inside one side of an edge-aligned multi-cell opening."""
  coords=[];at=op['at'];st=op['start'];end=st+op['length']
  if op['orient']=='H':
   for r in (at-1,at):
    if 0<=r<self.h and 0<=st<self.w and self.zone[r,st]==z:
     direction=-1 if r<at else 1
     for d in range(depth):
      rr=r+direction*d
      for c in range(st,min(end,self.w)):
       if 0<=rr<self.h and self.zone[rr,c]==z:coords.append((rr,c))
  else:
   for c in (at-1,at):
    if 0<=c<self.w and 0<=st<self.h and self.zone[st,c]==z:
     direction=-1 if c<at else 1
     for d in range(depth):
      cc=c+direction*d
      for r in range(st,min(end,self.h)):
       if 0<=cc<self.w and self.zone[r,cc]==z:coords.append((r,cc))
  return coords
 def _opening_side_depth(self):
  return 1  # single-cell-thick barricades independent of resolution
 def _compute_opening_proximity(self):
  field=np.zeros((self.h,self.w),dtype=np.float32)
  reach=max(1,int(math.ceil(2.2*self.resolution)))
  for op in self.openings:
   if op['kind'] not in ('door','window','exit'):continue
   for z in (op['zone_a'],op['zone_b']):
    if z is None:continue
    for r,c in self._portal_cells(op,z,depth=reach):
     d=(abs(r-(op['at']-1 if op['orient']=='H' and r<op['at'] else op['at'] if op['orient']=='H' else r)) if op['orient']=='H' else abs(c-(op['at']-1 if c<op['at'] else op['at'])))
     val=max(0.,1-d/max(1,reach))
     if val>field[r,c]:field[r,c]=val
  return field
 def _purge_debris_fields(self):
  if not hasattr(self,'debris') or not np.any(self.debris):return
  mask=self.debris>0
  self.oxygen[mask]=0
  self.smoke[mask]=0
  self.fire[mask]=0
  self.fuel[mask]=0
  if hasattr(self,'draft'):self.draft[mask]=0
  if hasattr(self,'flame_noise'):self.flame_noise[mask]=0
 def _fill_opening_side(self,op,z):
  changed=False
  for r,c in self._portal_cells(op,z,depth=self._opening_side_depth()):
   if not self.debris[r,c]:
    self.debris[r,c]=1;changed=True
  return changed
 def _refresh_opening_blocks(self):
  changed=False
  structure_changed=False
  for op in self.openings:
   sides=[z for z in (op['zone_a'],op['zone_b']) if z is not None]
   blocked=False
   for z in sides:
    cells=self._portal_cells(op,z,depth=self._opening_side_depth())
    if any(self.debris[r,c] for r,c in cells):
     blocked=True
     if self._fill_opening_side(op,z):changed=True
   if op.get('blocked',False)!=blocked:
    op['blocked']=blocked;structure_changed=True
  if structure_changed:
   self._apply_openings();changed=True
  if changed and hasattr(self,'oxygen'):
   self._purge_debris_fields();self.rebuild_draft()
  return changed
 def _spawn_initial_blockages(self):
  """Barricade randomly selected door, window, and exit approaches."""
  eligible=[o for o in self.openings if o['kind'] in ('door','window','exit')]
  pct=self.params['initial_blocked_percent']
  n=min(len(eligible),int(math.floor(len(eligible)*pct/100+self.rng.random())))
  selected=self.rng.choice(len(eligible),size=n,replace=False) if n else []
  self.initial_blockages=[]
  for idx in selected:
   op=eligible[int(idx)]
   side=int(self.rng.choice([z for z in (op['zone_a'],op['zone_b']) if z is not None]))
   for r,c in self._portal_cells(op,side,depth=self._opening_side_depth()):self.debris[r,c]=1
   self.initial_blockages.append(op['id'])
  self._refresh_opening_blocks()
 def clear_debris(self,r,c,radius=0):
  if not(0<=r<self.h and 0<=c<self.w):raise ValueError('outside building')
  r0=max(0,r-radius);r1=min(self.h,r+radius+1)
  c0=max(0,c-radius);c1=min(self.w,c+radius+1)
  changed=bool(np.any(self.debris[r0:r1,c0:c1]))
  touched=[]
  for op in self.openings:
   for z in (op['zone_a'],op['zone_b']):
    if z is None:continue
    cells=self._portal_cells(op,z,depth=self._opening_side_depth())
    if any(r0<=rr<r1 and c0<=cc<c1 for rr,cc in cells):
     touched.append((op,z,cells))
  self.debris[r0:r1,c0:c1]=0
  for op,z,cells in touched:
   for rr,cc in cells:self.debris[rr,cc]=0
  if not self._refresh_opening_blocks() and changed:self.rebuild_draft()
  self._purge_debris_fields()
 def _apply_openings(self):
  self.east[:]=((self.zone[:,:-1]==self.zone[:,1:]) &
                (self.zone[:,:-1]>=0))
  self.south[:]=((self.zone[:-1,:]==self.zone[1:,:]) &
                 (self.zone[:-1,:]>=0))
  for op in self.openings:
   if op['kind']=='exit':continue
   permeability=(1. if op['open'] and not op.get('blocked',False) else .012) if op['kind']=='door' else 0.
   if op['kind']=='window':continue # exterior, handled by draft exchange
   lo=op['start'];hi=lo+op['length'];at=op['at']
   if op['orient']=='H':self.south[at-1,lo:hi]=permeability
   else:self.east[lo:hi,at-1]=permeability
 def toggle(self,oid):
  match=next((o for o in self.openings if o['id']==int(oid)),None)
  if match is None:raise ValueError('invalid opening id')
  match['open']=not match['open'];self._apply_openings();self.rebuild_draft()
  return match
 def _cells_for_opening(self,op,z):
  """Within-space inward cells, all openings occupy integer-length edge spans."""
  at=op['at'];ss=op['start'];n=op['length'];H,W=self.h,self.w
  if op['orient']=='H':
   candidates=[(at-1,ss+n//2),(at,ss+n//2)]
  else:candidates=[(ss+n//2,at-1),(ss+n//2,at)]
  return next(((r,c) for r,c in candidates if 0<=r<H and 0<=c<W and self.zone[r,c]==z),None)
 def _draw_gaussian(self,out,zone,r,c,sigma,weight):
  dx=1/self.resolution;radius=max(2,int(math.ceil(3.2*sigma/dx)))
  r0=max(0,r-radius);r1=min(self.h,r+radius+1);c0=max(0,c-radius);c1=min(self.w,c+radius+1)
  yy,xx=np.ogrid[r0:r1,c0:c1]
  dis2=((yy-r)*dx)**2+((xx-c)*dx)**2
  vals=weight*np.exp(-dis2/(2*sigma*sigma))
  patch=out[r0:r1,c0:c1];mask=self.zone[r0:r1,c0:c1]==zone
  np.maximum(patch,np.where(mask,vals,0),out=patch)
 def _draw_segment(self,out,z,start,end,weight,axis_x=None,axis_y=None):
  # Continuous line from port center to port center; includes collinear pairs.
  q=self.resolution; a=np.array(start,dtype=float); b=np.array(end,dtype=float)
  length=np.linalg.norm(b-a)
  sigma=max(.1,self.params['draft_width']*.5)
  radius=math.ceil(3.2*sigma*q)
  r0=max(0,int(min(a[0],b[0]))-radius);r1=min(self.h,int(max(a[0],b[0]))+radius+2)
  c0=max(0,int(min(a[1],b[1]))-radius);c1=min(self.w,int(max(a[1],b[1]))+radius+2)
  yy,xx=np.ogrid[r0:r1,c0:c1]
  dr=b[0]-a[0];dc=b[1]-a[1];length_sq=max(1e-8,length*length)
  t=np.clip(((yy-a[0])*dr+(xx-a[1])*dc)/length_sq,0,1)
  dist_sq=(yy-(a[0]+t*dr))**2+(xx-(a[1]+t*dc))**2
  side=np.exp(-.5*(dist_sq/(sigma*q)**2)**self.params['draft_falloff'])
  middle=self.params['draft_midpoint_factor']
  longitudinal=middle+(1-middle)*(np.abs(2*t-1)**1.5)
  vals=weight*longitudinal*side
  contribution=np.where(self.zone[r0:r1,c0:c1]==z,vals,0)
  out[r0:r1,c0:c1]+=contribution
  if axis_x is not None:
   axis_x[r0:r1,c0:c1]+=contribution*(dc*dc/length_sq)
   axis_y[r0:r1,c0:c1]+=contribution*(dr*dr/length_sq)
 def _ventilation_reachable(self):
  """Event-driven flood fill from exterior openings through open, debris-free
  passages. Prevents draft sources from replenishing isolated air compartments.
  Performed on topology edits, never at every physics step.
  """
  reachable=np.zeros((self.h,self.w),dtype=bool)
  todo=deque()
  for op in self.openings:
   if not op['open'] or op.get('blocked',False) or op['zone_b'] is not None:continue
   for r,c in self._portal_cells(op,op['zone_a']):
    if not self.debris[r,c] and not reachable[r,c]:
     reachable[r,c]=True;todo.append((r,c))
  while todo:
   r,c=todo.popleft()
   for nr,nc,conduct in ((r-1,c,self.south[r-1,c] if r>0 else 0),
                         (r+1,c,self.south[r,c] if r<self.h-1 else 0),
                         (r,c-1,self.east[r,c-1] if c>0 else 0),
                         (r,c+1,self.east[r,c] if c<self.w-1 else 0)):
    if conduct<.5 or not(0<=nr<self.h and 0<=nc<self.w):continue
    if not reachable[nr,nc] and not self.debris[nr,nc]:
     reachable[nr,nc]=True;todo.append((nr,nc))
  return reachable
 def rebuild_draft(self):
  """Recalculate ALL unique pairs of open ports per rectangle, with bounded
  source strengths and diminishing overlap contributions. No hidden outdoor cells.
  """
  ports=defaultdict(list);edges=defaultdict(list);outdoors=[]
  for op in self.openings:
   if not op['open'] or op.get('blocked',False):continue
   a,b=op['zone_a'],op['zone_b']
   for z in (a,b):
    if z is None:continue
    point=self._cells_for_opening(op,z)
    if point is not None:ports[z].append((op['id'],point,op['kind']))
   if b is None:
    outdoors.append((a,min(3.,op['length']/self.resolution),op['kind']))
   else:
    connection=.60+.20*(1-math.exp(-.8*op['length']/self.resolution))
    edges[a].append((b,connection));edges[b].append((a,connection))
  # Each outside connection contributes independently, attenuated by the most
  # conductive simple route through interior doors. No cumulative cycle boosting.
  source_effects=defaultdict(list)
  for z,width,kind in outdoors:
   strength=(.64 if kind=='window' else .79)*(1-math.exp(-1.0*width))
   best={z:strength};heap=[(-strength,z)]
   while heap:
    neg,u=heapq.heappop(heap);val=-neg
    if val+1e-9<best[u]:continue
    for v,k in edges[u]:
     candidate=val*k
     if candidate>best.get(v,0)+1e-9:
      best[v]=candidate;heapq.heappush(heap,(-candidate,v))
   for k,v in best.items():source_effects[k].append(v)
  fresh={z:float(1-np.prod(1-np.array(v))) for z,v in source_effects.items()}
  draft=np.zeros_like(self.oxygen);axis_x=np.zeros_like(draft);axis_y=np.zeros_like(draft);num=0
  for z,items in ports.items():
   strength=fresh.get(z,0)
   if strength<=1e-8:continue
   part=np.zeros_like(self.oxygen)
   gauss=np.zeros_like(self.oxygen)
   px=np.zeros_like(self.oxygen);py=np.zeros_like(self.oxygen)
   pairs=len(items)*(len(items)-1)//2
   for i,(_,p,kind) in enumerate(items):
    if kind in ('window','exit'):
     self._draw_gaussian(gauss,z,*p,sigma=.62,weight=strength)
    for _,p2,_ in items[i+1:]:
     self._draw_segment(part,z,p,p2,strength,px,py);num+=1
   # All pairs are accounted for. Their aggregate is normalized using a
   # sublinear saturation curve: extra ports strengthen a zone only modestly.
   # This avoids corridor brightness exploding with O(n^2) connector counts.
   scale=1.+.88*(pairs**.84)
   zm=self.zone==z
   draft[zm]=np.clip(gauss[zm]*.65+(1-np.exp(-part[zm]/scale)),0,.85)
   axis_x[zm]=px[zm]/np.maximum(part[zm],1e-8)
   axis_y[zm]=py[zm]/np.maximum(part[zm],1e-8)
  reachable=self._ventilation_reachable()
  self.draft=np.where(reachable,draft,0.).astype(np.float32)
  # Directional, unsigned orientation: there is no assumed wind direction.
  # Horizontal segments enhance x-axis fire transfer; vertical, y-axis.
  self.draft_axis_x=np.where(reachable,axis_x,0.)
  self.draft_axis_y=np.where(reachable,axis_y,0.)
  self.draft_segments=num
  self.draft_port_count=sum(len(v) for v in ports.values())
  self.draft_source_count=len(outdoors)
  self.max_draft=float(self.draft.max())
 def select_debris_landing(self,fr,fc):
  """One candidate landing, constrained to the fire's room.

  Bias only changes WHERE an event lands; its overall rate remains unchanged.
  """
  q=self.resolution;z=int(self.zone[fr,fc]);bias=self.params['debris_opening_bias']
  candidates=[];scores=[]
  for _ in range(16):
   angle=float(self.rng.uniform(0,2*math.pi))
   dist=float(self.rng.uniform(.45,3.2))*q
   rr=int(round(fr+dist*math.sin(angle)));cc=int(round(fc+dist*math.cos(angle)))
   if 0<=rr<self.h and 0<=cc<self.w and self.zone[rr,cc]==z and not self.debris[rr,cc]:
    candidates.append((rr,cc))
    scores.append(1+bias*float(self._opening_proximity[rr,cc]))
  if not candidates:return (fr,fc)
  weights=np.asarray(scores,dtype=float);weights/=weights.sum()
  return candidates[int(self.rng.choice(len(candidates),p=weights))]
 def make_debris(self,r,c,severity=1.,force=False):
  """Rasterize a fire-adjacent physical line to nearest grid-cell centers in ONE room."""
  r=int(r);c=int(c)
  if not(0<=r<self.h and 0<=c<self.w):return 0
  z=int(self.zone[r,c]);q=self.resolution; p=self.params
  # Structural debris is fire-triggered, never distributed arbitrarily.
  search=int(math.ceil(4.1*q));r0s=max(0,r-search);r1s=min(self.h,r+search+1);c0s=max(0,c-search);c1s=min(self.w,c+search+1)
  near=(self.fire[r0s:r1s,c0s:c1s]>=p['fire_threshold'])&(self.zone[r0s:r1s,c0s:c1s]==z)
  if not force and not np.any(near):return 0
  mult=1+p['debris_severity_scale']*max(0,min(1,float(severity)))
  length=min(4.8,float(self.rng.lognormal(math.log(p['debris_length_mean']),p['debris_length_spread']))*mult)
  thick=min(2.5,float(self.rng.lognormal(math.log(p['debris_thickness_mean']),p['debris_thickness_spread']))*math.sqrt(mult))
  angle=self.rng.uniform(0,math.pi)
  # Clip each endpoint independently to the inside of this room.
  # Unlike symmetric shrinking, this still permits a fragment centered
  # directly beside a door/window to extend *inward* into the room.
  rect=self.geometry_rooms[z];sa=math.sin(angle);ca=math.cos(angle)
  half=length*q*.5;dr=sa*half;dc=ca*half
  rlo=rect['top'];rhi=rlo+rect['height']-1
  clo=rect['left'];chi=clo+rect['width']-1
  def clip_endpoint(vr,vc):
   t=1.0
   if vr>0:t=min(t,(rhi-r)/vr)
   elif vr<0:t=min(t,(rlo-r)/vr)
   if vc>0:t=min(t,(chi-c)/vc)
   elif vc<0:t=min(t,(clo-c)/vc)
   t=max(0.,t)
   return r+vr*t,c+vc*t
   
  ra,ca=clip_endpoint(-dr,-dc)
  rb,cb=clip_endpoint(dr,dc)
  actual=math.hypot(rb-ra,cb-ca)/q
  radius=int(math.ceil(max(abs(ra-r),abs(rb-r),abs(ca-c),abs(cb-c))+thick*q+2))
  r0=max(0,r-radius);r1=min(self.h,r+radius+1)
  c0=max(0,c-radius);c1=min(self.w,c+radius+1)
  yy,xx=np.ogrid[r0:r1,c0:c1]
  vr=rb-ra;vc=cb-ca
  t=np.clip(((yy-ra)*vr+(xx-ca)*vc)/max(1e-8,vr*vr+vc*vc),0,1)
  d2=(yy-ra-t*vr)**2+(xx-ca-t*vc)**2
  mask=(d2<=(max(.5,thick*q*.5))**2)&(self.zone[r0:r1,c0:c1]==z)
  # Do NOT carve away opening-adjacent cells: a single touching pixel
  # must block the complete opening on that same side, via refresh below.
  patch=self.debris[r0:r1,c0:c1]
  before=np.count_nonzero(patch);patch[mask]=1
  self.debris_lines.append({'r':r,'c':c,'angle':float(angle),'length':round(actual,2),'thickness':round(thick,2),'fire':round(float(severity),2)})
  added=int(np.count_nonzero(patch)-before)
  if added:
   self._purge_debris_fields()
   if not self._refresh_opening_blocks():self.rebuild_draft()
  return added
 def ignite(self,r,c,intensity=.82):
  if not(0<=r<self.h and 0<=c<self.w):raise ValueError('outside building')
  if self.debris[r,c]:return
  self.fire[r,c]=max(float(self.fire[r,c]),float(intensity))
 def visual_fire(self):
  """Visible heat field. Never renders a cell that is inactive in physics."""
  f=self.fire;threshold=self.params['fire_threshold']
  if self.params['fire_core_flicker']<=0:return np.where(f>=threshold,f,0).astype(np.float32)
  core_start=max(threshold+.12,.36)
  hot=np.clip((f-core_start)/max(.01,1-core_start),0,1)**.6
  flicker=self.params['fire_core_flicker']*hot*self.flame_noise
  return np.where(f>=threshold,np.clip(f+flicker,threshold,1),0).astype(np.float32)


# Native ABI bindings (direct memory views, no serialization)
P_KEYS=(
 'fire_threshold','fire_spread','fire_growth','fire_decay','fuel_burn',
 'oxygen_consumption','smoke_generation','oxygen_diffusion','smoke_diffusion',
 'smoke_draft_boost','smoke_oxygen_drift','draft_oxygen_rate','smoke_settling',
 'fire_draft_boost','fire_core_flicker','fire_flicker_memory','draft_smoke_clearance',
 'damage_rate','debris_event_rate','debris_opening_bias','water_strength','water_radius',
 'draft_width','draft_midpoint_factor','draft_falloff',
 'wood_ratio','fuel_to_oxygen','concrete_flammability','wood_flammability','smoke_flammable_yield',
 'smoke_starvation_boost','smoke_fire_escape','water_cooling','water_base_evaporation',
 'water_fire_evaporation','water_heat_radius','flashback_min_mass','flashback_min_oxygen',
 'flashback_delay','flashback_strength','flashback_min_flammability','flashback_min_smoke','smoke_turbulence',
 'debris_length_mean','debris_length_spread','debris_thickness_mean','debris_thickness_spread','debris_severity_scale',
)
FIELD_NAMES=('oxygen','smoke','fire','fuel','damage','draft','flame_noise','draft_axis_x','draft_axis_y','smoke_flammability','smoke_flammable_mass','water_level')

class Person(C.Structure):
 _fields_=[('id',C.c_int32),('r',C.c_int32),('c',C.c_int32),('kind',C.c_int32),
           ('health',C.c_float),('rescued',C.c_int32),('dead',C.c_int32),('carried_by',C.c_int32),('stabilized',C.c_int32)]

class Opening(C.Structure):
 _fields_=[(name,C.c_int32) for name in ('id','kind','orient','at','start','length','za','zb','opened','blocked')]

class NativeKernel:
 def __init__(self,env):
  folder=Path(__file__).resolve().parent/'native'
  library=folder/('sim_native.dll' if sys.platform=='win32' else ('libsim_native.dylib' if sys.platform=='darwin' else 'libsim_native.so'))
  if not library.exists():
   raise RuntimeError(f'Native library not found: {library}. Run: python native/build.py')
  try:
   register_native_dll_dirs()
   self.lib=C.CDLL(str(library))
  except OSError as exc:
   if getattr(exc,'winerror',None)==4551:
    raise RuntimeError('Windows Application Control blocked the native FlameBox DLL (WinError 4551). Windows security denied loading the native library.') from exc
   raise
  self._bind()
  self.handle=self.lib.sim_new(env.h,env.w,env.resolution,C.c_uint64(env.seed))
  if not self.handle:raise MemoryError('Could not initialize C++ simulation')
  cuda_name='flamebox_gpu.dll' if sys.platform=='win32' else ('libflamebox_gpu.dylib' if sys.platform=='darwin' else 'libflamebox_gpu.so')
  os.environ['FLAMEBOX_GPU_LIBRARY']=str(folder/cuda_name)
  self.solver=os.getenv('FLAMEBOX_SOLVER','fast')
  gpu_supported=bool(self.lib.sim_gpu_available(self.handle))
  if self.solver=='auto':self.solver='cuda' if gpu_supported else 'parallel'
  if self.solver=='cuda' and not gpu_supported:
   self.lib.sim_free(self.handle);self.handle=None
   raise RuntimeError('CUDA solver requested, but the CUDA plugin is missing or cannot load. Run SETUP_GPU.cmd (Windows) or python Runtime/FlameBox/native/gpu_setup.py --build, then inspect BUILD_NATIVE.log. CUDA Toolkit nvcc and Windows MSVC C++ tools are required; MSYS2 g++ alone cannot compile CUDA.')
  self.lib.sim_solver_mode(self.handle,{'reference':0,'fast':1,'parallel':2,'cuda':3,'turbo':4}.get(self.solver,1))
  self.gpu_device=(self.lib.sim_gpu_device(self.handle) or b'').decode('utf-8',errors='replace') if self.solver=='cuda' else None
  self.cpu_threads=self.lib.sim_cpu_threads(self.handle)
  self.shape=(env.h,env.w)
  self.fields={}
  for k,name in enumerate(FIELD_NAMES):
   array=np.ctypeslib.as_array(self.lib.sim_field(self.handle,k),shape=(env.h*env.w,)).reshape(self.shape)
   array[:]=getattr(env,name) if hasattr(env,name) else 0
   self.fields[name]=array
  self.material=np.ctypeslib.as_array(self.lib.sim_material(self.handle),shape=(env.h*env.w,)).reshape(self.shape)
  self.material[:]=env.material
  self.debris=np.ctypeslib.as_array(self.lib.sim_debris(self.handle),shape=(env.h*env.w,)).reshape(self.shape)

  self.debris[:]=env.debris
  self.set_params(env.params)
  self.sync_edges(env)
  self.set_openings(env.openings)
  self.set_people(env.victims,env.agents)
 def _bind(self):
  lib=self.lib;ptr=C.c_void_p
  f=C.POINTER(C.c_float);i16=C.POINTER(C.c_int16);i32=C.POINTER(C.c_int32)
  specs={
   'sim_new':([C.c_int,C.c_int,C.c_int,C.c_uint64],ptr),
   'sim_free':([ptr],None),
   'sim_field':([ptr,C.c_int],f),
   'sim_debris':([ptr],C.POINTER(C.c_uint8)),
   'sim_material':([ptr],C.POINTER(C.c_uint8)),
   'sim_flashbacks':([ptr],C.c_int),
   'sim_rescan_water':([ptr],None),
   'sim_pending_flashbacks':([ptr],C.c_int),
   'sim_last_flash_cells':([ptr],C.c_int),
   'sim_zone':([ptr],i16),
   'sim_edges':([ptr,f,f,i16],None),
   'sim_proximity':([ptr,f],None),
   'sim_refresh':([ptr],None),
   'sim_params':([ptr,f,C.c_int],None),
   'sim_solver_mode':([ptr,C.c_int],None),
   'sim_solver_substeps':([ptr],C.c_int),
   'sim_cpu_threads':([ptr],C.c_int),
   'sim_gpu_available':([ptr],C.c_int),
   'sim_gpu_active':([ptr],C.c_int),
   'sim_gpu_device':([ptr],C.c_char_p),
   'sim_gpu_error':([ptr],C.c_char_p),
   'sim_profile_enable':([ptr,C.c_int],None),
   'sim_profile_read':([ptr,C.POINTER(C.c_double)],None),
   'sim_step':([ptr,C.c_int],C.c_int),
   'sim_event_r':([ptr],C.c_int),
   'sim_event_c':([ptr],C.c_int),
   'sim_last_events':([ptr],C.c_int),
   'sim_get_blocked':([ptr,i32,C.c_int],None),
   'sim_ticks':([ptr],C.c_int),
   'sim_max_draft':([ptr],C.c_float),
   'sim_route':([ptr,C.c_int,C.c_int,C.c_int,C.c_int,i32,C.c_int],C.c_int),
   'sim_route_world':([ptr,C.c_int,C.c_int,C.c_int,C.c_int,i32,C.c_int],C.c_int),
   'sim_water':([ptr,C.c_int,C.c_int,C.c_float,C.c_float],None),
   'sim_water_cone':([ptr,C.c_int,C.c_int,C.c_int,C.c_int,C.c_float,C.c_float],None),
   'sim_people':([ptr,C.c_int,C.POINTER(Person),C.c_int],None),
   'sim_people_count':([ptr,C.c_int],C.c_int),
   'sim_people_get':([ptr,C.c_int,C.POINTER(Person)],None),
   'sim_actions':([ptr,i32,i32,C.c_int],C.c_int),
   'sim_observe':([ptr,C.c_int,C.c_int,f],C.c_int),
   'sim_observe_batch':([ptr,C.c_int,f],C.c_int),
   'sim_move_mask_batch':([ptr,i32],C.c_int),
   'sim_grass_margin':([ptr],C.c_int),
   'sim_move_mask':([ptr,C.c_int],C.c_int),
   'sim_visibility':([ptr,C.c_int,C.POINTER(C.c_uint8)],C.c_int),
   'sim_openings':([ptr,C.POINTER(Opening),C.c_int],None),
   'sim_draft_segments':([ptr],C.c_int),
   'sim_draft_ports':([ptr],C.c_int),
   'sim_draft_sources':([ptr],C.c_int),
  }
  for name,(args,restype) in specs.items():
   method=getattr(lib,name);method.argtypes=args;method.restype=restype
 def close(self):
  if getattr(self,'handle',None):
   self.lib.sim_free(self.handle);self.handle=None
 def __del__(self):
  try:self.close()
  except Exception:pass
 def set_params(self,p):
  arr=np.array([p[key] for key in P_KEYS],dtype=np.float32)
  self.lib.sim_params(self.handle,arr.ctypes.data_as(C.POINTER(C.c_float)),len(arr))
 def update_draft(self,draft,x,y):
  for key,value in [('draft',draft),('draft_axis_x',x),('draft_axis_y',y)]:
   np.copyto(self.fields[key],np.asarray(value,dtype=np.float32))
  self.lib.sim_refresh(self.handle)
 def sync_edges(self,env):
  H,W=self.shape
  east=np.zeros((H,W),np.float32);south=np.zeros((H,W),np.float32)
  east[:,:-1]=env.east;south[:-1,:]=env.south
  zone=np.ascontiguousarray(env.zone,dtype=np.int16)
  self.lib.sim_edges(self.handle,east.ctypes.data_as(C.POINTER(C.c_float)),south.ctypes.data_as(C.POINTER(C.c_float)),zone.ctypes.data_as(C.POINTER(C.c_int16)))
  prox=np.ascontiguousarray(env._opening_proximity,dtype=np.float32)
  self.lib.sim_proximity(self.handle,prox.ctypes.data_as(C.POINTER(C.c_float)))
 def set_openings(self,openings):
  arr=(Opening*len(openings))()
  for i,op in enumerate(openings):
   p=arr[i];p.id=int(op['id']);p.kind={'door':0,'window':1,'exit':2}[op['kind']]
   p.orient=1 if op['orient']=='H' else 0
   p.at=int(op['at']);p.start=int(op['start']);p.length=int(op['length'])
   p.za=int(op['zone_a']);p.zb=-1 if op['zone_b'] is None else int(op['zone_b'])
   p.opened=int(op['open']);p.blocked=int(op.get('blocked',False))
  self.lib.sim_openings(self.handle,arr,len(arr))
 def blocked_states(self,n):
  out=np.empty(n,dtype=np.int32)
  self.lib.sim_get_blocked(self.handle,out.ctypes.data_as(C.POINTER(C.c_int32)),n)
  return out
 def draft_stats(self):
  return tuple(getattr(self.lib,key)(self.handle) for key in ('sim_draft_segments','sim_draft_ports','sim_draft_sources'))
 def sync_debris(self):self.lib.sim_refresh(self.handle)
 def step(self,n):
  completed=self.lib.sim_step(self.handle,int(n))
  if completed<0:
   reason=(self.lib.sim_gpu_error(self.handle) or b'Unknown CUDA failure').decode('utf-8',errors='replace')
   raise RuntimeError('CUDA physics execution failed: '+reason)
  return completed,self.lib.sim_event_r(self.handle),self.lib.sim_event_c(self.handle),self.lib.sim_last_events(self.handle)
 @property
 def ticks(self):return int(self.lib.sim_ticks(self.handle))
 @property
 def max_draft(self):return float(self.lib.sim_max_draft(self.handle))
 def route(self,a,b,world=False):
  H,W=self.shape
  if world or not(0<=a[0]<H and 0<=a[1]<W and 0<=b[0]<H and 0<=b[1]<W):
   q=int(self.lib.sim_grass_margin(self.handle))
   cap=(H+2*q)*(W+2*q)
   arr=np.empty(2*cap,np.int32)
   length=self.lib.sim_route_world(self.handle,int(a[0]),int(a[1]),int(b[0]),int(b[1]),arr.ctypes.data_as(C.POINTER(C.c_int32)),cap)
   return [(int(r),int(c)) for r,c in arr[:max(0,2*length)].reshape(-1,2)] if length>0 else []
  cap=self.shape[0]*self.shape[1]
  arr=np.empty(cap,np.int32)
  length=self.lib.sim_route(self.handle,int(a[0]),int(a[1]),int(b[0]),int(b[1]),arr.ctypes.data_as(C.POINTER(C.c_int32)),cap)
  if length<=0:return []
  W=self.shape[1]
  return [(int(j)//W,int(j)%W) for j in arr[:length]]
 def water(self,r,c,strength,radius):self.lib.sim_water(self.handle,r,c,C.c_float(strength),C.c_float(radius))
 def water_cone(self,r,c,dx,dy,strength,radius):self.lib.sim_water_cone(self.handle,r,c,int(dx),int(dy),C.c_float(strength),C.c_float(radius))
 def set_people(self,victims,agents):
  for type_,records in [(0,victims),(1,agents)]:
   persons=(Person*len(records))()
   for i,record in enumerate(records):
    p=persons[i];p.id=int(record['id']);p.r=int(record['r']);p.c=int(record['c'])
    p.kind=ord(record.get('category','A'))-ord('A') if not type_ else int(record.get('role',i))
    p.health=float(record.get('health',1.));p.dead=int(record.get('dead',False));p.rescued=int(record.get('rescued',False));p.carried_by=int(record.get('carried_by',-1));p.stabilized=int(record.get('stabilized',False))
   self.lib.sim_people(self.handle,type_,persons,len(records))
 def people(self,type_):
  n=self.lib.sim_people_count(self.handle,type_)
  arr=(Person*n)()
  if n:self.lib.sim_people_get(self.handle,type_,arr)
  records=[]
  for p in arr:
   d={'id':p.id,'r':p.r,'c':p.c,'health':float(p.health),'dead':bool(p.dead),'rescued':bool(p.rescued),'carried_by':int(p.carried_by),'stabilized':bool(p.stabilized)}
   if not type_:d['category']='ABC'[p.kind%3]
   else:d['role']=p.kind
   records.append(d)
  return records
 def rescan_water(self):self.lib.sim_rescan_water(self.handle)
 def flashbacks(self):return int(self.lib.sim_flashbacks(self.handle))
 def pending_flashbacks(self):return int(self.lib.sim_pending_flashbacks(self.handle))
 def last_flash_cells(self):return int(self.lib.sim_last_flash_cells(self.handle))
 def move_mask(self,agent):return int(self.lib.sim_move_mask(self.handle,int(agent)))
 def visibility(self,agent):
  result=np.zeros(self.shape,dtype=np.uint8)
  ptr=result.ctypes.data_as(C.POINTER(C.c_uint8))
  if self.lib.sim_visibility(self.handle,int(agent),ptr):raise IndexError('rescuer index out of range')
  return result

 def observe(self,agent,radius):
  r=int(radius);out=np.zeros((15,2*r+1,2*r+1),np.float32)
  ptr=out.ctypes.data_as(C.POINTER(C.c_float))
  if self.lib.sim_observe(self.handle,int(agent),r,ptr):raise ValueError('invalid observation request')
  return out
 def observe_batch(self,count,radius=7):
  r=int(radius);grid=np.zeros((count,15,2*r+1,2*r+1),np.float32)
  masks=np.empty(count,np.int32)
  if count:
   if self.lib.sim_observe_batch(self.handle,r,grid.ctypes.data_as(C.POINTER(C.c_float)))!=count:raise ValueError('invalid batched observations')
   self.lib.sim_move_mask_batch(self.handle,masks.ctypes.data_as(C.POINTER(C.c_int32)))
  return grid,masks
 def actions(self,commands,targets=None):
  commands=np.ascontiguousarray(commands,np.int32)
  targets=np.ascontiguousarray(targets if targets is not None else np.zeros(len(commands),dtype=np.int32),dtype=np.int32)
  if commands.shape!=targets.shape:raise ValueError('commands and targets shapes must match')
  ptr=C.POINTER(C.c_int32)
  result=self.lib.sim_actions(self.handle,commands.ctypes.data_as(ptr),targets.ctypes.data_as(ptr),len(commands))
  self.sync_debris()
  return int(result)


# Public engine: all runtime simulation steps are delegated to native C++.
class Environment(GeometryEnvironment):
 def reset(self):
  old=getattr(self,'_core',None)
  self._core=None
  if old is not None:
   old.close()
   for field in (*FIELD_NAMES, 'debris', 'material'):
    if hasattr(self,field):delattr(self,field)
  super().reset()
  self._core=NativeKernel(self)
  self._adopt_fields()
  return self
 def _adopt_fields(self):
  for name in FIELD_NAMES:setattr(self,name,self._core.fields[name])
  self.debris=self._core.debris
  self.material=self._core.material
  self.max_draft=self._core.max_draft
 def rebuild_draft(self):
  # Geometry updates happen only at generation / changed doors / new debris.
  if getattr(self,'_core',None) is None:
   super().rebuild_draft()
  else:
   self._core.sync_edges(self)
   self._core.set_openings(self.openings)
   self._adopt_fields()
   self.draft_segments,self.draft_port_count,self.draft_source_count=self._core.draft_stats()
 def set_params(self,values):
  super().set_params(values)
  if getattr(self,'_core',None) is not None:
   self._core.set_params(self.params)
   if any(k.startswith('draft_') for k in values):self.rebuild_draft()
 def tick(self,count=1):
  if not 0<=int(count)<=1000000:raise ValueError('invalid count')
  left=int(count)
  while left:
   done,r,c,events=self._core.step(left)
   if done<=0:raise RuntimeError('native step did not progress')
   left-=done;self.ticks=self._core.ticks
   if events:
    for op,blocked in zip(self.openings,self._core.blocked_states(len(self.openings))):
     op['blocked']=bool(blocked)
    self._apply_openings()
    self._native_debris_events=getattr(self,'_native_debris_events',0)+events
  self.victims=self._core.people(0)
  self.agents=self._core.people(1)
  return self
 def toggle(self,oid):
  out=super().toggle(oid)
  return out
 def _refresh_opening_blocks(self):
  out=super()._refresh_opening_blocks()
  if getattr(self,'_core',None) is not None:
   self._core.sync_debris()
  return out
 def clear_debris(self,r,c,radius=0):
  super().clear_debris(r,c,radius)
  self._core.sync_debris()
 def make_debris(self,r,c,severity=1.,force=False):
  added=super().make_debris(r,c,severity,force)
  if getattr(self,'_core',None) is not None and added:self._core.sync_debris()
  return added
 def water(self,r,c,strength=None,radius=None):
  if not(0<=r<self.h and 0<=c<self.w):raise ValueError('outside grid')
  self._core.water(int(r),int(c),1.0,float(self.params['water_radius'] if radius is None else radius))
 def spray(self,r,c,dx,dy):
  if not(0<=r<self.h and 0<=c<self.w):raise ValueError('outside grid')
  dx,dy=int(dx),int(dy)
  if (dx,dy)==(0,0) or dx not in (-1,0,1) or dy not in (-1,0,1):raise ValueError('direction must be one of eight neighbors')
  self._core.water_cone(int(r),int(c),dx,dy,1.0,float(self.params['water_radius']))
 def pathfind(self,start,goal,world=False):return self._core.route(start,goal,world=world)
 def action_batch(self,commands,targets=None,physics_ticks=1):
  """Headless RL action interface. 0 wait, 1 up, 2 right, 3 down, 4 left,
  5 water, 6 treat, 7 rescue on grass, 8 clear debris, 10 pick up/drop victim.
  Targets are linear cell indices for action 8; rescue flags are validated here.
  Returns {actions_succeeded, rewards, done, agents, victims, ticks}.
  """
  commands=np.asarray(commands,dtype=np.int32)
  if commands.ndim!=1 or len(commands)!=len(self.agents):raise ValueError('expected one action per agent')
  targets=np.asarray(targets if targets is not None else np.zeros(len(commands)),dtype=np.int32).copy()
  # C++ enforces grass rescue eligibility; no Python-generated rescue flags.
  agents=self._core.people(1)
  for k,a in enumerate(agents):
   if False and commands[k]==7:
    valid=False
    for op in self.openings:
     if op['kind']!='exit' or not op['open'] or op.get('blocked'):continue
     valid=valid or any(abs(r-a['r'])+abs(c-a['c'])<=1 for r,c in self._portal_cells(op,op['zone_a']))
    targets[k]=int(valid)
  before=sum(v['rescued'] for v in self._core.people(0))
  # Clear-opening semantics must remove the whole barricade, not one cell.
  # Topology edits remain rare and are handled by the geometry adapter.
  cleared=0
  for k,command in enumerate(commands):
   if command!=8:continue
   target=int(targets[k]);a=agents[k]
   if 0<=target<self.h*self.w and abs(target//self.w-a['r'])+abs(target%self.w-a['c'])<=1 and self.debris.flat[target]:
    self.clear_debris(target//self.w,target%self.w);cleared+=1
   commands[k]=0
  succeeded=self._core.actions(commands,targets)+cleared
  self.tick(physics_ticks)
  saved=sum(v['rescued'] for v in self.victims)
  return {'actions_succeeded':succeeded,'reward':float(10*(saved-before)),
          'done':all(v['rescued'] or v['dead'] for v in self.victims),
          'agents':self.agents,'victims':self.victims,'ticks':self.ticks}
 def observe_all(self,visibility_radius=7):
  """One C++ call for all agent grids/movement masks and one people call."""
  agents=self._core.people(1);grids,masks=self._core.observe_batch(len(agents),visibility_radius)
  result=[]
  for i,a in enumerate(agents):
   moves=[bool(int(masks[i])&(1<<k)) for k in range(4)]
   result.append({'grid':grids[i],
       'agent':np.array([a['r']/self.h,a['c']/self.w,a['health'],a['role']/3],np.float32),
       'actions_mask':np.array([True,*moves,*([True]*4),False,True],dtype=bool)})
  return result
 def observe(self,agent_id,visibility_radius=7):
  """C++ strict line-of-sight observation; 15 channels including flammability, water, material, entities and passability.
  Zeroed unknown cells can be distinguished from real zero values by channel 5.
  """
  agent=self._core.people(1)[int(agent_id)]
  grid=self._core.observe(agent_id,int(visibility_radius))
  r,c=agent['r'],agent['c']
  bits=self._core.move_mask(agent_id)
  moves=[bool(bits & (1<<k)) for k in range(4)]
  return {'grid':grid,'agent':np.array([r/self.h,c/self.w,agent['health'],agent['role']/3],np.float32),
          'actions_mask':np.array([True,*moves,*([True]*4),False,True],dtype=bool)}
 def __del__(self):
  try:
   if getattr(self,'_core',None) is not None:self._core.close()
  except Exception:pass
