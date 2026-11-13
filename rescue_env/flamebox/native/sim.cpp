// C++17 simulation kernel. Field storage and agent/victim lifetimes are native.
// Cell graph: dense 4-connected lattice with per-face conductance; zones are node metadata.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <queue>
#include <random>
#include <vector>
#include <array>
#include <limits>
#include <stdexcept>
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <string>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif


// Optional CUDA plugin: dynamically loaded, so the CPU reference engine works
// without a CUDA Toolkit, CUDA runtime, GPU driver, or supported NVIDIA card.
using GpuNew = void*(*)(int,int);
using GpuFree = void(*)(void*);
using GpuStep = int(*)(void*,float* const*,const float* const*,const uint8_t*,const uint8_t*,const float*,int,int,bool);
using GpuAvailable = int(*)();
using GpuError = const char*(*)();
using GpuDevice = const char*(*)();
struct GpuPlugin {
 void* module=nullptr;GpuNew create=nullptr;GpuFree destroy=nullptr;
 GpuStep step=nullptr;GpuAvailable available=nullptr;GpuError error=nullptr;GpuDevice device=nullptr;
 bool load(){
  if(module)return available && available()>0;
  const char* path=std::getenv("FLAMEBOX_GPU_LIBRARY");
  if(!path || !*path)return false;
  #ifdef _WIN32
   module=(void*)LoadLibraryExA(path,nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
   if(!module)return false;
   auto symbol=[&](const char* name){return (void*)GetProcAddress((HMODULE)module,name);};
  #else
   module=dlopen(path,RTLD_NOW|RTLD_LOCAL);
   if(!module)return false;
   auto symbol=[&](const char* name){return dlsym(module,name);};
  #endif
  create=(GpuNew)symbol("flamebox_cuda_new");
  destroy=(GpuFree)symbol("flamebox_cuda_free");
  step=(GpuStep)symbol("flamebox_cuda_step");
  available=(GpuAvailable)symbol("flamebox_cuda_available");
  error=(GpuError)symbol("flamebox_cuda_error");
  device=(GpuDevice)symbol("flamebox_cuda_device");
  return create&&destroy&&step&&available&&available()>0;
 }
 ~GpuPlugin(){
  if(module){
   #ifdef _WIN32
   FreeLibrary((HMODULE)module);
   #else
   dlclose(module);
   #endif
  }
 }
};

struct Person {int32_t id, r, c, kind; float health; int32_t rescued=0, dead=0, carried_by=-1, stabilized=0;};
struct Opening {int32_t id,kind,orient,at,start,length,za,zb,opened,blocked;};
struct Kernel {
 int H,W,N,q,grass_margin,ticks=0, event_r=-1,event_c=-1,last_events=0;
 std::array<std::vector<float>,12> a; // O, smoke, fire, fuel, damage, draft, noise, draft X/Y, smoke flam fraction/mass, water
 std::vector<uint8_t> material; // 0 concrete, 1 wood: floor only
 std::vector<float> smokeMassDelta,integral,escapeRate,escapeFraction;
 std::vector<int> wetIndices;
 std::vector<uint8_t> wetMark;
 struct Flash {int due;std::vector<int> cells;};
 std::vector<Flash> scheduled;
 int flashCount=0, lastFlashCells=0;
 int wetCells=0;
 bool staticCoeffValid=false;int cachedSubsteps=-1;
 bool parallelSmoke=false;
 bool profiling=false;
 std::array<double,7> profileMs{};
 using Clock=std::chrono::steady_clock;
 Clock::time_point checkpoint(){return Clock::now();}
 void accumulate_time(int stage,Clock::time_point started){
  if(profiling)profileMs[stage]+=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
 }
 int solverMode=1; // 0 reference; 1 fast; 2 parallel CPU; 3 CUDA; 4 turbo parallel CPU (experimental CFL)
 GpuPlugin gpu;
 void* gpuContext=nullptr;
 bool gpuFailure=false;
 std::string gpuLastError;
 bool gpu_ready(){return gpu.load();}
 bool gpu_init(){
  if(gpuContext)return true;
  if(!gpu_ready())return false;
  gpuContext=gpu.create(H,W);
  return gpuContext!=nullptr;
 }
 ~Kernel(){if(gpuContext&&gpu.destroy)gpu.destroy(gpuContext);}

 std::vector<int> debrisCells;
 std::vector<int> hotCells,heatTouched;
 std::vector<uint8_t> heatSeen,fireSeen;
 std::vector<uint8_t> debris;
 std::vector<int16_t> zone;
 std::vector<float> east,south,proximity,flux_e,flux_s;
 std::vector<float> delta,heat,coarse,oxyE,oxyS,smkE,smkS,fireE,fireS,smokeDecay;
 // Graph view of open conductance faces, partitioned into four conflict-free colors.
 // Built only when topology changes; vectors list face source cells in row-major order.
 std::array<std::vector<int>,4> openFaceIds;
 std::vector<Person> victims,agents;
 // Visibility depends on the observer's cell and the current topology, not
 // on smoke-frame updates. This cache avoids full perimeter raycasts per frame.
 uint64_t topologyVersion=0;
 mutable std::vector<std::vector<uint8_t>> visibilityCache;
 mutable std::vector<std::array<int64_t,3>> visibilityKeys;
 std::vector<Opening> openings;
 std::vector<std::array<int,4>> regionBounds;
 int draftSegments=0,draftPorts=0,draftSources=0;
 std::mt19937_64 rng;
 std::normal_distribution<float> gauss{0.f,1.f};
 // Numeric parameter order is defined by Python wrapper in flamebox.py.
 std::array<float,48> p{};
 int noisyH,noisyW;
 float maxDraft=0;
 Kernel(int h,int w,int resolution,uint64_t seed):H(h),W(w),N(h*w),q(resolution),grass_margin(3*resolution),rng(seed^0x6a09e667f3bcc909ULL) {
  for(auto &v:a)v.assign(N,0.f);
  debris.assign(N,0);zone.assign(N,0);east.assign(N,0);south.assign(N,0);material.assign(N,0);wetMark.assign(N,0);
  heatSeen.assign(N,0);fireSeen.assign(N,0);
  proximity.assign(N,0);flux_e.assign(N,0);flux_s.assign(N,0);
  delta.assign(N,0);heat.assign(N,0);smokeMassDelta.assign(N,0);escapeRate.assign(N,0);escapeFraction.assign(N,0);integral.assign((H+1)*(W+1),0);
  for(auto *v:{&oxyE,&oxyS,&smkE,&smkS,&fireE,&fireS,&smokeDecay})v->assign(N,0.f);
  float step=std::max(1.f,1.45f*q);
  noisyH=int(std::floor((H-1)/step))+2;
  noisyW=int(std::floor((W-1)/step))+2;
  coarse.assign(noisyH*noisyW,0);
 }
 void conduct() {
  ++topologyVersion;
  staticCoeffValid=false;
  std::fill(flux_e.begin(),flux_e.end(),0.f);std::fill(flux_s.begin(),flux_s.end(),0.f);
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   int i=r*W+c;
   if(debris[i])continue;
   if(c+1<W && !debris[i+1])flux_e[i]=east[i];
   if(r+1<H && !debris[i+W])flux_s[i]=south[i];
  }
  for(auto &faces:openFaceIds)faces.clear();
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   const int i=r*W+c;
   if(c+1<W && flux_e[i]>0.f)openFaceIds[c&1].push_back(i);
   if(r+1<H && flux_s[i]>0.f)openFaceIds[2+(r&1)].push_back(i);
  }
  maxDraft=0;for(int i=0;i<N;i++)maxDraft=std::max(maxDraft,a[5][i]);
 }
 // Event-driven all-pairs ventilation field. Geometry is the small portal graph;
 // only the resulting field is rasterized on the dense pixel grid.
 int portal_cell(const Opening &op,int z)const{
  int at=op.at,center=op.start+op.length/2;
  if(op.orient==1){
   for(int r:{at-1,at})if(valid(r,center) && zone[r*W+center]==z)return r*W+center;
  }else{
   for(int c:{at-1,at})if(valid(center,c) && zone[center*W+c]==z)return center*W+c;
  }
  return -1;
 }
 void draw_gaussian(std::vector<float>&out,int z,int i,float sigma,float strength){
  int r=i/W,c=i%W;float dx=1.f/q;
  int radius=std::max(2,int(std::ceil(3.2f*sigma/dx)));
  for(int y=std::max(0,r-radius);y<std::min(H,r+radius+1);y++)for(int x=std::max(0,c-radius);x<std::min(W,c+radius+1);x++){
   int j=y*W+x;if(zone[j]!=z)continue;
   float d2=((y-r)*(y-r)+(x-c)*(x-c))*dx*dx;
   float v=strength*std::exp(-d2/(2*sigma*sigma));
   out[j]=std::max(out[j],v);
  }
 }
 void draw_segment(std::vector<float>&part,std::vector<float>&px,std::vector<float>&py,
                   int z,int ia,int ib,float weight){
  float ar=ia/W,ac=ia%W,br=ib/W,bc=ib%W;
  float dr=br-ar,dc=bc-ac,L2=std::max(1e-8f,dr*dr+dc*dc);
  float sigma=std::max(.1f,p[22]*.5f);
  int rad=int(std::ceil(3.2f*sigma*q));
  int r0=std::max(0,int(std::min(ar,br))-rad),r1=std::min(H,int(std::max(ar,br))+rad+2);
  int c0=std::max(0,int(std::min(ac,bc))-rad),c1=std::min(W,int(std::max(ac,bc))+rad+2);
  float inv=1.f/((sigma*q)*(sigma*q));
  float orientX=dc*dc/L2,orientY=dr*dr/L2;
  for(int r=r0;r<r1;r++)for(int c=c0;c<c1;c++){
   int i=r*W+c;if(zone[i]!=z)continue;
   float t=std::clamp(((r-ar)*dr+(c-ac)*dc)/L2,0.f,1.f);
   float dR=r-(ar+t*dr),dC=c-(ac+t*dc);
   float side=std::exp(-.5f*std::pow((dR*dR+dC*dC)*inv,p[24]));
   float mid=p[23]+(1-p[23])*std::pow(std::abs(2*t-1),1.5f);
   float value=weight*mid*side;
   part[i]+=value;px[i]+=value*orientX;py[i]+=value*orientY;
  }
 }
 void rebuild_draft(){
  struct Port {int id,cell,kind;};
  std::vector<std::vector<Port>> ports;
  std::vector<std::vector<std::pair<int,float>>> graph;
  int maxZone=0;for(auto z:zone)maxZone=std::max(maxZone,int(z));
  ports.resize(maxZone+1);graph.resize(maxZone+1);
  struct Outside {int z,kind;float width;};
  std::vector<Outside> outside;
  draftSegments=0;draftSources=0;draftPorts=0;
  for(const auto &op:openings){
   if(!op.opened || op.blocked)continue;
   for(int z:{op.za,op.zb}){
    if(z<0||z>maxZone)continue;
    int pos=portal_cell(op,z);
    if(pos>=0)ports[z].push_back({op.id,pos,op.kind});
   }
   if(op.zb<0)outside.push_back({op.za,op.kind,std::min(3.f,float(op.length)/q)});
   else{
    float conn=.60f+.20f*(1-std::exp(-.8f*float(op.length)/q));
    graph[op.za].push_back({op.zb,conn});graph[op.zb].push_back({op.za,conn});
   }
  }
  draftSources=int(outside.size());
  for(const auto &p_:ports)draftPorts+=int(p_.size());
  std::vector<float> freshProduct(maxZone+1,1.f);
  for(const auto &out:outside){
   float strength=(out.kind==1?.64f:.79f)*(1-std::exp(-out.width));
   std::vector<float> best(maxZone+1,0.f);
   std::priority_queue<std::pair<float,int>> pq;
   pq.push({strength,out.z});best[out.z]=strength;
   while(!pq.empty()){
    auto [val,u]=pq.top();pq.pop();
    if(val+1e-9f<best[u])continue;
    for(auto [v,w]:graph[u]){
     float nv=val*w;
     if(nv>best[v]+1e-9f){best[v]=nv;pq.push({nv,v});}
    }
   }
   for(int z=0;z<=maxZone;z++)if(best[z]>0)freshProduct[z]*=(1-best[z]);
  }
  std::vector<float>part(N,0),gauss(N,0),px(N,0),py(N,0);
  std::vector<float> &draft=a[5],&axisX=a[7],&axisY=a[8];
  std::fill(draft.begin(),draft.end(),0.f);
  std::fill(axisX.begin(),axisX.end(),0.f);
  std::fill(axisY.begin(),axisY.end(),0.f);
  for(int z=0;z<=maxZone;z++){
   float strength=1-freshProduct[z];if(strength<1e-8f)continue;
   std::fill(part.begin(),part.end(),0.f);
   std::fill(gauss.begin(),gauss.end(),0.f);
   std::fill(px.begin(),px.end(),0.f);
   std::fill(py.begin(),py.end(),0.f);
   auto &v=ports[z];int pairs=int(v.size()*(v.size()-1)/2);
   for(int j=0;j<(int)v.size();j++){
    if(v[j].kind==1 ||v[j].kind==2)draw_gaussian(gauss,z,v[j].cell,.62f,strength);
    for(int k=j+1;k<(int)v.size();k++){
     draw_segment(part,px,py,z,v[j].cell,v[k].cell,strength);
     draftSegments++;
    }
   }
   float scale=1+.88f*std::pow(float(pairs),.84f);
   for(int i=0;i<N;i++)if(zone[i]==z){
    draft[i]=std::clamp(gauss[i]*.65f+(1-std::exp(-part[i]/scale)),0.f,.85f);
    axisX[i]=px[i]/std::max(part[i],1e-8f);
    axisY[i]=py[i]/std::max(part[i],1e-8f);
   }
  }
  // 4-neighbor BFS from every exterior source, not through closed doors/debris.
  std::vector<uint8_t> reached(N,0);std::queue<int> bfs;
  for(auto &op:openings){
   if(!op.opened||op.blocked||op.zb>=0)continue;
   if(op.orient==1){
    for(int c=op.start;c<op.start+op.length;c++)for(int r:{op.at-1,op.at}){
     if(valid(r,c)){int i=r*W+c;if(zone[i]==op.za&&!debris[i]&&!reached[i]){reached[i]=1;bfs.push(i);}}
    }
   }else{
    for(int r=op.start;r<op.start+op.length;r++)for(int c:{op.at-1,op.at}){
     if(valid(r,c)){int i=r*W+c;if(zone[i]==op.za&&!debris[i]&&!reached[i]){reached[i]=1;bfs.push(i);}}
    }
   }
  }
  while(!bfs.empty()){
   int i=bfs.front();bfs.pop();int r=i/W,c=i%W;
   for(int j:{c?i-1:-1,c+1<W?i+1:-1,r?i-W:-1,r+1<H?i+W:-1}){
    if(j<0||reached[j]||!traversable(i,j))continue;
    reached[j]=1;bfs.push(j);
   }
  }
  for(int i=0;i<N;i++)if(!reached[i]||debris[i])draft[i]=axisX[i]=axisY[i]=0.f;
  conduct();
 }
 // Native room-bounded falling-debris rasterization and automatic full-span
 // barricades. Called from C++ physics, not via Python per-tick callbacks.
 void refresh_bounds(){
  int mz=0;for(auto z:zone)mz=std::max(mz,int(z));
  regionBounds.assign(mz+1,{H,W,-1,-1});
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   if(zone[r*W+c]<0)continue;
   auto &b=regionBounds[zone[r*W+c]];
   b[0]=std::min(b[0],r);b[1]=std::min(b[1],c);
   b[2]=std::max(b[2],r);b[3]=std::max(b[3],c);
  }
 }
 void reapply_openings(){
  std::fill(east.begin(),east.end(),0.f);
  std::fill(south.begin(),south.end(),0.f);
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   int i=r*W+c;
   if(c+1<W && zone[i]>=0 && zone[i]==zone[i+1])east[i]=1.f;
   if(r+1<H && zone[i]>=0 && zone[i]==zone[i+W])south[i]=1.f;
  }
  for(auto &o:openings){
   if(o.kind!=0 || o.zb<0)continue;
   float cond=(o.opened && !o.blocked)?1.f:.012f;
   for(int j=o.start;j<o.start+o.length;j++){
    if(o.orient==1 && o.at>0 && o.at<H && j>=0 && j<W)south[(o.at-1)*W+j]=cond;
    if(o.orient==0 && o.at>0 && o.at<W && j>=0 && j<H)east[j*W+o.at-1]=cond;
   }
  }
  conduct();
 }
 void block_openings_from_debris(){
  int depth=1; // opening barricades occupy one cell in depth
  bool changed=false;
  for(auto &op:openings){
   bool blocked=false;
   for(int z:{op.za,op.zb}){
    if(z<0)continue;
    bool touching=false;
    if(op.orient==1){
     for(int baseR:{op.at-1,op.at})for(int c=op.start;c<op.start+op.length;c++){
      if(!valid(baseR,c)||zone[baseR*W+c]!=z)continue;
      int dir=baseR<op.at?-1:1;
      for(int d=0;d<depth;d++){
       int r=baseR+dir*d;
       if(valid(r,c)&&zone[r*W+c]==z&&debris[r*W+c])touching=true;
      }
     }
     if(touching){
      for(int baseR:{op.at-1,op.at})for(int c=op.start;c<op.start+op.length;c++){
       if(!valid(baseR,c)||zone[baseR*W+c]!=z)continue;
       int dir=baseR<op.at?-1:1;
       for(int d=0;d<depth;d++){int r=baseR+dir*d;if(valid(r,c)&&zone[r*W+c]==z)debris[r*W+c]=1;}
      }
     }
    }else{
     for(int baseC:{op.at-1,op.at})for(int r=op.start;r<op.start+op.length;r++){
      if(!valid(r,baseC)||zone[r*W+baseC]!=z)continue;
      int dir=baseC<op.at?-1:1;
      for(int d=0;d<depth;d++){
       int c=baseC+dir*d;
       if(valid(r,c)&&zone[r*W+c]==z&&debris[r*W+c])touching=true;
      }
     }
     if(touching){
      for(int baseC:{op.at-1,op.at})for(int r=op.start;r<op.start+op.length;r++){
       if(!valid(r,baseC)||zone[r*W+baseC]!=z)continue;
       int dir=baseC<op.at?-1:1;
       for(int d=0;d<depth;d++){int c=baseC+dir*d;if(valid(r,c)&&zone[r*W+c]==z)debris[r*W+c]=1;}
      }
     }
    }
    blocked|=touching;
   }
   if(int(blocked)!=op.blocked){op.blocked=blocked;changed=true;}
  }
  if(changed)reapply_openings();
  purge();conduct();rebuild_draft();
 }
 void generate_debris(int fr,int fc){
  int fireIdx=fr*W+fc,z=zone[fireIdx];
  if(z<0||z>=(int)regionBounds.size())return;
  std::uniform_real_distribution<float> u(0.f,1.f);
  std::vector<int> candidates;std::vector<float> weights;
  for(int k=0;k<16;k++){
   float angle=u(rng)*6.28318530718f;
   float distance=(.45f+u(rng)*2.75f)*q;
   int r=std::lround(fr+distance*std::sin(angle));
   int c=std::lround(fc+distance*std::cos(angle));
   if(!valid(r,c)||zone[r*W+c]!=z||debris[r*W+c])continue;
   candidates.push_back(r*W+c);weights.push_back(1+p[19]*proximity[r*W+c]);
  }
  int place=fireIdx;
  if(!candidates.empty()){
   float sum=std::accumulate(weights.begin(),weights.end(),0.f),v=u(rng)*sum;
   for(size_t k=0;k<candidates.size();k++){v-=weights[k];if(v<=0 ||k+1==candidates.size()){place=candidates[k];break;}}
  }
  int r=place/W,c=place%W;
  float mult=1+p[47]*std::clamp(a[2][fireIdx],0.f,1.f);
  std::lognormal_distribution<float> lengthDist(std::log(std::max(.01f,p[43])),p[44]);
  std::lognormal_distribution<float> thickDist(std::log(std::max(.01f,p[45])),p[46]);
  float length=std::min(4.8f,lengthDist(rng)*mult);
  float thick=std::min(2.5f,thickDist(rng)*std::sqrt(mult));
  float theta=u(rng)*3.14159265359f,dr=std::sin(theta)*length*q*.5f,dc=std::cos(theta)*length*q*.5f;
  auto &b=regionBounds[z];
  auto endpoint=[&](float dy,float dx){
   float t=1.f;
   if(dy>0)t=std::min(t,(b[2]-r)/dy);else if(dy<0)t=std::min(t,(b[0]-r)/dy);
   if(dx>0)t=std::min(t,(b[3]-c)/dx);else if(dx<0)t=std::min(t,(b[1]-c)/dx);
   t=std::max(t,0.f);
   return std::pair<float,float>{r+dy*t,c+dx*t};
  };
  auto aa=endpoint(-dr,-dc),bb=endpoint(dr,dc);
  float dy=bb.first-aa.first,dx=bb.second-aa.second;
  float norm=std::max(1e-8f,dy*dy+dx*dx);
  int radius=int(std::ceil(std::max({std::abs(aa.first-r),std::abs(bb.first-r),std::abs(aa.second-c),std::abs(bb.second-c)})+thick*q+2));
  int added=0;
  for(int rr=std::max(0,r-radius);rr<std::min(H,r+radius+1);rr++)for(int cc=std::max(0,c-radius);cc<std::min(W,c+radius+1);cc++){
   int i=rr*W+cc;if(zone[i]!=z||debris[i])continue;
   float t=std::clamp(((rr-aa.first)*dy+(cc-aa.second)*dx)/norm,0.f,1.f);
   float vr=rr-aa.first-t*dy,vc=cc-aa.second-t*dx;
   float thickness=std::max(.5f,thick*q*.5f);
   if(vr*vr+vc*vc<=thickness*thickness){debris[i]=1;added++;}
  }
  if(added){last_events++;block_openings_from_debris();}
 }
 void purge(){for(int i=0;i<N;i++)if(debris[i]){for(int k=0;k<12;k++)a[k][i]=0;}}
 void refresh_debris_cells(){
  debrisCells.clear();
  for(int i=0;i<N;i++)if(debris[i])debrisCells.push_back(i);
 }
 void purge_debris_cells(){
  for(int i:debrisCells)for(int k=0;k<12;k++)a[k][i]=0.f;
 }
 // Equivalent 5-point diffusion stencil: independent destination cells.
 // Coefficients are evaluated for each undirected face just once, preserving
 // conservative pairwise exchange up to floating-point summation order.
 void oxygen_stencil(){
  const auto &old=a[0];
  const int w=W,h=H;
  #pragma omp parallel for schedule(static) if(N>=32768)
  for(int r=0;r<h;r++){
   int base=r*w;
   for(int c=0;c<w;c++){
    const int i=base+c;
    float center=old[i],sum=0.f;
    if(c+1<w)sum+=oxyE[i]*(old[i+1]-center);
    if(c>0)sum+=oxyE[i-1]*(old[i-1]-center);
    if(r+1<h)sum+=oxyS[i]*(old[i+w]-center);
    if(r>0)sum+=oxyS[i-w]*(old[i-w]-center);
    delta[i]=center+sum;
   }
  }
  std::memcpy(a[0].data(),delta.data(),N*sizeof(float));
 }
 void noise(){
  if(p[14]<=0 && p[42]<=0)return;
  for(float &c:coarse)c=gauss(rng);
  float step=std::max(1.f,1.45f*q),rho=p[15],alpha=std::sqrt(std::max(0.f,1-rho*rho));
  for(int r=0;r<H;r++){
   float yr=r/step;int rr=std::min(noisyH-2,int(yr));float fr=yr-rr;
   for(int c=0;c<W;c++){
    float xc=c/step;int cc=std::min(noisyW-2,int(xc));float fc=xc-cc;
    float v0=coarse[rr*noisyW+cc]*(1-fc)+coarse[rr*noisyW+cc+1]*fc;
    float v1=coarse[(rr+1)*noisyW+cc]*(1-fc)+coarse[(rr+1)*noisyW+cc+1]*fc;
    int i=r*W+c;a[6][i]=rho*a[6][i]+alpha*(v0*(1-fr)+v1*fr);
   }
  }
 }
 // Each face is evaluated once, equal-and-opposite updates conserve total gas.
 void transport(int field,float dt,bool isSmoke){
  auto &f=a[field];const auto &ec=isSmoke?smkE:oxyE, &sc=isSmoke?smkS:oxyS;
  if(isSmoke){
   auto &mass=a[10];
   // Per-cell GATHER: only the source cell writes its outgoing fire-driven
   // displacement budget, so this pass is race-free and NUMA-friendly.
   // Precompute expm1 once per source cell rather than repeatedly per face.
   // The reference solver keeps its original traversal/rounding order.
   if(solverMode==0){
    std::fill(escapeRate.begin(),escapeRate.end(),0.f);
    for(int r=0;r<H;r++)for(int c=0;c<W;c++){
     int i=r*W+c;
     if(c+1<W&&flux_e[i]>0){int j=i+1;float d=(a[2][i]-a[2][j])*flux_e[i];
      if(d>0)escapeRate[i]+=d;else escapeRate[j]-=d;}
     if(r+1<H&&flux_s[i]>0){int j=i+W;float d=(a[2][i]-a[2][j])*flux_s[i];
      if(d>0)escapeRate[i]+=d;else escapeRate[j]-=d;}
    }
   }else{
    #pragma omp parallel for schedule(static) if(N>=32768)
    for(int i=0;i<N;i++){
     const float center=a[2][i];
     if(center<=0.f){escapeRate[i]=0.f;escapeFraction[i]=0.f;continue;}
     const int r=i/W,c=i%W;
     float total=0.f;
     if(c+1<W)total+=std::max(0.f,(center-a[2][i+1])*flux_e[i]);
     if(c>0)total+=std::max(0.f,(center-a[2][i-1])*flux_e[i-1]);
     if(r+1<H)total+=std::max(0.f,(center-a[2][i+W])*flux_s[i]);
     if(r>0)total+=std::max(0.f,(center-a[2][i-W])*flux_s[i-W]);
     escapeRate[i]=total;
     escapeFraction[i]=total>1e-12f?
      -std::expm1(-p[31]*dt*total*(q/3.f))/total:0.f;
    }
   }
   // Flux limitation operates on each face; smoke & combustible portion remain
   // conservative, nonnegative, and <=1 without a global clipping mass sink.
   auto apply=[&](int i,int j,float coef,float conduct){
    // Empty neighboring cells cannot exchange smoke or its constituents.
    // This cheap check bypasses the expensive transport/mixing math for
    // the large smoke-free portion of a building without altering results.
    if(coef<=0 || (f[i]==0.f && f[j]==0.f))return;
    float x=coef*(f[j]-f[i]);
    float v=p[10]*dt*q*(a[0][i]-a[0][j])*conduct;
    x-=v*(v>0?f[i]:f[j]);
    float d=a[2][i]-a[2][j];
    if(d!=0){int src=d>0?i:j;
     // Outward smoke transport responds to a spatial fire gradient.
     // Neighbor intensity difference is proportional to dx; multiply by
     // resolution to preserve a fixed physical transport velocity. /3
     // preserves the existing reference dynamics at resolution 3.
     float frac=solverMode==0?
      (escapeRate[src]>1e-12f?-std::expm1(-p[31]*dt*escapeRate[src]*(q/3.f))/escapeRate[src]:0.f):
      escapeFraction[src];
     x+=(d>0?-1.f:1.f)*std::abs(d)*conduct*f[src]*frac;
    }
    if(x>0){ // j -> i
     x=std::min(x,std::min(f[j],1.f-f[i]));
     if(x>0){
      float frac=f[j]>1e-10f?std::clamp(mass[j]/f[j],0.f,1.f):0;
      float moved=std::min(mass[j],frac*x);
      f[j]-=x;f[i]+=x;mass[j]-=moved;mass[i]+=moved;
     }
    }else if(x<0){ // i -> j
     x=std::min(-x,std::min(f[i],1.f-f[j]));
     if(x>0){
      float frac=f[i]>1e-10f?std::clamp(mass[i]/f[i],0.f,1.f):0;
      float moved=std::min(mass[i],frac*x);
      f[i]-=x;f[j]+=x;mass[i]-=moved;mass[j]+=moved;
     }
    }
    // Independent constituent interdiffusion. Equal SMOKE levels can still
    // exchange flammability. The paired exchange conserves BOTH smoke and
    // the flammable mass, and never takes more than a cell contains.
    float fi=f[i],fj=f[j];
    if(fi<=1.e-8f||fj<=1.e-8f)return;
    float ci=std::clamp(mass[i]/fi,0.f,1.f),cj=std::clamp(mass[j]/fj,0.f,1.f);
    float ex=std::min(.30f,std::max(0.f,coef))*std::min(fi,fj)*(cj-ci);
    if(ex>0){
     ex=std::min(ex,std::min(mass[j],fi-mass[i]));
    }else{
     ex=-std::min(-ex,std::min(mass[i],fj-mass[j]));
    }
    mass[i]+=ex;mass[j]-=ex;
   };
   if((solverMode==2 || solverMode==4) && parallelSmoke){
    // Pre-indexed, conflict-free sparse edge graph: skip blocked/impermeable
    // faces altogether. Graph updates occur only when conductance changes.
    #pragma omp parallel if(N>=32768)
    {
     for(int phase=0;phase<4;phase++){
      const auto &faces=openFaceIds[phase];
      const int count=(int)faces.size();
      #pragma omp for schedule(static)
      for(int k=0;k<count;k++){
       int i=faces[k];
       if(phase<2)apply(i,i+1,ec[i],flux_e[i]);
       else apply(i,i+W,sc[i],flux_s[i]);
      }
     }
    }
   }else{
    for(int r=0;r<H;r++)for(int c=0;c<W;c++){
     int i=r*W+c;
     if(c+1<W)apply(i,i+1,ec[i],flux_e[i]);
     if(r+1<H)apply(i,i+W,sc[i],flux_s[i]);
    }
   }
   return;
  }
  std::fill(delta.begin(),delta.end(),0.f);
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   int i=r*W+c;
   if(c+1<W&&ec[i]!=0){int j=i+1;float flux=ec[i]*(f[j]-f[i]);delta[i]+=flux;delta[j]-=flux;}
   if(r+1<H&&sc[i]!=0){int j=i+W;float flux=sc[i]*(f[j]-f[i]);delta[i]+=flux;delta[j]-=flux;}
  }
  for(int i=0;i<N;i++)f[i]+=delta[i];
 }

 // Opening-triggered only: both sides are independently evaluated, not every tick.
 void evaluate_flashback(const Opening &op){
  for(int z:{op.za,op.zb}){
   if(z<0)continue;
   std::vector<int> candidates;
   double smokeSum=0,flammable=0,oxy=0;
   if(z>=(int)regionBounds.size())continue;
   auto b=regionBounds[z];
   for(int r=b[0];r<=b[2];r++)for(int c=b[1];c<=b[3];c++){
    int i=r*W+c;if(zone[i]!=z||debris[i])continue;
    if(a[1][i]>=p[41]&&a[9][i]>=p[40]){candidates.push_back(i);smokeSum+=a[1][i];flammable+=a[10][i];oxy+=a[0][i];}
   }
   // Concentrations live on cells, but flashback needs physical inventory.
   // Summing cell values without dividing by q*q made high-resolution
   // buildings artificially more likely to flash back.
   if(candidates.empty() || flammable/(float(q)*q)<p[36])continue;
   // A fireball must have a minimum amount of *flammable* smoke, and events
   // must never grow without bound even in an enormous generated building.
   // Bound event area in physical units, not in number of cells. Keep
   // the original resolution-3 footprint (~228 physical square units).
   const int MAX_FLASH_CELLS=std::min(N,228*q*q);
   if(int(candidates.size())>MAX_FLASH_CELLS){
    const size_t stride=(candidates.size()+MAX_FLASH_CELLS-1)/MAX_FLASH_CELLS;
    std::vector<int> sampled;sampled.reserve(MAX_FLASH_CELLS);
    for(size_t ci=0;ci<candidates.size()&&sampled.size()<MAX_FLASH_CELLS;ci+=stride)sampled.push_back(candidates[ci]);
    candidates.swap(sampled);
   }
   // Estimate new oxygen admitted through this particular opening over the
   // configured ignition delay. One event-time zone scan, NEVER per physics tick.
   int other=op.za==z?op.zb:op.za;
   float otherO=1.f;
   if(other>=0 && other<(int)regionBounds.size()){
    auto b=regionBounds[other];double total=0;int cells=0;
    for(int r=b[0];r<=b[2];r++)for(int c=b[1];c<=b[3];c++){
     int i=r*W+c;if(zone[i]==other&&!debris[i]){total+=a[0][i];cells++;}
    }
    otherO=cells?float(total/cells):0.f;
   }
   float currentO=float(oxy/candidates.size());
   float mixing=1.f-std::exp(-p[11]*p[38]*std::max(.25f,float(op.length)/q)*2.f);
   float predictedO=currentO+mixing*std::max(0.f,otherO-currentO);
   if(predictedO<p[37])continue;
   // Do not duplicate already pending cells in this compartment.
   bool pending=false;for(const auto &f:scheduled)if(!f.cells.empty()&&zone[f.cells[0]]==z)pending=true;
   if(!pending)scheduled.push_back({ticks+std::max(1,int(std::round(p[38]))),std::move(candidates)});
  }
 }
 void process_flashbacks(){
  for(size_t k=0;k<scheduled.size();){
   if(scheduled[k].due>ticks){k++;continue;}
   int count=0;
   for(int i:scheduled[k].cells){
    if(debris[i]||a[1][i]<p[41]||a[9][i]<p[40])continue;
    // Combustible smoke cannot supply unlimited energy: both its available
    // flammable mass and the oxygen in this cell bound the reaction. Recheck
    // at ignition because the room may have been vented or extinguished.
    float available=std::min(a[10][i],a[0][i]*.75f);
    if(available<.025f)continue;
    float consumed=std::min(available,.22f);
    a[10][i]-=consumed;a[1][i]=std::max(0.f,a[1][i]-.4f*consumed);
    a[0][i]=std::max(0.f,a[0][i]-.65f*consumed);
    a[2][i]=std::max(a[2][i],std::min(p[39],.28f+1.6f*consumed));count++;
   }
   if(count){flashCount++;lastFlashCells=count;}
   scheduled.erase(scheduled.begin()+k);
  }
 }
 void evaporate_water(float dt){
  if(!wetCells)return;
  // Choose between sparse direct neighborhoods and a dense prefix sum.
  // A few wet cells should not force a scan of the full map.
  int rad=std::max(1,int(std::ceil(p[35]*q)));
  bool sparseWet=(solverMode!=0 && uint64_t(wetCells)*uint64_t(2*rad+1)*uint64_t(2*rad+1)<uint64_t(N)/2);
  if(!sparseWet){
  std::fill(integral.begin(),integral.end(),0.f);
  for(int r=0;r<H;r++){
   float rowsum=0;
   for(int c=0;c<W;c++){
    int i=r*W+c;rowsum+=a[2][i];
    integral[(r+1)*(W+1)+c+1]=integral[r*(W+1)+c+1]+rowsum;
   }
  }
  }
  size_t out=0;
  for(int i:wetIndices)if(a[11][i]>1e-6f && !debris[i]){
   int r=i/W,c=i%W,r0=std::max(0,r-rad),r1=std::min(H,r+rad+1);
   int c0=std::max(0,c-rad),c1=std::min(W,c+rad+1);
   float fireSum=0.f;
   if(sparseWet){
    for(int yy=r0;yy<r1;yy++)for(int xx=c0;xx<c1;xx++)fireSum+=a[2][yy*W+xx];
   }else fireSum=integral[r1*(W+1)+c1]-integral[r0*(W+1)+c1]-integral[r1*(W+1)+c0]+integral[r0*(W+1)+c0];
   // Normalize by effective physical footprint; no dependence on resolution.
   float avg=fireSum/float((r1-r0)*(c1-c0));
   float evaporation=dt*(p[33]+p[34]*avg)*a[11][i];
   a[11][i]=std::max(0.f,a[11][i]-evaporation);
   if(a[11][i]<1e-5f)a[11][i]=0;
   if(a[11][i]>0.f)wetIndices[out++]=i;
   else wetMark[i]=0;
   // Wet cells cool during each combustion substep, before growth and spread.
   // Evaporation here only updates the remaining water inventory.
  }
  for(size_t k=0;k<wetIndices.size();k++)if(debris[wetIndices[k]] || a[11][wetIndices[k]]<=0.f)wetMark[wetIndices[k]]=0;
  wetIndices.resize(out);wetCells=int(out);
 }

 void physics_tick(){
  auto t0=checkpoint();
  noise();
  accumulate_time(0,t0);
  t0=checkpoint();
  // Oxygen uses an explicit stencil and determines the necessary CFL steps.
  // Smoke uses bounded, conservative face transfers; no extra smoke-CFL
  // substeps are required. This is crucial at resolutions 3 and 4.
  int n=std::max({1,int(std::ceil(4*p[7]*q*q/(solverMode==0?.42f:solverMode==4?1.0f:.77f))),int(std::ceil(4*p[10]*q/.5f))});
  // The fast stencil has a positivity bound of approximately 4*D*q*q
  // for fully open faces; /0.77 keeps a safety margin while avoiding the
  // reference solver's excess oxygen microsteps at high resolution.
  if(solverMode!=0)n=std::max(4,n);
  if(solverMode!=0){
   refresh_debris_cells();
   hotCells.clear();heatTouched.clear();
   std::fill(heat.begin(),heat.end(),0.f);
   std::fill(fireSeen.begin(),fireSeen.end(),0);
   std::fill(heatSeen.begin(),heatSeen.end(),0);
   for(int i=0;i<N;i++)if(a[2][i]>=p[0] && !debris[i]){
    hotCells.push_back(i);fireSeen[i]=1;
   }
  }
  float dt=1.f/n;
  auto &oxygen=a[0],&smoke=a[1],&fire=a[2],&fuel=a[3],&damage=a[4],&draft=a[5],&fnoise=a[6],&axisX=a[7],&axisY=a[8];
  // Fast idle-gas path: if the entire grid has no smoke or active fire,
  // there is no smoke to transport or generate this tick. Avoid all smoke
  // face transfers and decay on such scenarios. Oxygen still equilibrates.
  bool smokeNeeded=false;
  int smokeOccupied=0;
  if(solverMode!=0){
   for(int i=0;i<N;i++){
    smokeOccupied+=(a[1][i]>1e-6f);
    smokeNeeded|=(a[1][i]>1e-8f || a[2][i]>=p[0]);
   }
  }else for(int i=0;i<N;i++)if(a[1][i]>1e-8f || a[2][i]>=p[0]){smokeNeeded=true;break;}
  parallelSmoke=((solverMode==2 || solverMode==4) && N>=50000);
  // Topology and drafts are constant through this physics tick. Precompute
  // every unchanged coefficient ONCE, not dozens of times per CFL substep.
  const bool refreshStatic=!staticCoeffValid || cachedSubsteps!=n || solverMode==0;
  #pragma omp parallel for schedule(static) if(N>=32768 && solverMode!=0)
  for(int r=0;r<H;r++)for(int c=0;c<W;c++){
   int i=r*W+c;
   if(refreshStatic){
   oxyE[i]=flux_e[i]*p[7]*dt*q*q;
   oxyS[i]=flux_s[i]*p[7]*dt*q*q;
   fireE[i]=c+1<W?flux_e[i]*(zone[i]!=zone[i+1]?2.5f:1.f)*(1+p[13]*std::min(draft[i],draft[i+1])*(.15f+.85f*std::min(axisX[i],axisX[i+1]))):0;
   fireS[i]=r+1<H?flux_s[i]*(zone[i]!=zone[i+W]?2.5f:1.f)*(1+p[13]*std::min(draft[i],draft[i+W])*(.15f+.85f*std::min(axisY[i],axisY[i+W]))):0;
   smokeDecay[i]=std::exp(-dt*(p[12]+p[16]*draft[i]));
   }
   smkE[i]=smokeNeeded&&c+1<W?flux_e[i]*p[8]*dt*q*q*(1+.5f*p[9]*(draft[i]+draft[i+1]))
    *(1+p[42]*(1-.5f*(a[1][i]+a[1][i+1]))*std::clamp(1+.25f*fnoise[i],.5f,1.5f)):0;
   smkS[i]=smokeNeeded&&r+1<H?flux_s[i]*p[8]*dt*q*q*(1+.5f*p[9]*(draft[i]+draft[i+W]))
    *(1+p[42]*(1-.5f*(a[1][i]+a[1][i+W]))*std::clamp(1+.25f*fnoise[i],.5f,1.5f)):0;
  }
  staticCoeffValid=true;cachedSubsteps=n;
  accumulate_time(1,t0);
  if(solverMode==3){
   float* fields[12];for(int i=0;i<12;i++)fields[i]=a[i].data();
   const float* coeff[9]={oxyE.data(),oxyS.data(),smkE.data(),smkS.data(),
    fireE.data(),fireS.data(),flux_e.data(),flux_s.data(),smokeDecay.data()};
   if(!gpu_init() || gpu.step(gpuContext,fields,coeff,material.data(),debris.data(),
                             p.data(),q,n,smokeNeeded)!=0){
    gpuFailure=true;
    gpuLastError=gpu.error?gpu.error():"CUDA initialization or backend load failure";
    return;
   }
   t0=checkpoint();
   purge_debris_cells();
   evaporate_water(1.f);ticks++;process_flashbacks();
   for(auto &v:victims){
    if(v.dead||v.rescued||!valid(v.r,v.c))continue;
    int i=v.r*W+v.c;
    v.health=std::max(0.f,v.health-.0018f*smoke[i]-.003f*fire[i]);
    if(v.health==0)v.dead=1;
   }
   accumulate_time(6,t0);
   return;
  }
  for(int sub=0;sub<n;sub++){
   t0=checkpoint();
   if(solverMode!=0)oxygen_stencil();else transport(0,dt,false);
   accumulate_time(2,t0);
   t0=checkpoint();
   if(smokeNeeded)transport(1,dt,true);
   accumulate_time(3,t0);
   t0=checkpoint();
   const float threshold=p[0];
   if(solverMode!=0){
    for(int i:heatTouched){heat[i]=0.f;heatSeen[i]=0;}
    heatTouched.clear();
    // An active-flame frontier gives exact neighbor heat contributions without
    // traversing every fire face in a mostly unburned building.
    auto deposit=[&](int target,float contribution){
     if(contribution==0.f)return;
     if(!heatSeen[target]){heatSeen[target]=1;heatTouched.push_back(target);}
     heat[target]+=contribution;
    };
    for(int i:hotCells){
     if(fire[i]<p[0] || debris[i])continue;
     const int r=i/W,c=i%W;const float F=fire[i];
     if(c+1<W)deposit(i+1,F*fireE[i]);
     if(c>0)deposit(i-1,F*fireE[i-1]);
     if(r+1<H)deposit(i+W,F*fireS[i]);
     if(r>0)deposit(i-W,F*fireS[i-W]);
    }
   }else{
   std::fill(heat.begin(),heat.end(),0.f);
   float boost=p[13];
   for(int r=0;r<H;r++)for(int c=0;c<W;c++){
    int i=r*W+c;
    if(c+1<W && fireE[i]!=0){
     int j=i+1;float f=fireE[i];
     if(fire[i]>=threshold)heat[j]+=fire[i]*f;
     if(fire[j]>=threshold)heat[i]+=fire[j]*f;
    }
    if(r+1<H && fireS[i]!=0){
     int j=i+W;float f=fireS[i];
     if(fire[i]>=threshold)heat[j]+=fire[i]*f;
     if(fire[j]>=threshold)heat[i]+=fire[j]*f;
    }
   }
   }
   accumulate_time(4,t0);
   t0=checkpoint();
   #pragma omp parallel for schedule(static) if(N>=32768 && solverMode!=0)
   for(int i=0;i<N;i++){
    if(debris[i])continue;
    float O=oxygen[i],F=fire[i],fuel0=fuel[i];
    O+=dt*p[11]*draft[i]*(1-O);
    smoke[i]*=smokeDecay[i];a[10][i]*=smokeDecay[i];
    // No combustion, fuel use, soot production, or charring is possible
    // without either an existing flame or incoming heat. Oxygen replenishment
    // and pre-existing smoke decay have already been applied above.
    if(F==0.f && heat[i]==0.f){
     oxygen[i]=O;
     a[9][i]=smoke[i]>1e-8f?std::clamp(a[10][i]/smoke[i],0.f,1.f):0.f;
     continue;
    }
    float active=F>=threshold?F:0;
    float flammability=material[i]?p[28]:p[27];
    // Below 8% oxygen, burning must fail; never create energy by merely
    // conducting heat into an oxygen-starved cell.
    float oxygen_factor=std::clamp((O-.025f)/.22f,0.f,1.15f);
    // Liquid water has an immediate *thermal* effect during each solver
    // substep, rather than only once after an entire fire-growth tick.
    // Wet fuel also resists reignition by its burning neighbors.  Neither
    // effect modifies fuel inventory or depends on the renderer's frame rate.
    float wet=std::clamp(a[11][i],0.f,1.f);
    float ignitionShield=1.f-.90f*wet;
    float growth=p[2]*active*fuel0*flammability*oxygen_factor*ignitionShield;
    if(p[14]>0){
     float core_start=std::max(threshold+.12f,.36f);
     float hot=std::clamp((active-core_start)/std::max(.01f,1-core_start),0.f,1.f);
     growth*=std::clamp(1+.6f*p[14]*hot*fnoise[i],.72f,1.28f);
    }
    // Fire is a propagating reaction front, NOT a conservative diffusion
    // field. Neighbor activation is proportional to 1/dx = resolution;
    // dx^-2 made high-resolution maps ignite artificially fast. The
    // coefficient is normalized to preserve the resolution-3 tuning.
    float spread=p[1]*heat[i]*fuel0*flammability*oxygen_factor*(1-F)*dt*q*.36f*ignitionShield;
    float starvation=O<.08f?.065f*(1.f-O/.08f):0.f;
    // p[32] is a per-physical-tick quenching coefficient. Water suppresses
    // strong established flames, not only small embers; the dt multiplier
    // makes its effect independent of the resolution-dependent substeps.
    float quench=dt*p[32]*wet*(.30f+.70f*F);
    F=std::clamp(F+dt*(growth-(p[3]+starvation)*F)+spread-quench,0.f,1.f);
    float burning=F>=threshold?F:0;
    float burningRate=burning*flammability*fuel0*(p[4]/.006f);
    float oxygenBurn=std::min(O,dt*p[5]*burningRate);
    float fuelBurn=std::min(fuel0,oxygenBurn*p[26]);
    fuel[i]=std::max(0.f,fuel0-fuelBurn);
    O=std::max(0.f,O-oxygenBurn);
    float generated=dt*p[6]*burningRate;
    float room=std::max(0.f,1.f-smoke[i]);
    generated=std::min(room,generated);
    smoke[i]+=generated;
    // Smoke flammability is a PROPERTY of smoke, not a second gas species.
    float fuelFraction=std::clamp(p[29]*(1+p[30]*(1-std::clamp(O/.30f,0.f,1.f))),0.f,1.f);
    a[10][i]=std::min(smoke[i],a[10][i]+generated*fuelFraction);
    damage[i]=std::clamp(damage[i]+dt*p[17]*burning,0.f,1.f);
    if(fuel[i]<.02f)F*=.93f;
    fire[i]=F;oxygen[i]=O;
    a[9][i]=smoke[i]>1e-8f?std::clamp(a[10][i]/smoke[i],0.f,1.f):0.f;
   }
   if(solverMode!=0){
    for(int i:heatTouched)if(fire[i]>=p[0] && !fireSeen[i] && !debris[i]){
     hotCells.push_back(i);fireSeen[i]=1;
    }
    purge_debris_cells();
   }else purge();
   accumulate_time(5,t0);
  }
  t0=checkpoint();
  evaporate_water(1.f);
  ticks++;
  process_flashbacks();
  // Victim state is authoritative here, never updated in the Python server.
  for(auto &v:victims){
   if(v.dead||v.rescued)continue;
   if(!valid(v.r,v.c))continue;
   int i=v.r*W+v.c;
   v.health=std::max(0.f,v.health-.0018f*smoke[i]-.003f*fire[i]);
   if(v.health==0)v.dead=1;
  }
  accumulate_time(6,t0);
 }
 // Runs until count reached or debris event needs Python geometry callback.
 int step(int count){
  event_r=event_c=-1;last_events=0;
  int done=0;
  for(;done<count;){
   physics_tick();if(gpuFailure)return -1;done++;
   if(p[18]<=0)continue;
   int nactive=0;double meanDamage=0,totalWeight=0;
   for(int i=0;i<N;i++)if(a[2][i]>=p[0] && !debris[i]){nactive++;meanDamage+=a[4][i];totalWeight+=1+p[19]*proximity[i];}
   if(!nactive)continue;
   meanDamage/=nactive;
   double expectation=std::min(2.,double(p[18])*nactive/(q*q)*(meanDamage+.08));
   // Standard Poisson, event handled by host only when necessary.
   std::poisson_distribution<int> poiss(expectation);
   if(poiss(rng)<=0)continue;
   std::uniform_real_distribution<double> sample(0,totalWeight);
   double choice=sample(rng);int selected=-1;
   for(int i=0;i<N;i++)if(a[2][i]>=p[0] && !debris[i]){
    choice-=1+p[19]*proximity[i];if(choice<=0){selected=i;break;}
   }
   if(selected>=0){event_r=selected/W;event_c=selected%W;generate_debris(event_r,event_c);}
  }
  return done;
 }
 bool valid(int r,int c)const{return r>=0&&r<H&&c>=0&&c<W;}
 bool world(int r,int c)const{return r>=-grass_margin&&c>=-grass_margin&&r<H+grass_margin&&c<W+grass_margin;}
 bool grass(int r,int c)const{return world(r,c)&&!valid(r,c);}
 bool exterior_pass(int r,int c,int nr,int nc)const{
  // On dynamic<->static boundary, check the actual exterior opening span.
  for(const auto &o:openings){if(o.kind!=2||!o.opened||o.blocked||o.zb>=0)continue;
   if(o.orient==1 && ((r==-1&&nr==0)||(r==0&&nr==-1)||(r==H-1&&nr==H)||(r==H&&nr==H-1))
      &&c==nc&&c>=o.start&&c<o.start+o.length&&o.at==(r<0||nr<0?0:H))return true;
   if(o.orient==0 && ((c==-1&&nc==0)||(c==0&&nc==-1)||(c==W-1&&nc==W)||(c==W&&nc==W-1))
      &&r==nr&&r>=o.start&&r<o.start+o.length&&o.at==(c<0||nc<0?0:W))return true;
  }return false;
 }
 bool move_possible(int r,int c,int nr,int nc)const{
  if(!world(nr,nc)||!world(r,c)||std::abs(r-nr)+std::abs(c-nc)!=1)return false;
  bool inside=valid(r,c),destination=valid(nr,nc);
  if(!inside&&!destination)return true;
  if(inside&&destination)return traversable(r*W+c,nr*W+nc);
  if(inside && debris[r*W+c])return false;
  if(destination && debris[nr*W+nc])return false;
  return exterior_pass(r,c,nr,nc);
 }

 bool traversable(int i,int j)const{
  if(debris[j])return false;
  if(j==i+1)return flux_e[i]>=.5f;
  if(j==i-1)return flux_e[j]>=.5f;
  if(j==i+W)return flux_s[i]>=.5f;
  if(j==i-W)return flux_s[j]>=.5f;
  return false;
 }
 int route(int r,int c,int gr,int gc,int32_t*out,int capacity){
  if(!valid(r,c)||!valid(gr,gc)||debris[r*W+c]||debris[gr*W+gc])return 0;
  int start=r*W+c,target=gr*W+gc;
  struct Q {int f,g,i;bool operator<(const Q&o)const{return f>o.f;}};
  std::priority_queue<Q> open;std::vector<int> best(N,INT32_MAX),parent(N,-1);
  auto heuristic=[&](int i){return std::abs(i/W-gr)+std::abs(i%W-gc);};
  best[start]=0;open.push({heuristic(start),0,start});
  while(!open.empty()){
   auto n=open.top();open.pop();if(n.g!=best[n.i])continue;
   if(n.i==target)break;
   int rr=n.i/W,cc=n.i%W;
   int neigh[4]={cc>0?n.i-1:-1,cc+1<W?n.i+1:-1,rr>0?n.i-W:-1,rr+1<H?n.i+W:-1};
   for(int j:neigh)if(j>=0 && traversable(n.i,j) && n.g+1<best[j]){
    best[j]=n.g+1;parent[j]=n.i;open.push({n.g+1+heuristic(j),n.g+1,j});
   }
  }
  if(best[target]==INT32_MAX)return 0;
  int length=best[target]+1;if(length>capacity)return -length;
  for(int i=target,k=length-1;i>=0;k--){out[k]=i;if(i==start)break;i=parent[i];}
  return length;
 }
 // A* over an IMPLICIT exterior ring, not a simulated gas grid. Outside
 // nodes are generated on demand; only the building interior has hazard fields.
 int route_world(int r,int c,int gr,int gc,int32_t*out,int capacity)const{
  if(!world(r,c)||!world(gr,gc))return 0;
  int HH=H+2*grass_margin,WW=W+2*grass_margin,NN=HH*WW;
  auto index=[&](int y,int x){return (y+grass_margin)*WW+(x+grass_margin);};
  int start=index(r,c),target=index(gr,gc);
  struct Node{int f,g,i;bool operator<(const Node &b)const{return f>b.f;}};
  std::priority_queue<Node> pq;
  std::vector<int> dist(NN,INT32_MAX),parent(NN,-1);
  auto h=[&](int id){return std::abs(id/WW-gr-grass_margin)+std::abs(id%WW-gc-grass_margin);};
  dist[start]=0;pq.push({h(start),0,start});
  const int dr[4]={-1,0,1,0},dc[4]={0,1,0,-1};
  while(!pq.empty()){
   auto cur=pq.top();pq.pop();if(cur.g!=dist[cur.i])continue;
   if(cur.i==target)break;
   int rr=cur.i/WW-grass_margin,cc=cur.i%WW-grass_margin;
   for(int k=0;k<4;k++){
    int nr=rr+dr[k],nc=cc+dc[k];if(!move_possible(rr,cc,nr,nc))continue;
    int ni=index(nr,nc);if(cur.g+1<dist[ni]){
     dist[ni]=cur.g+1;parent[ni]=cur.i;pq.push({dist[ni]+h(ni),dist[ni],ni});
    }
   }
  }
  if(dist[target]==INT32_MAX)return 0;
  int length=dist[target]+1;
  if(length>capacity)return -length;
  for(int at=target,k=length-1;k>=0;k--){out[2*k]=at/WW-grass_margin;out[2*k+1]=at%WW-grass_margin;at=parent[at];}
  return length;
 }
 // Strict local line-of-sight observation, no hidden hazard data.
 // Ch. 0..4: O, smoke, fire, draft, debris; 5 visible; 6 victim type; 7 teammate; 8..11 N,E,S,W passability; 12 smoke flammability; 13 water; 14 wood material.
 bool line_of_sight(int r,int c,int tr,int tc)const{
  if(!valid(r,c)||!valid(tr,tc))return false;
  int old=r*W+c,steps=4*std::max(std::abs(tr-r),std::abs(tc-c));
  if(steps==0)return true;
  for(int k=1;k<=steps;k++){
   float t=float(k)/steps;
   int nr=std::clamp(int(std::floor(r+.5f+t*(tr-r))),0,H-1);
   int nc=std::clamp(int(std::floor(c+.5f+t*(tc-c))),0,W-1);
   int ni=nr*W+nc;
   if(ni==old)continue;
   int rr=old/W,cc=old%W;
   if(nr!=rr && nc!=cc){
    int alongRow=rr*W+nc,alongCol=nr*W+cc;
    if(!traversable(old,alongRow)||!traversable(old,alongCol)||
       !traversable(alongRow,ni)||!traversable(alongCol,ni))return false;
   }else if(!traversable(old,ni))return false;
   old=ni;
  }
  return old==tr*W+tc;
 }
 // Full-grid, 360-degree visibility for one rescuer. Rays advance cell-by-cell
 // through the exact native face graph. O(perimeter * ray length), not O(N^1.5).
 // Grass is rendered as always visible by Orb and is not part of this mask.
 int visibility(int agent,uint8_t*out)const{
  if(!out || agent<0 || agent>=int(agents.size()))return -1;
  const auto &a0=agents[agent];
  if(visibilityCache.size()<agents.size()){
   visibilityCache.resize(agents.size());
   visibilityKeys.resize(agents.size(),{-1,-1,-1});
  }
  const std::array<int64_t,3> key{a0.r,a0.c,static_cast<int64_t>(topologyVersion)};
  if(visibilityKeys[agent]==key && visibilityCache[agent].size()==size_t(N)){
   std::memcpy(out,visibilityCache[agent].data(),N);return 0;
  }
  std::fill(out,out+N,uint8_t(0));
  if(!valid(a0.r,a0.c))return 0;
  out[a0.r*W+a0.c]=1;
  const double sx=double(a0.c)+.5,sy=double(a0.r)+.5;
  auto shoot=[&](double tx,double ty){
   const double dx=tx-sx,dy=ty-sy;
   if(dx==0 && dy==0)return;
   int x=a0.c,y=a0.r;
   const int ax=(dx>0)?1:-1,ay=(dy>0)?1:-1;
   const double invx=(dx==0)?std::numeric_limits<double>::infinity():std::abs(1./dx);
   const double invy=(dy==0)?std::numeric_limits<double>::infinity():std::abs(1./dy);
   double tmx=(dx==0)?invx:.5*invx,tmy=(dy==0)?invy:.5*invy;
   for(int step=0;step<H+W+2;step++){
    const int old=y*W+x;
    if(tmx<tmy-1e-12){
     int nx=x+ax;if(nx<0||nx>=W||!traversable(old,y*W+nx))break;
     x=nx;tmx+=invx;
    }else if(tmy<tmx-1e-12){
     int ny=y+ay;if(ny<0||ny>=H||!traversable(old,ny*W+x))break;
     y=ny;tmy+=invy;
    }else{
     int nx=x+ax,ny=y+ay;
     if(nx<0||nx>=W||ny<0||ny>=H)break;
     int hx=y*W+nx,vy=ny*W+x,target=ny*W+nx;
     // A ray exactly through a corner cannot see diagonally through two walls.
     if(!traversable(old,hx)||!traversable(old,vy)||
        !traversable(hx,target)||!traversable(vy,target))break;
     x=nx;y=ny;tmx+=invx;tmy+=invy;
    }
    out[y*W+x]=1;
   }
  };
  // Two sub-cell samples per boundary cell avoid holes in the visibility fan.
  for(int x=0;x<W;x++)for(double offset:{.25,.75}){
   shoot(double(x)+offset,-.5);shoot(double(x)+offset,double(H)+.5);
  }
  for(int y=0;y<H;y++)for(double offset:{.25,.75}){
   shoot(-.5,double(y)+offset);shoot(double(W)+.5,double(y)+offset);
  }
  visibilityCache[agent].assign(out,out+N);
  visibilityKeys[agent]=key;
  return 0;
 }
 int move_mask(int agent)const{
  if(agent<0||agent>=(int)agents.size())return 0;
  auto &a0=agents[agent];int mask=0;
  const int dr[4]={-1,0,1,0},dc[4]={0,1,0,-1};
  for(int k=0;k<4;k++){
   int r=a0.r+dr[k],c=a0.c+dc[k];if(!move_possible(a0.r,a0.c,r,c))continue;
   bool occupied=false;for(int j=0;j<(int)agents.size();j++)if(j!=agent&&agents[j].r==r&&agents[j].c==c)occupied=true;
   if(!occupied)mask|=1<<k;
  }return mask;
 }
 int observe(int agent,int radius,float*out)const{
  if(agent<0 || agent>=(int)agents.size() || radius<1 ||radius>64)return -1;
  int side=2*radius+1,plane=side*side;
  std::fill(out,out+15*plane,0.f);
  auto &ag=agents[agent];
  for(int y=-radius;y<=radius;y++)for(int x=-radius;x<=radius;x++){
   int nr=ag.r+y,nc=ag.c+x;
   if(!valid(ag.r,ag.c)||!valid(nr,nc) || y*y+x*x>radius*radius || !line_of_sight(ag.r,ag.c,nr,nc))continue;
   int pos=(y+radius)*side+(x+radius),idx=nr*W+nc;
   out[0*plane+pos]=a[0][idx];out[1*plane+pos]=a[1][idx];
   out[2*plane+pos]=a[2][idx];out[3*plane+pos]=a[5][idx];
   out[4*plane+pos]=debris[idx];out[5*plane+pos]=1.f;
   out[12*plane+pos]=a[9][idx];out[13*plane+pos]=a[11][idx];out[14*plane+pos]=float(material[idx]);
   out[8*plane+pos]=(nr>0 && flux_s[idx-W]>=.5f)?1.f:0.f;
   out[9*plane+pos]=(nc+1<W && flux_e[idx]>=.5f)?1.f:0.f;
   out[10*plane+pos]=(nr+1<H && flux_s[idx]>=.5f)?1.f:0.f;
   out[11*plane+pos]=(nc>0 && flux_e[idx-1]>=.5f)?1.f:0.f;
  }
  // Entity channels are populated only on visible cells.
  for(auto &v:victims){
   if(v.dead||v.rescued)continue;
   int dy=v.r-ag.r,dx=v.c-ag.c;
   if(std::abs(dy)>radius||std::abs(dx)>radius)continue;
   int pos=(dy+radius)*side+dx+radius;
   if(out[5*plane+pos]!=0.f)out[6*plane+pos]=(v.kind%3+1)/3.f;
  }
  for(int j=0;j<(int)agents.size();j++){
   if(j==agent)continue;
   int dy=agents[j].r-ag.r,dx=agents[j].c-ag.c;
   if(std::abs(dy)>radius||std::abs(dx)>radius)continue;
   int pos=(dy+radius)*side+dx+radius;
   if(out[5*plane+pos]!=0.f)out[7*plane+pos]=1.f;
  }
  return 0;
 }
 // Water is line-of-sight transported through native face conductances.
 // Unlike a same-room mask, this also passes through OPEN doors/windows but
 // never jumps a closed face, an outer wall or debris. Exact-corner crossings
 // require both adjacent faces, so diagonal jets cannot leak through corners.
 bool water_path(int r,int c,int tr,int tc)const{
  if(!valid(r,c)||!valid(tr,tc)||debris[r*W+c]||debris[tr*W+tc])return false;
  int y=r,x=c,dy=tr-r,dx=tc-c;
  if(!dy&&!dx)return true;
  const int sy=(dy>0?1:-1),sx=(dx>0?1:-1);
  const double invx=dx?1./std::abs(dx):std::numeric_limits<double>::infinity();
  const double invy=dy?1./std::abs(dy):std::numeric_limits<double>::infinity();
  double tx=.5*invx,ty=.5*invy;
  auto face=[&](int ar,int ac,int br,int bc)->bool{
   if(!valid(ar,ac)||!valid(br,bc))return false;
   int a0=ar*W+ac,b0=br*W+bc;
   if(debris[a0]||debris[b0])return false;
   if(bc==ac+1&&br==ar)return flux_e[a0]>1e-6f;
   if(bc==ac-1&&br==ar)return flux_e[b0]>1e-6f;
   if(br==ar+1&&bc==ac)return flux_s[a0]>1e-6f;
   if(br==ar-1&&bc==ac)return flux_s[b0]>1e-6f;
   return false;
  };
  for(int step=0;step<std::abs(dy)+std::abs(dx)+2 && (y!=tr||x!=tc);step++){
   if(tx<ty-1e-12){if(!face(y,x,y,x+sx))return false;x+=sx;tx+=invx;}
   else if(ty<tx-1e-12){if(!face(y,x,y+sy,x))return false;y+=sy;ty+=invy;}
   else{
    if(!face(y,x,y,x+sx)||!face(y,x,y+sy,x) ||
       !face(y,x+sx,y+sy,x+sx)||!face(y+sy,x,y+sy,x+sx))return false;
    x+=sx;y+=sy;tx+=invx;ty+=invy;
   }
  }
  return y==tr&&x==tc;
 }
 void wet_cell(int i,float strength){
  if(!wetMark[i]){wetMark[i]=1;wetIndices.push_back(i);wetCells++;}
  a[11][i]=std::max(a[11][i],std::clamp(strength,0.f,1.f));
 }
 // Existing radial water action used by training agents: now obeys barriers.
 void water(int r,int c,float strength,float radius){
  if(!valid(r,c)||debris[r*W+c]||radius<0.f)return;
  int rad=std::max(0,int(std::ceil(radius*q)));
  for(int y=std::max(0,r-rad);y<std::min(H,r+rad+1);y++)for(int x=std::max(0,c-rad);x<std::min(W,c+rad+1);x++){
   if(std::hypot(float(y-r),float(x-c))/q>radius+1e-8f)continue;
   if(water_path(r,c,y,x))wet_cell(y*W+x,strength);
  }
 }
 // Orb water tool: physical-radius 8-direction triangular spray.
 // Cone half-width expands linearly with distance from the nozzle.
 void water_cone(int r,int c,int dx,int dy,float strength,float radius){
  if(!valid(r,c)||debris[r*W+c]||radius<0.f||(!dx&&!dy))return;
  dx=(dx>0)-(dx<0);dy=(dy>0)-(dy<0);
  float unit=1.f/std::sqrt(float(dx*dx+dy*dy)),ax=dx*unit,ay=dy*unit;
  float distance=radius*q;
  int rad=std::max(0,int(std::ceil(distance)));
  for(int y=std::max(0,r-rad);y<std::min(H,r+rad+1);y++)for(int x=std::max(0,c-rad);x<std::min(W,c+rad+1);x++){
   float vx=float(x-c),vy=float(y-r);
   float forward=vx*ax+vy*ay,lateral=std::abs(vx*ay-vy*ax);
   if(forward<-.001f||forward>distance+.01f || lateral>.55f*forward+.35f ||
      vx*vx+vy*vy>distance*distance+.01f)continue;
   if(water_path(r,c,y,x))wet_cell(y*W+x,strength);
  }
 }

 // Agent coordinates can enter a static three-physical-unit grass ring.
 // Actions 0 wait; 1..4 N/E/S/W; 5 water; 6 treat; 7 legacy rescue;
 // 8 debris; 10 pick up / drop adjacent victim. Moving carrier moves victim.
 int actions(const int32_t*commands,const int32_t*targets,int n){
  int success=0;
  const int dr[4]={-1,0,1,0},dc[4]={0,1,0,-1};
  for(int k=0;k<std::min(n,int(agents.size()));k++){
   auto &a0=agents[k];if(a0.dead)continue;
   int action=commands[k];
   if(action>=1&&action<=4){
    int nr=a0.r+dr[action-1],nc=a0.c+dc[action-1];
    if(move_possible(a0.r,a0.c,nr,nc)){
     bool occupied=false;for(int j=0;j<(int)agents.size();j++)if(j!=k&&agents[j].r==nr&&agents[j].c==nc)occupied=true;
     if(!occupied){
      a0.r=nr;a0.c=nc;success++;
      for(auto &v:victims)if(v.carried_by==k&&!v.dead&&!v.rescued){
       v.r=nr;v.c=nc;
       if(grass(nr,nc) && (v.kind!=2||v.stabilized)){v.rescued=1;v.carried_by=-1;}
      }
     }
    }
   }
   if(action==5&&valid(a0.r,a0.c)){water(a0.r,a0.c,p[20],p[21]);success++;}
   if(action==6){for(auto &v:victims)if(!v.dead&&!v.rescued&&std::abs(v.r-a0.r)+std::abs(v.c-a0.c)<=1){v.health=std::min(1.f,v.health+.15f);v.stabilized=1;success++;break;}}
   if(action==7 && grass(a0.r,a0.c)){
    for(auto &v:victims)if(!v.dead&&!v.rescued&&v.carried_by==k &&(v.kind!=2||v.stabilized)){
     v.rescued=1;v.carried_by=-1;success++;break;
    }
   }
   if(action==8&&targets&&valid(a0.r,a0.c)){
    int t=targets[k];if(t>=0&&t<N&&std::abs(t/W-a0.r)+std::abs(t%W-a0.c)<=1&&debris[t]){debris[t]=0;success++;}
   }
   if(action==10){
    bool dropping=false;
    for(auto &v:victims)if(v.carried_by==k&&!v.dead&&!v.rescued){v.carried_by=-1;dropping=true;success++;break;}
    if(!dropping)for(auto &v:victims)if(!v.dead&&!v.rescued&&v.carried_by<0&&std::abs(v.r-a0.r)+std::abs(v.c-a0.c)<=1){
     v.carried_by=k;v.r=a0.r;v.c=a0.c;success++;if(grass(v.r,v.c)&&(v.kind!=2||v.stabilized)){v.rescued=1;v.carried_by=-1;}break;
    }
   }
  }
  return success;
 }
};
#ifdef _WIN32
#define FLAMEBOX_API __declspec(dllexport)
#else
#define FLAMEBOX_API
#endif
extern "C" {
 FLAMEBOX_API void* sim_new(int h,int w,int resolution,uint64_t seed){try{return new Kernel(h,w,resolution,seed);}catch(...){return nullptr;}}
 FLAMEBOX_API void sim_free(void*p){delete static_cast<Kernel*>(p);}
 FLAMEBOX_API int sim_cells(void*p){return static_cast<Kernel*>(p)->N;}
 FLAMEBOX_API float* sim_field(void*p,int index){auto*s=static_cast<Kernel*>(p);return index>=0&&index<12?s->a[index].data():nullptr;}
 FLAMEBOX_API uint8_t*sim_debris(void*p){return static_cast<Kernel*>(p)->debris.data();}
 FLAMEBOX_API int16_t*sim_zone(void*p){return static_cast<Kernel*>(p)->zone.data();}
 FLAMEBOX_API void sim_edges(void*p,const float*e,const float*s,const int16_t*z){auto*b=static_cast<Kernel*>(p);int n=b->N;std::copy(e,e+n,b->east.begin());std::copy(s,s+n,b->south.begin());std::copy(z,z+n,b->zone.begin());b->refresh_bounds();b->conduct();}
 FLAMEBOX_API void sim_proximity(void*p,const float*x){auto*b=static_cast<Kernel*>(p);std::copy(x,x+b->N,b->proximity.begin());}
 FLAMEBOX_API void sim_refresh(void*p){auto*b=static_cast<Kernel*>(p);b->purge();b->conduct();}
 FLAMEBOX_API void sim_params(void*p,const float*x,int n){auto*b=static_cast<Kernel*>(p);for(int i=0;i<std::min(n,int(b->p.size()));i++)b->p[i]=x[i];b->staticCoeffValid=false;}
 FLAMEBOX_API void sim_solver_mode(void*p,int mode){
  static_cast<Kernel*>(p)->solverMode=(mode>=0&&mode<=4?mode:1);
 }
 FLAMEBOX_API int sim_gpu_available(void*p){return static_cast<Kernel*>(p)->gpu_ready()?1:0;}
 FLAMEBOX_API int sim_gpu_active(void*p){auto*k=static_cast<Kernel*>(p);return k->solverMode==3 && !k->gpuFailure && k->gpu_ready()?1:0;}
 FLAMEBOX_API const char* sim_gpu_device(void*p){auto*k=static_cast<Kernel*>(p);return k->gpu_ready()&&k->gpu.device?k->gpu.device():"No CUDA device";}
 FLAMEBOX_API const char* sim_gpu_error(void*p){return static_cast<Kernel*>(p)->gpuLastError.c_str();}
 FLAMEBOX_API int sim_cpu_threads(void*){
  #ifdef _OPENMP
   return omp_get_max_threads();
  #else
   return 1;
  #endif
 }
 FLAMEBOX_API void sim_profile_enable(void*p,int enabled){auto*b=static_cast<Kernel*>(p);b->profiling=enabled!=0;b->profileMs.fill(0.0);}
 FLAMEBOX_API void sim_profile_read(void*p,double*out){auto*b=static_cast<Kernel*>(p);if(out)std::copy(b->profileMs.begin(),b->profileMs.end(),out);}
 FLAMEBOX_API int sim_solver_substeps(void*p){auto*b=static_cast<Kernel*>(p);return std::max({b->solverMode!=0?4:1,int(std::ceil(4*b->p[7]*b->q*b->q/(b->solverMode==0?.42f:b->solverMode==4?1.0f:.77f))),int(std::ceil(4*b->p[10]*b->q/.5f))});}
 FLAMEBOX_API int sim_step(void*p,int count){return static_cast<Kernel*>(p)->step(count);}
 FLAMEBOX_API int sim_event_r(void*p){return static_cast<Kernel*>(p)->event_r;}
 FLAMEBOX_API int sim_event_c(void*p){return static_cast<Kernel*>(p)->event_c;}
 FLAMEBOX_API int sim_last_events(void*p){return static_cast<Kernel*>(p)->last_events;}
 FLAMEBOX_API void sim_get_blocked(void*p,int32_t*out,int n){auto*b=static_cast<Kernel*>(p);for(int i=0;i<n;i++)out[i]=i<(int)b->openings.size()?b->openings[i].blocked:0;}
 FLAMEBOX_API int sim_ticks(void*p){return static_cast<Kernel*>(p)->ticks;}
 FLAMEBOX_API float sim_max_draft(void*p){return static_cast<Kernel*>(p)->maxDraft;}
 FLAMEBOX_API int sim_route_world(void*p,int r,int c,int y,int x,int32_t*out,int cap){return static_cast<Kernel*>(p)->route_world(r,c,y,x,out,cap);}
 FLAMEBOX_API int sim_route(void*p,int r,int c,int y,int x,int32_t*out,int cap){return static_cast<Kernel*>(p)->route(r,c,y,x,out,cap);}
 FLAMEBOX_API void sim_water(void*p,int r,int c,float strength,float radius){static_cast<Kernel*>(p)->water(r,c,strength,radius);}
 FLAMEBOX_API void sim_water_cone(void*p,int r,int c,int dx,int dy,float strength,float radius){static_cast<Kernel*>(p)->water_cone(r,c,dx,dy,strength,radius);}
 FLAMEBOX_API void sim_openings(void*p,const Opening*x,int n){
  auto*b=static_cast<Kernel*>(p);
  // Detect CLOSED -> OPEN transitions; do NOT check flashback on reset.
  std::vector<Opening> previous=b->openings;
  b->openings.assign(x,x+n);
  for(const auto &op:b->openings)if(op.opened && !op.blocked){
   auto it=std::find_if(previous.begin(),previous.end(),[&](const Opening &old){return old.id==op.id;});
   if(it!=previous.end()&&(!it->opened||it->blocked))b->evaluate_flashback(op);
  }
  b->rebuild_draft();
 }
 FLAMEBOX_API uint8_t*sim_material(void*p){return static_cast<Kernel*>(p)->material.data();}
 FLAMEBOX_API void sim_rescan_water(void*p){
  auto*b=static_cast<Kernel*>(p);b->wetIndices.clear();std::fill(b->wetMark.begin(),b->wetMark.end(),0);
  for(int i=0;i<b->N;i++)if(b->a[11][i]>1e-6f&&!b->debris[i]){b->wetMark[i]=1;b->wetIndices.push_back(i);}
  b->wetCells=int(b->wetIndices.size());
 }
 FLAMEBOX_API int sim_flashbacks(void*p){return static_cast<Kernel*>(p)->flashCount;}
 FLAMEBOX_API int sim_pending_flashbacks(void*p){return int(static_cast<Kernel*>(p)->scheduled.size());}
 FLAMEBOX_API int sim_last_flash_cells(void*p){return static_cast<Kernel*>(p)->lastFlashCells;}
 FLAMEBOX_API int sim_draft_segments(void*p){return static_cast<Kernel*>(p)->draftSegments;}
 FLAMEBOX_API int sim_draft_ports(void*p){return static_cast<Kernel*>(p)->draftPorts;}
 FLAMEBOX_API int sim_draft_sources(void*p){return static_cast<Kernel*>(p)->draftSources;}
 FLAMEBOX_API void sim_people(void*p,int type,const Person*x,int n){auto*b=static_cast<Kernel*>(p);auto &v=type?b->agents:b->victims;v.assign(x,x+n);}
 FLAMEBOX_API int sim_people_count(void*p,int type){auto*b=static_cast<Kernel*>(p);return type?int(b->agents.size()):int(b->victims.size());}
 FLAMEBOX_API void sim_people_get(void*p,int type,Person*out){auto*b=static_cast<Kernel*>(p);auto &v=type?b->agents:b->victims;std::copy(v.begin(),v.end(),out);}
 FLAMEBOX_API int sim_grass_margin(void*p){return static_cast<Kernel*>(p)->grass_margin;}
 FLAMEBOX_API int sim_move_mask(void*p,int agent){return static_cast<Kernel*>(p)->move_mask(agent);}
 FLAMEBOX_API int sim_visibility(void*p,int agent,uint8_t*out){return static_cast<Kernel*>(p)->visibility(agent,out);}
 FLAMEBOX_API int sim_observe(void*p,int agent,int radius,float*out){return static_cast<Kernel*>(p)->observe(agent,radius,out);}
 FLAMEBOX_API int sim_observe_batch(void*p,int radius,float*out){
  auto*b=static_cast<Kernel*>(p);int length=15*(2*radius+1)*(2*radius+1);
  for(int k=0;k<int(b->agents.size());k++)if(b->observe(k,radius,out+k*length))return -1;
  return int(b->agents.size());
 }
 FLAMEBOX_API int sim_move_mask_batch(void*p,int32_t*out){
  auto*b=static_cast<Kernel*>(p);for(int k=0;k<int(b->agents.size());k++)out[k]=b->move_mask(k);
  return int(b->agents.size());
 }
 FLAMEBOX_API int sim_actions(void*p,const int32_t*a,const int32_t*targets,int n){return static_cast<Kernel*>(p)->actions(a,targets,n);}
}
