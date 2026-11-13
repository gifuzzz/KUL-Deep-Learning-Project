// FlameBox CUDA backend (experimental). Requires CUDA Toolkit/nvcc and NVIDIA driver.
// The complete coupled numerical substep is executed on device; host exchanges
// simulation state at physical tick boundaries for CPU events/agents/visualization.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <initializer_list>

#ifdef _WIN32
#define FB_EXPORT extern "C" __declspec(dllexport)
#else
#define FB_EXPORT extern "C"
#endif

namespace {
constexpr int FIELD_COUNT=12;
constexpr int DATA_COUNT=FIELD_COUNT+12; // fields, scratch oxygen, heat, oxyE/S, smkE/S, fireE/S, fluxE/S
struct State {int H,W,N;float *d[DATA_COUNT]{};uint8_t *material=nullptr,*debris=nullptr;float *parameters=nullptr;};
thread_local char error_message[512] = "";
__device__ inline float clampf(float x,float lo,float hi){return fminf(hi,fmaxf(lo,x));}
__global__ void oxygen_kernel(const float* oxygen,float* next,const float* east,const float* south,int H,int W,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 int r=i/W,c=i%W;float center=oxygen[i],sum=0.f;
 if(c+1<W)sum+=east[i]*(oxygen[i+1]-center);
 if(c>0)sum+=east[i-1]*(oxygen[i-1]-center);
 if(r+1<H)sum+=south[i]*(oxygen[i+W]-center);
 if(r>0)sum+=south[i-W]*(oxygen[i-W]-center);
 next[i]=center+sum;
}
__global__ void smoke_escape(const float* fire,const float* fe,const float* fs,float *fraction,
                              const float* p,float dt,int H,int W,int N,int q){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 float center=fire[i];if(center<=0){fraction[i]=0;return;}
 int r=i/W,c=i%W;float rate=0;
 if(c+1<W)rate+=fmaxf(0.f,(center-fire[i+1])*fe[i]);
 if(c>0)rate+=fmaxf(0.f,(center-fire[i-1])*fe[i-1]);
 if(r+1<H)rate+=fmaxf(0.f,(center-fire[i+W])*fs[i]);
 if(r>0)rate+=fmaxf(0.f,(center-fire[i-W])*fs[i-W]);
 fraction[i]=rate>1.e-12f ? -expm1f(-p[31]*dt*rate*(q/3.f))/rate : 0.f;
}
__device__ inline void face_apply(int i,int j,float coef,float conduct,float* smoke,float* mass,
             const float* oxygen,const float* fire,const float* fraction,const float* p,float dt,int q){
 if(coef<=0 || (smoke[i]==0.f && smoke[j]==0.f))return;
 float fi=smoke[i],fj=smoke[j],mi=mass[i],mj=mass[j];
 float x=coef*(fj-fi);
 float v=p[10]*dt*q*(oxygen[i]-oxygen[j])*conduct;
 x-=v*(v>0?fi:fj);
 float d=fire[i]-fire[j];
 if(d!=0){int src=d>0?i:j;
  x+=(d>0?-1.f:1.f)*fabsf(d)*conduct*(d>0?fi:fj)*fraction[src];}
 if(x>0){x=fminf(x,fminf(fj,1.f-fi));if(x>0){
  float frac=fj>1.e-10f?clampf(mj/fj,0.f,1.f):0.f;
  float moved=fminf(mj,frac*x);fj-=x;fi+=x;mj-=moved;mi+=moved;
 }}else if(x<0){x=fminf(-x,fminf(fi,1.f-fj));if(x>0){
  float frac=fi>1.e-10f?clampf(mi/fi,0.f,1.f):0.f;
  float moved=fminf(mi,frac*x);fi-=x;fj+=x;mi-=moved;mj+=moved;
 }}
 if(fi>1.e-8f&&fj>1.e-8f){
  float ci=clampf(mi/fi,0.f,1.f),cj=clampf(mj/fj,0.f,1.f);
  float ex=fminf(.30f,fmaxf(0.f,coef))*fminf(fi,fj)*(cj-ci);
  if(ex>0)ex=fminf(ex,fminf(mj,fi-mi));
  else ex=-fminf(-ex,fminf(mi,fj-mj));
  mi+=ex;mj-=ex;
 }
 smoke[i]=fi;smoke[j]=fj;mass[i]=mi;mass[j]=mj;
}
// Per-phase cells never overlap; no atomics and strict local conservation.
__global__ void smoke_phase(int phase,int H,int W,int N,float* smoke,float* mass,
       const float* oxygen,const float* fire,const float* fractions,const float* p,
       const float* smkE,const float* smkS,const float* fluxE,const float* fluxS,float dt,int q){
 int k=blockIdx.x*blockDim.x+threadIdx.x;
 if(phase<2){int ncols=(W-phase)/2;if(ncols<=0)return;
  int rr=k/ncols,cc=phase+2*(k%ncols);if(rr>=H||cc+1>=W)return;
  int i=rr*W+cc;face_apply(i,i+1,smkE[i],fluxE[i],smoke,mass,oxygen,fire,fractions,p,dt,q);
 }else{int rr=(2*(k/W))+(phase-2),cc=k%W;if(rr+1>=H)return;
  int i=rr*W+cc;face_apply(i,i+W,smkS[i],fluxS[i],smoke,mass,oxygen,fire,fractions,p,dt,q);
 }
}
__global__ void heat_kernel(const float* fire,const float* fe,const float* fs,float* heat,
                             const float* p,int H,int W,int N){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
 int r=i/W,c=i%W;float x=0.f,threshold=p[0];
 if(c+1<W&&fire[i+1]>=threshold)x+=fire[i+1]*fe[i];
 if(c>0&&fire[i-1]>=threshold)x+=fire[i-1]*fe[i-1];
 if(r+1<H&&fire[i+W]>=threshold)x+=fire[i+W]*fs[i];
 if(r>0&&fire[i-W]>=threshold)x+=fire[i-W]*fs[i-W];
 heat[i]=x;
}
__global__ void combustion_kernel(int N,int q,float dt,const float* p,const uint8_t* material,
  const uint8_t* debris,float*oxygen,float*smoke,float*fire,float*fuel,float*damage,
  const float*draft,const float*noise,float*flam_fraction,float*flam_mass,const float*water,
  const float*smokeDecay,const float*heat){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N||debris[i])return;
 float O=oxygen[i],F=fire[i],fuel0=fuel[i];
 O+=dt*p[11]*draft[i]*(1.f-O);
 smoke[i]*=smokeDecay[i];flam_mass[i]*=smokeDecay[i];
 if(F==0.f&&heat[i]==0.f){oxygen[i]=O;
  flam_fraction[i]=smoke[i]>1.e-8f?clampf(flam_mass[i]/smoke[i],0.f,1.f):0.f;return;}
 float active=F>=p[0]?F:0.f;
 float flammability=material[i]?p[28]:p[27];
 float oxygen_factor=clampf((O-.025f)/.22f,0.f,1.15f);
 float wet=clampf(water[i],0.f,1.f),shield=1.f-.90f*wet;
 float growth=p[2]*active*fuel0*flammability*oxygen_factor*shield;
 if(p[14]>0){float core_start=fmaxf(p[0]+.12f,.36f);
  float hot=clampf((active-core_start)/fmaxf(.01f,1.f-core_start),0.f,1.f);
  growth*=clampf(1.f+.6f*p[14]*hot*noise[i],.72f,1.28f);}
 float spread=p[1]*heat[i]*fuel0*flammability*oxygen_factor*(1.f-F)*dt*q*.36f*shield;
 float starvation=O<.08f?.065f*(1.f-O/.08f):0.f;
 float quench=dt*p[32]*wet*(.30f+.70f*F);
 F=clampf(F+dt*(growth-(p[3]+starvation)*F)+spread-quench,0.f,1.f);
 float burning=F>=p[0]?F:0.f;
 float burningRate=burning*flammability*fuel0*(p[4]/.006f);
 float oxygenBurn=fminf(O,dt*p[5]*burningRate);
 float fuelBurn=fminf(fuel0,oxygenBurn*p[26]);
 fuel[i]=fmaxf(0.f,fuel0-fuelBurn);
 O=fmaxf(0.f,O-oxygenBurn);
 float generated=dt*p[6]*burningRate;
 generated=fminf(fmaxf(0.f,1.f-smoke[i]),generated);
 smoke[i]+=generated;
 float fuelFraction=clampf(p[29]*(1.f+p[30]*(1.f-clampf(O/.30f,0.f,1.f))),0.f,1.f);
 flam_mass[i]=fminf(smoke[i],flam_mass[i]+generated*fuelFraction);
 damage[i]=clampf(damage[i]+dt*p[17]*burning,0.f,1.f);
 if(fuel[i]<.02f)F*=.93f;
 fire[i]=F;oxygen[i]=O;
 flam_fraction[i]=smoke[i]>1.e-8f?clampf(flam_mass[i]/smoke[i],0.f,1.f):0.f;
}
inline bool check(cudaError_t status,const char* operation){
 if(status==cudaSuccess)return true;
 std::snprintf(error_message,sizeof(error_message),"%s: %s",operation,cudaGetErrorString(status));return false;
}
#define CUDA_CHECK(call) do{if(!check((call),#call))return -1;}while(0)
} // namespace

FB_EXPORT const char* flamebox_cuda_error(){return error_message;}
FB_EXPORT int flamebox_cuda_available(){int n=0;return cudaGetDeviceCount(&n)==cudaSuccess&&n>0?n:0;}
FB_EXPORT const char* flamebox_cuda_device(){static thread_local char name[256]="";
 cudaDeviceProp prop{};if(cudaGetDeviceProperties(&prop,0)==cudaSuccess)
 std::snprintf(name,sizeof(name),"%s",prop.name);else std::snprintf(name,sizeof(name),"CUDA unavailable");return name;}
FB_EXPORT void* flamebox_cuda_new(int H,int W){
 if(!flamebox_cuda_available())return nullptr;
 auto *s=new State{};s->H=H;s->W=W;s->N=H*W;
 for(float*& x:s->d)if(!check(cudaMalloc((void**)&x,s->N*sizeof(float)),"cudaMalloc fields")){goto fail;}
 if(!check(cudaMalloc((void**)&s->material,s->N),"cudaMalloc material"))goto fail;
 if(!check(cudaMalloc((void**)&s->debris,s->N),"cudaMalloc debris"))goto fail;
 if(!check(cudaMalloc((void**)&s->parameters,48*sizeof(float)),"cudaMalloc parameters"))goto fail;
 return s;
 fail:
 for(float* x:s->d)if(x)cudaFree(x);
 if(s->material)cudaFree(s->material);
 if(s->debris)cudaFree(s->debris);
 if(s->parameters)cudaFree(s->parameters);
 delete s;return nullptr;
}
FB_EXPORT void flamebox_cuda_free(void* ctx){auto*s=(State*)ctx;if(!s)return;
 for(float* x:s->d)if(x)cudaFree(x);
 if(s->material)cudaFree(s->material);
 if(s->debris)cudaFree(s->debris);
 if(s->parameters)cudaFree(s->parameters);
 delete s;
}
// a[0..11]; coefficients: 0 oxyE,1 oxyS,2 smkE,3 smkS,4 fireE,5 fireS,
// 6 fluxE,7 fluxS,8 smokeDecay. Remaining scratch buffers are allocated once.
FB_EXPORT int flamebox_cuda_step(void* ctx,float* const* fields,const float* const* coefficients,
                      const uint8_t* material,const uint8_t* debris,const float* params,
                      int q,int substeps,bool smokeNeeded){
 auto*s=(State*)ctx;if(!s||substeps<1)return -1;
 const int N=s->N,H=s->H,W=s->W,bs=256,blocks=(N+bs-1)/bs;
 float* d[12];
 // Keep authoritative host-only fields (draft axes) on the CPU.
 // Flammable fraction is fully recomputed by combustion each substep.
 // Reducing compulsory PCIe transfers matters for frequent interactive ticks.
 for(int i=0;i<12;i++){
  d[i]=s->d[i];
  if(i==7 || i==8 || i==9)continue;
  CUDA_CHECK(cudaMemcpy(d[i],fields[i],N*sizeof(float),cudaMemcpyHostToDevice));
 }
 for(int k=0;k<9;k++) CUDA_CHECK(cudaMemcpy(s->d[14+k],coefficients[k],N*sizeof(float),cudaMemcpyHostToDevice));
 CUDA_CHECK(cudaMemcpy(s->material,material,N,cudaMemcpyHostToDevice));
 CUDA_CHECK(cudaMemcpy(s->debris,debris,N,cudaMemcpyHostToDevice));
 CUDA_CHECK(cudaMemcpy(s->parameters,params,48*sizeof(float),cudaMemcpyHostToDevice));
 float* oxyScratch=s->d[12];float*heat=s->d[13];float*fraction=s->d[23];
 float dt=1.f/substeps;
 for(int sub=0;sub<substeps;sub++){
  oxygen_kernel<<<blocks,bs>>>(d[0],oxyScratch,s->d[14],s->d[15],H,W,N);
  float*temp=d[0];d[0]=oxyScratch;oxyScratch=temp;
  if(smokeNeeded){
   smoke_escape<<<blocks,bs>>>(d[2],s->d[20],s->d[21],fraction,s->parameters,dt,H,W,N,q);
   for(int phase=0;phase<4;phase++){
    int taskCount=phase<2?H*((W-phase)/2):((H-(phase-2))/2)*W;
    smoke_phase<<<(taskCount+bs-1)/bs,bs>>>(phase,H,W,N,d[1],d[10],d[0],d[2],fraction,
         s->parameters,s->d[16],s->d[17],s->d[20],s->d[21],dt,q);
   }
  }
  heat_kernel<<<blocks,bs>>>(d[2],s->d[18],s->d[19],heat,s->parameters,H,W,N);
  combustion_kernel<<<blocks,bs>>>(N,q,dt,s->parameters,s->material,s->debris,
   d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[9],d[10],d[11],s->d[22],heat);
 }
 CUDA_CHECK(cudaGetLastError());
 CUDA_CHECK(cudaDeviceSynchronize());
 // Only physics fields modified by GPU need a download. The other
 // arrays are CPU-owned and already current (draft, noise, axes, water).
 for(int i : {0,1,2,3,4,9,10})
  CUDA_CHECK(cudaMemcpy(fields[i],d[i],N*sizeof(float),cudaMemcpyDeviceToHost));
 return 0;
}
