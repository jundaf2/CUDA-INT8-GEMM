#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wcast-qual"
#define __NV_CUBIN_HANDLE_STORAGE__ static
#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#endif
#include "crt/host_runtime.h"
#include "test_gemm_i8.fatbin.c"
static void __device_stub__ZN6kernel9GEMMI8TCUILi256ELi256ELi32ELi64ELi64ELi2ELb1ELb0ELb1EEEvPKaS2_Paiii(const int8_t *, const int8_t *, int8_t *, int, int, int);
static void __nv_cudaEntityRegisterCallback(void **);
static void __sti____cudaRegisterAll(void) __attribute__((__constructor__));
static void __device_stub__ZN6kernel9GEMMI8TCUILi256ELi256ELi32ELi64ELi64ELi2ELb1ELb0ELb1EEEvPKaS2_Paiii(const int8_t *__par0, const int8_t *__par1, int8_t *__par2, int __par3, int __par4, int __par5){__cudaLaunchPrologue(6);__cudaSetupArgSimple(__par0, 0UL);__cudaSetupArgSimple(__par1, 8UL);__cudaSetupArgSimple(__par2, 16UL);__cudaSetupArgSimple(__par3, 24UL);__cudaSetupArgSimple(__par4, 28UL);__cudaSetupArgSimple(__par5, 32UL);__cudaLaunch(((char *)((void ( *)(const int8_t *, const int8_t *, int8_t *, int, int, int))kernel::GEMMI8TCU<(int)256, (int)256, (int)32, (int)64, (int)64, (int)2, (bool)1, (bool)0, (bool)1> )));}namespace kernel{

template<> __specialization_static void __wrapper__device_stub_GEMMI8TCU<256,256,32,64,64,2,true,false,true>( const ::int8_t *&__cuda_0,const ::int8_t *&__cuda_1,::int8_t *&__cuda_2,int &__cuda_3,int &__cuda_4,int &__cuda_5){__device_stub__ZN6kernel9GEMMI8TCUILi256ELi256ELi32ELi64ELi64ELi2ELb1ELb0ELb1EEEvPKaS2_Paiii( (const ::int8_t *&)__cuda_0,(const ::int8_t *&)__cuda_1,(::int8_t *&)__cuda_2,(int &)__cuda_3,(int &)__cuda_4,(int &)__cuda_5);}}
static void __nv_cudaEntityRegisterCallback(void **__T122){__nv_dummy_param_ref(__T122);__nv_save_fatbinhandle_for_managed_rt(__T122);__cudaRegisterEntry(__T122, ((void ( *)(const int8_t *, const int8_t *, int8_t *, int, int, int))kernel::GEMMI8TCU<(int)256, (int)256, (int)32, (int)64, (int)64, (int)2, (bool)1, (bool)0, (bool)1> ), _ZN6kernel9GEMMI8TCUILi256ELi256ELi32ELi64ELi64ELi2ELb1ELb0ELb1EEEvPKaS2_Paiii, (-1));}
static void __sti____cudaRegisterAll(void){__cudaRegisterBinary(__nv_cudaEntityRegisterCallback);}

#pragma GCC diagnostic pop
