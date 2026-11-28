#include <jni.h>
#include <vulkan/vulkan.h>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>
#include "probe_shader.h"
#include "matmul_shader.h"
#include "gemv_shader.h"
#include "gemv_subgroup16_shader.h"
#include "gemv_subgroup32_shader.h"
#include "gemv_fused4_shader.h"
#include "gemv_fused8_shader.h"
#include "gemv_fused16_shader.h"
#include <algorithm>
#include <cstring>

namespace {
void check(VkResult r,const char* step){
    if(r!=VK_SUCCESS)throw std::runtime_error(std::string(step)+" VkResult="+std::to_string(r));
}
struct Resources {
    VkInstance instance=VK_NULL_HANDLE;
    VkDevice device=VK_NULL_HANDLE;
    VkBuffer buffer=VK_NULL_HANDLE;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout=VK_NULL_HANDLE;
    VkDescriptorPool pool=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkShaderModule shader=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkCommandPool commands=VK_NULL_HANDLE;
    VkFence fence=VK_NULL_HANDLE;
    VkQueryPool queryPool=VK_NULL_HANDLE;
    void* mapped=nullptr;
    bool pending=false;
    ~Resources(){
        // A timed-out command may still be in flight. Do not destroy resources
        // referenced by it or wait indefinitely; keep them until process exit.
        if(pending)return;
        if(device){
            if(mapped)vkUnmapMemory(device,memory);
            if(queryPool)vkDestroyQueryPool(device,queryPool,nullptr);
            if(fence)vkDestroyFence(device,fence,nullptr);
            if(commands)vkDestroyCommandPool(device,commands,nullptr);
            if(pipeline)vkDestroyPipeline(device,pipeline,nullptr);
            if(shader)vkDestroyShaderModule(device,shader,nullptr);
            if(layout)vkDestroyPipelineLayout(device,layout,nullptr);
            if(pool)vkDestroyDescriptorPool(device,pool,nullptr);
            if(setLayout)vkDestroyDescriptorSetLayout(device,setLayout,nullptr);
            if(buffer)vkDestroyBuffer(device,buffer,nullptr);
            if(memory)vkFreeMemory(device,memory,nullptr);
            vkDestroyDevice(device,nullptr);
        }
        if(instance)vkDestroyInstance(instance,nullptr);
    }
};
std::atomic<bool> timedOut{false};
}
extern "C" JNIEXPORT jstring JNICALL
Java_nie_translator_seamless_VulkanProbe_nativeRun(JNIEnv* env,jclass,jint mode){
    std::ostringstream out;
    if(timedOut.load())return env->NewStringUTF("Direct Vulkan: previous queue failure; restart app before retry");
    Resources r;
    try{
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="Seamless compute diagnostic";app.apiVersion=VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};ici.pApplicationInfo=&app;
        check(vkCreateInstance(&ici,nullptr,&r.instance),"vkCreateInstance");
        uint32_t n=0;check(vkEnumeratePhysicalDevices(r.instance,&n,nullptr),"enumerate device count");
        if(!n)throw std::runtime_error("no Vulkan device");
        std::vector<VkPhysicalDevice> devices(n);check(vkEnumeratePhysicalDevices(r.instance,&n,devices.data()),"enumerate devices");
        VkPhysicalDevice physical=VK_NULL_HANDLE;uint32_t family=0,timestampBits=0;int best=-1;
        for(auto d:devices){
            VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(d,&p);
            uint32_t qn=0;vkGetPhysicalDeviceQueueFamilyProperties(d,&qn,nullptr);
            std::vector<VkQueueFamilyProperties> qs(qn);vkGetPhysicalDeviceQueueFamilyProperties(d,&qn,qs.data());
            int score=p.deviceType==VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU?3:p.deviceType==VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU?2:p.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU?0:1;
            for(uint32_t i=0;i<qn;i++)if(qs[i].queueCount&&(qs[i].queueFlags&VK_QUEUE_COMPUTE_BIT)&&score>best){physical=d;family=i;timestampBits=qs[i].timestampValidBits;best=score;break;}
        }
        if(!physical)throw std::runtime_error("no compute queue");
        VkPhysicalDeviceProperties prop{};vkGetPhysicalDeviceProperties(physical,&prop);
        VkPhysicalDeviceSubgroupProperties subgroupProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};props2.pNext=&subgroupProps;
        vkGetPhysicalDeviceProperties2(physical,&props2);
        const bool subgroupArithmetic=(subgroupProps.supportedStages&VK_SHADER_STAGE_COMPUTE_BIT)&&
            (subgroupProps.supportedOperations&VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
        out<<"Vulkan subgroup size: "<<subgroupProps.subgroupSize
           <<" | compute arithmetic: "<<(subgroupArithmetic?"yes":"no")<<"\\n";
        out<<"Vulkan device: "<<prop.deviceName<<" | type="<<prop.deviceType<<" | vendor=0x"<<std::hex<<prop.vendorID<<std::dec<<" | driver="<<prop.driverVersion<<" | API "<<VK_VERSION_MAJOR(prop.apiVersion)<<"."<<VK_VERSION_MINOR(prop.apiVersion)<<"."<<VK_VERSION_PATCH(prop.apiVersion)<<"\n";
        float priority=1.f;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;
        check(vkCreateDevice(physical,&di,nullptr,&r.device),"vkCreateDevice");
        VkQueue queue;vkGetDeviceQueue(r.device,family,0,&queue);
        if(timestampBits&&prop.limits.timestampComputeAndGraphics){
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType=VK_QUERY_TYPE_TIMESTAMP;qpi.queryCount=2;
            check(vkCreateQueryPool(r.device,&qpi,nullptr,&r.queryPool),"timestamp query pool");
        }
        // Positive modes are tiled GEMM; negative -1..-32 are repeated
        // baseline GEMV dispatches; -104/-108/-116 are fused 4/8/16-output kernels.
        const bool gemv=mode<0;
        const bool subgroup=mode==-201;
        const bool fused=mode==-104||mode==-108||mode==-116;
        const int dispatches=fused||subgroup?1:(gemv?-mode:1);
        const uint32_t outputsPerGroup=fused?uint32_t(-mode-100):4u;
        const int rows=gemv?1:mode;
        const bool validMode=mode==0||mode==1||mode==32||mode==-1||mode==-4||mode==-8||
            mode==-16||mode==-32||mode==-104||mode==-108||mode==-116||subgroup;
        if(!validMode)throw std::runtime_error("unsupported probe mode");
        constexpr uint32_t K=1024,N=8192;
        const uint32_t aCount=uint32_t(rows)*K;
        const uint32_t bCount=K*N/2;
        const uint32_t cCount=uint32_t(rows)*N;
        const VkDeviceSize bytes=rows?VkDeviceSize(aCount+bCount+cCount)*4:16*sizeof(float);
        if(bytes>prop.limits.maxStorageBufferRange)throw std::runtime_error("storage buffer limit");
        if(rows&&!fused&&!subgroup&&(prop.limits.maxComputeWorkGroupInvocations<128||prop.limits.maxComputeWorkGroupSize[0]<(gemv?32u:16u)||prop.limits.maxComputeWorkGroupSize[1]<(gemv?4u:8u)))throw std::runtime_error("workgroup limit");
        if(subgroup&&(!subgroupArithmetic||(subgroupProps.subgroupSize!=16&&subgroupProps.subgroupSize!=32)||
            prop.limits.maxComputeWorkGroupInvocations<subgroupProps.subgroupSize*4||
            prop.limits.maxComputeWorkGroupSize[0]<subgroupProps.subgroupSize||prop.limits.maxComputeWorkGroupSize[1]<4))
            throw std::runtime_error("subgroup arithmetic/size 16 or 32 unsupported");
        if(fused&&(prop.limits.maxComputeWorkGroupInvocations<32||prop.limits.maxComputeWorkGroupSize[0]<32))throw std::runtime_error("fused workgroup limit");
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};bi.size=bytes;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(r.device,&bi,nullptr,&r.buffer),"vkCreateBuffer");
        VkMemoryRequirements req{};vkGetBufferMemoryRequirements(r.device,r.buffer,&req);
        VkPhysicalDeviceMemoryProperties mp{};vkGetPhysicalDeviceMemoryProperties(physical,&mp);
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<mp.memoryTypeCount;i++)if((req.memoryTypeBits&(1u<<i))&&((mp.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))){type=i;break;}
        if(type==UINT32_MAX)throw std::runtime_error("no host-visible coherent storage-buffer memory");
        VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};mi.allocationSize=req.size;mi.memoryTypeIndex=type;
        check(vkAllocateMemory(r.device,&mi,nullptr,&r.memory),"vkAllocateMemory");
        check(vkBindBufferMemory(r.device,r.buffer,r.memory,0),"vkBindBufferMemory");
        check(vkMapMemory(r.device,r.memory,0,VK_WHOLE_SIZE,0,&r.mapped),"vkMapMemory");
        auto values=static_cast<float*>(r.mapped);
        // Binary-exact inputs, nonconstant signs, reproducible reference.
        auto av=[](uint32_t m,uint32_t k){return float(int((m*7+k*3)%17)-8)/16.f;};
        auto wi=[](uint32_t k,uint32_t n){return (k*13u+n*7u+(k*n)%19u)%4u;};
        const uint16_t halfBits[]={0x3400,0xb400,0x3800,0xb800};
        const float weights[]={.25f,-.25f,.5f,-.5f};
        if(rows){
            auto words=static_cast<uint32_t*>(r.mapped);
            for(uint32_t m=0;m<uint32_t(rows);m++)for(uint32_t k=0;k<K;k++){
                float f=av(m,k);std::memcpy(words+m*K+k,&f,4);
            }
            if(gemv){
                // Physically transpose B, without changing any of its values.
                // Each output column is contiguous in reduction dimension.
                for(uint32_t n=0;n<N;n++)for(uint32_t k=0;k<K;k+=2)
                    words[aCount+(n*K+k)/2]=uint32_t(halfBits[wi(k,n)])|(uint32_t(halfBits[wi(k+1,n)])<<16);
            }else{
                for(uint32_t k=0;k<K;k++)for(uint32_t n=0;n<N;n+=2)
                    words[aCount+(k*N+n)/2]=uint32_t(halfBits[wi(k,n)])|(uint32_t(halfBits[wi(k,n+1)])<<16);
            }
            std::fill(words+aCount+bCount,words+aCount+bCount+cCount,0x7fc00000u);
        }else for(int i=0;i<16;i++)values[i]=float(i+1);
        VkDescriptorSetLayoutBinding binding{};binding.binding=0;binding.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;binding.descriptorCount=1;binding.stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};sli.bindingCount=1;sli.pBindings=&binding;
        check(vkCreateDescriptorSetLayout(r.device,&sli,nullptr,&r.setLayout),"descriptor layout");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pi.maxSets=1;pi.poolSizeCount=1;pi.pPoolSizes=&ps;
        check(vkCreateDescriptorPool(r.device,&pi,nullptr,&r.pool),"descriptor pool");
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};ai.descriptorPool=r.pool;ai.descriptorSetCount=1;ai.pSetLayouts=&r.setLayout;
        VkDescriptorSet set;check(vkAllocateDescriptorSets(r.device,&ai,&set),"descriptor allocation");
        VkDescriptorBufferInfo dbi{r.buffer,0,bytes};VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};w.dstSet=set;w.dstBinding=0;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&dbi;
        vkUpdateDescriptorSets(r.device,1,&w,0,nullptr);
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT,0,4};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&r.setLayout;
        if(rows&&!fused&&!subgroup){pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&range;}
        check(vkCreatePipelineLayout(r.device,&pli,nullptr,&r.layout),"pipeline layout");
        const uint32_t* shaderCode=probe_shader;
        size_t shaderBytes=sizeof(probe_shader);
        if(subgroup){
            if(subgroupProps.subgroupSize==16){shaderCode=gemv_subgroup16_shader;shaderBytes=sizeof(gemv_subgroup16_shader);}
            else {shaderCode=gemv_subgroup32_shader;shaderBytes=sizeof(gemv_subgroup32_shader);}
        }else if(fused){
            if(outputsPerGroup==4){shaderCode=gemv_fused4_shader;shaderBytes=sizeof(gemv_fused4_shader);}
            else if(outputsPerGroup==8){shaderCode=gemv_fused8_shader;shaderBytes=sizeof(gemv_fused8_shader);}
            else {shaderCode=gemv_fused16_shader;shaderBytes=sizeof(gemv_fused16_shader);}
        }else if(gemv){shaderCode=gemv_shader;shaderBytes=sizeof(gemv_shader);}
        else if(rows){shaderCode=matmul_shader;shaderBytes=sizeof(matmul_shader);}
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};si.codeSize=shaderBytes;si.pCode=shaderCode;
        check(vkCreateShaderModule(r.device,&si,nullptr,&r.shader),"shader module");
        VkComputePipelineCreateInfo pci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pci.layout=r.layout;pci.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pci.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pci.stage.module=r.shader;pci.stage.pName="main";
        check(vkCreateComputePipelines(r.device,VK_NULL_HANDLE,1,&pci,nullptr,&r.pipeline),"compute pipeline");
        VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};cpi.queueFamilyIndex=family;
        check(vkCreateCommandPool(r.device,&cpi,nullptr,&r.commands),"command pool");
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};cai.commandPool=r.commands;cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;cai.commandBufferCount=1;
        VkCommandBuffer cmd;check(vkAllocateCommandBuffers(r.device,&cai,&cmd),"command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};begin.flags=0;
        check(vkBeginCommandBuffer(cmd,&begin),"begin command");
        if(r.queryPool){
            vkCmdResetQueryPool(cmd,r.queryPool,0,2);
            vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,r.queryPool,0);
        }
        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,r.pipeline);
        vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,r.layout,0,1,&set,0,nullptr);
        if(rows){
            uint32_t m=uint32_t(rows);
            if(!fused&&!subgroup)vkCmdPushConstants(cmd,r.layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&m);
            if(gemv){
                for(int d=0;d<dispatches;d++){
                    if(d){
                        VkMemoryBarrier waw{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        waw.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;waw.dstAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
                        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&waw,0,nullptr,0,nullptr);
                    }
                    vkCmdDispatch(cmd,N/outputsPerGroup,1,1);
                }
            }else vkCmdDispatch(cmd,N/16,(m+7)/8,1);
        }else vkCmdDispatch(cmd,1,1,1);
        if(r.queryPool)vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,r.queryPool,1);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
        check(vkEndCommandBuffer(cmd),"end command");
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};check(vkCreateFence(r.device,&fi,nullptr,&r.fence),"fence");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&cmd;
        std::vector<double> wallUs,gpuUs;
        // The tiny diagnostic multiplies its buffer in place. Submit it exactly once;
        // repeating the command would multiply by four and fail the 2x assertion.
        const int warmupSubmissions=rows?std::max(1,(32+dispatches-1)/dispatches):0;
        const int measuredSubmissions=rows?11:1;
        if(rows){
            auto words=static_cast<uint32_t*>(r.mapped);
            std::fill(words+aCount+bCount,words+aCount+bCount+cCount,0x7fc00000u);
        }
        for(int repeat=0;repeat<warmupSubmissions+measuredSubmissions;repeat++){
            if(repeat)check(vkResetFences(r.device,1,&r.fence),"reset fence");
            auto t=std::chrono::steady_clock::now();
            check(vkQueueSubmit(queue,1,&submit,r.fence),"queue submit");
            r.pending=true;
            VkResult wait=vkWaitForFences(r.device,1,&r.fence,VK_TRUE,5000000000ULL);
            if(wait!=VK_SUCCESS){timedOut.store(true);throw std::runtime_error("fence wait VkResult="+std::to_string(wait)+"; restart app before retry");}
            r.pending=false;
            double elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t).count();
            if(repeat>=warmupSubmissions){
                wallUs.push_back(elapsed);
                if(r.queryPool){
                    uint64_t ts[2]={0,0};
                    VkResult qr=vkGetQueryPoolResults(r.device,r.queryPool,0,2,sizeof(ts),ts,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
                    if(qr!=VK_SUCCESS)throw std::runtime_error("timestamp query result VkResult="+std::to_string(qr));
                    uint64_t delta=ts[1]-ts[0];
                    if(timestampBits<64)delta&=((uint64_t(1)<<timestampBits)-1);
                    gpuUs.push_back(double(delta)*prop.limits.timestampPeriod/1000.0);
                }
            }
        }
        if(rows){
            // Validate the final output only after all timings to avoid CPU-side
            // reference work perturbing device clocks between measured submits.
            auto words=static_cast<uint32_t*>(r.mapped);
            for(uint32_t i=0;i<cCount;i++){
                float f;std::memcpy(&f,words+aCount+bCount+i,4);
                if(!std::isfinite(f))throw std::runtime_error("unwritten or nonfinite matrix output");
            }
            for(uint32_t j=0;j<64;j++){
                uint32_t m=j%uint32_t(rows), n=(j*127u+j*j*11u)%N;
                float expected=0;
                for(uint32_t k=0;k<K;k++)expected+=av(m,k)*weights[wi(k,n)];
                float got;std::memcpy(&got,words+aCount+bCount+m*N+n,4);
                if(std::abs(got-expected)>1e-4f)throw std::runtime_error("matrix readback mismatch");
            }
        }else for(int i=0;i<16;i++)if(!std::isfinite(values[i])||std::abs(values[i]-2.f*(i+1))>0.0001f)throw std::runtime_error("compute readback mismatch");
        std::sort(wallUs.begin(),wallUs.end());
        std::sort(gpuUs.begin(),gpuUs.end());
        if(rows){
            double wallMedian=wallUs[wallUs.size()/2];
            double gpuMedian=gpuUs.empty()?0:gpuUs[gpuUs.size()/2];
            double kernelUs=gpuUs.empty()?wallMedian:gpuMedian;
            out<<"Vulkan "<<(subgroup?"GEMV subgroup-reduction":fused?("GEMV fused-"+std::to_string(outputsPerGroup)):gemv?"GEMV output-major":"MatMul tiled")
               <<" "<<rows<<"x1024x8192: PASS | FP16 stored / FP32 accumulate | "<<dispatches<<" dispatch(es)/submit"
               <<" | median GPU "<<(gpuUs.empty()?std::string("unavailable"):std::to_string(gpuMedian/1000.0)+" ms")
               <<" | median queue+fence "<<wallMedian/1000.0<<" ms | min queue+fence "<<wallUs.front()/1000.0
               <<" ms | GPU us/dispatch "<<(gpuUs.empty()?0:gpuMedian/dispatches)
               <<" | "<<(2.0*rows*K*N*dispatches/std::max(1.0,kernelUs)/1000.0)<<" GFLOP/s"
               <<" | 32 warmup dispatches, 11 measured submissions; excludes setup";
            if(dispatches>1)out<<" | repeated same weights; final output checked";
        }else out<<"Direct Vulkan compute/readback: PASS ("<<wallUs[0]<<" us queue + fence; synthetic only)";
        if(prop.deviceType==VK_PHYSICAL_DEVICE_TYPE_CPU)out<<" [software CPU device]";
    }catch(const std::exception& e){out<<"Direct Vulkan: FAIL "<<e.what();}
    return env->NewStringUTF(out.str().c_str());
}
