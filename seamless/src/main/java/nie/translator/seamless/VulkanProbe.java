package nie.translator.seamless;

/** Independent Vulkan compute, not ONNX Runtime, no model or network. */
final class VulkanProbe {
    static String run(){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            return nativeRun(0);
        }catch(LinkageError e){return "Direct Vulkan probe unavailable: "+e;}
    }
    static String matmul(int rows){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            return nativeRun(rows);
        }catch(LinkageError e){return "Vulkan MatMul unavailable: "+e;}
    }
    static String gemv(boolean batched){return matmul(batched?-8:-1);}
    private static native String nativeRun(int mode);
}
