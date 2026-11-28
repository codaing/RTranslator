package nie.translator.seamless;

/** Independent Vulkan compute, not ONNX Runtime, no model or network. */
final class VulkanProbe {
    static String run(){
        try{
            System.loadLibrary("seamless_vulkan_probe");
            return nativeRun();
        }catch(LinkageError e){return "Direct Vulkan probe unavailable: "+e;}
    }
    private static native String nativeRun();
}
