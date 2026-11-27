package nie.translator.seamless;

import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtException;

final class CpuTuning {
    /** Opt-in experiment: keep inactive session thread pools from busy spinning. */
    static void quiet(OrtSession.SessionOptions options,boolean xnnpack)throws OrtException{
        options.setIntraOpNumThreads(xnnpack?1:4);
        options.setInterOpNumThreads(1);
        options.addConfigEntry("session.intra_op.allow_spinning","0");
        options.addConfigEntry("session.inter_op.allow_spinning","0");
    }
}
