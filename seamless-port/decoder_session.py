"""Host decoder memory experiment, preserving FP32 arithmetic."""
import onnxruntime as ort

def configure(options):
    # Avoid retaining expanded DequantizeLinear weights through constant
    # folding/prepacking and reduce allocator high-water retention.
    # These settings may slow inference and do not guarantee lower RSS.
    options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern=False
    options.add_session_config_entry("session.disable_prepacking","1")
    return options


def configure_native(options):
    """The exact host configuration used by the passing native INT8 test."""
    options.enable_cpu_mem_arena=False
    options.enable_mem_pattern=False
    options.add_session_config_entry("session.disable_prepacking","1")
    return options


def configure_fp16(options, profile):
    """Independent host experiment; no mobile rollout until measured."""
    if profile=="default":
        return options
    if profile not in ("no-retention","basic-no-retention"):
        raise ValueError("unknown FP16 session profile")
    configure_native(options)
    if profile=="basic-no-retention":
        options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return options
