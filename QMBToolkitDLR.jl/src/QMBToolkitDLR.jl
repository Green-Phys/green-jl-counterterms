module QMBToolkitDLR

using Lehmann
using QMBToolkitED

# Workspace structures (format-independent)
include("workspace.jl")

# Dense operations (current implementation)
include("dense/convolution.jl")
include("dense/reflection.jl")

# BlockDense operations (Phase 4 implementation)
include("blockdense/tau2dlr.jl")
include("blockdense/dlr2tau.jl")
include("blockdense/convolution.jl")
include("blockdense/reflection.jl")

# Re-export from Lehmann
export DLRGrid, dlr2tau, tau2dlr

# Workspace exports
export kernel_tau, KernelCache, compute_kernel_cache, ConvolutionWorkspace

# Dense operation exports
export do_convolution_dlr2tau!
export do_reflection_tau2tau

# BlockDense type and operation exports
export BlockDenseArray
export BlockDenseConvolutionWorkspace, create_blockdense_convolution_workspace
export tau2dlr_blockdense, dlr2tau_blockdense
export do_convolution_dlr2tau_blockdense!, do_reflection_blockdense, do_reflection_tau2tau_blockdense

end
