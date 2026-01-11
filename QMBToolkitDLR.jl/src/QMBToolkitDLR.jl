module QMBToolkitDLR

using Lehmann

# Workspace structures
include("workspace.jl")

# Dense operations
include("convolution.jl")
include("reflection.jl")

# Re-export from Lehmann
export DLRGrid, dlr2tau, tau2dlr

# Workspace exports
export kernel_tau, KernelCache, compute_kernel_cache, ConvolutionWorkspace

# Dense operation exports
export do_convolution_dlr2tau!
export do_reflection_dlr2tau, do_reflection_tau2tau

end
