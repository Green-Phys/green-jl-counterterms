module QMBToolkitDLR

using Lehmann

# Workspace structures
include("workspace.jl")

# Dense operations
include("convolution.jl")
include("reflection.jl")

# Re-export from Lehmann (including in-place versions)
export DLRGrid, dlr2tau, tau2dlr, dlr2tau!, tau2dlr!, DLRWorkspace, ensure_kernel_τ_lu!

# Workspace exports
export kernel_tau, KernelCache, compute_kernel_cache, ConvolutionWorkspace

# Dense operation exports
export do_convolution_dlr2tau!
export do_reflection_tau2tau!

end
