"""
DLR operations for DensePropagator.

Provides tau2dlr, dlr2tau, reflection, and convolution operations for dense propagators.
These are trivial wrappers around QMBToolkitDLR functions that work on Array{T,3}.
"""

using QMBToolkitDLR

#=============================================================================
# DLR Transformations
=============================================================================#

"""
    tau2dlr_dense!(dest::DensePropagator{T}, dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T

In-place transform dense propagator from tau space to DLR representation.
Writes result to dest.
"""
function tau2dlr_dense!(dest::DensePropagator{T}, dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T
    # QMBToolkitDLR.tau2dlr is allocating, so we call it and copy to dest
    result = QMBToolkitDLR.tau2dlr(dlr_grid, prop.data, tau_grid)
    copyto!(dest.data, result)
    return dest
end

"""
    tau2dlr_dense(dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform dense propagator from tau space to DLR representation (allocating version).
"""
function tau2dlr_dense(dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T
    coeffs = QMBToolkitDLR.tau2dlr(dlr_grid, prop.data, tau_grid)
    return DensePropagator{T}(coeffs)
end

"""
    dlr2tau_dense(dlr_grid::DLRGrid, prop_dlr::DensePropagator{T}, tau_grid::Vector{Float64}) where T

Transform dense propagator from DLR representation to tau space.
"""
function dlr2tau_dense(dlr_grid::DLRGrid, prop_dlr::DensePropagator{T}, tau_grid::Vector{Float64}) where T
    vals = QMBToolkitDLR.dlr2tau(dlr_grid, prop_dlr.data, tau_grid)
    return DensePropagator{T}(vals)
end

#=============================================================================
# Reflection Operations
=============================================================================#

"""
    do_reflection_tau2tau_dense(dlr_grid::DLRGrid, prop_tau::DensePropagator{T}) where T

Perform reflection operation on tau-space dense data: F(tau) = H(beta - tau).
First transforms to DLR, then applies reflection.
"""
function do_reflection_tau2tau_dense(dlr_grid::DLRGrid, prop_tau::DensePropagator{T}) where T
    reflected = QMBToolkitDLR.do_reflection_tau2tau(dlr_grid, prop_tau.data)
    return DensePropagator{T}(reflected)
end

#=============================================================================
# Convolution Operations
=============================================================================#

"""
    do_convolution_dlr2tau_dense!(dest::DensePropagator{T}, workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T

Compute convolution H = F * G for dense propagators in DLR representation.
Writes result to dest in tau space. This is the preferred in-place version.
"""
function do_convolution_dlr2tau_dense!(dest::DensePropagator{T}, workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T
    result = QMBToolkitDLR.do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_dlr.data, G_dlr.data)
    copyto!(dest.data, result)
    return dest
end

"""
    do_convolution_dlr2tau_dense!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T

Compute convolution H = F * G for dense propagators in DLR representation.
Returns result in tau space as a new DensePropagator (allocating version).
"""
function do_convolution_dlr2tau_dense!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T
    result = QMBToolkitDLR.do_convolution_dlr2tau!(workspace, dlr_grid, kernel_cache, F_dlr.data, G_dlr.data)
    return DensePropagator{T}(copy(result))  # Must copy since workspace.H_tau is reused
end

#=============================================================================
# Tau Extraction Operations
=============================================================================#

"""
    extract_at_tau_dense(propagator::DensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where T

Extract propagator value at specific tau by DLR interpolation, return as dense matrix.
"""
function extract_at_tau_dense(propagator::DensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where T
    # Convert to DLR coefficients
    prop_dlr = tau2dlr_dense(dlr_grid, propagator, dlr_grid.tau)

    # Evaluate at the requested tau
    prop_at_tau = dlr2tau_dense(dlr_grid, prop_dlr, [tau])

    # Return the single time slice as a matrix
    return prop_at_tau.data[1, :, :]
end

#=============================================================================
# Interface Implementations
=============================================================================#

# DLR transformations - interface versions
tau2dlr(dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T = tau2dlr_dense(dlr_grid, prop, tau_grid)
tau2dlr!(dest::DensePropagator{T}, dlr_grid::DLRGrid, prop::DensePropagator{T}, tau_grid::Vector{Float64}) where T = tau2dlr_dense!(dest, dlr_grid, prop, tau_grid)
dlr2tau(dlr_grid::DLRGrid, prop_dlr::DensePropagator{T}, tau_grid::Vector{Float64}) where T = dlr2tau_dense(dlr_grid, prop_dlr, tau_grid)

# Reflection
do_reflection_tau2tau(dlr_grid::DLRGrid, prop_tau::DensePropagator{T}) where T = do_reflection_tau2tau_dense(dlr_grid, prop_tau)

# Extract at tau
extract_at_tau(propagator::DensePropagator{T}, tau::Float64, dlr_grid::DLRGrid) where T = extract_at_tau_dense(propagator, tau, dlr_grid)

# Convolution
do_convolution_dlr2tau!(workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T = do_convolution_dlr2tau_dense!(workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
do_convolution_dlr2tau!(dest::DensePropagator{T}, workspace::ConvolutionWorkspace{T}, dlr_grid::DLRGrid, kernel_cache::KernelCache, F_dlr::DensePropagator{T}, G_dlr::DensePropagator{T}) where T = do_convolution_dlr2tau_dense!(dest, workspace, dlr_grid, kernel_cache, F_dlr, G_dlr)
