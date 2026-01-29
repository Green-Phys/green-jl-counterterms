"""
Reflection operations for dense representation.

Layout: (dim1, dim2, n_tau) for tau-space, (dim1, dim2, r) for DLR representation.
"""

using Lehmann
using LinearAlgebra

"""
    do_reflection_tau2tau!(dest::AbstractArray{T,3}, src::AbstractArray{T,3},
                           reflection_matrix::AbstractMatrix, workspace::DLRWorkspace{T}) where T

In-place reflection operation using pre-computed reflection matrix: dest = src(β - τ).

This is the most efficient version - performs reflection in a single matrix multiply
instead of the tau2dlr + dlr2tau approach. The reflection matrix R = K(β-τ,ω) * inv(K(τ,ω))
is pre-computed once and stored in ConvolutionWorkspace.

# Arguments
- `dest`: Output array in tau space (dim1, dim2, n_tau)
- `src`: Input array in tau space (dim1, dim2, n_tau)
- `reflection_matrix`: Pre-computed R = K(β-τ,ω) * inv(K(τ,ω)), shape (n_tau, n_tau)
- `workspace`: DLRWorkspace for buffer allocation

# Performance
This version saves one LU solve compared to the tau2dlr + dlr2tau approach,
providing ~20-30% speedup in the reflection step.
"""
function do_reflection_tau2tau!(
    dest::AbstractArray{T,3},
    src::AbstractArray{T,3},
    reflection_matrix::AbstractMatrix,
    workspace::DLRWorkspace{T},
) where {T<:Number}
    dim1, dim2, n_tau = size(src)

    @boundscheck begin
        @assert size(dest) == size(src) "dest size $(size(dest)) != src size $(size(src))"
        @assert size(reflection_matrix) == (n_tau, n_tau) "reflection_matrix size $(size(reflection_matrix)) != expected ($n_tau, $n_tau)"
        @assert workspace.n_dlr >= n_tau "Workspace n_dlr $(workspace.n_dlr) < required $n_tau"
        @assert workspace.n_other >= dim1 * dim2 "Workspace n_other $(workspace.n_other) < required $(dim1*dim2)"
    end

    # Use workspace buffers for the matrix operation
    # src(dim1, dim2, n_tau) -> buffer_in(n_tau, dim1*dim2)
    buffer_in = view(workspace.buffer1, 1:n_tau, 1:(dim1*dim2))
    buffer_out = view(workspace.buffer2, 1:n_tau, 1:(dim1*dim2))

    # Copy input to contiguous buffer
    Lehmann._tensor3_to_matrix!(buffer_in, src)

    # Single matrix multiply: dest = reflection_matrix * src
    mul!(buffer_out, reflection_matrix, buffer_in)

    # Copy result back
    Lehmann._matrix_to_tensor3!(dest, buffer_out)

    return dest
end
