"""
Stub implementations for BlockDense operations in SOEXCA.

These will be implemented in Phase 5.
For now, they provide clear error messages when BlockDense format is attempted.
"""

function propagator_convolution_buffers_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function gen_propagator_given_order_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function eval_propagator_given_order_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function multiplication_convolution_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function multiplication_convolution_reverse_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function apply_hybridization_inplace_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end

function operator_multiplication_blockdense(args...; kwargs...)
    error("BlockDense format is not yet implemented (Phase 5). Use :dense format.")
end
