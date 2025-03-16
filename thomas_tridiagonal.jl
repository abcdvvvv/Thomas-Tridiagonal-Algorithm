"""
A concise, lightweight, high-performance Julia implementation of the Thomas Algorithm. You can copy the code to use in your scripts or modules.
"""

const tridiagonal_buffer = Vector{Vector{Float64}}(undef, nthreads())

function get_tridiagonal_buffer(n::Int)
    tid = threadid()
    if !isassigned(tridiagonal_buffer, tid)
        U⁺ = tridiagonal_buffer[tid] = vec(n - 1)
    else
        U⁺ = tridiagonal_buffer[tid]
        if length(U⁺) != n - 1
            resize!(U⁺, n - 1)
        end
    end
    return U⁺
end

function thomas_tridiagonal!(x::AbstractVector, L::AbstractVector, D::AbstractVector,
    U::AbstractVector, b::AbstractVector)
    n = length(b)
    U⁺ = get_tridiagonal_buffer(n)
    # Forward sweep
    D⁺ = D[1]
    x[1] = b[1] / D[1]
    @inbounds @simd for i = 2:n
        U⁺[i-1] = U[i-1] / D⁺
        D⁺ = D[i] - L[i] * U⁺[i-1]
        x[i] = (b[i] - L[i] * x[i-1]) / D⁺
    end
    # Back substitution
    @inbounds @simd for i = n-1:-1:1
        x[i] -= U⁺[i] * x[i+1]
    end
    return x
end

thomas_tridiagonal(L::AbstractVector, D::AbstractVector, U::AbstractVector,
    b::AbstractVector) = thomas_tridiagonal!(similar(b), L, D, U, b)
