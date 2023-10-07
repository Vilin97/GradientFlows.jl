struct Blob{S,F,A} <: Solver
    score_values::S
    ε::F
    allocated_memory::A
end
Blob(ε=0.025) = Blob(nothing, ε, nothing)

struct BlobAllocMemCPU{T}
    diff_norm2s::Matrix{T}
    mol_sum::Vector{T}
    mols::Matrix{T}
end

"Initialize solver."
function initialize(solver::Blob, score_values::Matrix{T}) where {T}
    n = size(score_values, 2)
    diff_norm2s = zeros(T, n, n)
    mol_sum = zeros(T, n)
    mols = zeros(T, n, n)
    allocated_memory = BlobAllocMemCPU(diff_norm2s, mol_sum, mols)
    Blob(copy(score_values), T(solver.ε), allocated_memory)
end

"Fill in solver.score_values."
function update!(solver::Blob{S,F,A}, integrator) where {S,F,A<:BlobAllocMemCPU}
    @unpack score_values, ε, allocated_memory = solver
    @unpack diff_norm2s, mol_sum, mols = allocated_memory
    u = integrator.u
    d, n = size(u)

    diff_norm2s .= 0
    mol_sum .= 0
    score_values .= 0
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        diff_norm2s[p, q] += (u[k, p] - u[k, q])^2
    end
    @tturbo for p in 1:n, q in 1:n
        mols[p, q] = exp(-diff_norm2s[p, q] / ε) / sqrt((π * ε)^d)
        mol_sum[p] += mols[p, q]
    end
    @tturbo for p in 1:n, q in 1:n, k in 1:d
        fac = -2.0 / ε * mols[p, q]
        diff_k = u[k, p] - u[k, q]
        score_values[k, p] += fac * diff_k / mol_sum[p]
        score_values[k, p] += fac * diff_k / mol_sum[q]
    end
    nothing
end