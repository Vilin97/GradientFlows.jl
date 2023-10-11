function Mol(ε, x, u::AbstractArray{T,2}) where {T}
    res = zero(eltype(u))
    d = length(x)
    eps_recip = 1 / ε
    for x_q in eachcol(u)
        res += exp(-sum(abs2, x .- x_q) * eps_recip)
    end
    res / sqrt((π * ε)^d)
end

rec_epsilon(d, n) = n^(-2 / (d + 4))
kde(x, u::AbstractMatrix; ε=rec_epsilon(size(u)...), kwargs...) = (@assert length(x) == size(u, 1); Mol(ε, x, u) / size(u, 2))