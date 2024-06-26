"Σ n^(-2 / (d + 4)), from Equation 3.17 in https://bookdown.org/egarpor/NP-UC3M/kde-ii-bwd.html (first constant is ~1)"
function kde_bandwidth(u)
    d, n = size(u)
    Σ = cov(u')
    Σ .* n^(-2 / (d + 4))
end

kde(x, u; h=kde_bandwidth(u)) = kde(x, u, h)
kde(x, u, h::Number) = kde(x, u, h*I(size(u,1)))
function kde(x, u, h::AbstractMatrix)
    res = zero(eltype(u))
    d = length(x)
    h_inv = inv(h)
    for x_q in eachcol(u)
        res += exp(-normsq(x, x_q, h_inv) / 2)
    end
    res / (sqrt((2π)^d * det(h)) * size(u, 2))
end
