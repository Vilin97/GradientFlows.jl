score(ρ::MultivariateDistribution, u::AbstractArray{T,1}) where {T} = gradlogpdf(ρ, u)
score(ρ::MultivariateDistribution, u::AbstractArray{T,2}) where {T} = reshape(hcat([score(ρ, @view u[:, i]) for i in axes(u, 2)]...), size(u))
score(ρ::UnivariateDistribution, u::Number) = gradlogpdf(ρ, u)
score(ρ::Product, u::AbstractArray{T,1}) where {T} = [score(ρᵢ, uᵢ) for (ρᵢ,uᵢ) in zip(ρ.v, u)]
score(ρ::Product, u::AbstractArray{T,2}) where {T} = reshape(hcat([score(ρ, @view u[:, i]) for i in axes(u, 2)]...), size(u))