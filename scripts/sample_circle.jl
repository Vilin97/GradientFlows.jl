using Random
using Distributions
import Distributions: sample, pdf
using Plots

# Plot settings
default(size=(800, 800))

struct CircleDist
    mean::Array{Float64}
    radius::Float64
    noise::Float64
end

function sample(ring::CircleDist, n_samples::Int; seed::Union{Nothing, Int}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    samples = rand(MvNormal(I(2)), n_samples)
    radius_dist = Normal(ring.radius, ring.noise)
    samples = [ring.mean+rand(radius_dist)*x/norm(x) for x in eachcol(samples)]

    return samples
end

function score(ring::CircleDist, x::AbstractArray)
    r = sum(abs2, points, dims=1)
    return x .* (ring.radius ./ r .- 1)
end

# Example usage
ring = CircleDist([0.5,0.5], 0.5, 0.01)

# Sample points from the distribution
n_samples = 1000
points = hcat(sample(ring, n_samples)...)

# Plot the sampled points
scatter(points[1, :], points[2, :], legend=false, title="Ring Distribution Samples")

# Probability density function for a set of points
x = rand(MvNormal(ring.mean, ring.radius/5),1000)  # Random points for PDF calculation

# Gradient of the log of the PDF (scores)
scores_values = score(ring, x)

println("Scores: ", scores_values[:, 1:5])  # Display the first 5 score values

plot_vector_field(x_ -> score(ring, x_), points)