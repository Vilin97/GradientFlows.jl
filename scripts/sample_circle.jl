using Random
using Distributions, Flux
import Distributions: sample, pdf
using Plots, LinearAlgebra
using Zygote: pullback

"= ∑ᵢ ∇⋅s(xᵢ)"
function divergence(f, v::AbstractMatrix)
    res = zero(eltype(v))
    fv, ∂f = pullback(f, v)
    for i in axes(v,1)
        seed = zero.(fv)
        seed[i,:] .= 1
        res += sum(∂f(seed)[1][i,:])
    end
    return res
end
"= ∑ᵢ s(xᵢ)² + 2∇⋅s(xᵢ)"
true_score_matching_loss(s, u) = (sum(abs2, s(u)) + 2 * divergence(s, u)) / size(u, 2)


# Plot settings
default(size=(800, 800))

function plot_vector_field(s, samples; mesh_size=41, alpha=0.1, lims=(-1,1))
    s = s |> cpu
    samples = samples |> cpu
    x = range(lims..., length=mesh_size)
    y = range(lims..., length=mesh_size)
    sxy = [s([x_, y_]) for x_ in x, y_ in y]
    sxy = reshape(sxy, :)
    # normalize sxy so that the longest arrow is 1/mesh_size
    sxy = 2 .* sxy ./ maximum(norm.(sxy)) / mesh_size
    sx = [v[1] for v in sxy]
    sy = [v[2] for v in sxy]
    xx = reshape(repeat(x, 1, mesh_size), :)
    yy = reshape(repeat(y', mesh_size, 1), :)
    
    quiver(xx, yy, quiver=(sx, sy), label="score approximation")
    scatter!(samples[1, :], samples[2, :], alpha=alpha, markersize=2, label="Samples")
end

plot_vector_field(x -> -x, [0,0])

struct CircleDist
    mean::Array{Float64}
    radius::Float64
    noise::Float64
end

function sample(ring::CircleDist, n_samples::Int; seed::Union{Nothing, Int}=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    samples = randn(2, n_samples)
    radius_dist = Normal(ring.radius, ring.noise)
    for (i,x) in enumerate(eachcol(samples))
        # x/norm(x) is a point on the unit circle
        r = rand(radius_dist)
        samples[:,i] = ring.mean + r * x / norm(x)
    end
    return samples
end

function pdf(ring::CircleDist, x::AbstractArray)
    r = reshape(sqrt.(sum(abs2, x,dims=1) .+ eps()),:) # vector of norms
    return pdf(Normal(ring.radius, ring.noise), r)
end

function score(ring::CircleDist, x::AbstractArray)
    r = sqrt.(sum(abs2, x .- ring.mean, dims=1) .+ eps())
    return (x .- ring.mean) .* (ring.radius ./ r .- 1) ./ ring.noise^2
end

# Example usage
ring = CircleDist([0.,0.], 1., 0.1)

# Sample points from the distribution
n_samples = 10000
points = sample(ring, n_samples)

# Plot the sampled points
scatter(points[1, :], points[2, :], legend=false, title="Ring Distribution Samples")

plt=plot_vector_field(x_ -> score(ring, x_), points; lims=(-0.9,0.9));
plot(plt, title="True score function")

using Flux, CUDA
function score_matching_loss(s, u, ζ, α)
    denoise_val = dot(s(u .+ α .* ζ) .- s(u .- α .* ζ), ζ) / α
    su = s(u)
    return (dot(su, su) + denoise_val) / size(u, 2)
end

u = points |> gpu
depth=3; activation = softsign
s = Chain(
    Dense(2 => 128, activation),
    [Dense(128 => 128, activation) for _ in 1:depth-1]...,
    Dense(128 => 2)
) |> gpu
optimiser = ADAM(1e-4)
optimiser_state = Flux.setup(optimiser, s)
epochs = 1000
denoising_alpha = 0.1f0
verbose = 2
plt = plot_vector_field(s, u);
plot!(plt, title="Score approximation at epoch 0")
plots = []
push!(plots, plt)
batchsize = 500
data_loader = Flux.DataLoader(u, batchsize=min(size(u, 2), batchsize), partial=false, buffer=true)
ζ = similar(u[:,1:batchsize]) |> gpu
for epoch in 1:epochs
    for x in data_loader
        randn!(ζ)
        loss_value, grads = Flux.withgradient(s -> score_matching_loss(s, x, ζ, denoising_alpha), s)
        Flux.update!(optimiser_state, s, grads[1])
    end
    if epoch % 100 == 0
        verbose > 1 && @info "Epoch $(lpad(epoch, 2)), true loss = $(true_score_matching_loss(s, u))."
        plt = plot_vector_field(s, u)
        plot!(plt, title="Score approximation at epoch $epoch")
        push!(plots, plt)
    end
end
layer_dimensions(s) = [2,[length(layer.bias) for layer in s.layers[1:end-1]]...,2]
plt = plot_vector_field(s, u;alpha=0.1,lims=(-1.,1.));
plot!(plt, title="Score approximation with NN $(layer_dimensions(s))")

using GradientFlows
GradientFlows.save("data/models/circle.jld2", s|>cpu)
s = GradientFlows.load("data/models/circle.jld2")

"s ≈ ∇log p, where p is the target density"
function circle_problem(n, solver_; t0::F=0.0, t_end::F=1.0, dt::F=0.01, rng=GradientFlows.DEFAULT_RNG, kwargs...) where {F}
    ring = CircleDist([3.,0.], 1., 0.1)
    f!(du, u, prob, t) = (du .= score(ring, u) .- prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = nothing
    params = nothing
    ρ0 = MvNormal(I(2))
    u0 = rand(rng, ρ0, n)
    name = "circle"
    solver = GradientFlows.initialize(solver_, u0, GradientFlows.score(ρ0, u0), name; kwargs...)
    diffusion_coefficient(u, params) = 1
    covariance(t, params) = nothing
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, diffusion_coefficient, covariance)
end

solver=SBTM(learning_rate=1e-4, epochs=15, denoising_alpha=0.1f0, init_batch_size=500, init_loss_tolerance=1f-5, init_max_iterations=10^5, verbose=2)
prob = circle_problem(10^3, solver)
sol=solve(prob)
scatter(sol[1][1,:], sol[1][2,:], label="Initial distribution",alpha=0.6);
scatter!(sol[end][1,:], sol[end][2,:], label="Final distribution",alpha=0.3)
plt = plot!(title="SBTM sampling from circle", size=(800,800))
# savefig(plt, "data/sampling/circle.png")