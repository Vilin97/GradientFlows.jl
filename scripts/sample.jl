# use SBTM for sampling
# Plan:
# 1. sample from https://github.com/ATISLabs/SyntheticDatasets.jl?tab=readme-ov-file
# 2. train an NN to learn the score of the target. Plot it in 2d with arrows.
# 3. save the NN in a file.
# 4. use the NN as the score of the target in SBTM, i.e. as the drift b.

using Random
using Distributions

function make_checkerboard(n::Int, noise::Float64, rows::Int, columns::Int)
    samples = zeros(2, n)
    cell_width = 1.0 / columns
    cell_height = 1.0 / rows

    count = 0
    while count < n
        # Choose a random cell
        cell_x = rand(1:columns)
        cell_y = rand(1:rows)

        # Check if the cell is white in the checkerboard pattern
        if (cell_x + cell_y) % 2 == 0
            # Choose a random point within the cell
            point_x = (cell_x - 1) * cell_width + rand() * cell_width
            point_y = (cell_y - 1) * cell_height + rand() * cell_height

            # Add Gaussian noise
            noise_x = rand(Normal(0, noise))
            noise_y = rand(Normal(0, noise))

            samples[1, count + 1] = point_x + noise_x
            samples[2, count + 1] = point_y + noise_y
            count += 1
        end
    end

    return samples
end

# Example usage:
n = 2000
noise = 0.0
rows = 3
columns = 3

samples = make_checkerboard(n, noise, rows, columns)

# Display the generated samples
using Plots
scatter(samples[1, :], samples[2, :], alpha=0.6, markersize=2, label="Samples", xlabel="x", ylabel="y", title="Checkerboard Samples")

### train NN
using LinearAlgebra
function plot_vector_field(s, samples; mesh_size=21)
    x = range(0, 1, length=mesh_size)
    y = range(0, 1, length=mesh_size)
    sxy = [s([x_, y_]) for x_ in x, y_ in y]
    sxy = reshape(sxy, :)
    # normalize sxy so that the longest arrow is 1/mesh_size
    sxy = sxy ./ maximum(norm.(sxy)) / mesh_size
    sx = [v[1] for v in sxy]
    sy = [v[2] for v in sxy]
    xx = reshape(repeat(x, 1, mesh_size), :)
    yy = reshape(repeat(y', mesh_size, 1), :)

    quiver(xx, yy, quiver=(sx, sy), label="score approximation")
    scatter!(samples[1, :], samples[2, :], alpha=0.6, markersize=2, label="Samples")
end

using Flux, JLD2
using GradientFlows
u = samples
s = mlp(2;depth=4)
ζ = similar(u)
optimiser = ADAM(1e-3)
optimiser_state = Flux.setup(optimiser, s)
epochs = 1000
denoising_alpha = 0.2
verbose = 2
plots = []
plt = plot_vector_field(s, u);
plot!(plt, title="Score approximation at epoch 0")
push!(plots, plt)
for epoch in 1:epochs
    randn!(ζ)
    loss_value, grads = Flux.withgradient(s -> GradientFlows.score_matching_loss(s, u, ζ, denoising_alpha), s)
    Flux.update!(optimiser_state, s, grads[1])
    true_loss_value = GradientFlows.true_score_matching_loss(s, u)
    verbose > 1 && epoch%100==0 && @info "Epoch $(lpad(epoch, 2)), approximate loss = $(GradientFlows.pretty(loss_value,7)), true loss = $(GradientFlows.pretty(true_loss_value,7))."
    if epoch % 100 == 0
        plt = plot_vector_field(s, u)
        plot!(plt, title="Score approximation at epoch $epoch")
        push!(plots, plt)
    end
end
layer_dimensions(s) = [2,[length(layer.bias) for layer in s.layers[1:end-1]]...,2]
plt1=plot(plots[[1,2,4,8]]..., size = (2000, 2000), plot_title="Score approximation with NN $(layer_dimensions(s))")
plots[end]
GradientFlows.save("data/models/checkerboard_3/d_2/n_2000.jld2", s)


"s ≈ ∇log p, where p is the target density"
function checkerboard_problem(n, solver_; t0::F=0.0, t_end::F=1., dt::F=0.1, rng=GradientFlows.DEFAULT_RNG, kwargs...) where {F}
    s = GradientFlows.load("data/models/checkerboard_3.jld2")
    f!(du, u, prob, t) = (du .= s(u) .- prob.solver.score_values)
    tspan = (t0, t_end)
    ρ(t, params) = nothing
    params = nothing
    ρ0 = Product(fill(Uniform(), 2))
    u0 = rand(rng, ρ0, n)
    name = "checkerboard_3"
    solver = GradientFlows.initialize(solver_, u0, s(u0), name; kwargs...)
    diffusion_coefficient(u, params) = 1
    covariance(t, params) = nothing
    return GradFlowProblem(f!, ρ0, u0, ρ, tspan, dt, params, solver, name, diffusion_coefficient, covariance)
end

prob = checkerboard_problem(200, SBTM(verbose=2,epochs=0,init_max_iterations=10^2))
solution = solve(prob)
