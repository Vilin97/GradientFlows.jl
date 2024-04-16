struct SBTM{S,NN,OPT,T,A,L,OS} <: Solver
    score_values::S
    s::NN
    optimiser::OPT
    epochs::Int
    denoising_alpha::T
    init_batch_size::Int
    init_loss_tolerance::T
    init_max_iterations::Int
    allocated_memory::A
    verbose::Int
    logger::L
    optimiser_state::OS
end

struct SBTMAllocMem{M}
    ζ::M
end

function SBTM(s::Union{Chain,Nothing}; learning_rate=4e-4, epochs=25, denoising_alpha=0.4, init_batch_size=2^8, init_loss_tolerance=1e-4, init_max_iterations=10^5, allocated_memory=nothing, verbose=0, logger=Logger(1), optimiser_state=nothing)
    return SBTM(nothing, s, Adam(learning_rate), epochs, denoising_alpha, init_batch_size, init_loss_tolerance, init_max_iterations, allocated_memory, verbose, logger, optimiser_state)
end
SBTM(; kwargs...) = SBTM(nothing; kwargs...)

function initialize(solver::SBTM, u0::AbstractMatrix{F}, score_values::AbstractMatrix{F}, problem_name; kwargs...) where {F}
    ζ = similar(u0)
    allocated_memory = SBTMAllocMem(ζ)
    if isnothing(solver.s)
        s = best_model(problem_name, size(u0, 1); kwargs...)
    else
        s = solver.s
    end
    logger = Logger(solver.logger.log_level, score_values)
    optimiser_state = Flux.setup(solver.optimiser, s)
    SBTM(copy(score_values), s, solver.optimiser, solver.epochs, solver.denoising_alpha, solver.init_batch_size, solver.init_loss_tolerance, solver.init_max_iterations, allocated_memory, solver.verbose, logger, optimiser_state)
end

function train_s!(solver::SBTM, u, score_values)
    @unpack s, init_batch_size, init_loss_tolerance, init_max_iterations, verbose, optimiser_state = solver

    verbose > 1 && println("Training NN for $(size(u, 2)) particles.")
    verbose > 1 && println("Batch size = $init_batch_size, loss tolerance = $init_loss_tolerance, max iterations = $init_max_iterations. \n$s")
    data_loader = Flux.DataLoader((data=u, label=score_values), batchsize=min(size(u, 2), init_batch_size))

    iteration = 0
    epoch = 0
    while iteration < init_max_iterations
        loss = l2_error_normalized(s(u), score_values)
        (loss < init_loss_tolerance) && break
        verbose > 1 && epoch % 100 == 0 && println("Epoch $(lpad(epoch, 5)) iteration $(lpad(iteration, 6)) loss $loss")
        epoch += 1
        for (x, y) in data_loader
            batch_loss, grads = withgradient(s -> l2_error_normalized(s(x), y), s)
            if iteration >= init_max_iterations
                break
            end
            iteration += 1
            Flux.update!(optimiser_state, s, grads[1])
        end
    end
    final_loss = l2_error_normalized(s(u), score_values)
    verbose > 0 && println("Trained NN in $iteration iterations. Loss = $final_loss.")
    nothing
end

function reset!(solver::SBTM, u0, score_values)
    train_s!(solver, u0, score_values)
    solver.score_values .= score_values
    nothing
end

function update!(solver::SBTM, integrator)
    @unpack score_values, s, optimiser, epochs, denoising_alpha, allocated_memory, verbose, logger, optimiser_state = solver
    @unpack ζ = allocated_memory
    prob = integrator.p
    u = integrator.u
    log!(logger, integrator)

    D = prob.diffusion_coefficient(u, prob.params)
    for epoch in 1:epochs
        randn!(ζ)
        loss_value, grads = withgradient(s -> score_matching_loss(s, u, ζ, denoising_alpha, D), s)
        Flux.update!(optimiser_state, s, grads[1])
        verbose > 1 && println("Epoch $(lpad(epoch, 2)), loss = $loss_value.")
    end
    score_values .= s(u)
    if verbose > 0 && integrator.iter % 10 == 0
        train_loss = pretty(score_matching_loss(s, u, ζ, denoising_alpha), 7)
        if !isnothing(true_dist(integrator.p, integrator.t))
            test_loss = pretty(l2_error_normalized(score_values, true_score(prob, integrator.t, integrator.u)), 7)
            println("Time $(integrator.t) test loss = $test_loss train loss = $train_loss")
        else
            println("Time $(integrator.t) train loss = $train_loss")
        end
    end
    nothing
end

"Fisher divergence on CPU: ∑ᵢ (s(xᵢ) - yᵢ)² / ∑ᵢ |y|²"
l2_error_normalized(y_hat, y) = sum(abs2, y_hat .- y) / sum(abs2, y)

"≈ ( |√D s(u)|² + 2∇⋅Ds(u) ) / n"
function score_matching_loss(s, u, ζ, α, D=1)
    denoise_val = dot(s(u .+ α .* ζ) .- s(u .- α .* ζ), D, ζ) / α
    su = s(u)
    return (dot(su, D, su) + denoise_val) / size(u, 2)
end

function mlp(d::Int; depth, width=100, activation=softsign, rng=DEFAULT_RNG)
    if depth == 0
        return Chain(Dense(d => d, init=Flux.glorot_normal(rng))) |> f64
    end
    return Chain(
        Dense(d => width, activation, init=Flux.glorot_normal(rng)),
        repeat([Dense(width => width, activation, init=Flux.glorot_normal(rng))], depth - 1)...,
        Dense(width => d, init=Flux.glorot_normal(rng))
    ) |> f64
end

### Display ###
function Base.show(io::IO, solver::SBTM)
    Base.print(io, "SBTM")
end

hidden_layer_dimensions(solver::SBTM) = [length(layer.bias) for layer in solver.s.layers[1:end-1]]

name(solver::SBTM) = "sbtm"

long_name(solver::SBTM) = "sbtm η=$(solver.optimiser.eta)"
