struct ASBTM{S,NN,OPT,T,A,L,OS,SS} <: Solver
    score_values::S
    s::NN
    optimiser::OPT
    denoising_alpha::T
    init_batch_size::Int
    init_loss_tolerance::T
    init_max_iterations::Int
    allocated_memory::A
    verbose::Int
    logger::L
    optimiser_state::OS
    stopping_strategy::SS
end

struct ASBTMAllocMem{M}
    ζ::M
end

struct StoppingStrategy{F}
    # train at most max_epochs epochs
    max_epochs::Int
    # stop if the relative improvement in the loss is less than loss_tolerance
    loss_tolerance::F
    # evaluate the true loss every loss_eval_frequency epochs
    loss_eval_frequency::Int
end
StoppingStrategy(loss_eval_frequency) = StoppingStrategy(1000, 0.005/loss_eval_frequency, loss_eval_frequency)
StoppingStrategy() = StoppingStrategy(5)

function ASBTM(s::Union{Chain,Nothing}; learning_rate=1e-4, denoising_alpha=0.4, init_batch_size=2^8, init_loss_tolerance=1e-4, init_max_iterations=10^4, allocated_memory=nothing, verbose=0, logger=Logger(1), optimiser_state=nothing, stopping_strategy=StoppingStrategy())
    return ASBTM(nothing, s, Adam(learning_rate), denoising_alpha, init_batch_size, init_loss_tolerance, init_max_iterations, allocated_memory, verbose, logger, optimiser_state, stopping_strategy)
end
ASBTM(; kwargs...) = ASBTM(nothing; kwargs...)

function initialize(solver::ASBTM, u0::AbstractMatrix{F}, score_values::AbstractMatrix{F}, problem_name; kwargs...) where {F}
    ζ = similar(u0)
    allocated_memory = ASBTMAllocMem(ζ)
    if isnothing(solver.s)
        s = best_model(problem_name, size(u0, 1); kwargs...)
    else
        s = solver.s
    end
    logger = Logger(solver.logger.log_level, score_values)
    optimiser_state = Flux.setup(solver.optimiser, s)
    ASBTM(copy(score_values), s, solver.optimiser, solver.denoising_alpha, solver.init_batch_size, solver.init_loss_tolerance, solver.init_max_iterations, allocated_memory, solver.verbose, logger, optimiser_state, solver.stopping_strategy)
end

function train_s!(solver::ASBTM, u, score_values)
    @unpack s, init_batch_size, init_loss_tolerance, init_max_iterations, verbose, optimiser_state = solver

    verbose > 1 && @info "Training NN for $(size(u, 2)) particles."
    verbose > 1 && @info "Batch size = $init_batch_size, loss tolerance = $init_loss_tolerance, max iterations = $init_max_iterations. \n$s"
    data_loader = Flux.DataLoader((data=u, label=score_values), batchsize=min(size(u, 2), init_batch_size))

    iteration = 0
    epoch = 0
    while iteration < init_max_iterations
        loss = l2_error_normalized(s(u), score_values)
        (loss < init_loss_tolerance) && break
        verbose > 1 && epoch % 100 == 0 && @info "Epoch $(lpad(epoch, 5)) iteration $(lpad(iteration, 6)) loss $loss"
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
    verbose > 0 && @info "Trained NN in $iteration iterations. Loss = $final_loss."
    nothing
end

function reset!(solver::ASBTM, u0, score_values)
    train_s!(solver, u0, score_values)
    solver.score_values .= score_values
    nothing
end

function update!(solver::ASBTM, integrator)
    @unpack score_values, s, denoising_alpha, allocated_memory, verbose, logger, optimiser_state, stopping_strategy = solver
    @unpack max_epochs, loss_tolerance, loss_eval_frequency = stopping_strategy
    @unpack ζ = allocated_memory
    prob = integrator.p
    u = integrator.u
    log!(logger, integrator)

    D = prob.diffusion_coefficient(u, prob.params)
    true_loss_value = true_score_matching_loss(s, u)
    for epoch in 1:max_epochs
        randn!(ζ)
        loss_value, grads = withgradient(s -> score_matching_loss(s, u, ζ, denoising_alpha, D), s)
        Flux.update!(optimiser_state, s, grads[1])
        if epoch % loss_eval_frequency == 0
            new_true_loss_value = true_score_matching_loss(s, u)
            ΔL = true_loss_value - new_true_loss_value
            if ΔL < loss_tolerance
                verbose > 1 && @info "ΔL=$(rpad(round(ΔL, digits=5),9)) < $loss_tolerance. Converged in $epoch epochs."
                break
            end
            true_loss_value = new_true_loss_value
            epoch==max_epochs && verbose > 1 && @info "ΔL=$(pretty(ΔL, 6)) > $loss_tolerance. Did not converge in $max_epochs epochs."
        end
    end
    score_values .= s(u)
    if verbose > 0 && integrator.iter % 10 == 0
        train_loss = pretty(score_matching_loss(s, u, ζ, denoising_alpha), 7)
        if !isnothing(true_dist(integrator.p, integrator.t))
            test_loss = pretty(l2_error_normalized(score_values, true_score(prob, integrator.t, integrator.u)), 7)
            @info "Time $(integrator.t) test loss = $test_loss train loss = $train_loss"
        else
            @info "Time $(integrator.t) train loss = $train_loss"
        end
    end
    nothing
end

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
# TODO: this should probably depend on D
"= ∑ᵢ s(xᵢ)² + 2∇⋅s(xᵢ)"
true_score_matching_loss(s, u) = (sum(abs2, s(u)) + 2 * divergence(s, u)) / size(u, 2)


### Display ###
function Base.show(io::IO, solver::ASBTM)
    Base.print(io, "ASBTM")
end

hidden_layer_dimensions(solver::ASBTM) = [length(layer.bias) for layer in solver.s.layers[1:end-1]]

name(solver::ASBTM) = "asbtm"

long_name(solver::ASBTM) = "asbtm η=$(solver.optimiser.eta)"
