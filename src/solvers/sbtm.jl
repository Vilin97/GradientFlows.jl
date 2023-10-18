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
    logger::L
    optimiser_state::OS
end

struct SBTMAllocMem{M}
    ζ::M
end

# TODO: add other values to log
struct Logger
    verbose::Int
end
Logger() = Logger(0)

function SBTM(s::Chain; optimiser=Adam(1.0f-3), epochs=25, denoising_alpha=0.4f0, init_batch_size=2^8, init_loss_tolerance=1.0f-4, init_max_iterations=10^5, allocated_memory=nothing, logger=Logger(), optimiser_state=Flux.setup(optimiser, s))
    return SBTM(nothing, s, optimiser, epochs, denoising_alpha, init_batch_size, init_loss_tolerance, init_max_iterations, allocated_memory, logger, optimiser_state)
end

function initialize(solver::SBTM, u0::AbstractMatrix{Float32}, score_values::AbstractMatrix{Float32})
    train_s!(solver, u0, score_values)
    ζ = similar(u0)
    allocated_memory = SBTMAllocMem(ζ)
    SBTM(score_values, solver.s, solver.optimiser, solver.epochs, solver.denoising_alpha, solver.init_batch_size, solver.init_loss_tolerance, solver.init_max_iterations, allocated_memory, solver.logger, solver.optimiser_state)
end

function train_s!(solver::SBTM, u, score_values)
    @unpack s, init_batch_size, init_loss_tolerance, init_max_iterations, logger, optimiser_state = solver
    @unpack verbose = logger

    verbose > 1 && println("Initializing NN for $(size(u, 2)) particles.")
    verbose > 2 && println("Batch size = $init_batch_size, loss tolerance = $init_loss_tolerance, max iterations = $init_max_iterations. \n$s")
    data_loader = Flux.DataLoader((data=u, label=score_values), batchsize=min(size(u, 2), init_batch_size))

    iteration = 1
    epoch = 1
    while iteration < init_max_iterations
        for (x, y) in data_loader
            # TODO: this is type-unstable for some reason. Investigate.
            batch_loss, grads = withgradient(s -> l2_error_normalized(s, x, y), s)
            if iteration >= init_max_iterations
                break
            end
            iteration += 1
            Flux.update!(optimiser_state, s, grads[1])
        end
        loss = l2_error_normalized(s, u, score_values)
        verbose > 1 && epoch % 100 == 0 && println("Epoch $(lpad(epoch, 6)), loss $loss")
        if loss < init_loss_tolerance
            break
        end
        epoch += 1
    end
    final_loss = l2_error_normalized(s, u, score_values)
    verbose > 0 && println("Initialized NN in $iteration iterations. Loss = $final_loss.")
    nothing
end

function reset!(solver::SBTM, u0, score_values)
    train_s!(solver, u0, score_values)
    solver.score_values .= score_values
    nothing
end

function update!(solver::SBTM, integrator)
    @unpack score_values, s, optimiser, epochs, denoising_alpha, allocated_memory, logger, optimiser_state = solver
    @unpack ζ = allocated_memory
    @unpack verbose = logger

    u = integrator.u
    for epoch in 1:epochs
        randn!(ζ)
        loss_value, grads = withgradient(s -> score_matching_loss(s, u, ζ, denoising_alpha), s)
        Flux.update!(optimiser_state, s, grads[1])
        verbose > 1 && println("Epoch $(lpad(epoch, 2)), loss = $loss_value.")
    end
    verbose > 0 && integrator.iter % 100 == 0 && println("Time $(integrator.t), loss = $(score_matching_loss(s, u, ζ, denoising_alpha)).")
    score_values .= s(u)
    nothing
end

"Fisher divergence on CPU: ∑ᵢ (s(xᵢ) - yᵢ)² / |y|²"
l2_error_normalized(s, x, y) = normsq(s(x), y) / sum(abs2, y)

"≈ ( |s(u)|² + 2∇⋅s(u) ) / |u|²"
function score_matching_loss(s, u, ζ, α)
    denoise_val = (s(u .+ α .* ζ) ⋅ ζ - s(u .- α .* ζ) ⋅ ζ) / α
    return (sum(abs2, s(u)) + denoise_val) / size(u, 2)
end

function mlp(d::Int; depth=2, width=100, activation=softsign, rng=DEFAULT_RNG)
    return Chain(
        Dense(d => width, activation, init=Flux.glorot_normal(rng)),
        repeat([Dense(width => width, activation, init=Flux.glorot_normal(rng))], depth - 1)...,
        Dense(width => d, init=Flux.glorot_normal(rng))
    )
end
