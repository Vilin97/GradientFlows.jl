mutable struct GradFlowExperiment{P,V,S,F,T}
    problem::P
    saveat::V
    solution::Vector{S} # solution[time_index][d, n] isa Float
    L2_error::F
    mean_norm_error::F
    cov_norm_error::F
    cov_trace_error::F
    timer::T
end

function Base.show(io::IO, experiment::GradFlowExperiment)
    @unpack problem, saveat, solution, L2_error, mean_norm_error, cov_norm_error, cov_trace_error = experiment
    width = 6
    d, n = size(problem.u0)
    print(io, "$(problem.name) d=$d n=$(rpad(n,n_WIDTH)) $(lpad(problem.solver, SOLVER_NAME_WIDTH)): |ρ∗ϕ - ρ*|₂ = $(short_string(L2_error,width)) |E(ρ)-E(ρ*)|₂ = $(short_string(mean_norm_error,width)) |Σ-Σ'|₂ = $(short_string(cov_norm_error,width)) |tr(Σ)-tr(Σ')| = $(short_string(cov_trace_error,width))")
end

function GradFlowExperiment(problem::GradFlowProblem; saveat=problem.tspan)
    solution = Vector{typeof(problem.u0)}(undef, 0)
    F = eltype(problem.u0)
    return GradFlowExperiment(problem, saveat, solution, zero(F), zero(F), zero(F), zero(F), TimerOutput())
end

function solve!(experiment::GradFlowExperiment)
    @unpack problem, saveat, solution = experiment
    d, n = size(problem.u0)
    reset_timer!(DEFAULT_TIMER)
    @timeit DEFAULT_TIMER "$(problem.name) d=$d n=$(rpad(n,n_WIDTH)) $(problem.solver)" experiment.solution = solve(problem, saveat=saveat).u
    experiment.timer = deepcopy(DEFAULT_TIMER)
    nothing
end
