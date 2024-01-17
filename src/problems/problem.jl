struct GradFlowProblem{FU,D,M,DF,F,P,S,DC,C}
    f!::FU # f!(du, u, prob :: GradFlowProblem, t) 
    ρ0::D # initial distribution
    u0::M # sample of initial distribution
    ρ::DF  # ρ(t, params) target distribution, if known
    tspan::Tuple{F,F} # (t0, t_end)
    dt::F # time step
    params::P # physical parameters of the problem
    solver::S # the solver to use
    name::String # short name, e.g. "landau" or "diffusion"
    diffusion_coefficient::DC # diffusion_coefficient(u, params) = A∗u in Landau equation, = D in diffusion and FPE
    covariance::C # covariance(t, params) is the covariance matrix of the target distribution at time t
end

function Base.show(io::IO, prob::GradFlowProblem)
    print(io, "$(prob.name) with \n tspan = $(prob.tspan) \n  dt = $(prob.dt) \n  params = $(prob.params) \n  solver = $(prob.solver)")
end

true_dist(prob::GradFlowProblem, t) = prob.ρ(t, prob.params)
true_score(prob::GradFlowProblem, t, u) = score(true_dist(prob, t), u)

"Reset the problem to its initial state so it can be solved again."
function reset!(problem::GradFlowProblem)
    reset!(problem.solver, problem.u0, score(problem.ρ0, problem.u0))
    nothing
end

"Set u0."
function set_u0!(problem::GradFlowProblem, u0)
    problem.u0 .= u0
    reset!(problem)
    nothing
end