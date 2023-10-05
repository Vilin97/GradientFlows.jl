abstract type Solver end

# initialize(solver, score_values) initialize the solver with the score values
# update!(solver, integrator) fill in solver.score_values

struct Exact <: Solver
    score_values

    Exact(score_values) = new(score_values)
    Exact() = new(nothing)
end

function Base.show(io::IO, solver::Exact)
    Base.print(io, "Exact")
end

function initialize(solver::Exact, score_values)
    Exact(score_values)
end

function update!(solver::Exact, integrator)
    prob = integrator.p
    t = integrator.t
    true_dist = prob.ρ(t, prob.params)
    solver.score_values .= score(true_dist, integrator.u)
    nothing
end
