"Exact solver uses the true solution to compute the score values. It is used as a baseline."
struct Exact{S} <: Solver
    score_values::S
end
Exact() = Exact(nothing)

"Initialize solver."
function initialize(::Exact, u0, score_values, problem_name)
    Exact(copy(score_values))
end

"Fill in solver.score_values."
function update!(solver::Exact, integrator)
    prob = integrator.p
    t = integrator.t
    true_dist = prob.ρ(t, prob.params)
    solver.score_values .= score(true_dist, integrator.u)
    nothing
end

function Base.show(io::IO, ::Exact)
    Base.print(io, "Exact")
end
name(solver::Exact) = "exact"