"Exact solver uses the true solution to compute the score values. It is used as a baseline."
struct Exact{S} <: Solver
    score_values::S
end
Exact() = Exact(nothing)

"Initialize solver."
function initialize(::Exact, u0, score_values, problem_name; kwargs...)
    Exact(copy(score_values))
end

"Fill in solver.score_values."
function update!(solver::Exact, integrator)
    prob = integrator.p
    solver.score_values .= true_score(prob, integrator.t, integrator.u)
    nothing
end

function Base.show(io::IO, ::Exact)
    Base.print(io, "Exact")
end
name(solver::Exact) = "exact"