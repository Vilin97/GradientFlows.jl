abstract type Solver end

# initialize(solver, score_values) initialize the solver with the score values
# update!(solver, integrator) fill in solver.score_values

function Base.show(io::IO, solver::S) where {S<:Solver}
    Base.print(io, "$(typeof(solver))")
end