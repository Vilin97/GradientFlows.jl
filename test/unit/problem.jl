using GradientFlows
problem = diffusion(2, 10, Exact())

@test "$problem" isa String
@test true_dist(problem, problem.tspan[1]) isa MvNormal
@test set_solver(problem, Blob()) == diffusion_problem(2, 10, Blob())