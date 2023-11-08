emp_mean(u::AbstractMatrix) = vec(mean(u, dims=2))
emp_cov(u::AbstractMatrix) = cov(u, dims=2)
emp_abs_moment(u::AbstractMatrix, k::Int) = sum(sum(abs2, u, dims=1) .^ (k/2)) / size(u, 2)