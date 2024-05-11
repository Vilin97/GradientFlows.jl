using Flux, Zygote, LinearAlgebra, Distributions, Random, Plots, JLD2
using Flux.OneHotArrays: onehot

score(ρ::MultivariateDistribution, u::AbstractArray{T,1}) where {T} = gradlogpdf(ρ, u)
score(ρ::MultivariateDistribution, u::AbstractArray{T,2}) where {T} = reshape(hcat([score(ρ, @view u[:, i]) for i in axes(u, 2)]...), size(u))
# divergence(s, u) = tr(jacobian(s, u)[1])
function divergence(f, v)
    res = zero(eltype(v))
    for vi in eachcol(v)
        _, ∂f = pullback(f, vi)
        res += sum(eachindex(vi)) do i
            ∂fᵢ = ∂f(onehot(i, eachindex(vi)))
            sum(x -> x[i], ∂fᵢ)
        end
    end
    return res
end
l2_error_normalized(y_hat, y) = sum(abs2, y_hat .- y) / sum(abs2, y)
true_score_matching_loss(s, u) = (sum(abs2, s(u)) + 2 * divergence(s, u)) / size(u, 2)
function score_matching_loss(s, u, ζ, α)
    denoise_val = dot(s(u .+ α .* ζ) .- s(u .- α .* ζ), ζ) / α
    su = s(u)
    return (dot(su, su) + denoise_val) / size(u, 2)
end
model_filename(problem_name, d, n; dir="data") = joinpath(dir, "models", lowercase(problem_name), "d_$(d)", "n_$(n).jld2")
function best_model(problem_name, d; kwargs...)
    dir = dirname(model_filename(problem_name, d, 1; kwargs...))
    filenames = readdir(dir)
    if isempty(filenames)
        return nothing
    end
    get_n(f) = parse(Int, f[3:findfirst('.', f)-1])
    highest_n_filename = maximum(f -> (get_n(f), f), filenames)[2]
    return load(joinpath(dir, highest_n_filename))
end


struct StoppingStrategy{F}
    # train at least min_epochs epochs
    min_epochs::Int
    # train at most max_epochs epochs
    max_epochs::Int
    # stop if the relative improvement in the loss is less than loss_tolerance
    loss_tolerance::F
    # evaluate the true loss every loss_eval_frequency epochs
    loss_eval_frequency::Int
end
StoppingStrategy(loss_eval_frequency) = StoppingStrategy(0, 400, 0.005/loss_eval_frequency, loss_eval_frequency)
StoppingStrategy() = StoppingStrategy(5)

function train_s!(s, u, dist; α=0.4, η=1e-4, verbose=1, stopping_strategy)
    optimiser_state = Flux.setup(Adam(η), s)
    ζ = similar(u)
    train_losses = Float64[]
    test_losses = Float64[]
    true_train_loss = Float64[]
    last_epoch = 0
    true_loss_value = true_score_matching_loss(s, u)
    for epoch in 1:stopping_strategy.max_epochs
        randn!(ζ)
        loss_value, grads = withgradient(s -> score_matching_loss(s, u, ζ, α), s)
        Flux.update!(optimiser_state, s, grads[1])
        test_loss = l2_error_normalized(s(u), score(dist, u))
        verbose > 1 && @info "Epoch $(lpad(epoch, 2)), test loss = $(round(test_loss,digits=6)) train loss = $(round(loss_value,digits=7))."
        push!(train_losses, loss_value)
        push!(test_losses, test_loss)
        push!(true_train_loss, true_loss_value)
        last_epoch = epoch
        if epoch % stopping_strategy.loss_eval_frequency == 0
            new_true_loss_value = true_score_matching_loss(s, u)
            if true_loss_value - new_true_loss_value < stopping_strategy.loss_tolerance
                verbose > 0 && @info "ΔL=$(round(true_loss_value - new_true_loss_value, digits=6)) < $(stopping_strategy.loss_tolerance) Converged in $last_epoch epochs."
                break
            end
            true_loss_value = new_true_loss_value
        end
    end
    verbose > 0 && last_epoch==stopping_strategy.max_epochs && @info "Converged in $last_epoch epochs."
    return train_losses, test_losses, true_train_loss
end
# fpe
function get_losses(d, n, Δt, t_end, stopping_strategy=StoppingStrategy())
    # Random.seed!(123)
    ρ(t) = MvNormal((1 - exp(-2t)) * I(d)) # true distribution
    t = 0.1
    u = rand(ρ(t), n)
    s = best_model("fpe", d)["single_stored_object"] # load the best model
    train_losses = Vector{Float64}[]
    test_losses = Vector{Float64}[]
    true_train_losses = Vector{Float64}[]

    # move particles
    while t < t_end
        # @show l2_error_normalized(s(u), score(ρ(t), u))
        train_loss, test_loss, true_train_loss = train_s!(s, u, ρ(t); stopping_strategy=stopping_strategy)
        # @show l2_error_normalized(s(u), score(ρ(t), u))
        push!(train_losses, train_loss)
        push!(test_losses, test_loss)
        push!(true_train_losses, true_train_loss)
        t += Δt
        u = u - Δt .* (u + s(u))
    end
    return train_losses, test_losses, true_train_losses
end

function get_plots(train_losses, test_losses, true_train_losses; d, n, Δt)
    # train_losses = exp_average.(train_losses, r)
    train_losses = vcat(train_losses...)
    test_losses = vcat(test_losses...)
    true_train_losses = vcat(true_train_losses...)
    train_loss_ups = [i for i in 1:length(true_train_losses)-1 if true_train_losses[i+1] > true_train_losses[i]]
    total_epochs = sum(x -> length(x), losses[1])

    plt = plot(train_losses, label="train loss", xlabel="epoch", ylabel="loss", title="Averaged loss, d=$d, n=$n, Δt=$Δt, $total_epochs total epochs", size=(800,400));
    scatter!(plt, train_loss_ups, [true_train_losses[i] for i in train_loss_ups], label="train loss up, #=$(length(train_loss_ups))", color=:red, markersize=1);
    plot!(plt, true_train_losses, label="true train loss")
    plot!(plt, test_losses ./ Δt, label="test loss / $Δt")
    return plt
end

d=2; n=2000; Δt=0.01; t_end=0.5
@time losses = get_losses(d, n, Δt, t_end)

plt=get_plots(losses...;d=d,n=n,Δt=Δt)
plot([t[end] for t in losses[2]])
true_losses = losses[3]
normalized = [l .- l[1] for l in true_losses]
plot(normalized)




function divergence1(f, v)
    res = zero(eltype(v))
    for vi in eachcol(v)
        _, ∂f = pullback(f, vi)
        res += sum(eachindex(vi)) do i
            ∂fᵢ = ∂f(onehot(i, eachindex(vi)))
            sum(x -> x[i], ∂fᵢ)
        end
    end
    return res
end
function divergence2(f, v)
    res = zero(eltype(v))
    for vi in eachcol(v)
        _, ∂f = pullback(f, vi)
        res += sum(eachindex(vi)) do i
            ∂f(onehot(i, eachindex(vi)))[1][i]
        end
    end
    return res
end
x = randn(2,5000)
s = Chain(Dense(2=>100, softsign), Dense(100=>2)) |> f64
divergence(s,x) # compile
@time divergence(s,x)
divergence2(s,x) # compile
@time divergence2(s,x)