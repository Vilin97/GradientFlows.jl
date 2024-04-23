function rejection_sample(target_dist::Distribution, proposal_dist, M, rng)
    f(x) = pdf(target_dist, x)
    rejection_sample(f, proposal_dist, M, rng)
end
function rejection_sample(target_pdf::Function, proposal_dist, M, rng)
    f(x) = target_pdf(x)
    g(x) = pdf(proposal_dist, x)
    while true
        x = rand(rng, proposal_dist)
        if M * g(x) < f(x)
            error("M = $M is too low: $(M*g(x)) = Mg(x) < f(x) = $(f(x)) for x = $x.")
        end
        if rand(rng) * M * g(x) < f(x) # accept with probability f(x)/Mg(x)
            return x
        end
    end
end