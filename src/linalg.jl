"|x-y|^2"
normsq(x, y) = sum(ab -> abs2(ab[1] - ab[2]), zip(x, y))