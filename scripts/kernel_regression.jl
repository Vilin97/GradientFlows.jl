# Kernel regression with a 2d checkerboard pattern
using KernelFunctions
using LinearAlgebra
using Distributions

## Plotting
using Plots;

function kernel_ridge_regression(k, X, y, Xstar, lambda)
    K = kernelmatrix(k, X)
    kstar = kernelmatrix(k, Xstar, X)
    return kstar * ((K + lambda * I) \ y)
end;



# Now adapt to 2d checkerboard pattern
default()
num_rows = 4; num_columns = 4
train_mesh = 0:0.02:0.999
x_train = hcat(reshape([[a,b] for a in train_mesh, b in train_mesh], :)...)
scatter(x_train[1,:], x_train[2,:]; label="grid")
is_white(x, num_rows, num_columns) = (floor(Int, num_rows * x[1]) + floor(Int, num_columns * x[2])) % 2 == 0
whites=hcat([x for x in eachcol(x_train) if is_white(x,num_rows, num_columns)]...)
blacks = hcat([x for x in eachcol(x_train) if !is_white(x,num_rows, num_columns)]...)
scatter(whites[1,:],whites[2,:]; label="white", c="white", markersize=2)
scatter!(blacks[1,:],blacks[2,:]; label="black", c="black", markersize=2)

y_train = [is_white(x,num_rows, num_columns) for x in eachcol(x_train)]
test_mesh = train_mesh
x_test = hcat(reshape([[a,b] for a in test_mesh, b in test_mesh], :)...)
kernel = SqExponentialKernel(metric=WeightedEuclidean([200,200]))
y_pred = kernel_ridge_regression(kernel, x_train, y_train, x_test, 0.00001)
y_pred_mat = reshape(y_pred, length(test_mesh), length(test_mesh))
heatmap(y_pred_mat)

# now need to take ∇log y_pred somehow
# can do a grid-based approach

# Another way: train an NN to approximate the target distribution and autodiff to get the score