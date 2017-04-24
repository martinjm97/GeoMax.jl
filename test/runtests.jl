using JManOpt
using StaticArrays
using Base.Test

# function elementwise_close(a, b, tol)
#     d = a - b
#     return length(d[abs.(d) .< tol]) == prod(size(d))
# end
# elementwise_close(a, b) = elementwise_close(a, b, 0.01)

# TODO replace this with `isapprox(a, b)` (or `all(isapprox.(a, b))`)
array_almost_equal(a, b, decimal) = all(abs.(a-b) .< 1.5 * exp10(-decimal))
array_almost_equal(a, b) = array_almost_equal(a, b, 6)

# Test Sphere
m = 100
n = 50
s = JManOpt.Sphere(SVector(m, n))
@test JManOpt.dim(s) == m * n - 1
@test JManOpt.typicaldist(s) == pi
x = rand(s)
y = rand(s)
# TODO figure out tensor_double_dot in 2x2 also how to do approx in code/atom
#@test JManOpt.dist(s, x, y) == acos(JManOpt.tensor_double_dot(x, y))

x = rand(s)
u = JManOpt.randvec(s, x)
v = JManOpt.randvec(s, x)
@test isapprox(sum(u .* v), JManOpt.inner(s, x, u, v))

# TODO test proj

#  Construct a random point X on the manifold.
X = randn(m, n)
X = X / vecnorm(X)

#  Construct a vector H in the ambient space.
H = randn(m, n)

#  Compare the projections.
@test array_almost_equal(H - X * trace(dot(H, X')), JManOpt.egrad2rgrad(s, X, H), 2)

# TODO Test ehess
# x = rand(s)
# u = randvec(s, x)
# egrad = randn(m, n)
# ehess = randn(m, n)
# testing.ehess2rhess(proj)(x, egrad, ehess, u), ehess2rhess(s, x, egrad, ehess, u)

x = rand(s)
u = JManOpt.randvec(s, x)

@test isapprox(norm(JManOpt.retr(s, x, u)), 1)

u = u * 1e-6
xretru = JManOpt.retr(s, x, u)
@test array_almost_equal(JManOpt.retr(s, x, u), x + u)

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.norm(s, x, u), norm(u))

# test rand
x = rand(s)
@test isapprox(norm(x), 1)
y = rand(s)
@test norm(x - y) > 1e-3

# test randvec
x = rand(s)
u = JManOpt.randvec(s, x)
v = JManOpt.randvec(s, x)
# TODO need tensordot
#np_testing.assert_almost_equal(np.tensordot(x, u), 0)
@test norm(u - v) > 1e-3

x = rand(s)
y = rand(s)
u = JManOpt.randvec(s, x)
@test array_almost_equal(JManOpt.transp(s, x, y, u), JManOpt.proj(s, y, u))


x = rand(s)
x = JManOpt.randvec(s, x)
@test array_almost_equal(u, exp(s, x, log(s, x, u)))


x = rand(s)
u = randvec(s, x)
ulogexp = log(s, x, exp(s, x, u))
@test array_almost_equal(u, ulogexp)

x = rand(s)
y = rand(s)
z = pairmean(s, x, y)
@test array_almost_equal(dist(s, x, z), dist(s, y, z))
