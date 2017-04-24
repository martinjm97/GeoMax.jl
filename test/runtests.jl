using JManOpt
using Base.Test

function elementwise_close(a, b, tol)
    d = a - b
    return length(d[abs.(d) .< tol]) == prod(size(d))
end
elementwise_close(a, b) = elementwise_close(a, b, 0.01)

# Test Sphere
m = 100
n = 50
s = JManOpt.Sphere([m, n])
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
@test elementwise_close(H - X * trace(dot(H, X')), JManOpt.egrad2rgrad(s, X, H))

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
@test elementwise_close(JManOpt.retr(s, x, u), x + u)

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
@test isapprox(norm(u - v) > 1e-3)

x = rand(s)
y = rand(s)
u = JManOpt.randvec(s, x)
@test elementwise_close(JManOpt.transp(s, x, y, u), JManOpt.proj(s, y, u))


X = rand(s)
U = JManOpt.randvec(s, X)
elementwise_close(U, exp(s, X, log(s, X, U)))

def test_log_exp_inverse(self):
    s = self.man
    X = s.rand()
    U = s.randvec(X)
    Ulogexp = s.log(X, s.exp(X, U))
    np_testing.assert_array_almost_equal(U, Ulogexp)

def test_pairmean(self):
    s = self.man
    X = s.rand()
    Y = s.rand()
    Z = s.pairmean(X, Y)
    np_testing.assert_array_almost_equal(dist(s, X, Z), dist(s, Y, Z))
