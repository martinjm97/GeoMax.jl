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
X ./= vecnorm(X)

#  Construct a matrix H in the ambient space.
H = randn(m, n)

#  Compare the projections.
@test isapprox(H - X * trace(X' * H), JManOpt.egrad2rgrad(s, X, H))

# TODO Test ehess
x = rand(s)
u = JManOpt.randvec(s, x)
egrad = randn(m, n)
ehess = randn(m, n)
#testing.ehess2rhess(proj)(x, egrad, ehess, u), ehess2rhess(s, x, egrad, ehess, u)

x = rand(s)
u = JManOpt.randvec(s, x)

@test isapprox(vecnorm(JManOpt.retr(s, x, u)), 1)

u = u * 1e-6
@test isapprox(JManOpt.retr(s, x, u), x + u, atol = 1e-5)

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.norm(s, x, u), vecnorm(u))

# test rand
x = rand(s)
@test isapprox(vecnorm(x), 1)
y = rand(s)
@test norm(x - y) > 1e-3

# test randvec
x = rand(s)
u = JManOpt.randvec(s, x)
v = JManOpt.randvec(s, x)
# TODO need tensordot for 2d
#np_testing.assert_almost_equal(np.tensordot(x, u), 0)
#@test isapprox(JManOpt.tensor_single_dot(u,v))
@test norm(u - v) > 1e-3

x = rand(s)
y = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.transp(s, x, y, u), JManOpt.proj(s, y, u))

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(u, exp(s, x, log(s, x, u)))

x = rand(s)
u = JManOpt.randvec(s, x)
ulogexp = log(s, x, exp(s, x, u))
@test isapprox(u, ulogexp)

x = rand(s)
y = rand(s)
z = JManOpt.pairmean(s, x, y)
@test isapprox(JManOpt.dist(s, x, z), JManOpt.dist(s, y, z))
