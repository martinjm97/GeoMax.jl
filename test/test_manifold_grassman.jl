n = 5
p = 2
k = 1
s = JManOpt.Grassmann(n, p, k)

#s.proj = lambda x, u: u - npa.dot(x, npa.dot(x.T, u))

#TODO log has dimension error maybe svd
x = rand(s)
y = rand(s)
@test isapprox(JManOpt.dist(s, x, y, JManOpt.norm(s, x, log(s, x, y))))

# Test this function at some randomly generated point.
x = rand(s)
u = JManOpt.randvec(s, x)
egrad = randn(s.n, s.p)
ehess = randn(s.n, s.p)
# TODO FIGURE OUT EHESS
#@test isapprox(JManOpt.ehess2rhess(s.proj)(x, egrad, ehess, u, ehess2rhess(s, x, egrad, ehess, u))

# Test that the result is on the manifold and that for small
# tangent vectors it has little effect.
x = rand(s)
u = JManOpt.randvec(s, x)

#TODO retr has dimension error maybe svd

#xretru = JManOpt.retr(s, x, u)

#@test isapprox(multiprod(multitransp(xretru), xretru, eye(s.p),atol=1e-10)

u = u * 1e-6
xretru = JManOpt.retr(s, x, u)
@test isapprox(xretru, x + u)

# def test_egrad2rgrad(s):

# def test_norm(s):


# Just make sure that things generated are on the manifold and that
# if you generate two they are not equal.
X = rand(s)
@test isapprox(multiprod(multitransp(X), X, eye(s.p), atol=1e-10)
Y = rand(s)
@test JManOpt.vecnorm(X .- Y) > 1e-6

# def test_randvec(s):

# def test_transp(s):

x = rand(s)
y = rand(s)
u = JManOpt.log(s, x, y)
z = JManOpt.exp(s, x, u)
@test isapprox(0, JManOpt.dist(s, y, z), atol = 1e-5)


x = rand(s)
u = JManOpt.randvec(s, x)
y = JManOpt.exp(s, x, u)
v = JManOpt.log(s, x, y)
# Check that the manifold difference between the tangent vectors u and
# v is 0
@test isapprox(0, JManOpt.norm(s, x, u - v))

# def test_pairmean(s):
    # s = s.man
    # X = s.rand()
    # Y = s.rand()
    # Z = s.pairmean(X, Y)
    # np_testing.assert_array_almost_equal(s.dist(X, Z), s.dist(Y, Z))


# Now test the Multigrassman
n = 5
p = 2
k = 3
s = JManOpt.Grassmann(m, n, k)

#s.proj = lambda x, u: u - npa.dot(x, npa.dot(x.T, u))

@test isapprox JManOpt.dim(s) == s.k * (s.n * s.p - s.p^2)

@test isapprox(JManOpt.typicaldist(s), sqrt(s.p * s.k))

x = rand(s)
y = rand(s)
@test isapprox(JManOpt.dist(s, x, y), JManOpt.norm(s, x, JManOpt.log(s, x, y)))

X = rand(s)
A = JManOpt.randvec(s, X)
B = JManOpt.randvec(s, X)
@test isapprox(sum(A * B), JManOpt.inner(s, X, A, B))


# Construct a random point X on the manifold.
X = rand(s)

# Construct a vector H in the ambient space.
H = randn(s.k, s.n, s.p)

# Compare the projections.
Hproj = H - multiprod(X, multiprod(multitransp(X), H))
@test isapprox(Hproj, JManOpt.proj(s, X, H))

# Test that the result is on the manifold and that for small
# tangent vectors it has little effect.
x = rand(s)
u = randvec(s, x)
xretru = retr(s, x, u)
@test isapprox(multiprod(multitransp(xretru), xretru),multieye(s.k, s.p), atol=1e-10)

u = u * 1e-6
xretru = JManOpt.retr(s, x, u)
@test isapprox(xretru, x + u)

# def test_egrad2rgrad(s):

x = rand(s)
u = randvec(s, x)
@test isapprox(norm(s, x, u), vecnorm(u))

# Just make sure that things generated are on the manifold and that
# if you generate two they are not equal.
X = rand(s)
@test isapprox(multiprod(multitransp(X), X), multieye(s.k, s.p), atol=1e-10)
Y = rand(s)
@test isapprox(vecnorm(X .- Y) > 1e-6)

# Make sure things generated are in tangent space and if you generate
# two then they are not equal.
X = rand(s)
U = JManOpt.randvec(s, X)
@test isapprox(multisym(multiprod(multitransp(X), U)), zeros(s.k, s.p, s.p), atol=1e-10)
V = JManOpt.randvec(s, X)
@test isapprox(vecnorm(U - V) > 1e-6)

# def test_transp(s):

x = rand(s)
y = rand(s)
u = log(s, x, y)
z = exp(s, x, u)
@test isapprox(0, JManOpt.dist(s, y, z))

x = rand(s)
u = JManOpt.randvec(s, x)
y = JManOpt.exp(s, x, u)
v = JManOpt.log(s, x, y)
# Check that the manifold difference between the tangent vectors u and
# v is 0
@test isapprox(0, JManOpt.norm(x, u - v))
