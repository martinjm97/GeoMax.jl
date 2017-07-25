m = 20
n = 2
k = 1
s = GeoMax.Stiefel(m, n, k)
#self.proj = lambda x, u: u - npa.dot(x, npa.dot(x.T, u) +
#                                     npa.dot(u.T, x)) / 2

@test GeoMax.dim(s) == 0.5 * n* (2 * m- n- 1)

#  test_typicaldist():

#  test_dist(self):

X = qr(randn(m, n))[1]
A, B = randn(2, m, n)
@test isapprox(sum(A * B), GeoMax.inner(s, X, A, B))


# Construct a random point X on the manifold.
X = randn(m, n)
X = qr(X)[1]

# Construct a vector H in the ambient space.
H = randn(m, n)

# Compare the projections.
Hproj = H - dot(X,dot(H, X') + dot(X, H')) / 2
@test isapprox(Hproj, GeoMax.proj(s,X, H))


# Just make sure that things generated are on the manifold and that
# if you generate two they are not equal.
X = rand(s)
@test isapprox(dot(X, X'), eye(n), atol=1e-10)
Y = rand(s)
@test vecnorm(X - Y) > 1e-6

# Make sure things generated are in tangent space and if you generate
# two then they are not equal.
X = rand(s)
U = randvec(s,X)
@test isapprox(GeoMax.multisym(dot(U, X')), zeros(n, n), atol=1e-10)
V = randvec(s,X)
assert vecnorm(U - V) > 1e-6

# Test that the result is on the manifold and that for small
# tangent vectors it has little effect.
x = rand(s)
u = randvec(s, x)

xretru = GeoMax.retr(s,x, u)
@test isapprox(dot(xretru, xretru'), eye(n,n), atol=1e-10)

u = u * 1e-6
xretru = GeoMax.retr(s,x, u)
@test isapprox(xretru, x + u)

# Test this function at some randomly generated point.
x = rand(s)
u = randvec(s, x)
egrad = randn(m, n)
ehess = randn(m, n)

#TODO learn how to test EHESS
#@test isapprox(testing.ehess2rhess(self.proj)(x, egrad,
#                                                          ehess, u),
#                           man.ehess2rhess(x, egrad, ehess, u))


#  test_egrad2rgrad():
x = rand(s)
u = randvec(s, x)
@test isapprox(norm(s, x, u), vecnorm(u))

#  test_transp(self):

# Check that exp lies on the manifold and that exp of a small vector u
# is close to x + u.
s = man
x = rand(s)
u = randvec(s, x)

xexpu = exp(s, x, u)
@test isapprox(dot(xexpu, xexpu'), eye(n,n), atol=1e-10)

u = u * 1e-6
xexpu = exp(s, x, u)
@test isapprox(xexpu, x + u)

#  test_exp_log_inverse
#  test_log_exp_inverse
#  test_pairmean


####################
# Multidim Steifel #
####################
m = 10
n = 3
k = 3
s = Stiefel(m, n, k)

@test man.dim == 0.5 * k * n* (2 * m - n - 1)

@test isapprox(typicaldist(s), sqrt(n * k))

#  test_dist(self):

X = rand(s)
A = randvec(s,X)
B = randvec(s,X)
@test isapprox(sum(A .* B), GeoMax.inner(s, X, A, B))

# Construct a random point X on the manifold.
X = rand(s)

# Construct a vector H in the ambient space.
H = randn(k, m, n)

# Compare the projections.
Hproj = H - multiprod(X, multiprod(GeoMax.multitransp(X), H) +
                      multiprod(GeoMax.multitransp(H), X)) / 2
@test isapprox(Hproj, GeoMax.proj(s, X, H))

# Just make sure that things generated are on the manifold and that
# if you generate two they are not equal.
X = rand(s)
@test isapprox(multiprod(GeoMax.multitransp(X), X), multieye(k, n), atol=1e-10)
Y = rand(s)
@test vecnorm(X - Y) > 1e-6

# Make sure things generated are in tangent space and if you generate
# two then they are not equal.
X = rand(s)
U = randvec(s,X)
@test isapprox(GeoMax.multisym(multiprod(GeoMax.multitransp(X), U)), zeros(k, n, n),atol=1e-10)
V = randvec(s,X)
@test vecnorm(U - V) > 1e-6

# Test that the result is on the manifold and that for small
# tangent vectors it has little effect.
x = rand(s)
u = randvec(s, x)

xretru = retr(s, x, u)

@test isapprox(multiprod(GeoMax.multitransp(xretru), xretru), multieye(s.k, n), atol=1e-10)

u = u * 1e-6
xretru = GeoMax.retr(s,x, u)
@test isapprox(xretru, x + u)

#  test_egrad2rgrad(self):

x = rand(s)
u = randvec(s, x)
@test isapprox(norm(s, x, u), vecnorm(u))

#  test_transp(self):

# Check that exp lies on the manifold and that exp of a small vector u
# is close to x + u.
s = man
x = rand(s)
u = randvec(s, x)

xexpu = exp(s, x, u)
@test isapprox(multiprod(GeoMax.multitransp(xexpu), xexpu), multieye(k, n), atol=1e-10)

u = u * 1e-6
xexpu = exp(s, x, u)
@test isapprox(xexpu, x + u)

# test_exp_log_inverse

# test_log_exp_inverse

# test_pairmean
