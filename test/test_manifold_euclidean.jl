m = 10
n = 5
s = GeoMax.Euclidean([m, n])
sm = s.shape[1]
sn = s.shape[2]

@test GeoMax.dim(s) == sm * sn

@test isapprox(GeoMax.typicaldist(s), sqrt(sm * sn))

x, y = randn(2, sm, sn)
@test isapprox(GeoMax.dist(s, x, y), vecnorm(x .- y))

x = rand(s)
y = GeoMax.randvec(s, x)
z = GeoMax.randvec(s, x)
@test isapprox(sum(y .* z), GeoMax.inner(s, x, y, z))

x = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.proj(s, x, u), u)

x = rand(s)
u = GeoMax.randvec(s, x)
egrad, ehess = randn(2, sm, sn)
@test isapprox(GeoMax.ehess2rhess(s, x, egrad, ehess, u), ehess)

x = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.retr(s, x, u), x + u)

x = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.egrad2rgrad(s, x, u), u)

x = rand(s)
u = randn(sm, sn)
@test isapprox(sqrt(sum(u.^2)), GeoMax.norm(s, x, u))

x = rand(s)
y = rand(s)
@test size(x) == (sm, sn)
@test vecnorm(x - y) > 1e-6

x = rand(s)
u = GeoMax.randvec(s, x)
v = GeoMax.randvec(s, x)
@test size(u) == (sm, sn)
@test isapprox(vecnorm(u), 1)
@test vecnorm(u - v) > 1e-6

x = rand(s)
y = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.transp(s, x, y, u), u)

X = rand(s)
Y = rand(s)
Yexplog = exp(s, X, log(s, X, Y))
@test isapprox(Y, Yexplog)

X = rand(s)
U = GeoMax.randvec(s, X)
Ulogexp = GeoMax.log(s, X, exp(s, X, U))
@test isapprox(U, Ulogexp)

X = rand(s)
Y = rand(s)
Z = GeoMax.pairmean(s, X, Y)
@test isapprox(GeoMax.dist(s, X, Z), GeoMax.dist(s, Y, Z))

##################
# Test Symmetric #
##################

k = 5
n = 10
s = GeoMax.Symmetric(n, k)

@test GeoMax.dim(s) == 0.5 * k * n * (n + 1)

@test isapprox(GeoMax.typicaldist(s), sqrt(GeoMax.dim(s)))

x, y = randn(2, k, n, n)
@test isapprox(GeoMax.dist(s, x, y), vecnorm(x - y))

x = rand(s)
y = GeoMax.randvec(s, x)
z = GeoMax.randvec(s, x)
@test isapprox(sum(y .* z), GeoMax.inner(s, x, y, z))

x = rand(s)
u = randn(n, n, k)
@test isapprox(GeoMax.proj(s, x, u), GeoMax.multisym(u))

x = rand(s)
u = GeoMax.randvec(s, x)
a = randn(n, n, k, 2)
egrad, ehess = a[:,:,:,1], a[:,:,:,2]
@test isapprox(GeoMax.ehess2rhess(s,x, egrad, ehess, u), GeoMax.multisym(ehess))

x = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.retr(s, x, u), x + u)

x = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.egrad2rgrad(s, x, u), u)

x = rand(s)
u = randn(n, n, k)
@test isapprox(sqrt(sum(u.^2)), norm(s, x, u))

x = rand(s)
y = rand(s)
@test size(x) == (n, n, k)
@test isapprox(x, GeoMax.multisym(x))
@test vecnorm(x - y) > 1e-6

x = rand(s)
u = GeoMax.randvec(s, x)
v = GeoMax.randvec(s, x)
@test size(u) == (n, n, k)
@test isapprox(u, GeoMax.multisym(u))
@test isapprox(vecnorm(u), 1)
@test vecnorm(u - v) > 1e-6

x = rand(s)
y = rand(s)
u = GeoMax.randvec(s, x)
@test isapprox(GeoMax.transp(s, x, y, u), u)

X = rand(s)
Y = rand(s)
Yexplog = exp(s, X, log(s, X, Y))
@test isapprox(Y, Yexplog)

X = rand(s)
U = GeoMax.randvec(s, X)
Ulogexp = log(s, X, exp(s, X, U))
@test isapprox(U, Ulogexp)

X = rand(s)
Y = rand(s)
Z = GeoMax.pairmean(s, X, Y)
@test isapprox(GeoMax.dist(s, X, Z), GeoMax.dist(s, Y, Z))
