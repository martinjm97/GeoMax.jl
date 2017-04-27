m = 10
n = 5
s = JManOpt.Euclidean([m, n])
sm = s.shape[1]
sn = s.shape[2]

@test JManOpt.dim(s) == sm * sn

@test isapprox(JManOpt.typicaldist(s), sqrt(sm * sn))

x, y = randn(2, sm, sn)
@test isapprox(JManOpt.dist(s, x, y), vecnorm(x .- y))

x = rand(s)
y = JManOpt.randvec(s, x)
z = JManOpt.randvec(s, x)
@test isapprox(sum(y .* z), JManOpt.inner(s, x, y, z))

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.proj(s, x, u), u)

x = rand(s)
u = JManOpt.randvec(s, x)
egrad, ehess = randn(2, sm, sn)
@test isapprox(JManOpt.ehess2rhess(s, x, egrad, ehess, u), ehess)

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.retr(s, x, u), x + u)

x = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.egrad2rgrad(s, x, u), u)

x = rand(s)
u = randn(sm, sn)
@test isapprox(sqrt(sum(u.^2)), JManOpt.norm(s, x, u))

x = rand(s)
y = rand(s)
@test size(x) == (sm, sn)
@test vecnorm(x - y) > 1e-6

x = rand(s)
u = JManOpt.randvec(s, x)
v = JManOpt.randvec(s, x)
@test size(u) == (sm, sn)
@test isapprox(vecnorm(u), 1)
@test vecnorm(u - v) > 1e-6

x = rand(s)
y = rand(s)
u = JManOpt.randvec(s, x)
@test isapprox(JManOpt.transp(s, x, y, u), u)

X = rand(s)
Y = rand(s)
Yexplog = exp(s, X, log(s, X, Y))
@test isapprox(Y, Yexplog)

X = rand(s)
U = JManOpt.randvec(s, X)
Ulogexp = JManOpt.log(s, X, exp(s, X, U))
@test isapprox(U, Ulogexp)

X = rand(s)
Y = rand(s)
Z = JManOpt.pairmean(s, X, Y)
@test isapprox(JManOpt.dist(s, X, Z), JManOpt.dist(s, Y, Z))
