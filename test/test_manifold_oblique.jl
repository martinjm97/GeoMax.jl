m = 100
n = 50
s = JManOpt.Oblique(m, n)

# test_dim:
# test_typicaldist
# test_dist
# test_inner
# test_proj
# test_ehess2rhess
# test_retr
# test_egrad2rgra
# test_norm
# test_rand
# test_randvec
# test_transp

x = rand(s)
y = rand(s)
u = log(s, x, y)
z = exp(s, x, u)
@test isapprox(0, JManOpt.dist(s, y, z), atol = 1e-6)

x = rand(s)
u = randvec(s, x)
y = exp(s, x, u)
v = log(s, x, y)
# Check that the manifold difference between the tangent vectors u and
# v is 0
@test isapprox(0, norm(s, x, u - v))

X = rand(s)
Y = rand(s)
Z = JManOpt.pairmean(s, X, Y)
@test isapprox(JManOpt.dist(s, X, Z), JManOpt.dist(s, Y, Z))
