m = 100
n = 50
s = Oblique(m, n)

# def test_dim(self):

# def test_typicaldist(self):

# def test_dist(self):

# def test_inner(self):

# def test_proj(self):

# def test_ehess2rhess(self):

# def test_retr(self):

# def test_egrad2rgrad(self):

# def test_norm(self):

# def test_rand(self):

# def test_randvec(self):

# def test_transp(self):

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
