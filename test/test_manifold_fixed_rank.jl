m = 10
n = 5
k = 3
s = GeoMax.FixedRank(m, n, k)

@test GeoMax.dim(s) == (m + n - k) * k

@test GeoMax.dim(s) == GeoMax.typicaldist(s)

a = rand(s)
x = GeoMax.randvec(s, a)
y = GeoMax.randvec(s, a)


x = rand(s)
a = GeoMax.randvec(s,x)
b = GeoMax.randvec(s,x)
# First embed in the ambient space
A = dot(dot(a[2], x[3]), x[1]) + dot(x[3], a[1]) + dot(a[3]', x[1])
B = dot(x[1], dot(b[2], x[3])) + dot(b[1], x[3]) + dot(x[1], b[3]')
trueinner = sum(A * B)
@test isapprox(trueinner, GeoMax.inner(s, x, a, b))

x = rand(s)
v = randn(m, n)

g = GeoMax.proj(s, x, v)
# Check that g is a true tangent vector
@test isapprox(dot(g[1]', x[1]), zeros(k, k),atol=1e-6)
@test isapprox(dot(g[3]', x[3]'), zeros(k, k), atol=1e-6)

# Verify that proj gives the closest point within the tangent space
# by displacing the result slightly and checking that this increases
# the distance.
x = rand(s)
v = randn(m, n)

g = GeoMax.proj(x, v)
# Displace g a little
g_disp = g + 0.01 * GeoMax.randvec(x)

# Return to the ambient representation
g = GeoMax.tangent2ambient(x, g)
g_disp = GeoMax.tangent2ambient(x, g_disp)
g = dot(g[3]', dot(g[1], g[2]))
g_disp = dot(g_disp[3]', dot(g_disp[2], g_disp[1]))

@test norm(g - v) < norm(g_disp - v)

# Verify that proj leaves tangent vectors unchanged
x = rand(s)
u = GeoMax.randvec(s,x)
A = GeoMax.proj(x, GeoMax.tangent2ambient(x, u))
B = u
# diff = [A[k]-B[k] for k in xrange(len(A))]
@test isapprox(A[1], B[1])
@test isapprox(A[2], B[2])
@test isapprox(A[3], B[3])

x = rand(s)
u = GeoMax.randvec(s,x)
@test isapprox(sqrt(GeoMax.inner(x, u, u)), norm(x, u))

x = rand(s)
y = rand(s)
@test shape(x[1]) == (m, k)
@test shape(x[2]) == (k,)
@test shape(x[3]) == (k, n)
@test isapprox(dot(x[1], x[1]'), eye(k), atol=1e-6)
@test isapprox(dot(x[3]', x[3]), eye(k), atol=1e-6)

@test norm(x[0] - y[0]) > 1e-6
@test norm(x[1] - y[1]) > 1e-6
@test norm(x[2] - y[2]) > 1e-6

x = rand(s)
y = rand(s)
u = GeoMax.randvec(x)
A = GeoMax.transp(x, y, u)
B = GeoMax.proj(y, GeoMax.tangent2ambient(x, u))
diff = [A[k]-B[k] for k in range(len(A))]
@test isapprox(GeoMax.norm(y, diff), 0)


z = randn(m, n)

# Set u, s, v so that z = u.dot(s).dot(v.T)
u, s, v = svd(z)
s = diag(s)
v = v'

w = randn(n, n)

@test isapprox(dot(w, z), GeoMax._apply_ambient(z, w))
@test isapprox(dot(w, z), GeoMax._apply_ambient((u, s, v), w))

z = randn(n, m)

# Set u, s, v so that z = u.dot(s).dot(v.T)
u, s, v = svd(z)
s = diag(s)
v = v'

w = randn(n, n)

@test isapprox(dot(w, z'), GeoMax._apply_ambient_transpose(z, w))
@test isapprox(dot(w, z'), GeoMax._apply_ambient_transpose((u, s, v), w))

x = rand(s)
z = GeoMax.randvec(s, x)

z_ambient = (dot(x[3], dot(z[2], x[1])) + dot(x[3], z[1]) +
             dot(z[3]', x[1]))

u, s, v = GeoMax.tangent2ambient(x, z)

@test isapprox(z_ambient, dot(v, dot(u, s)))

# test_ehess2rhess()

# Test that the result is on the manifold and that for small
# tangent vectors it has little effect.
x = rand(s)
u = GeoMax.randvec(s, x)

y = GeoMax.retr(s, x, u)

@test isapprox(dot(y[1]', y[1]), eye(k), atol=1e-6)
@test isapprox(dot(y[3] ,y[3]'), eye(k), atol=1e-6)

u = u * 1e-6
y = GeoMax.retr(s, x, u)
y = y[1].dot(diag(y[2])).dot(y[3])

u = GeoMax.tangent2ambient(x, u)
u = dot(u[3]', dot(u[1], u[2]))
x = dot(dot(x[1], diag(x[2])), x[3])

@test isapprox(y, x + u, atol=1e-5)

# TODO test egrad2rgrad
# Verify that egrad2rgrad and proj are equivalent.
# x = rand(s)
# u, s, vt = x
#
# i = eye(k)
#
# f = 1 / (s[..., newaxis, :]**2 - s[..., :, newaxis]**2 + i)
#
# du = random.randn(m, k)
# ds = random.randn(k)
# dvt = random.randn(k, n)
#
# Up = (dot(dot(eye(m) - dot(u, u.T), du),
#              linalg.inv(diag(s))))
# M = (dot(f * (dot(u.T, du) - dot(du.T, u)), diag(s)) +
#      dot(diag(s), f * (dot(vt, dvt.T) - dot(dvt, vt.T))) +
#      diag(ds))
# Vp = (dot(dot(eye(n) - dot(vt.T, vt), dvt.T),
#              linalg.inv(diag(s))))
#
# up, m, vp = m.egrad2rgrad(x, (du, ds, dvt))
#
# @test isapprox(Up, up)
# @test isapprox(M, m)
# @test isapprox(Vp, vp)

x = rand(s)
u = GeoMax.randvec(s,x)

# Check that u is a tangent vector
@test size(u[1]) == (m, k)
@test size(u[2]) == (k, k)
@test size(u[3]) == (n, k)
@test isapprox(dot(u[1]', x[1]), zeros(k, k), atol=1e-6)
@test isapprox(dot(u[3]', x[3]'), zeros(k, k), atol=1e-6)

v = GeoMax.randvec(s,x)

@test isapprox(GeoMax.norm(x, u), 1)
@test GeoMax.norm(x, u - v) > 1e-6
