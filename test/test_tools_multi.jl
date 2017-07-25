a = ones(2,2,2)
a[:,:,1] += 2
a[:,:,2] += 4
e_val, e_vec = GeoMax.eig(a)
@test isapprox(e_val, [0.0 0.0; 6.0 10.0])

v = [ -0.707107  0.707107
  0.707107  0.707107]
@test isapprox(e_vec[:,:,1],  v, atol=1e-5)

a = ones(2,2,2)
@test GeoMax.multiprod(a, 2a) == 4*ones(2,2,2)

a = eye(4)
@test GeoMax.multiprod(a, 2a) == 2*eye(4)

a = reshape(1:10,2,5)
@test a' == GeoMax.multitransp(a)

a = reshape(1:12,2,2,3)
@test [1 2; 3 4] == GeoMax.multitransp(a)[:,:,1]

a = reshape(1:8,2,2,2)
@test [1 2.5; 2.5 4] == GeoMax.multisym(a)[:,:,1]

k = 3
n = 2
me = GeoMax.multieye(k,n)
@test size(me) == (n,n, k)
@test me[:,:,1] == eye(n)

@test isapprox(GeoMax.multilog(100*eye(4)), 4.60517*eye(4), atol = 1e-6)


a = eye(2)
@test isapprox(GeoMax.multiexp(a), Base.e .* eye(2))

a = [1 2; 2 4]
@test isapprox(GeoMax.multiexp(a), [30.4826 58.9653; 58.9653 118.931], atol=1e-3)

a = eye(2)
@test isapprox(GeoMax.multilog(a), zeros(2,2))

a = 5*eye(3)
a[1,2] = 4
a[2,1] = 4
b = [1.09861  1.09861  0.0;
    1.09861  1.09861  0.0;
    0.0      0.0      1.60944]
@test isapprox(GeoMax.multilog(a), b, atol = 1e-5)

a = GeoMax.multieye(2,2)
a[1,2,1] = 1
b = ones(2, 2, 2)
q = GeoMax.solve(a, b)
@test q[:,:,1] == [0.0 0.0; 1.0 1.0]
@test q[:,:,2] == [1.0 1.0; 1.0 1.0]

a = GeoMax.multieye(2,2)
a[:,:,1] = 3a[:,:,1]
q = cholfact(a)
@test isapprox(q[:,:,1], [1.73205 0.0; 0.0 1.73205], atol = 1e-5)
@test isapprox(q[:,:,2], [1.0 0.0; 0.0 1.0], atol = 1e-5)
