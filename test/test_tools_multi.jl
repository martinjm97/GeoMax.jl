a = ones(2,2,2)
a[:,:,1] += 2
a[:,:,2] += 4
e_val, e_vec = eig(a)
@test isapprox(e_val, [0.0 8.0; 0.0 8.0])

v = [-0.857493 -0.707107; 0.514496 -0.707107]
@test isapprox(e_vec[:,:,1],  v, atol=1e-5)

a = ones(2,2,2)
@test JManOpt.multiprod(a, 2a) == 4*ones(2,2,2)

a = eye(4)
@test JManOpt.multiprod(a, 2a) == 2*eye(4)

a = reshape(1:10,2,5)
@test a' == JManOpt.multitransp(a)

a = reshape(1:12,2,2,3)
@test [1 3; 5 7; 9 11] == JManOpt.multitransp(a)[1,:,:]

a = reshape(1:8,2,2,2)
@test [1 4; 4 7] == JManOpt.multisym(a)[1,:,:]

k = 3
n = 2
me = JManOpt.multieye(k,n)
@test size(me) == (k,n,n)
@test me[1,:,:] == eye(n)

@test isapprox(JManOpt.multilog(100*eye(4)), 4.60517*eye(4), atol = 1e-6)


a = eye(2)
@test isapprox(JManOpt.multiexp(a), Base.e .* eye(2))

a = [1 2; 2 4]
@test isapprox(JManOpt.multiexp(a), [30.4826 58.9653; 58.9653 118.931], atol=1e-3)

a = eye(2)
@test isapprox(JManOpt.multilog(a), zeros(2,2))

a = 5*eye(3)
a[1,2] = 4
a[2,1] = 4
b = [1.09861  1.09861  0.0;
    1.09861  1.09861  0.0;
    0.0      0.0      1.60944]
@test isapprox(JManOpt.multilog(a), b, atol = 1e-5)

a = JManOpt.multieye(2,2)
a[1,2,1] = 1
b = ones(2, 2, 2)
@test JManOpt.solve(a, b) == [[1.0 1.0; 0.0 0.0],[1.0 1.0; 1.0 1.0]]
