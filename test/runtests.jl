using JManOpt
using Base.Test

# Test Sphere
m = 100
n = 50
s = JManOpt.Sphere([m, n])
@test JManOpt.dim(s) == m * n -1
