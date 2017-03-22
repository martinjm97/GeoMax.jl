# TODO consider doing it with pycall and einsum
# how to einsum
using Einsum

function multiprod(A::Array, B::Array)
    if length(size(A)) == 2
        return dot(A, B)
    end
    # np.einsum('ijk,ikl->ijl', A, B)
    @einsum X[i,j,l] = A[i,j,k]*B[i,k,l]
    return X
end

function multitransp(A::Array)
    if ndims(A) == 2:
        return transpose(A)
    return permutedims(A, [1, 3, 2])
end

function multisym(A::Array)
    return 0.5 * (A + multitransp(A))
end

function multieye(k::Int,n::Int)
    return repeat(eye(n), (k, 1, 1))
end

# TODO check if required to type cast to pos def and hermetian respectively
function multilog(A::Array)
    if !LinAlg.isposdef(A)
        throw("not implemented")
    end
    l, v = Linalg.eig(A)
    l =  reshape(log.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end

function multiexp(A::Array)
    if !LinAlg.issymmetric(A)
        throw("not implemented")
    end
    l, v = Linalg.eig(A)
    l =  reshape(exp.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end
