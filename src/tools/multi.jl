# MAKE one line functions FOR STYLE

# TODO write manually

multiprod(A::AbstractMatrix, B::AbstractMatrix) = dot(A,B)
# macroexpand(@einsum)
function multiprod(A::AbstractArray{<:Number,3}, B::AbstractArray{<:Number, 3})
    # np.einsum('ijk,ikl->ijl', A, B)
    @einsum X[i,j,l] := A[i,j,k]*B[i,k,l]
    return X
end

#todo rewrite witH ::AbstractMatrix
function multitransp(A)
    if ndims(A) == 2
        return transpose(A)
    end
    return permutedims(A, [1, 3, 2])
end

function multisym(A)
    return 0.5 * (A + multitransp(A))
end

function multieye(k,n)
    return repeat(eye(n), (k, 1, 1))
end

# TODO maybe do type checking of matrices.
function multilog(A)
    @assert LinAlg.issymmetric(A) "not implemented"
    @assert LinAlg.isposdef(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(log.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v)
end

function multiexp(A)
    @assert LinAlg.issymmetric(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(exp.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end
