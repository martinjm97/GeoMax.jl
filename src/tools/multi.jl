multiprod(A::AbstractMatrix, B::AbstractMatrix) = dot(A,B)

function multiprod(A::AbstractArray{<:Number,3}, B::AbstractArray{<:Number, 3})
    # Einstein contraction 'ijk,ikl->ijl'
    X = Array{Any}(size(A,1), size(A,2), size(B,3))
    for r = 1:size(B,3)
        for j = 1:size(A,2)
            for i = 1:size(A,1)
                s = 0
                for k = 1:size(A,3)
                    s += A[i,j,k] * B[i,k,r]
                end
                X[i,j,r] = s
            end
        end
    end
    return X
end

# TODO rewrite witH ::AbstractMatrix
function multitransp(A::AbstractArray)
    if ndims(A) == 2
        return A'
    end
    return permutedims(A, [1, 3, 2])
end

function multisym(A::AbstractArray)
    return 0.5 * (A + multitransp(A))
end

function multieye(k,n)
    return repeat(eye(n), (k, 1, 1))
end

# TODO maybe do type checking of matrices.
function multilog(A::AbstractArray)
    @assert LinAlg.issymmetric(A) "not implemented"
    @assert LinAlg.isposdef(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(log.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end

function multiexp(A::AbstractArray)
    @assert LinAlg.issymmetric(A) "not implemented"
    l, v = Linalg.eig(Hermetian(A))
    l =  reshape(exp.(l), (size(l)...,1))
    return multiprod(v, l * multitransp(v))
end
