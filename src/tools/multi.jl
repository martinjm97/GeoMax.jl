multiprod(A::AbstractMatrix, B::AbstractMatrix) = dot(A,B)
multiprod(A::Float64, B::AbstractMatrix) = A * B
multiprod(A::AbstractMatrix, B::Float64) = A * B
function multiprod(a::AbstractArray{A,3}, b::AbstractArray{B,3}) where {A,B}
    c = similar(a, promote_type(A, B), size(a, 1), size(a, 2), size(b, 3))
    return multiprod!(c, a, b)
end

function multiprod!(c::AbstractArray{C,3},
                    a::AbstractArray{<:Number,3},
                    b::AbstractArray{<:Number,3}) where C<:Number
    for r = 1:size(c, 3)
        for j = 1:size(c, 2)
            for i = 1:size(c, 1)
                s = zero(C)
                for k = 1:size(a, 3)
                    s += a[i,j,k] * b[i,k,r]
                end
                c[i,j,r] = s
            end
        end
    end
    return c
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

# TODO maybe do type checking of matrices. Split into errors and Hermetian
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
