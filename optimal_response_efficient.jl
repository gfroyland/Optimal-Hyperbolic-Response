using LinearAlgebra, SparseArrays, Statistics, Distances, FFTW, ForwardDiff, Arpack, ProgressMeter, JLD2, CairoMakie

function fft_and_reorder(Afine, 𝐊, d)
    #Afine is a general NxN array of values on a fine grid
    #Â is the Fourier transform of Afine, subsampled to size n, and reshaped to 1D with ordering matching L
    N = size(Afine, 1)
    n = size(𝐊, 1)
    Âfine = fftshift(fft(Afine))
    Â = zeros(Complex, n^2)
    for 𝐢 ∈ 𝐊
        Â[d[𝐢]] = Âfine[𝐢[2]+N÷2+1, 𝐢[1]+N÷2+1]
    end

    return Â
end

function optimal_response_efficient(n)

    #notation:
    #x is a 2-vector on the 2-torus
    #𝐢, 𝐣, 𝐤 are 2-vectors of Fourier indices
    #𝐊 is a 2D array of 2-vectors of Fourier indices
    #e is standard scalar-valued Fourier basis function in 2D space
    #T is the map on the 2-torus
    #L is the transfer operator representation in Fourier space

    # for simplicity, this code computes the conjugation of the optimal coefficients, namely 
    # ā⁽¹⁾ₖ = -∫ c⋅(I-L)⁻¹L(∇⋅(f₀(x)(DₓT₀)⁻¹(eₖ,0)(x)))) dx and
    # ā⁽¹⁾ₗ = -∫ c⋅(I-L)⁻¹L(∇⋅(f₀(x)(DₓT₀)⁻¹(0,eₗ)(x)))) dx
    # the right hand sides of the above expressions are ultimately conjugated just prior to storage to obtain a⁽¹⁾ₖ and a⁽¹⁾ₗ

    #define map on 2-torus
    δ = 0.0
    T(x) = mod.([2x[1] + x[2] + 2δ * cos(2π * x[1]), x[1] + x[2] + δ * sin(4π * x[2] + 1)], 1)
    Tlift(x) = [2x[1] + x[2] + 2δ * cos(2π * x[1]), x[1] + x[2] + δ * sin(4π * x[2] + 1)]

    #define objective function
    c(x) = cos(2π * x[1]) + cos(2π * x[2])  #max at fixed point [0,0] and min at [0.5,0.5].
    #c(x) = exp(-peuclidean(x, [0.1796, 0.4023], [1, 1])^2 / 0.1^2) + exp(-peuclidean(x, [0.7877, 0.5852], [1, 1])^2 / 0.1^2)   #period-2 orbit stabilisation

    #Fourier modes in 2D space
    e(𝐤, x) = exp(2π * im * (𝐤 ⋅ x))

    #fine grid size = 4 × fine grid size
    N = 4n

    #2D array of 2D Fourier indices
    𝐊 = [[i, j] for j = -n÷2+1:n÷2, i = -n÷2+1:n÷2]

    #create a dictionary to index elements of 𝐊 by integers 1,2,...,n^2
    #d[𝐤] yields an integer index i∈{1,2,...,n^2} used later to index entries of L
    #𝐊[i] inverts the indexing, yielding the 2 Fourier indices 𝐤∈𝐊 corresponding to matrix index i∈{1,2,...,n^2}
    d = Dict([(𝐊[i], i) for i = 1:n^2])

    #fine spatial grid of 2-vectors x on 2-torus
    finespacerange = (1/2:N-1/2) / N
    xfine = [[x₁, x₂] for x₂ ∈ finespacerange, x₁ ∈ finespacerange]

    #function outputting Fourier coefficients of 2D Fejer kernel. Input 𝐤 is a 2-vector
    F̂(𝐤) = (1 - abs(𝐤[1]) / (n / 2 + 1)) * (1 - abs(𝐤[2]) / (n / 2 + 1))

    #compute image of fine grid on 2-torus
    Txfine = T.(xfine)

    #initialise transfer operator matrix representation on coarse Fourier indices
    L = zeros(ComplexF64, n^2, n^2)

    #construct L
    println("Constructing transfer operator...")
    @showprogress Threads.@threads for 𝐢 ∈ 𝐊
        #calculate fft of e(-𝐢)∘T on xfine
        ê𝐢T = fftshift(fft([e(-𝐢, x) for x ∈ Txfine]) / N^2)
        for 𝐣 ∈ 𝐊
            #compute product of Fejer kernel Fourier coefficient and e∘T Fourier coefficient
            L[d[𝐢], d[𝐣]] = F̂(𝐢) * ê𝐢T[-𝐣[2]+N÷2+1, -𝐣[1]+N÷2+1]
        end
    end

    println("Eigensolving...")
    @time λ, v̂ = eigs(sparse(L), nev=1, which=:LM)
    #f̂ is the leading eigenvector in frequency space
    println("Assembling leading eigenfunction in space...")
    f̂ = v̂[:, 1]
    #linearly combine the elementary Fourier basis elements according to f̂
    f(x) = sum(f̂[d[𝐤]] * e(𝐤, x) for 𝐤 ∈ 𝐊)
    #evaluate the above linear combination on the fine spatial grid
    ffine = f.(xfine)
    #alter phase to maximise real part
    ψ = -angle(transpose(ffine[:]) * ffine[:]) / 2
    ffine = ffine * exp(im * ψ)
    parity = sign(real(mean(ffine)))
    ffine = ffine * parity
    ffineplot = normalize(real(ffine), 1) * N^2

    #plot
    println("Plotting...")
    srbfig = Figure(size=(450, 400))
    srbax = Axis(srbfig[1, 1], autolimitaspect=1)
    heatmap!(srbax, finespacerange, finespacerange, ffineplot', colormap=:Blues)
    Colorbar(srbfig[1, 2], limits=(0, maximum(ffineplot)), colormap=:Blues)
    display(srbfig)

    #compute (I-L)⁻¹ restricted to to zero-mean subspace
    #in frequency space, just delete row and column corresponding to the [0,0] mode and compute inverse directly
    restrind = setdiff(1:n^2, d[[0, 0]])
    unitresolvent = inv(I - L[restrind, restrind])

    #the term below ought to conjugate c, but c is real, so we forego this conjugation
    ĉordered = fft_and_reorder(c.(xfine), 𝐊, d)
    premult = transpose(ĉordered[restrind]) * unitresolvent * L[restrind, restrind]
    #need ForwardDiff to perform real and imaginary parts separately
    ∇ffine = parity * exp(im * ψ) * (ForwardDiff.gradient.(x -> real(f(x)), xfine) + ForwardDiff.gradient.(x -> imag(f(x)), xfine) * im)

    invDT = x -> inv(ForwardDiff.jacobian(Tlift, x))
    DinvDT = x -> ForwardDiff.jacobian(invDT, x)
    divinvDT(x) = [DinvDT(x)[1, 1] + DinvDT(x)[2, 2], DinvDT(x)[3, 1] + DinvDT(x)[4, 2]]

    invDTfine = invDT.(xfine)
    divinvDTfine = divinvDT.(xfine)

    #premultiply those terms in the 𝐤 loops below that don't depend on 𝐤
    #transposes are simply to make row vectors for the later inner products of two vectors
    term1prelim = transpose.(∇ffine) .* invDTfine
    term2prelim = transpose.(ffine .* divinvDTfine)
    term3prelim = ffine .* invDTfine

    ∂e1(𝐤, x) = 2π * im * [𝐤[1]*e(𝐤, x) 𝐤[2]*e(𝐤, x); 0 0]
    ∂e2(𝐥, x) = 2π * im * [0 0; 𝐥[1]*e(𝐥, x) 𝐥[2]*e(𝐥, x)]

    #initialise arrays
    a1 = zeros(ComplexF64, n, n)
    a2 = zeros(ComplexF64, n, n)
    termsum1 = zeros(ComplexF64, n, n)
    termsum2 = zeros(ComplexF64, n, n)

    #set scale factor γ in the Sobolev H⁵ norm
    γ = 0.02

    scale(𝐤) = sum((2π * γ)^(2m) * norm(𝐤)^(2m) for m = 0:7)

    #compute Fourier coefficients of the x-component of the optimal vector field
    println("Computing optimal Fourier coefficients...")
    @showprogress Threads.@threads for 𝐤 ∈ 𝐊
        term1 = term1prelim .* [[e(𝐤, x), 0] for x ∈ xfine]
        term2 = term2prelim .* [[e(𝐤, x), 0] for x ∈ xfine]
        term3 = tr.(term3prelim .* [∂e1(𝐤, x) for x ∈ xfine])
        fftallterms = fft_and_reorder(term1 + term2 + term3, 𝐊, d)
        #store result;  we need to apply the conjugation to a1 because the prior code computes its conjugate
        a1[d[𝐤]] = -conj(premult * fftallterms[restrind] / scale(𝐤))
        termsum1[d[𝐤]] = fftallterms[d[[0, 0]]]
    end

    #compute Fourier coefficients of the y-component of the optimal vector field
    @showprogress Threads.@threads for 𝐤 ∈ 𝐊
        term1 = term1prelim .* [[0, e(𝐤, x)] for x ∈ xfine]
        term2 = term2prelim .* [[0, e(𝐤, x)] for x ∈ xfine]
        term3 = tr.(term3prelim .* [∂e2(𝐤, x) for x ∈ xfine])
        fftallterms = fft_and_reorder(term1 + term2 + term3, 𝐊, d)
        #store result;  we need to apply the conjugation to a2 because the prior code computes its conjugate
        a2[d[𝐤]] = -conj(premult * fftallterms[restrind] / scale(𝐤))
        termsum2[d[𝐤]] = fftallterms[d[[0, 0]]]
    end

    #put together to make a vector field
    Ṫ(x) = sum(a1[d[𝐤]] * [e(𝐤, x), 0] for 𝐤 ∈ 𝐊) + sum(a2[d[𝐤]] * [0, e(𝐤, x)] for 𝐤 ∈ 𝐊)

    #coarse spatial grid of 2-vectors x on 2-torus
    coarsespacerange = (1/2:n-1/2) / n
    xcoarse = [[x₁, x₂] for x₂ ∈ coarsespacerange, x₁ ∈ coarsespacerange]

    #visualation of the optimal vector field on coarse points and their images
    xcoarselist = xcoarse[:]    #xcoarse in vector form (vector of 2-vectors)
    Ṫcoarse = [Ṫ(x) for x ∈ xcoarse]
    Ṫcoarselist = [Ṫ(x) for x ∈ xcoarse][:]    #the optimal vector field listed as a vector of 2-vectors at coarse points

    #create points and vectors for the vector-field plot
    points = Point2f.(xcoarselist)
    vectors = Vec2f.(real.(Ṫcoarselist))

    #compute a scalefactor for the visual length of the vector field arrows
    scalefactor = (√2 / n) / maximum(norm.(Ṫcoarselist))  #scale so the largest component is the grid spacing

    # set up figure axis and plot optimal vector field
    arrowfig = Figure(size=(425, 425))
    arrowax = Axis(arrowfig[1, 1], autolimitaspect=1)
    arrows!(arrowax, points, vectors, lengthscale=scalefactor, arrowsize=6, align=:tail)
    display(arrowfig)
    save("optimalvffig.png", arrowfig, px_per_unit=5)

    #plot optimal vector field on top of the SRB measure
    arrows!(srbax, points, vectors, lengthscale=scalefactor, arrowsize=6, align=:tail)
    display(srbfig)
    save("optimalvfsrbfig.png", srbfig, px_per_unit=5)

    return a1, a2, Ṫ, Ṫcoarse, ffine, L

end