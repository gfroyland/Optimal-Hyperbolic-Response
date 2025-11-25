function response_compare(n, Ṫ, ffine0)

    #n is the number of Fourier modes in each coordinate direction
    #Ṫ is a function that returns a 2-vector of real numbers (namely the vector field)
    #ffine0 is the SRB measure of T₀ evaluated on the fine grid

    δ = 0.00
    #define perturbed version of T with tiny increment of Ṫ added (divide by 8e10 for period 2)
    T(x) = mod.([2x[1] + x[2] + 2δ * cos(2π * x[1]), x[1] + x[2] + δ * sin(4π * x[2] + 1)] + real(Ṫ(x)) / 9e10, 1)
    
    #define objective function 
    c(x) = cos(2π * x[1]) + cos(2π * x[2])  #max at fixed point [0,0] and min at [0.5,0.5]
    #c(x) = sin(2π * (x[1]))^2 + cos(2π * (x[2] - 0.5))  #period 2 stabilisation

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

    #PREPARE FOR PLOTTING LEADING EIGENFUNCTION
    println("Eigensolving...")
    @time λ, v̂ = eigs(sparse(L), nev=10, which=:LM, maxiter=10000)
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
    parity = sign(real(ffine[1]))
    ffine = ffine * parity

    #normalise
    ffine0 = normalize(real(ffine0), 1) * N^2
    ffine1 = normalize(real(ffine), 1) * N^2

    #plot
    println("Plotting...")
    srbfig = Figure(size=(450, 400))
    srbax = Axis(srbfig[1, 1], autolimitaspect=1)
    heatmap!(srbax, finespacerange, finespacerange, ffine1', colormap=:Blues)
    Colorbar(srbfig[1, 2], limits=(0, maximum(ffine1)), colormap=:Blues)
    display(srbfig)
    save("perturbed_SRB_fig.png", srbfig, px_per_unit=5)

    oldexpectation = mean(c.(xfine) .* real(ffine0))
    newexpectation = mean(c.(xfine) .* real(ffine1))

    return oldexpectation, newexpectation, ffine0, ffine1, λ, v̂

end