using LinearAlgebra, SparseArrays, Statistics, FFTW, Arpack, ForwardDiff, ProgressMeter, JLD2, CairoMakie

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

    #define map on 2-torus
    δ = 0.01
    T(x) = mod.([2x[1] + x[2] + 2δ * cos(2π * x[1]), x[1] + x[2] + δ * sin(4π * x[2] + 1)], 1)
    Tlift(x) = [2x[1] + x[2] + 2δ * cos(2π * x[1]), x[1] + x[2] + δ * sin(4π * x[2] + 1)]

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
    #inespacerange = (0:N-1) / N
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
        ê𝐢T = fftshift(fft([e(-𝐢, x) for x ∈ Txfine]) / N^2)  #T.(xfine) should be a 2d array of 2-vectors.  I had to do fftshift for indexing in line 49 below
        #if norm(ê𝐢T) * F̂(𝐢) / N > 1e-10
        for 𝐣 ∈ 𝐊
            #compute product of Fejer kernel Fourier coefficient and e∘T Fourier coefficient
            L[d[𝐢], d[𝐣]] = F̂(𝐢) * ê𝐢T[-𝐣[2]+N÷2+1, -𝐣[1]+N÷2+1]
            #compute pure Fourier truncation without Fejer kernel smoothing
            #L[d[𝐢], d[𝐣]] = ê𝐢T[-𝐣[1]+N÷2+1, -𝐣[2]+N÷2+1]
        end
    end

    #PREPARE FOR PLOTTING LEADING EIGENFUNCTION
    println("Eigensolving...")
    @time λ, v̂ = eigs(sparse(L), nev=1, which=:LM)
    #f̂ is the leading eigenvector in frequency space
    println("Assembling leading eigenfunction in space...")
    f̂ = v̂[:, 1]
    #linearly combine the elementary Fourier basis elements according to f̂
    f(x) = sum(f̂[d[𝐤]] * e(𝐤, x) for 𝐤 ∈ 𝐊)
    #evaluate the above linear combination on the fine spatial grid
    ffine = f.(xfine)
    #maximise real part
    ψ = -angle(transpose(ffine[:]) * ffine[:]) / 2
    ffine = ffine * exp(im * ψ)
    parity = sign(real(ffine[1]))
    ffine = ffine * parity
    ffineplot = normalize(real(ffine), 1) * N^2

    #plot
    println("Plotting...")
    #display(Plots.heatmap(finespacerange, finespacerange, real(μfine)', colormap=:Blues))
    srbfig = Figure(size=(450, 400))
    srbax = Axis(srbfig[1, 1], autolimitaspect=1)
    #NOTE THAT MAKIE NEEDS A TRANSPOSE ON THE GRIDDED FFINE TO PLOT IT CORRECTLY
    @time heatmap!(srbax, finespacerange, finespacerange, ffineplot', colormap=:Blues)
    Colorbar(srbfig[1, 2], limits=(0, maximum(ffineplot)), colormap=:Blues)
    display(srbfig)

    #compute (I-L)⁻¹ restricted to to zero-mean subspace
    #in frequency space, just delete row and column corresponding to the [0,0] mode and compute inverse directly
    restrind = setdiff(1:n^2, d[[0, 0]])
    unitresolvent = inv(I - L[restrind, restrind])

    # c(x) = cos(2π * x[1]) + cos(2π * x[2])  #max at fixed point [0,0] and min at [0.5,0.5]
    c(x) = sin(2π * (x[1]))^2 + cos(2π * (x[2] - 0.5))  #period 2 stabilisation

    cfig = Figure(size=(450, 400))
    cax = Axis(cfig[1, 1], autolimitaspect=1)
    #NOTE THAT MAKIE NEEDS A TRANSPOSE ON THE GRIDDED FFINE TO PLOT IT CORRECTLY
    heatmap!(cax, finespacerange, finespacerange, c.(xfine)', colormap=:RdBu)
    Colorbar(cfig[1, 2], limits=(minimum(c.(xfine)), maximum(c.(xfine))), colormap=:RdBu)
    display(cfig)
    save("cfig.pdf", cfig)

    ĉordered = fft_and_reorder(c.(xfine), 𝐊, d)
    premult = ĉordered[restrind]' * unitresolvent * L[restrind, restrind]

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
    γ = 0.025
    #γ = 0.02
    #γ = 0.015
   
    scale(𝐤) = sum((2π * γ)^(2m) * norm(𝐤)^(2m) for m = 0:5)

    #compute Fourier coefficients of the x-component of the optimal vector field
    @showprogress Threads.@threads for 𝐤 ∈ 𝐊

        term1 = term1prelim .* [[e(𝐤, x), 0] for x ∈ xfine]
        term2 = term2prelim .* [[e(𝐤, x), 0] for x ∈ xfine]
        term3 = tr.(term3prelim .* [∂e1(𝐤, x) for x ∈ xfine])
        fftallterms = fft_and_reorder(term1 + term2 + term3, 𝐊, d)
        #store result
        a1[d[𝐤]] = premult * fftallterms[restrind] / scale(𝐤)
        termsum1[d[𝐤]] = fftallterms[d[[0, 0]]]

    end

    #compute Fourier coefficients of the y-component of the optimal vector field
    @showprogress Threads.@threads for 𝐤 ∈ 𝐊

        term1 = term1prelim .* [[0, e(𝐤, x)] for x ∈ xfine]
        term2 = term2prelim .* [[0, e(𝐤, x)] for x ∈ xfine]
        term3 = tr.(term3prelim .* [∂e2(𝐤, x) for x ∈ xfine])
        fftallterms = fft_and_reorder(term1 + term2 + term3, 𝐊, d)
        #store result
        a2[d[𝐤]] = premult * fftallterms[restrind] / scale(𝐤)
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
    
    #x,y components of the points to plot the base of the arrows
    arrowx = [xcoarselist[i][1] for i ∈ eachindex(xcoarselist)]
    arrowy = [xcoarselist[i][2] for i ∈ eachindex(xcoarselist)]
    #x,y components of the arrow directions
    arrowu = [Ṫcoarselist[i][1] for i ∈ eachindex(xcoarselist)]
    arrowv = [Ṫcoarselist[i][2] for i ∈ eachindex(xcoarselist)]
   
    arrowfig = Figure(size=(425, 425))
    arrowax = Axis(arrowfig[1, 1], autolimitaspect=1)

    #compute a scalefactor for the visual length of the vector field arrows
    scalefactor = (1 / n) / max(maximum(real(arrowu)), maximum(real(arrowv)))  #scale so the largest component is the grid spacing

    #plot optimal vector field
    arrows!(arrowax, arrowx, arrowy, real(arrowu), real(arrowv), lengthscale=scalefactor, arrowsize=6)
    display(arrowfig)
    save("arrowfig.pdf", arrowfig)

    #plot optimal vector field on top of the SRB measure
    arrows!(srbax, arrowx, arrowy, real(arrowu), real(arrowv), lengthscale=scalefactor, arrowsize=6)
    display(srbfig)
    save("testfig.pdf", srbfig)

    #plot stable manifold on top of the vector field and SRB measure
    scatter!(srbax, Tuple.(extendedsegment), markersize=4, color=:red)
    display(srbfig)
    save("totalfig.pdf", srbfig)

    return a1, a2, Ṫ, Ṫcoarse, ffine

end
