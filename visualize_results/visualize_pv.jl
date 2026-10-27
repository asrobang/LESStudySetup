using LESStudySetup
using CairoMakie
using Printf
using FFTW
using Oceananigans.Grids: xnodes, ynodes, znodes, xspacings, yspacings
using Oceananigans.Fields: interior
using LESStudySetup.Diagnostics: load_subdomain_snapshot, load_snapshots, isotropic_powerspectrum
using MathTeXEngine
using JLD2 #, CUDA
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20261019_hy/"

# --- Set parameters ---
set_value!(; Δh = 4.8828125)    # horizontal spacing
f = parameters.f;               # Coriolis parameter
Q = 40                          # surface sensible heat flux? 
h₀ = 60                         # height of the convective BL or mixing depth? 
ρ₀ = parameters.ρ₀              # reference density
cₚ = parameters.cp              # heat capacity
α = parameters.α                # thermal expansion
g = parameters.g                # gravity
wₛ = (α * g * Q * h₀ / (ρ₀ * cₚ))^(1/3)     # convective velocity scale [m/s]?
dx = 4.88281                    # [m] horizontal grid size
dy = 4.88281                    # [m] horizontal grid size
dz = 1.125                      # [m] vertical grid size

# --- Helper Functions ---

function compute_dAdx_dAdy(A,dx,dy) 
    A_hpad = hcat(A, A[:, 1:1])      # pad the first column at the rightmost (Ny,Nx+1)
    A_vpad = vcat(A, A[1:1, :])      # pad the first row at the bottom (Ny+1,Nx)
    dAdx = (A_hpad[:, 2:end] .- A_hpad[:, 1:end-1]) ./ dx
    dAdy = (A_vpad[2:end, :] .- A_vpad[1:end-1, :]) ./ dy
    
    return dAdx, dAdy
end

function compute_dAdz(A_above,A_below,dz)
    dAdz = (A_above - A_below) ./ (2*dz)

    return dAdz
end

function lowpass_filter(field, k, smooth)
    """
        lowpass_filter(field, k, smooth)

    Apply a sharp spectral cutoff filter to `field`, removing all horizontal
    scales smaller than `smooth` (a physical length, e.g. in meters — NOT a
    number of grid cells).

    Cutoff wavenumber:
        K = 2π / smooth

    This generalizes the "4Δx smoothing → K = π/(2Δx)" recipe: if smooth = 4Δx,
    then K = π/(2·(smooth/4)) = 2π/smooth, so K = 2π/smooth works for any
    smoothing scale.

    Assumes a horizontally uniform grid. Works on 2D (Nx, Ny) slices or
    3D (Nx, Ny, Nz) fields (filtered level-by-level in the horizontal).
    """

    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field,:,:,k))  # bring to CPU physical-space array
    Nx, Ny = size(data, 1), size(data, 2)
    Lx, Ly = Nx * Δx, Ny * Δy

    # # ONLY if the domain being transformed is periodic 
    # # Angular wavenumbers associated with the horizontal FFT
    # kx = 2π .* fftfreq(Nx, 1/dx)
    # ky = 2π .* fftfreq(Ny, 1/dy)

    # DCT used for zero-gradient domain edges 
    # Cosine-transform wavenumbers (Neumann / zero-gradient assumption):
    # note π, not 2π — the fundamental mode has period 2L, not L
    kx = π .* (0:Nx-1) ./ Lx
    ky = π .* (0:Ny-1) ./ Ly

    K = 2π / smooth          # cutoff wavenumber for this smoothing scale
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    mask = κ² .<= K^2        # true where κ ≤ K (keep), false where κ > K (zero)

    filtered = similar(data)

    # # ONLY if the domain being transformed is periodic 
    # f̂ = fft(data)
    # f̂ .*= mask
    # filtered .= real.(ifft(f̂))

    # DCT used for zero-gradient domain edges 
    f̂ = FFTW.r2r(data, FFTW.REDFT10)   # forward DCT-II
    f̂ .*= mask
    filtered = FFTW.r2r(f̂, FFTW.REDFT01) ./ (4 * Nx * Ny)  # inverse DCT-III + normalization

    return filtered
end

function bandpass_filter(field, dx, dy, cutoffs)
    """
        bandpass_filter(field, dx, dy, cutoffs)

    Split `field` into multiple horizontal bands using a series of sharp
    spectral cutoffs (κ = √(k² + l²), periodic FFT in x, y).

    `cutoffs` should be given as physical smoothing scales (same units as
    dx, dy), sorted from largest scale to smallest, e.g.:

        cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]

    which corresponds to wavenumber cutoffs K = 2π / cutoffs, i.e.
    K = [2π/10⁴, 2π/10³, 2π/10², 2π/10¹, 2π/10⁰].

    Returns a Dict mapping each band (as a string label) to the filtered field
    containing only that band's wavenumbers. Bands are:

        band 1: κ ≤ K[1]                     (largest scales, > cutoffs[1])
        band 2: K[1] < κ ≤ K[2]
        band 3: K[2] < κ ≤ K[3]
        ...
        band N+1: κ > K[N]                   (smallest scales, < cutoffs[end])
    """

    Nx, Ny = size(field, 1), size(field, 2)

    kx = 2π .* fftfreq(Nx, 1/dx)
    ky = 2π .* fftfreq(Ny, 1/dy)
    κ² = (kx .^ 2) .+ (ky' .^ 2)

    # cutoff wavenumbers, sorted increasing (since cutoffs is sorted decreasing)
    K = 2π ./ cutoffs
    @assert issorted(K) "cutoffs must be sorted from largest scale to smallest (e.g. [1e4, 1e3, ..., 1e0])"

    f̂ = fft(field)
    results = Dict{String, Array{Float64}}()

    # band 1: everything below the first (smallest) cutoff wavenumber
    mask = κ² .<= K[1]^2
    results["κ ≤ 2π/$(cutoffs[1])"] = real.(ifft(f̂ .* mask))

    # middle bands: between consecutive cutoffs
    for n in 1:length(K)-1
        mask = (κ² .> K[n]^2) .& (κ² .<= K[n+1]^2)
        results["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])"] = real.(ifft(f̂ .* mask))
    end

    # last band: everything above the largest cutoff wavenumber
    mask = κ² .> K[end]^2
    results["κ > 2π/$(cutoffs[end])"] = real.(ifft(f̂ .* mask))

    return results
end

function gaussian_filter(field, k, smooth; periodic_x=true, periodic_y=true)
    """
        gaussian_filter(field, k, smooth; periodic_x=true, periodic_y=true)

    Gaussian low-pass filter (width Δ = `smooth`, physical length) applied in the
    horizontal to level `k` of `field`. Transfer function exp(-κ²Δ²/24).

    Non-periodic directions are mirror-padded (half-sample symmetric reflection,
    appropriate for cell-centered data) before the FFT. This is equivalent to a
    zero-gradient condition at those edges and avoids wraparound contamination.
    """
    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field, :, :, k))
    Nx, Ny = size(data)

    # Mirror-pad non-periodic directions
    padded = periodic_x ? data : vcat(data, reverse(data, dims=1))
    padded = periodic_y ? padded : hcat(padded, reverse(padded, dims=2))
    Mx, My = size(padded)

    # Wavenumbers for the padded array
    kx = 2π .* rfftfreq(Mx, 1/Δx)
    ky = 2π .* fftfreq(My, 1/Δy)
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    Ĝ = @. exp(-κ² * smooth^2 / 24)

    f̂ = rfft(padded)
    f̂ .*= Ĝ
    filtered = irfft(f̂, Mx)

    return filtered[1:Nx, 1:Ny]   # drop the mirrored copy
end

# Map each frequency to the band-max value it falls into
function bandmax_color_vals(freqs, band_max, K)
    color_vals = similar(freqs, Float64)
    for (i, κ) in enumerate(freqs)
        if κ <= K[1]
            color_vals[i] = band_max[1]
        elseif κ > K[end]
            color_vals[i] = band_max[end]
        else
            band_idx = findfirst(n -> κ <= K[n+1], 1:length(K)-1) + 1
            color_vals[i] = band_max[band_idx]
        end
    end
    return color_vals
end

function compute_load_pv(pv_cache_file, filehead, fileparam, iteration, dx, dy, dz, α, g, f, filter_scale)
    if isfile(pv_cache_file)
        println("Loading cached PV variables from $pv_cache_file...")
        x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f =
            load(pv_cache_file, "x", "y", "horizontalq_uf", "verticalq_uf", "q_uf",
                "horizontalq_f", "verticalq_f", "q_f")
    else
        # --- Compute gradients for T ---

        output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        x, y, _ = nodes(snapshot[:T])                           # call this one time to get x and y coords

        T_uf = copy(interior(snapshot[:T],:,:,70))              # 2048×2048 Matrix{Float32}
        T_above = copy(interior(snapshot[:T],:,:,71))
        T_below = copy(interior(snapshot[:T],:,:,69))
        dTdx, dTdy = compute_dAdx_dAdy(T_uf,dx,dy) 
        dTdz = compute_dAdz(T_above,T_below,dz)
        println("Computed unfiltered T gradients")
        T_uf, T_above, T_below = nothing, nothing, nothing      # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # T_f = lowpass_filter(snapshot[:T], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        # T_f_above = lowpass_filter(snapshot[:T], 71, filter_scale)
        # T_f_below = lowpass_filter(snapshot[:T], 69, filter_scale)
        T_f = gaussian_filter(snapshot[:T], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        T_f_above = gaussian_filter(snapshot[:T], 71, filter_scale)
        T_f_below = gaussian_filter(snapshot[:T], 69, filter_scale)
        dTdx_f, dTdy_f = compute_dAdx_dAdy(T_f,dx,dy) 
        dTdz_f = compute_dAdz(T_f_above,T_f_below,dz)
        println("Computed filtered T gradients")
        T_f, T_f_above, T_f_below = nothing, nothing, nothing   # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Compute gradients for u ---

        output_filename = filehead * "subdomain_u_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        u_uf = copy(interior(snapshot[:u],:,:,70)[1:end-1, :])        # initially 2049×2048 Matrix{Float32}
        u_above = copy(interior(snapshot[:u],:,:,71)[1:end-1, :])
        u_below = copy(interior(snapshot[:u],:,:,69)[1:end-1, :])
        dudx, dudy = compute_dAdx_dAdy(u_uf,dx,dy) 
        dudz = compute_dAdz(u_above,u_below,dz)
        println("Computed unfiltered u gradients")
        u_uf, u_above, u_below = nothing, nothing, nothing      # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # u_f = lowpass_filter(snapshot[:u], 70, filter_scale)[1:end-1, :]  # initially 2049×2048 Matrix{Float32}
        # u_f_above = lowpass_filter(snapshot[:u], 71, filter_scale)[1:end-1, :]
        # u_f_below = lowpass_filter(snapshot[:u], 69, filter_scale)[1:end-1, :]
        u_f = gaussian_filter(snapshot[:u], 70, filter_scale)[1:end-1, :]  # initially 2049×2048 Matrix{Float32}
        u_f_above = gaussian_filter(snapshot[:u], 71, filter_scale)[1:end-1, :]
        u_f_below = gaussian_filter(snapshot[:u], 69, filter_scale)[1:end-1, :]
        dudx_f, dudy_f = compute_dAdx_dAdy(u_f,dx,dy) 
        dudz_f = compute_dAdz(u_f_above,u_f_below,dz)
        println("Computed filtered u gradients")
        u_f, u_f_above, u_f_below = nothing, nothing, nothing   # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Compute gradients for v ---

        output_filename = filehead * "subdomain_v_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        v_uf = copy(interior(snapshot[:v],:,:,70)[:, 1:end-1])        # initially 2048×2049 Matrix{Float32}
        v_above = copy(interior(snapshot[:v],:,:,71)[:, 1:end-1])
        v_below = copy(interior(snapshot[:v],:,:,69)[:, 1:end-1])
        dvdx, dvdy = compute_dAdx_dAdy(v_uf,dx,dy) 
        dvdz = compute_dAdz(v_above,v_below,dz)
        println("Computed unfiltered v gradients")
        v_uf, v_above, v_below = nothing, nothing, nothing      # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # v_f = lowpass_filter(snapshot[:v], 70, filter_scale)[:, 1:end-1]  # initially 2048×2049 Matrix{Float32}
        # v_f_above = lowpass_filter(snapshot[:v], 71, filter_scale)[:, 1:end-1]
        # v_f_below = lowpass_filter(snapshot[:v], 69, filter_scale)[:, 1:end-1]
        v_f = gaussian_filter(snapshot[:v], 70, filter_scale)[:, 1:end-1]  # initially 2048×2049 Matrix{Float32}
        v_f_above = gaussian_filter(snapshot[:v], 71, filter_scale)[:, 1:end-1]
        v_f_below = gaussian_filter(snapshot[:v], 69, filter_scale)[:, 1:end-1]
        dvdx_f, dvdy_f = compute_dAdx_dAdy(v_f,dx,dy) 
        dvdz_f = compute_dAdz(v_f_above,v_f_below,dz)
        println("Computed filtered v gradients")
        v_f, v_f_above, v_f_below = nothing, nothing, nothing   # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Compute gradients for w ---

        output_filename = filehead * "subdomain_w_" * fileparam * "_iter$(iteration).jld2"      # Load T field
        snapshot = load_subdomain_snapshot(output_filename)

        w_uf = copy(interior(snapshot[:w],:,:,70))                    # 2048×2048 Matrix{Float32}
        w_above = copy(interior(snapshot[:w],:,:,71))
        w_below = copy(interior(snapshot[:w],:,:,69))
        dwdx, dwdy = compute_dAdx_dAdy(w_uf,dx,dy) 
        println("Computed unfiltered w gradients")
        w_uf, w_above, w_below = nothing, nothing, nothing      # free the variable
        GC.gc()                                                 # force the garbage collector to run immediately

        # w_f = lowpass_filter(snapshot[:w], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        # w_f_above = lowpass_filter(snapshot[:w], 71, filter_scale)
        # w_f_below = lowpass_filter(snapshot[:w], 69, filter_scale)
        w_f = gaussian_filter(snapshot[:w], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        w_f_above = gaussian_filter(snapshot[:w], 71, filter_scale)
        w_f_below = gaussian_filter(snapshot[:w], 69, filter_scale)
        dwdx_f, dwdy_f = compute_dAdx_dAdy(w_f,dx,dy) 
        println("Computed filtered w gradients")
        w_f, w_f_above, w_f_below = nothing, nothing, nothing   # free the variable
        snapshot = nothing
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Unfiltered Ertel PV (q_uf) ---

        # Compute unfiltered PV (q_uf)
        horizontalq_uf = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) )      # horizontal q
        verticalq_uf = (α*g) .* ( dTdz .* (dvdx.-dudy.+f) )                        # vertical q
        q_uf = horizontalq_uf + verticalq_uf

        # --- Filtered Ertel PV (q_f) ---

        # Compute filtered PV (q_f)
        horizontalq_f = (α*g) .* ( dTdx_f .* (dwdy_f.-dvdz_f) .+ dTdy_f .* (dudz_f.-dwdx_f) )   # horizontal q
        verticalq_f = (α*g) .* ( dTdz_f .* (dvdx_f.-dudy_f.+f) )                           # vertical q
        q_f = horizontalq_f + verticalq_f

        # --- Clean up memory ---
        dTdx, dTdy, dTdz, dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy = ntuple(i -> nothing, 11)
        dTdx_f, dTdy_f, dTdz_f, dudx_f, dudy_f, dudz_f, dvdx_f, dvdy_f, dvdz_f, dwdx_f, dwdy_f = ntuple(i -> nothing, 11)
        GC.gc()                                                 # force the garbage collector to run immediately

        # --- Save computed PV variables to cache so future runs can skip recomputation ---
        mkpath(filesave)
        jldsave(pv_cache_file; x, y, horizontalq_uf, verticalq_uf, q_uf,
                horizontalq_f, verticalq_f, q_f)
        println("Saved cached PV variables to $pv_cache_file")
    end # if isfile(pv_cache_file)

    return x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f
end 

function compute_load_pvspectra(pvspectra_cache_file, labels, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f)
    if isfile(pvspectra_cache_file)
        println("Loading cached PV spectra variables from $pvspectra_cache_file...")
        S_quf, S_qf, S_horquf, S_verquf, S_horqf, S_verqf =
            load(pvspectra_cache_file, "S_quf", "S_qf", "S_horquf", "S_verquf", "S_horqf", "S_verqf")
    else

        # Compute for spectra and band regimes for q_uf
        S_quf = isotropic_powerspectrum(q_uf, q_uf; Δx=dx, Δy=dy)
        println("Computed spectra of unfiltered q")
        # bands_quf = bandpass_filter(q_uf, dx, dy, cutoffs)      # compute bands
        # println("Computed spectra bands of unfiltered q")

        # Compute for spectra and band regimes for q_f
        S_qf = isotropic_powerspectrum(q_f, q_f; Δx=dx, Δy=dy)
        println("Computed spectra of filtered q")
        # bands_qf = bandpass_filter(q_f, dx, dy, cutoffs)        # compute bands
        # println("Computed spectra bands of filtered q")

        # Compute for spectra of the horizontal and vertical components of q_uf and q_f
        S_horquf = isotropic_powerspectrum(horizontalq_uf, horizontalq_uf; Δx=dx, Δy=dy)
        println("Computed spectra of horizontal component of unfiltered q")
        S_verquf = isotropic_powerspectrum(verticalq_uf, verticalq_uf; Δx=dx, Δy=dy)
        println("Computed spectra of vertical component of unfiltered q")
        S_horqf = isotropic_powerspectrum(horizontalq_f, horizontalq_f; Δx=dx, Δy=dy)
        println("Computed spectra of horizontal component of filtered q")
        S_verqf = isotropic_powerspectrum(verticalq_f, verticalq_f; Δx=dx, Δy=dy)
        println("Computed spectra of vertical component of filtered q")

        E0_horqf = S_horqf.spec[1]      # k_min value of filtered horizontal q spectra (region C's is the cross-region normalization)

        mkpath(filesave)
        jldsave(pvspectra_cache_file; S_quf, bands_quf, S_qf, bands_qf, S_horquf, S_verquf, S_horqf, S_verqf, E0_horqf)
        println("Saved cached PV spectra variables to $pvspectra_cache_file")
    end # if isfile(pvspectra_cache_file)

    return S_quf, S_qf, S_horquf, S_verquf, S_horqf, S_verqf
end

### -------------------------------------------------------------------------
### Hydrostatic (hy) simulation: full periodic domain, sliced to region A/B/C

hy_filehead = "/orcd/data/abodner/002/shared_datasets/hyles_output/"
hy_filename = hy_filehead * "hydrostatic_snapshots_hydrostatic_twin_simulation.jld2"
hy_metadata = hy_filehead * "experiment_hydrostatic_twin_simulation_metadata.jld2"
hy_dx, hy_dy = 156.25, 156.25     # [m] hy horizontal grid spacing
hy_Nx, hy_Ny = 640, 640           # hy full-domain size (periodic)

# Region index windows in the hy full domain (if left > right, the range wraps the periodic boundary)
function hy_region_ranges(region)
    if region == "A"
        xL, xR = 352, 608
        yB, yT = 192, 448
    elseif region == "B"
        xL, xR = 512, 128
        yB, yT = 480, 96
    elseif region == "C"
        xL, xR = 64, 320
        yB, yT = 192, 448
    else
        error("Subdomain not detected: region = \"$region\"; expected \"A\", \"B\", or \"C\".")
    end
    xrange = xL > xR ? vcat(xL:hy_Nx, 1:xR) : (xL:xR)
    yrange = yB > yT ? vcat(yB:hy_Ny, 1:yT) : (yB:yT)
    return xrange, yrange
end

# Forward differences along x (dim 1) and y (dim 2) of a periodic (Nx, Ny) array
function compute_dAdx_dAdy_periodic(A, dx, dy)
    dAdx = (circshift(A, (-1, 0)) .- A) ./ dx
    dAdy = (circshift(A, (0, -1)) .- A) ./ dy
    return dAdx, dAdy
end

function hy_compute_load_pv(pv_cache_file, region, snapshot_number, klev, dx, dy, dz, α, g, f)
    if isfile(pv_cache_file)
        println("Loading cached hy PV variables from $pv_cache_file...")
        x, y, horizontalq_uf, verticalq_uf, q_uf =
            load(pv_cache_file, "x", "y", "horizontalq_uf", "verticalq_uf", "q_uf")
    else
        xrange, yrange = hy_region_ranges(region)       # errors before loading data if region is invalid
        set_value!(; Δh = hy_dx)
        println("Loading hydrostatic data from $hy_filename...")
        snapshots = load_snapshots(hy_filename; metadata = hy_metadata)
        T = snapshots[:T][snapshot_number]
        u = snapshots[:u][snapshot_number]
        v = snapshots[:v][snapshot_number]
        w = snapshots[:w][snapshot_number]
        level(field, k) = Array(interior(field, :, :, k))

        # Horizontal gradients on the full periodic domain at klev; vertical gradients centered on klev
        dTdx, dTdy = compute_dAdx_dAdy_periodic(level(T, klev), dx, dy)
        dTdz = compute_dAdz(level(T, klev+1), level(T, klev-1), dz)
        _, dudy = compute_dAdx_dAdy_periodic(level(u, klev), dx, dy)
        dudz = compute_dAdz(level(u, klev+1), level(u, klev-1), dz)
        dvdx, _ = compute_dAdx_dAdy_periodic(level(v, klev), dx, dy)
        dvdz = compute_dAdz(level(v, klev+1), level(v, klev-1), dz)
        dwdx, dwdy = compute_dAdx_dAdy_periodic(level(w, klev), dx, dy)     # w at face klev, as in the nhy compute_load_pv
        println("Computed unfiltered hy gradients")

        # Unfiltered Ertel PV on the full domain, then sliced to the region
        horizontalq_uf = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) )      # horizontal q
        verticalq_uf = (α*g) .* ( dTdz .* (dvdx.-dudy.+f) )                        # vertical q
        q_uf = horizontalq_uf + verticalq_uf
        horizontalq_uf, verticalq_uf, q_uf = (A[xrange, yrange] for A in (horizontalq_uf, verticalq_uf, q_uf))

        # Monotonic region coordinates (unwrapped across the periodic boundary) for the heatmap
        xT, yT, _ = nodes(T)
        x = xT[first(xrange)] .+ (0:length(xrange)-1) .* dx
        y = yT[first(yrange)] .+ (0:length(yrange)-1) .* dy

        mkpath(filesave)
        jldsave(pv_cache_file; x, y, horizontalq_uf, verticalq_uf, q_uf)
        println("Saved cached hy PV variables to $pv_cache_file")
    end
    return x, y, horizontalq_uf, verticalq_uf, q_uf
end

### -------------------------------------------------------------------------
### Plotting functions

# For 10x10km tiles
function plot_filteredPV(snapshot, fileparam)
    filter_scale = 300                                      # meters, bound between submesoscale and BLT

    # filtered
    T_f = lowpass_filter(snapshot[:T], 70, filter_scale)                    # 2048×2048 Matrix{Float32}
    u_f = lowpass_filter(snapshot[:u], 70, filter_scale)[1:end-1, :]        # initially 2049×2048 Matrix{Float32}
    v_f = lowpass_filter(snapshot[:v], 70, filter_scale)[:, 1:end-1]        # initially 2048×2049 Matrix{Float32}
    w_f = lowpass_filter(snapshot[:w], 70, filter_scale)                    # 2048×2048 Matrix{Float32}
    T_f_above = lowpass_filter(snapshot[:T], 71, filter_scale)
    u_f_above = lowpass_filter(snapshot[:u], 71, filter_scale)[1:end-1, :]
    v_f_above = lowpass_filter(snapshot[:v], 71, filter_scale)[:, 1:end-1]
    w_f_above = lowpass_filter(snapshot[:w], 71, filter_scale)
    T_f_below = lowpass_filter(snapshot[:T], 69, filter_scale)
    u_f_below = lowpass_filter(snapshot[:u], 69, filter_scale)[1:end-1, :]
    v_f_below = lowpass_filter(snapshot[:v], 69, filter_scale)[:, 1:end-1]
    w_f_below = lowpass_filter(snapshot[:w], 69, filter_scale)

    dTdx, dTdy = compute_dAdx_dAdy(T_f,dx,dy) 
    dudx, dudy = compute_dAdx_dAdy(u_f,dx,dy) 
    dvdx, dvdy = compute_dAdx_dAdy(v_f,dx,dy) 
    dwdx, dwdy = compute_dAdx_dAdy(w_f,dx,dy) 
    dTdz = compute_dAdz(T_f_above,T_f_below,dz)
    dudz = compute_dAdz(u_f_above,u_f_below,dz)
    dvdz = compute_dAdz(v_f_above,v_f_below,dz)

    q_f = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) .+ dTdz .* (dvdx.-dudy.+f) ) 

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_f;
                rasterize = true, colormap = colormap, colorrange = (-4e-6, 4e-6))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * fileparam * "_q_f.png", fig; px_per_unit=4)

    # unfiltered! 
    T_uf = copy(interior(snapshot[:T],:,:,70))                    # 2048×2048 Matrix{Float32}
    u_uf = copy(interior(snapshot[:u],:,:,70)[1:end-1, :])        # initially 2049×2048 Matrix{Float32}
    v_uf = copy(interior(snapshot[:v],:,:,70)[:, 1:end-1])        # initially 2048×2049 Matrix{Float32}
    w_uf = copy(interior(snapshot[:w],:,:,70))                    # 2048×2048 Matrix{Float32}
    T_above = copy(interior(snapshot[:T],:,:,71))
    u_above = copy(interior(snapshot[:u],:,:,71)[1:end-1, :])
    v_above = copy(interior(snapshot[:v],:,:,71)[:, 1:end-1])
    w_above = copy(interior(snapshot[:w],:,:,71))
    T_below = copy(interior(snapshot[:T],:,:,69))
    u_below = copy(interior(snapshot[:u],:,:,69)[1:end-1, :])
    v_below = copy(interior(snapshot[:v],:,:,69)[:, 1:end-1])
    w_below = copy(interior(snapshot[:w],:,:,69))

    dTdx, dTdy = compute_dAdx_dAdy(T_uf,dx,dy) 
    dudx, dudy = compute_dAdx_dAdy(u_uf,dx,dy) 
    dvdx, dvdy = compute_dAdx_dAdy(v_uf,dx,dy) 
    dwdx, dwdy = compute_dAdx_dAdy(w_uf,dx,dy) 
    dTdz = compute_dAdz(T_above,T_below,dz)
    dudz = compute_dAdz(u_above,u_below,dz)
    dvdz = compute_dAdz(v_above,v_below,dz)

    q_uf = (α*g) .* ( dTdx .* (dwdy.-dvdz) .+ dTdy .* (dudz.-dwdx) .+ dTdz .* (dvdx.-dudy.+f) ) 

    # quick plot 
    x, y, _ = nodes(snapshot[:T])
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_uf;
                rasterize = true, colormap = colormap, colorrange = (-4e-6, 4e-6))
    Colorbar(fig[1, 2], hm)             # if you want colorbar
    save(filesave * fileparam * "_q_uf.png", fig; px_per_unit=4)

    # # filtered vs. unfiltered T 
    # # quick plot 
    # x, y, _ = nodes(snapshot[:T])
    # fig = Figure(size = (700, 640))     # if you want colorbar
    # ax = Axis(fig[1, 1]; aspect = DataAspect())
    # colormap = :balance
    # hm = heatmap!(ax, 1e-3x, 1e-3y, T_f;
    #             rasterize = true, colormap = colormap, colorrange = (19.5, 20.1))
    # Colorbar(fig[1, 2], hm)             # if you want colorbar
    # save(filesave * "T_f.png", fig; px_per_unit=4)

    # # quick plot 
    # x, y, _ = nodes(snapshot[:T])
    # fig = Figure(size = (700, 640))     # if you want colorbar
    # ax = Axis(fig[1, 1]; aspect = DataAspect())
    # colormap = :balance
    # hm = heatmap!(ax, 1e-3x, 1e-3y, T_uf;
    #             rasterize = true, colormap = colormap, colorrange = (19.5, 20.1))
    # Colorbar(fig[1, 2], hm)             # if you want colorbar
    # save(filesave * "T_uf.png", fig; px_per_unit=4)
end

function plot_fieldspectra(x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f; crange_static=(-5e-9,5e-9))
    # --- Unfiltered fields ---

    # Plot of q_uf field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(),
        title=fileparam * ": Unfiltered Ertel PV (min $(@sprintf("%.1e", minimum(q_uf))), max $(@sprintf("%.1e", maximum(q_uf))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_uf;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"q_{uf}")             # if you want colorbar
    save(filesave * fileparam * "_q_uf.png", fig; px_per_unit=4)
    println("Saved figure of unfiltered q")

    # Plot of horizontalq_uf field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(),
        title=fileparam * ": Unfiltered horizontal q term (min $(@sprintf("%.1e", minimum(horizontalq_uf))), max $(@sprintf("%.1e", maximum(horizontalq_uf))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, horizontalq_uf;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"horizontal q")             # if you want colorbar
    save(filesave * fileparam * "_horizontalq_uf.png", fig; px_per_unit=4)
    println("Saved figure of unfiltered horizontal q term")

    # Plot of verticalq_uf field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(),
        title=fileparam * ": Unfiltered vertical q term (min $(@sprintf("%.1e", minimum(verticalq_uf))), max $(@sprintf("%.1e", maximum(verticalq_uf))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, verticalq_uf;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"vertical q")             # if you want colorbar
    save(filesave * fileparam * "_verticalq_uf.png", fig; px_per_unit=4)
    println("Saved figure of unfiltered vertical q term")

    # --- Filtered fields ---

    # Plot of q_f field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), 
        title=fileparam * ": Filtered Ertel PV (min $(@sprintf("%.1e", minimum(q_f))), max $(@sprintf("%.1e", maximum(q_f))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, q_f;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"q_f")             # if you want colorbar
    save(filesave * fileparam * "_q_f.png", fig; px_per_unit=4)
    println("Saved figure of filtered q")

    # Plot of horizontalq_f field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), 
        title=fileparam * ": Filtered horizontal q term (min $(@sprintf("%.1e", minimum(horizontalq_f))), max $(@sprintf("%.1e", maximum(horizontalq_f))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, horizontalq_f;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"horizontal q")             # if you want colorbar
    save(filesave * fileparam * "_horizontalq_f.png", fig; px_per_unit=4)
    println("Saved figure of filtered horizontal q term")

    # Plot of verticalq_f field
    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect(), 
        title=fileparam * ": Filtered vertical q term (min $(@sprintf("%.1e", minimum(verticalq_f))), max $(@sprintf("%.1e", maximum(verticalq_f))))")
    colormap = :balance
    hm = heatmap!(ax, 1e-3x, 1e-3y, verticalq_f;
                rasterize = true, colormap = colormap, colorrange = crange_static)
    Colorbar(fig[1, 2], hm, label=L"vertical q")             # if you want colorbar
    save(filesave * fileparam * "_verticalq_f.png", fig; px_per_unit=4)
    println("Saved figure of filtered vertical q term")
end

function plot_spectra_colormag(filesave, fileparam, labels, K, S_quf, bands_quf, S_qf, bands_qf)
    # --- Compute frequencies and plot spectra for unfiltered q --- 

    freqs_quf = collect(S_quf.freq)
    global S0 = S_quf
    spec_normalized_quf = Real.(S_quf.spec ./ S0.spec[1])

    band_max_quf = [maximum(abs, bands_quf[label]) for label in labels]
    color_vals_quf = bandmax_color_vals(freqs_quf, band_max_quf, K)

    crange = extrema(color_vals_quf)        # Lock both lines to q_uf's color limits

    # Plot spectra colored by max(|q_uf|) in each band regime
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E_{q}(k)/E_{q} (k_{min})",
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title=L"q_{uf}, z=-2.8125 m", axis_kwargs1...)

    lp = lines!(ax, freqs_quf, spec_normalized_quf, color = color_vals_quf,
                colormap = :amp, colorrange = crange, linewidth = 2)

    Colorbar(fig[1,2], lp, label = L"\max {|q|} \text{ in band}")
    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * fileparam * "_spectracolor_quf.png", fig)
    println("Saved figure of q_uf spectra colored by frontal sharpness")

    # Free up memory
    S_quf, freqs_quf, spec_normalized_quf, bands_quf, band_max_quf, color_vals_quf = ntuple(i -> nothing, 6)
    GC.gc()                                                 # force the garbage collector to run immediately

    # --- Compute frequencies and plot spectra for filtered q --- 

    freqs_qf = collect(S_qf.freq)
    spec_normalized_qf = Real.(S_qf.spec ./ S0.spec[1])

    band_max_qf  = [maximum(abs, bands_qf[label])  for label in labels]
    color_vals_qf  = bandmax_color_vals(freqs_qf,  band_max_qf,  K)

    # Plot spectra colored by max(|q_f|) in each band regime
    fig = Figure(size = (600, 500))
    axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E_{q}(k)/E_{q} (k_{min})",
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
    ax = Axis(fig[1, 1]; title=L"q_{f}, z=-2.8125 m", axis_kwargs1...)

    lp2 = lines!(ax, freqs_qf, spec_normalized_qf, color = color_vals_qf,
                colormap = :amp, colorrange = crange, linewidth = 2, linestyle = :dash)

    Colorbar(fig[1,2], lp, label = L"\max {|q|} \text{ in band}")
    xlims!(ax, (10^-4.5, 10^0.5))
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    save(filesave * fileparam * "_spectracolor_qf.png", fig)
    println("Saved figure of q_f spectra colored by frontal sharpness")

    # Free up memory
    S_qf, freqs_qf, spec_normalized_qf, bands_qf, band_max_qf, color_vals_qf = ntuple(i -> nothing, 6)
    GC.gc()                                                 # force the garbage collector to run immediately
end

function plot_spectra_components(filesave, fileparam, S_quf, S_horquf, S_verquf, S_qf, S_horqf, S_verqf)
    S0 = S_quf.spec[1]          # normalize everything by unfiltered total q at k_min

    axis_kwargs = (xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E(k)/E_{q} (k_{min})",
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))

    # --- Plot spectra of total, horizontal, and vertical components of unfiltered q ---
    fig = Figure(size = (600, 500))
    # ax = Axis(fig[1, 1]; title=L"q_{uf}, z=-2.8125 m", axis_kwargs...)
    ax = Axis(fig[1, 1]; title=L"q, z=-2.8125 m", axis_kwargs...)

    # Shaded wavenumber bands (drawn first so they sit behind the lines)
    k_10km  = 2π / 10000     # λ = 10 km
    k_300m  = 2π / 300       # λ = 300 m
    k_right = 10^0.5         # right edge of the x-limits
    vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))   # light blue: k > 2π/10000
    vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))  # darker: k > 2π/300 (layers on top of the light band)

    lines!(ax, collect(S_quf.freq), Real.(S_quf.spec ./ S0); color = :purple, linestyle=:dash, linewidth = 2, label = L"q, unfiltered")
    lines!(ax, collect(S_horquf.freq), Real.(S_horquf.spec ./ S0); color = :red, linestyle=:dash, linewidth = 2, label = L"q_{h}, unfiltered")
    lines!(ax, collect(S_verquf.freq), Real.(S_verquf.spec ./ S0); color = :blue, linestyle=:dash, linewidth = 2, label = L"q_{v}, unfiltered")
    # # vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    # axislegend(ax; position = :lb)
    # save(filesave * fileparam * "_spectracomp_quf.png", fig)
    # println("Saved figure of q_uf component spectra")

    # # --- Plot spectra of total, horizontal, and vertical components of filtered q ---
    # fig = Figure(size = (600, 500))
    # ax = Axis(fig[1, 1]; title=L"q_{f}, z=-2.8125 m", axis_kwargs...)
    
    # # Shaded wavenumber bands (drawn first so they sit behind the lines)
    # k_10km  = 2π / 10000     # λ = 10 km
    # k_300m  = 2π / 300       # λ = 300 m
    # k_right = 10^0.5         # right edge of the x-limits
    # vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))   # light blue: k > 2π/10000
    # vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))  # darker: k > 2π/300 (layers on top of the light band)

    lines!(ax, collect(S_qf.freq), Real.(S_qf.spec ./ S0); color = :purple, linewidth = 2, label = L"q, filtered")
    lines!(ax, collect(S_horqf.freq), Real.(S_horqf.spec ./ S0); color = :red, linewidth = 2, label = L"q_{h}, filtered")
    lines!(ax, collect(S_verqf.freq), Real.(S_verqf.spec ./ S0); color = :blue, linewidth = 2, label = L"q_{v}, filtered")
    limits!(ax, (10^-4, 10^0), (1e-4, 1e2))
    # vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
    axislegend(ax; position = :lb)
    # save(filesave * fileparam * "_spectracomp_qf.png", fig)
    # println("Saved figure of q_f component spectra")

    save(filesave * fileparam * "_spectracomp_q.png", fig)
    println("Saved figure of q component spectra")
end

function plot_spectra_regions(filesave, iteration; regions = ["A", "B", "C"])
    colors = Makie.wong_colors()[1:3]           # one color per region
    spectra = Dict(region => load(filesave * "region$(region)_iter$(iteration)_pvspectra.jld2",
                                "S_quf", "S_qf", "S_horquf", "S_horqf", "S_verquf", "S_verqf")
                   for region in regions)       # only the spectra, not the large bands

    S0 = spectra["C"][4].spec[1]                # normalize everything by filtered horizontal q of region C at k_min

    # (index of unfiltered, index of filtered, title, file suffix)
    components = [(1, 2, L"q", "q"),
                  (3, 4, L"q_{h}", "horq"),
                  (5, 6, L"q_{v}", "verq")]

    for (iuf, ifilt, ttl, suffix) in components
        fig = Figure(size = (600, 500))
        ax = Axis(fig[1, 1]; title = ttl, xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E(k)/E_{q_{h,f}}^{C} (k_{min})",
                xscale = log10, yscale = log10)

        # Shaded wavenumber bands (drawn first so they sit behind the lines)
        k_10km  = 2π / 10000     # λ = 10 km
        k_300m  = 2π / 300       # λ = 300 m
        k_right = 10^0.5         # right edge of the x-limits
        vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))   # light blue: k > 2π/10000
        vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))  # darker: k > 2π/300 (layers on top of the light band)

        for (n, region) in enumerate(regions)
            S_uf, S_filt = spectra[region][iuf], spectra[region][ifilt]
            lines!(ax, collect(S_uf.freq), Real.(S_uf.spec ./ S0); color = colors[n], linewidth = 2,
                   linestyle = :dash, label = "Region $(region), unfiltered")
            lines!(ax, collect(S_filt.freq), Real.(S_filt.spec ./ S0); color = colors[n], linewidth = 2,
                   label = "Region $(region), filtered")
        end
        limits!(ax, (10^-4, 10^0), (1e-4, 1e2))
        # vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
        axislegend(ax; position = :lb)
        save(filesave * "regionABC_iter$(iteration)_spectra_$(suffix).png", fig)
        println("Saved figure of $(suffix) spectra for regions $(join(regions, ", "))")
    end
end

## Heatmaps of unfiltered hy q, horizontal q, and vertical q for a region
function plot_fieldspectra_hy(x, y, horizontalq_uf, verticalq_uf, q_uf, region, snapshot_number; crange_static=(-5e-9,5e-9))
    tag = "hy_region$(region)_snap$(snapshot_number)"
    for (field, name, ttl, lbl) in [(q_uf,           "q_uf",           "Unfiltered Ertel PV",          L"q_{uf}"),
                                    (horizontalq_uf, "horizontalq_uf", "Unfiltered horizontal q term", L"horizontal q"),
                                    (verticalq_uf,   "verticalq_uf",   "Unfiltered vertical q term",   L"vertical q")]
        fig = Figure(size = (700, 640))
        ax = Axis(fig[1, 1]; aspect = DataAspect(),
            title="hy region $(region): $(ttl) (min $(@sprintf("%.1e", minimum(field))), max $(@sprintf("%.1e", maximum(field))))")
        hm = heatmap!(ax, 1e-3x, 1e-3y, field;
                    rasterize = true, colormap = :balance, colorrange = crange_static)
        Colorbar(fig[1, 2], hm, label=lbl)
        save(filesave * tag * "_" * name * ".png", fig; px_per_unit=4)
        println("Saved figure of hy $(name)")
    end
end

## Spectra of unfiltered hy q and its horizontal / vertical components for a region
function plot_spectra_components_hy(horizontalq_uf, verticalq_uf, q_uf, region, snapshot_number, dx, dy)
    S_quf    = isotropic_powerspectrum(q_uf, q_uf; Δx=dx, Δy=dy)
    S_horquf = isotropic_powerspectrum(horizontalq_uf, horizontalq_uf; Δx=dx, Δy=dy)
    S_verquf = isotropic_powerspectrum(verticalq_uf, verticalq_uf; Δx=dx, Δy=dy)
    S0 = S_quf.spec[1]          # normalize everything by unfiltered total q at k_min

    fig = Figure(size = (600, 500))
    ax = Axis(fig[1, 1]; title=L"q, z=-2.8125 m", xlabel = "Wavenumber (rad⋅m⁻¹)",
            ylabel = L"E(k)/E_{q} (k_{min})", xscale = log10, yscale = log10)

    # Shaded wavenumber bands (drawn first so they sit behind the lines)
    k_10km  = 2π / 10000     # λ = 10 km
    k_300m  = 2π / 300       # λ = 300 m
    k_right = 10^0.5         # right edge of the x-limits
    vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))
    vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))

    lines!(ax, collect(S_quf.freq), Real.(S_quf.spec ./ S0); color = :purple, linestyle=:dash, linewidth = 2, label = L"q, unfiltered")
    lines!(ax, collect(S_horquf.freq), Real.(S_horquf.spec ./ S0); color = :red, linestyle=:dash, linewidth = 2, label = L"q_{h}, unfiltered")
    lines!(ax, collect(S_verquf.freq), Real.(S_verquf.spec ./ S0); color = :blue, linestyle=:dash, linewidth = 2, label = L"q_{v}, unfiltered")
    limits!(ax, (10^-4, 10^0), (1e-4, 1e2))
    axislegend(ax; position = :lb)

    save(filesave * "hy_region$(region)_snap$(snapshot_number)_spectracomp_q.png", fig)
    println("Saved figure of hy q component spectra")
end

## Compare unfiltered nhy, filtered nhy, and hy PV spectra (total, horizontal, vertical) across regions A, B, C
function plot_spectra_regions_hybrid(filesave, iteration, hy_snapshot, hy_klev; regions = ["A", "B", "C"])
    nhy_files = Dict(region => filesave * "region$(region)_iter$(iteration)_pvspectra.jld2" for region in regions)
    missing_files = [file for file in values(nhy_files) if !isfile(file)]
    isempty(missing_files) || error("Missing nhy PV spectra caches (run the nhy path for these regions first): $(join(missing_files, ", "))")

    colors = Makie.wong_colors()[1:length(regions)]     # one color per region
    nhy_spectra = Dict(region => load(nhy_files[region],
                                    "S_quf", "S_qf", "S_horquf", "S_horqf", "S_verquf", "S_verqf")
                       for region in regions)           # only the spectra, not the large bands

    hy_spectra = Dict{String, Any}()
    for region in regions
        pv_cache_file = filesave * "hy_region$(region)_snap$(hy_snapshot)_k$(hy_klev)_pv.jld2"
        _, _, horizontalq_uf, verticalq_uf, q_uf = hy_compute_load_pv(
            pv_cache_file, region, hy_snapshot, hy_klev, hy_dx, hy_dy, dz, α, g, f)
        hy_spectra[region] = (isotropic_powerspectrum(q_uf, q_uf; Δx=hy_dx, Δy=hy_dy),
                              isotropic_powerspectrum(horizontalq_uf, horizontalq_uf; Δx=hy_dx, Δy=hy_dy),
                              isotropic_powerspectrum(verticalq_uf, verticalq_uf; Δx=hy_dx, Δy=hy_dy))
    end

    S0 = nhy_spectra["C"][4].spec[1]            # normalize everything by filtered horizontal q of region C at k_min

    # (nhy unfiltered index, nhy filtered index, hy index, title, file suffix)
    components = [(1, 2, 1, L"q", "q"),
                  (3, 4, 2, L"q_{h}", "horq"),
                  (5, 6, 3, L"q_{v}", "verq")]

    for (iuf, ifilt, ihy, ttl, suffix) in components
        fig = Figure(size = (750, 500))
        ax = Axis(fig[1, 1]; title = ttl, xlabel = "Wavenumber (rad⋅m⁻¹)",
                ylabel = L"E(k)/E_{q_{h,f}}^{C} (k_{min})",
                xscale = log10, yscale = log10)

        # Shaded wavenumber bands (drawn first so they sit behind the lines)
        k_10km  = 2π / 10000     # λ = 10 km
        k_300m  = 2π / 300       # λ = 300 m
        k_right = 10^0.5         # right edge of the x-limits
        vspan!(ax, k_10km, k_right, color = (:lightblue, 0.15))
        vspan!(ax, k_300m, k_right, color = (:steelblue, 0.15))

        for (n, region) in enumerate(regions)
            S_uf, S_filt = nhy_spectra[region][iuf], nhy_spectra[region][ifilt]
            S_hy = hy_spectra[region][ihy]
            lines!(ax, collect(S_uf.freq), Real.(S_uf.spec ./ S0); color = colors[n], linewidth = 2,
                   linestyle = :dash, label = "Region $(region), nhy unfiltered")
            lines!(ax, collect(S_filt.freq), Real.(S_filt.spec ./ S0); color = colors[n], linewidth = 2,
                   label = "Region $(region), nhy filtered")
            lines!(ax, collect(S_hy.freq), Real.(S_hy.spec ./ S0); color = colors[n], linewidth = 2,
                   linestyle = :dot, label = "Region $(region), hy")
        end
        limits!(ax, (10^-4, 10^0), (1e-4, 1e2))
        # axislegend(ax; position = :lb, labelsize = 9)
        Legend(fig[1, 2], ax; labelsize = 9)
        save(filesave * "regionABC_iter$(iteration)_snap$(hy_snapshot)_spectra_hybrid_$(suffix).png", fig)
        println("Saved figure of hy/nhy $(suffix) spectra for regions $(join(regions, ", "))")
    end
end

### -------------------------------------------------------------------------
## Plot the heatmap plots of each subdomain tile

# ! ! ! ! ! EDIT THIS ONE TO SWITCH SIMULATION ! ! ! ! !
sim = "both"     # "hy", "nhy", or "both"
println("Simulation: $(sim)")

# Region A/B/C
region = "C"
println("--- Region $(region) ---")

if sim == "nhy"
    # 1. Define subdomain

    # Identify iteration (which denotes timestep)
    iteration = 164410
    println("Iteration: $(iteration)")

    # # 10km x 10km Tiles
    # i = 97
    # println("--- Subdomain $(i) ---")
    # fileparam = "subdomain" * string(i)
    # output_filename = filehead * fileparam * "_iter$(iteration).jld2"
    # snapshot = load_subdomain_snapshot(output_filename)

    fileparam = "region" * string(region)

    # ---

    # # 2a. Compute filtered potential vorticity (for 10x10km tiles where snapshot contains [:T,:u,:v,:w])
    # plot_filteredPV(snapshot, fileparam)

    # 2b. Load or compute PV variables

    filter_scale = 100                                      # meters, bound between submesoscale and BLT

    # # Cache of computed PV variables (x, y, horizontalq/verticalq/q, filtered & unfiltered)
    # pv_cache_file = filesave * fileparam * "_iter$(iteration)_pv.jld2"
    # # Load PV cache file if it exists, or compute PV
    # x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f = compute_load_pv(
    #             pv_cache_file, filehead, fileparam, iteration, dx, dy, dz, α, g, f, filter_scale)

    # # Plot field and spectra of region
    # plot_fieldspectra(x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f)

    # # Clear variables - TEMPORARY SINCE I KNOW PVSPECTRA CACHE EXISTS
    # horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f = ntuple(i -> nothing, 6)
    # GC.gc()                                                 # force the garbage collector to run immediately

    # ---

    # 3. Compute spectra for unfiltered and filtered q

    # Define cutoffs
    cutoffs = 10 .^ range(4, 0, length=100)                 # 100 log-spaced values from 1e4 down to 1e0
    K = 2π ./ cutoffs                                       # band edge wavenumbers, increasing
    labels = ["κ ≤ 2π/$(cutoffs[1])",
            ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
            "κ > 2π/$(cutoffs[end])"]

    # Cache of computed PV spectra/bands (filtered & unfiltered)
    pvspectra_cache_file = filesave * fileparam * "_iter$(iteration)_pvspectra.jld2"
    # # Load PV spectra cache file if it exists, or compute PV spectra
    # S_quf, S_qf, S_horquf, S_verquf, S_horqf, S_verqf = compute_load_pvspectra(
    #                     pvspectra_cache_file, labels, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f)

    # TEMPORARY SINCE I KNOW THE CACHE EXISTS
    println("Loading cached PV spectra variables from $pvspectra_cache_file...")
    S_quf, S_qf, S_horquf, S_verquf, S_horqf, S_verqf =
        load(pvspectra_cache_file, "S_quf", "S_qf", "S_horquf", "S_verquf", "S_horqf", "S_verqf")

    # # ---

    # # 4a. Color spectra plot according to max |q_uf| or max |q_f| by evaluating 100 band regimes
    # plot_spectra_colormag(filesave, fileparam, labels, K, S_quf, bands_quf, S_qf, bands_qf)

    # # ===== TEMPORARY: compute PV + spectra caches for regions A, B, C (delete after use) =====
    # for reg in ["A", "B", "C"]
    #     println("--- [temp] Region $(reg) ---")
    #     reg_param = "region" * reg
    #     reg_pv = compute_load_pv(filesave * reg_param * "_iter$(iteration)_pv.jld2",
    #                              filehead, reg_param, iteration, dx, dy, dz, α, g, f, filter_scale)
    #     _, _, hq_uf, vq_uf, tq_uf, hq_f, vq_f, tq_f = reg_pv
    #     compute_load_pvspectra(filesave * reg_param * "_iter$(iteration)_pvspectra.jld2",
    #                            labels, hq_uf, vq_uf, tq_uf, hq_f, vq_f, tq_f)
    #     reg_pv = hq_uf = vq_uf = tq_uf = hq_f = vq_f = tq_f = nothing   # free the 2048×2048 fields
    #     GC.gc()
    # end
    # # ===== END TEMPORARY =====

    # 4b. Plot spectra of total, horizontal, and vertical PV components
    plot_spectra_components(filesave, fileparam, S_quf, S_horquf, S_verquf, S_qf, S_horqf, S_verqf)

    # # 4c. Compare total/horizontal/vertical PV spectra across regions A, B, C
    # if all(region -> isfile(filesave * "region$(region)_iter$(iteration)_pvspectra.jld2"), ["A", "B", "C"])
    #     plot_spectra_regions(filesave, iteration)
    # else
    #     println("Skipping region comparison: not all region A/B/C spectra caches exist yet")
    # end

elseif sim == "hy"
    hy_snapshot = 121       # day 7.5 (matches nhy iteration 164410)
    hy_klev = 222           # z = -2.8125 m in the hy grid (cf. visualize_spectra_hybrid.jl)
    pv_cache_file = filesave * "hy_region$(region)_snap$(hy_snapshot)_k$(hy_klev)_pv.jld2"
    x, y, horizontalq_uf, verticalq_uf, q_uf = hy_compute_load_pv(
        pv_cache_file, region, hy_snapshot, hy_klev, hy_dx, hy_dy, dz, α, g, f)
    plot_fieldspectra_hy(x, y, horizontalq_uf, verticalq_uf, q_uf, region, hy_snapshot)
    plot_spectra_components_hy(horizontalq_uf, verticalq_uf, q_uf, region, hy_snapshot, hy_dx, hy_dy)

elseif sim == "both"
    nhy_iteration = 164410  # nhy day 7.5 (not `iteration`, which is imported from LESStudySetup/Oceananigans)
    hy_snapshot = 121       # hy day 7.5
    hy_klev = 222           # z = -2.8125 m in the hy grid
    plot_spectra_regions_hybrid(filesave, nhy_iteration, hy_snapshot, hy_klev)

else
    error("Unknown sim: $sim. Expected \"hy\", \"nhy\", or \"both\".")
end

