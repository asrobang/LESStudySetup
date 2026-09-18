using LESStudySetup
using CairoMakie
using Printf
using FFTW
using Oceananigans.Grids: xnodes, ynodes, znodes, xspacings, yspacings
using Oceananigans.Fields: interior
using LESStudySetup.Diagnostics: load_subdomain_snapshot, isotropic_powerspectrum
using MathTeXEngine
using JLD2 #, CUDA
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20260913_regionABC_figs/"

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

"""
    filter(field, smooth)

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
function filter(field, k, smooth)
    grid = field.grid
    Δx = minimum(xspacings(grid, Center()))
    Δy = minimum(yspacings(grid, Center()))

    data = Array(interior(field,:,:,k))  # bring to CPU physical-space array
    Nx, Ny = size(data, 1), size(data, 2)

    # Angular wavenumbers associated with the horizontal FFT
    kx = 2π .* fftfreq(Nx, 1/dx)
    ky = 2π .* fftfreq(Ny, 1/dy)

    K = 2π / smooth          # cutoff wavenumber for this smoothing scale
    κ² = (kx .^ 2) .+ (ky' .^ 2)
    mask = κ² .<= K^2        # true where κ ≤ K (keep), false where κ > K (zero)

    filtered = similar(data)

    f̂ = fft(data)
    f̂ .*= mask
    filtered .= real.(ifft(f̂))

    return filtered
end


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
function bandpass_filter(field, dx, dy, cutoffs)
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

        T_f = filter(snapshot[:T], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        T_f_above = filter(snapshot[:T], 71, filter_scale)
        T_f_below = filter(snapshot[:T], 69, filter_scale)
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

        u_f = filter(snapshot[:u], 70, filter_scale)[1:end-1, :]  # initially 2049×2048 Matrix{Float32}
        u_f_above = filter(snapshot[:u], 71, filter_scale)[1:end-1, :]
        u_f_below = filter(snapshot[:u], 69, filter_scale)[1:end-1, :]
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

        v_f = filter(snapshot[:v], 70, filter_scale)[:, 1:end-1]  # initially 2048×2049 Matrix{Float32}
        v_f_above = filter(snapshot[:v], 71, filter_scale)[:, 1:end-1]
        v_f_below = filter(snapshot[:v], 69, filter_scale)[:, 1:end-1]
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

        w_f = filter(snapshot[:w], 70, filter_scale)            # 2048×2048 Matrix{Float32}
        w_f_above = filter(snapshot[:w], 71, filter_scale)
        w_f_below = filter(snapshot[:w], 69, filter_scale)
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

function plot_filteredPV(snapshot, fileparam)
    filter_scale = 300                                      # meters, bound between submesoscale and BLT

    # filtered
    T_f = filter(snapshot[:T], 70, filter_scale)                    # 2048×2048 Matrix{Float32}
    u_f = filter(snapshot[:u], 70, filter_scale)[1:end-1, :]        # initially 2049×2048 Matrix{Float32}
    v_f = filter(snapshot[:v], 70, filter_scale)[:, 1:end-1]        # initially 2048×2049 Matrix{Float32}
    w_f = filter(snapshot[:w], 70, filter_scale)                    # 2048×2048 Matrix{Float32}
    T_f_above = filter(snapshot[:T], 71, filter_scale)
    u_f_above = filter(snapshot[:u], 71, filter_scale)[1:end-1, :]
    v_f_above = filter(snapshot[:v], 71, filter_scale)[:, 1:end-1]
    w_f_above = filter(snapshot[:w], 71, filter_scale)
    T_f_below = filter(snapshot[:T], 69, filter_scale)
    u_f_below = filter(snapshot[:u], 69, filter_scale)[1:end-1, :]
    v_f_below = filter(snapshot[:v], 69, filter_scale)[:, 1:end-1]
    w_f_below = filter(snapshot[:w], 69, filter_scale)

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

### -------------------------------------------------------------------------
## Plot the heatmap plots of each subdomain tile

# 1. Define subdomain

# Identify iteration (which denotes timestep)
iteration = 164410
println("Iteration: $(iteration)")

# # 10km x 10km Tiles
# i = 97
# println("--- Subdomain $(i) ---")
# fileparam = "subdomain" * string(i)
# output_filename = filehead * fileparam * "_iter$(iteration).jld2"

# Region A/B/C
region = "C"
println("--- Region $(region) ---")
fileparam = "region" * string(region)

# ---

# # 2. Load the snapshot
# output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"      # Load T field
# snapshot = load_subdomain_snapshot(output_filename)

# ---

# # 3. Compute filtered potential vorticity (for 10x10km tiles where snapshot contains [:T,:u,:v,:w])
# plot_filteredPV(snapshot, fileparam)

# --- Cache of computed PV variables (x, y, horizontalq/verticalq/q, filtered & unfiltered) ---
pv_cache_file = filesave * fileparam * "_iter$(iteration)_pv.jld2"

filter_scale = 300                                      # meters, bound between submesoscale and BLT

# Load PV cache file if it exists, or compute PV
x, y, horizontalq_uf, verticalq_uf, q_uf, horizontalq_f, verticalq_f, q_f = compute_load_pv(
            pv_cache_file, filehead, fileparam, iteration, dx, dy, dz, α, g, f, filter_scale)

# Plot of q_uf field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(),
    title=fileparam * ": Unfiltered Ertel PV (min $(@sprintf("%.1e", minimum(q_uf))), max $(@sprintf("%.1e", maximum(q_uf))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, q_uf;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"q_{uf}")             # if you want colorbar
save(filesave * fileparam * "_q_uf.png", fig; px_per_unit=4)
println("Saved figure of unfiltered q")

# Plot of horizontalq_uf field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(),
    title=fileparam * ": Unfiltered horizontal q term (min $(@sprintf("%.1e", minimum(horizontalq_uf))), max $(@sprintf("%.1e", maximum(horizontalq_uf))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, horizontalq_uf;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"horizontal q")             # if you want colorbar
save(filesave * fileparam * "_horizontalq_uf.png", fig; px_per_unit=4)
println("Saved figure of unfiltered horizontal q term")

# Plot of verticalq_uf field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(),
    title=fileparam * ": Unfiltered vertical q term (min $(@sprintf("%.1e", minimum(verticalq_uf))), max $(@sprintf("%.1e", maximum(verticalq_uf))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, verticalq_uf;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"vertical q")             # if you want colorbar
save(filesave * fileparam * "_verticalq_uf.png", fig; px_per_unit=4)
println("Saved figure of unfiltered vertical q term")

# Plot of q_f field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(), 
    title=fileparam * ": Filtered Ertel PV (min $(@sprintf("%.1e", minimum(q_f))), max $(@sprintf("%.1e", maximum(q_f))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, q_f;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"q_f")             # if you want colorbar
save(filesave * fileparam * "_q_f.png", fig; px_per_unit=4)
println("Saved figure of filtered q")

# Plot of horizontalq_f field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(), 
    title=fileparam * ": Filtered horizontal q term (min $(@sprintf("%.1e", minimum(horizontalq_f))), max $(@sprintf("%.1e", maximum(horizontalq_f))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, horizontalq_f;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"horizontal q")             # if you want colorbar
save(filesave * fileparam * "_horizontalq_f.png", fig; px_per_unit=4)
println("Saved figure of filtered horizontal q term")

# Plot of verticalq_f field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(), 
    title=fileparam * ": Filtered vertical q term (min $(@sprintf("%.1e", minimum(verticalq_f))), max $(@sprintf("%.1e", maximum(verticalq_f))))")
colormap = :balance
hm = heatmap!(ax, 1e-3x, 1e-3y, verticalq_f;
            rasterize = true, colormap = colormap, colorrange = (-5e-9, 5e-9))
Colorbar(fig[1, 2], hm, label=L"vertical q")             # if you want colorbar
save(filesave * fileparam * "_verticalq_f.png", fig; px_per_unit=4)
println("Saved figure of filtered vertical q term")

# # --- Compute and plot spectra for unfiltered q --- 

# # Define cutoffs
# cutoffs = 10 .^ range(4, 0, length=100)                 # 100 log-spaced values from 1e4 down to 1e0
# K = 2π ./ cutoffs                                       # band edge wavenumbers, increasing
# labels = ["κ ≤ 2π/$(cutoffs[1])",
#         ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
#         "κ > 2π/$(cutoffs[end])"]

# # --- Cache of computed PV spectra/bands (filtered & unfiltered) ---
# pvspectra_cache_file = filesave * fileparam * "_iter$(iteration)_pvspectra.jld2"

# if isfile(pvspectra_cache_file)
#     println("Loading cached PV spectra variables from $pvspectra_cache_file...")
#     S_quf, bands_quf, S_qf, bands_qf =
#         load(pvspectra_cache_file, "S_quf", "bands_quf", "S_qf", "bands_qf")
# else
#     # Compute for band regimes for |q_uf|
#     S_quf = isotropic_powerspectrum(q_uf, q_uf; Δx=dx, Δy=dy)
#     bands_quf = bandpass_filter(q_uf, dx, dy, cutoffs)      # compute bands

#     # Compute for band regimes for |q_f|
#     S_qf = isotropic_powerspectrum(q_f, q_f; Δx=dx, Δy=dy)
#     bands_qf = bandpass_filter(q_f, dx, dy, cutoffs)        # compute bands

#     mkpath(filesave)
#     jldsave(pvspectra_cache_file; S_quf, bands_quf, S_qf, bands_qf)
#     println("Saved cached PV spectra variables to $pvspectra_cache_file")
# end # if isfile(pvspectra_cache_file)

# freqs_quf = collect(S_quf.freq)
# global S0 = S_quf
# spec_normalized_quf = Real.(S_quf.spec ./ S0.spec[1])

# band_max_quf = [maximum(abs, bands_quf[label]) for label in labels]
# color_vals_quf = bandmax_color_vals(freqs_quf, band_max_quf, K)

# crange = extrema(color_vals_quf)        # Lock both lines to q_uf's color limits

# # Plot spectra colored by max(|q_uf|,|q_f|) in each band regime
# # Fig 1: in one plot 
# fig = Figure(size = (600, 500))
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
#             ylabel = L"E_{q}(k)/E_{q} (k_{min})",
#             xscale = log10, yscale = log10,
#             limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
# ax = Axis(fig[1, 1]; title=L"q_{uf}, z=-8.4375 m", axis_kwargs1...)

# lp = lines!(ax, freqs_quf, spec_normalized_quf, color = color_vals_quf,
#             colormap = :amp, colorrange = crange, linewidth = 2)

# Colorbar(fig[1,2], lp, label = L"\max {|q|} \text{ in band}")
# xlims!(ax, (10^-4.5, 10^0.5))
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
# save(filesave * fileparam * "_spectracolor_quf.png", fig)
# println("Saved figure of q_uf spectra colored by frontal sharpness")

# # Free up memory
# S_quf, freqs_quf, spec_normalized_quf, bands_quf, band_max_quf, color_vals_quf = ntuple(i -> nothing, 6)
# GC.gc()                                                 # force the garbage collector to run immediately

# # --- Compute spectra for unfiltered q and filtered q ---

# freqs_qf = collect(S_qf.freq)
# spec_normalized_qf = Real.(S_qf.spec ./ S0.spec[1])

# band_max_qf  = [maximum(abs, bands_qf[label])  for label in labels]
# color_vals_qf  = bandmax_color_vals(freqs_qf,  band_max_qf,  K)

# # Fig 2
# fig = Figure(size = (600, 500))
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
#             ylabel = L"E_{q}(k)/E_{q} (k_{min})",
#             xscale = log10, yscale = log10,
#             limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
# ax = Axis(fig[1, 1]; title=L"q_{f}, z=-8.4375 m", axis_kwargs1...)

# lp2 = lines!(ax, freqs_qf, spec_normalized_qf, color = color_vals_qf,
#             colormap = :amp, colorrange = crange, linewidth = 2, linestyle = :dash)

# Colorbar(fig[1,2], lp, label = L"\max {|q|} \text{ in band}")
# xlims!(ax, (10^-4.5, 10^0.5))
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
# save(filesave * fileparam * "_spectracolor_qf.png", fig)
# println("Saved figure of q_f spectra colored by frontal sharpness")