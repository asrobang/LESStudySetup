using LESStudySetup
using CairoMakie, Makie
using FFTW
using Printf, Dates, StatsBase
using Statistics: mean, std, quantile
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes, xspacings, yspacings
using Oceananigans.Fields: interior
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_subdomain_snapshot
using LESStudySetup.Diagnostics: load_snapshots, isotropic_powerspectrum, δ
using MathTeXEngine
set_theme!(theme_latexfonts(), fontsize=12, figure_padding = 10)
using JLD2 #, CUDA

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

# |\nabla b_h|: the buoyancy gradient magnitude typically only includes the horizontal buoyancy gradient
function compute_bgradmag(T,α,g,dx,dy)
    dTdx, dTdy = compute_dAdx_dAdy(T,dx,dy) 

    bgradmag = (α*g) .* sqrt.(dTdx.^2 .+ dTdy.^2)

    return bgradmag
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
region = "B"
println("--- Region $(region) ---")
fileparam = "region" * string(region)
output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"      # Load T field

# ---

# 2. Load the snapshot
snapshot = load_subdomain_snapshot(output_filename)

# ---

# 3. Compute horizontal buoyancy gradient magnitude |nabla b_h| and 
#    vertical buoyancy gradient db/dz
T_uf = copy(interior(snapshot[:T],:,:,70))
T_above = copy(interior(snapshot[:T],:,:,71))
T_below = copy(interior(snapshot[:T],:,:,69))
x, y, _ = nodes(snapshot[:T])

field = compute_bgradmag(T_uf,α,g,dx,dy)            # |nabla b_h| horizontal buoyancy gradient magnitude
dbdz = (α*g) .* compute_dAdz(T_above,T_below,dz)    # db/dz vertical buoyancy gradient

snapshot = nothing      # free the variable
T_uf = nothing
T_above = nothing
T_below = nothing
GC.gc()                 # force the garbage collector to run immediately

# ---

# 4. Plot the |nabla b_h| and db/dz fields and spectra

# Plot of |nabla b_h| field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"$|\nabla b_h|$")
colormap = :amp
m_field = maximum(abs, field)  # symmetric bound around 0
m_field = m_field == 0 ? 1e-10 : m_field   # avoid zero-width colorrange
hm = heatmap!(ax, 1e-3x, 1e-3y, field;
            rasterize = true, colormap = colormap, colorrange=(0,1e-5)) #colorrange=(0,m_field))
Colorbar(fig[1, 2], hm)             # if you want colorbar
save(filesave * fileparam * "_bgradmag_uf.png", fig; px_per_unit=4)
println("Saved figure of |nabla b_h| field")

# Plot of |nabla b_h| spectra 
S_bgradmag = isotropic_powerspectrum(field, field; Δx=dx, Δy=dy)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
            ylabel = L"E_{|\nabla b_h|}(k)/E_{|\nabla b_h|} (k_{min})",     # normalized wrt E at k_min
            xscale = log10, yscale = log10,
            limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)
global St0 = S_bgradmag
lines!(ax, S_bgradmag.freq, Real.(S_bgradmag.spec./St0.spec[1]))
xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
save(filesave * fileparam * "_spectra_bgradmag_uf.png", fig)
println("Saved figure of |nabla b_h| spectra")

# Plot of db/dz field
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"$\frac{\partial b}{\partial z}$")
colormap = :balance
m_dbdz = maximum(abs, dbdz)  # symmetric bound around 0
m_dbdz = m_dbdz == 0 ? 1e-10 : m_dbdz   # avoid zero-width colorrange
hm = heatmap!(ax, 1e-3x, 1e-3y, dbdz;
            rasterize = true, colormap = colormap, colorrange=(-1e-5,1e-5)) # colorrange=(-m_dbdz,m_dbdz))
Colorbar(fig[1, 2], hm)             # if you want colorbar
save(filesave * fileparam * "_dbdz_uf.png", fig; px_per_unit=4)
println("Saved figure of db/dz field")

# Plot of db/dz spectra
S_dbdz = isotropic_powerspectrum(dbdz, dbdz; Δx=dx, Δy=dy)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
            ylabel = L"E_{db/dz}(k)/E_{db/dz} (k_{min})",     # normalized wrt E at k_min
            xscale = log10, yscale = log10,
            limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)
global St0 = S_dbdz
lines!(ax, S_dbdz.freq, Real.(S_dbdz.spec./St0.spec[1]))
xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
save(filesave * fileparam * "_spectra_dbdz_uf.png", fig)
println("Saved figure of db/dz spectra")

# ---

# 5. Compute and plot bands of |nabla b_h| and db/dz, separated by wavenumber into 6 regimes

cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
bands = bandpass_filter(field, dx, dy, cutoffs)         # calculate bands for |nabla b_h|

# # Access a specific band
# bands["κ ≤ 2π/10000.0"]
# bands["2π/10000.0 < κ ≤ 2π/1000.0"]
# bands["2π/1000.0 < κ ≤ 2π/100.0"]
# bands["2π/100.0 < κ ≤ 2π/10.0"]
# bands["2π/10.0 < κ ≤ 2π/1.0"]

# labels = collect(keys(bands))   # unsorted
labels = ["κ ≤ 2π/10000.0", "2π/10000.0 < κ ≤ 2π/1000.0", "2π/1000.0 < κ ≤ 2π/100.0",   # sorted 
            "2π/100.0 < κ ≤ 2π/10.0", "2π/10.0 < κ ≤ 2π/1.0", "κ > 2π/1.0"]
n = length(labels)
ncols_ = 3
nrows_ = ceil(Int, n / ncols_)

# Plot of bands for |nabla b_h|
fig = Figure(size = (400*ncols_, 350*nrows_))
for (idx, label) in enumerate(labels)
    row = div(idx - 1, ncols_) + 1
    col = mod(idx - 1, ncols_) + 1

    ax = Axis(fig[row, col], title = label, aspect = DataAspect())
    data = bands[label]

    m = maximum(abs, data)  # symmetric bound around 0
    m = m == 0 ? 1e-10 : m   # avoid zero-width colorrange
    hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :balance, colorrange = (-1e-5, 1e-5))
    Colorbar(fig[row, col+ncols_], hm, width = 10; label = L"\text{reconstructed } $|\nabla b_h|$")
end
save(filesave * fileparam * "_bands_bgradmag_uf.png", fig)
println("Saved figure of |nabla b_h| field separated into 6 bands")

bands = nothing         # free the variable
GC.gc()                 # force the garbage collector to run immediately

bands_dbdz = bandpass_filter(dbdz, dx, dy, cutoffs)     # calculate bands for dbdz

# Plot of bands for dbdz
fig = Figure(size = (400*ncols_, 350*nrows_))
for (idx, label) in enumerate(labels)
    row = div(idx - 1, ncols_) + 1
    col = mod(idx - 1, ncols_) + 1

    ax = Axis(fig[row, col], title = label, aspect = DataAspect())
    data = bands_dbdz[label]

    m = maximum(abs, data)  # symmetric bound around 0
    m = m == 0 ? 1e-10 : m   # avoid zero-width colorrange
    hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :balance, colorrange = (-1e-5, 1e-5))
    Colorbar(fig[row, col+ncols_], hm, width = 10; label = L"\text{reconstructed } $\frac{\partial b}{\partial z}$")
end
save(filesave * fileparam * "_bands_dbdz_uf.png", fig)
println("Saved figure of db/dz field separated into 6 bands")

bands_dbdz = nothing    # free the variable
GC.gc()                 # force the garbage collector to run immediately

# # Sum bands together 
# band_sum = (bands["κ ≤ 2π/10000.0"] + bands["2π/10000.0 < κ ≤ 2π/1000.0"] + bands["2π/1000.0 < κ ≤ 2π/100.0"]
#             + bands["2π/100.0 < κ ≤ 2π/10.0"] + bands["2π/10.0 < κ ≤ 2π/1.0"])

# # quick plot of sum of bands 
# fig = Figure(size = (700, 640))     # if you want colorbar
# ax = Axis(fig[1, 1]; aspect = DataAspect(), title = L"Sum of bands of $|\nabla b_h|$")
# colormap = :amp
# hm = heatmap!(ax, 1e-3x, 1e-3y, band_sum;
#             rasterize = true, colormap = colormap, colorrange=(0,m_field))
# Colorbar(fig[1, 2], hm)             # if you want colorbar
# # save(filesave * "bgradmag_bandsum_uf.png", fig; px_per_unit=4)

# ---

# 6. Color spectra plot according to max bgradmag or max |db/dz| by evaluating 100 band regimes

# Compute for 100 band regimes for |nabla b_h|
freqs = collect(S_bgradmag.freq)
spec_normalized = Real.(S_bgradmag.spec ./ S_bgradmag.spec[1])
cutoffs = 10 .^ range(4, 0, length=100)             # 100 log-spaced values from 1e4 down to 1e0
bands = bandpass_filter(field, dx, dy, cutoffs)     # compute bands
K = 2π ./ cutoffs                                   # band edge wavenumbers, increasing
labels = ["κ ≤ 2π/$(cutoffs[1])",
        ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
        "κ > 2π/$(cutoffs[end])"]

band_max_bgradmag = [maximum(abs, bands[label]) for label in labels]  # from your bandpass_filter output on `field`
color_vals = similar(freqs, Float64)
for (i, κ) in enumerate(freqs)
    if κ <= K[1]
        color_vals[i] = band_max_bgradmag[1]
    elseif κ > K[end]
        color_vals[i] = band_max_bgradmag[end]
    else
        band_idx = findfirst(n -> κ <= K[n+1], 1:length(K)-1) + 1
        color_vals[i] = band_max_bgradmag[band_idx]
    end
end

bands = nothing         # free the variable
GC.gc()                 # force the garbage collector to run immediately

# Plot spectra colored by max |nabla b_h| in each band regime
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
            ylabel = L"E_{|\nabla b_h|}(k)/E_{|\nabla b_h|} (k_{min})",
            xscale = log10, yscale = log10,
            limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

lp = lines!(ax, freqs, spec_normalized, color = color_vals,
            colormap = :amp, linewidth = 2)

Colorbar(fig[1,2], lp, label = L"\max |\nabla b|\text{ in band}")

xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
save(filesave * fileparam * "_spectracolor_bgradmag_uf.png", fig)
println("Saved figure of |nabla b_h| spectra colored by frontal sharpness")

# Compute for 100 band regimes for |db/dz|
freqs = collect(S_dbdz.freq)
spec_normalized = Real.(S_dbdz.spec ./ S_dbdz.spec[1])
cutoffs = 10 .^ range(4, 0, length=100)                 # 100 log-spaced values from 1e4 down to 1e0
bands_dbdz = bandpass_filter(dbdz, dx, dy, cutoffs)     # compute bands
K = 2π ./ cutoffs                                       # band edge wavenumbers, increasing
labels = ["κ ≤ 2π/$(cutoffs[1])",
        ["2π/$(cutoffs[n]) < κ ≤ 2π/$(cutoffs[n+1])" for n in 1:length(cutoffs)-1]...,
        "κ > 2π/$(cutoffs[end])"]

band_max_dbdz = [maximum(abs, bands_dbdz[label]) for label in labels]
color_vals = similar(freqs, Float64)
for (i, κ) in enumerate(freqs)
    if κ <= K[1]
        color_vals[i] = band_max_dbdz[1]
    elseif κ > K[end]
        color_vals[i] = band_max_dbdz[end]
    else
        band_idx = findfirst(n -> κ <= K[n+1], 1:length(K)-1) + 1
        color_vals[i] = band_max_dbdz[band_idx]
    end
end

bands_dbdz = nothing    # free the variable
GC.gc()                 # force the garbage collector to run immediately

# Plot spectra colored by max |db/dz| in each band regime
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",
            ylabel = L"E_{\frac{\partial b}{\partial z}}(k)/E_{\frac{\partial b}{\partial z}} (k_{min})",
            xscale = log10, yscale = log10,
            limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

lp = lines!(ax, freqs, spec_normalized, color = color_vals,
            colormap = :amp, linewidth = 2)

Colorbar(fig[1,2], lp, label = L"\max {|\frac{\partial b}{\partial z}|} \text{ in band}")

xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
save(filesave * fileparam * "_spectracolor_dbdz_uf.png", fig)
println("Saved figure of db/dz spectra colored by frontal sharpness")