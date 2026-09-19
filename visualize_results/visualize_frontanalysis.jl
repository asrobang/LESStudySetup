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
filesave = "/home/asrobang/orcd/scratch/figures/20260824_pv/"

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
dx = 4.8828125                  # [m] nhy horizontal grid size
dy = 4.8828125                  # [m] nhy horizontal grid size
dz = 1.125                      # [m] nhy vertical grid size

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

function compute_bgradmag(T,T_above,T_below,α,g,dx,dy,dz)
    dTdx, dTdy = compute_dAdx_dAdy(T,dx,dy) 
    dTdz = compute_dAdz(T_above,T_below,dz)

    bgradmag = (α*g) .* sqrt.(dTdx.^2 .+ dTdy.^2 .+ dTdz.^2)

    return bgradmag
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
## Compute and plot front analysis 

# 1. Define file parameters
# Loop over subdomain files
iteration = 164410
println("Iteration: $(iteration)")

# --- 10km x 10km Tiles ---
i = 97 

println("--- Subdomain $(i) ---")

# 2. Define the filename of the saved snapshot
fileparam = "subdomain" * string(i)
output_filename = filehead * fileparam * "_iter$(iteration).jld2"

# 3. Load the snapshot
snapshot = load_subdomain_snapshot(output_filename)

# ---

T_uf = copy(interior(snapshot[:T],:,:,70))
T_above = copy(interior(snapshot[:T],:,:,71))
T_below = copy(interior(snapshot[:T],:,:,69))

field = compute_bgradmag(T_uf,T_above,T_below,α,g,dx,dy,dz)

# quick plot of field
x, y, _ = nodes(snapshot[:T])
fig = Figure(size = (700, 640))     # if you want colorbar
ax = Axis(fig[1, 1]; aspect = DataAspect())
colormap = :binary
hm = heatmap!(ax, 1e-3x, 1e-3y, field;
            rasterize = true, colormap = colormap, colorscale=log10)
Colorbar(fig[1, 2], hm)             # if you want colorbar
save(filesave * "bgradmag_uf.png", fig; px_per_unit=4)

# quick plot of spectra 
S_bgradmag = isotropic_powerspectrum(field, field; Δx=dx, Δy=dy)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
            ylabel = L"E_T(k)/E_T (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
            xscale = log10, yscale = log10,
            limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-2.8125 m", axis_kwargs1...)
global St0 = S_bgradmag
lines!(ax, S_bgradmag.freq, Real.(S_bgradmag.spec./St0.spec[1]))
xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
save(filesave * "spectra_bgradmag_uf.png", fig)

Nx, Ny = size(field, 1), size(field, 2)

# Angular wavenumbers associated with the horizontal FFT
kx = 2π .* fftfreq(Nx, 1/dx)
ky = 2π .* fftfreq(Ny, 1/dy)

cutoffs = [1e4, 1e3, 1e2, 1e1, 1e0]
bands = bandpass_filter(field, dx, dy, cutoffs)

# # access a specific band
# bands["κ ≤ 2π/10000.0"]
# bands["2π/10000.0 < κ ≤ 2π/1000.0"]
# bands["2π/1000.0 < κ ≤ 2π/100.0"]
# bands["2π/100.0 < κ ≤ 2π/10.0"]
# bands["2π/10.0 < κ ≤ 2π/1.0"]
# calculate extrema of specific band
extrema(bands["κ ≤ 2π/10000.0"])                # (3.728470647715559e-6, 3.728470647715559e-6)
extrema(bands["2π/10000.0 < κ ≤ 2π/1000.0"])    # (-3.097557008802669e-6, 1.0348733545685955e-5)
extrema(bands["2π/1000.0 < κ ≤ 2π/100.0"])      # (-1.2757192653560768e-5, 3.651896401880554e-5)
extrema(bands["2π/100.0 < κ ≤ 2π/10.0"])        # (-4.021673831348164e-5, 0.0001585605892923067)
extrema(bands["2π/10.0 < κ ≤ 2π/1.0"])          # (-2.105903123734332e-5, 2.2983862616691087e-5)

labels = collect(keys(bands))
n = length(labels)
ncols_ = 3
nrows_ = ceil(Int, n / ncols_)

# histogram distribution parameters
nbins = 100
field_min = 0.00    # 0.00     # hard-coded min, max
field_max = 0.00019    # 0.15     # hard-coded min, max
edges = range(field_min, field_max, length=nbins+1)
bin_centers = collect(edges[1:end-1] .+ step(edges)/2)  

# Plot bgradmag separated by wavenumber bands 
fig = Figure(size = (400*ncols_, 350*nrows_))
for (idx, label) in enumerate(labels)
    row = div(idx - 1, ncols_) + 1
    col = mod(idx - 1, ncols_) + 1

    ax = Axis(fig[row, col], title = label, aspect = DataAspect())
    data = bands[label]

    m = maximum(abs, data)  # symmetric bound around 0
    m = m == 0 ? 1e-10 : m   # avoid zero-width colorrange
    hm = heatmap!(ax, 1e-3x, 1e-3y, data, colormap = :balance, 
                colorrange = (-5e-5, 5e-5))
    Colorbar(fig[row, col+ncols_], hm, width = 10, label = L"|\nabla b|")

    # Convert the field into a distribution of the front sharpness
    counts = fit(Histogram, vec(abs.(bands[label])), edges).weights    # unnormalized counts of frontal sharpness fo the bins in each subdomain
    total = sum(counts)
    dist = total > 0 ? counts ./ total : zeros(nbins)   # normalise to probability density, distributions of frontal sharpness for each subdomain

    # Plot histograms of the front sharpness
    # Log scale
    fig2 = Figure()
    ax2 = Axis(fig[1, 1], xlabel = L"|\nabla b| \text{ front sharpness}", ylabel = "Probability density", yscale = log10)
    lines!(ax2, bin_centers, vec(dist).+ 1e-10)
    save(filesave * "bgradfront_logdist_$(idx)_iter$(iteration).png", fig2)

end
save(filesave * "bands_bgradmag_uf2.png", fig)      

# Convert the field into a distribution of the front sharpness
counts = fit(Histogram, vec(abs.(field)), edges).weights    # unnormalized counts of frontal sharpness fo the bins in each subdomain
total = sum(counts)
dist = total > 0 ? counts ./ total : zeros(nbins)   # normalise to probability density, distributions of frontal sharpness for each subdomain
max_field = maximum(field)                          # max temperature front for each subdomain
println("Maximum temperature front sharpness: $max_field")

# Plot histograms of the front sharpness
# Linear scale
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"|\nabla b| \text{ front sharpness}", ylabel = "Probability density")
lines!(ax, bin_centers, vec(dist))
save(filesave * "bgradfront_dist_iter$(iteration).png", fig)
# Log scale
fig = Figure()
ax = Axis(fig[1, 1], xlabel = L"|\nabla b| \text{ front sharpness}", ylabel = "Probability density", yscale = log10)
lines!(ax, bin_centers, vec(dist).+ 1e-10)
save(filesave * "bgradfront_logdist_iter$(iteration).png", fig)

# ---

# ### -------------------------------------------------------------------------