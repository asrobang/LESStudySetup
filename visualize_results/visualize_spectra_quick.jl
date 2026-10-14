using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_snapshots, load_subdomain_snapshot, isotropic_powerspectrum, δ
using JLD2
set_theme!(Theme(fontsize = 12))

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20260927_regionABC_redo/"

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

# --- File parameters ---
var = "T"               # Temperature
iteration = 164410      # Day 7.5 in nhy sim
klev = 70               # index of depth in nhy sim corresponding to z=-2.8125 m
depth = -2.8125         # meters

# --- Plotting functions ---
function plot_T_image(field, fileparam; Tmin=19.5, Tmax=20.1, k=70, colormap=:thermal)
    x, y, _ = nodes(field)

    fig = Figure(size = (700, 640))     # if you want colorbar
    ax = Axis(fig[1, 1]; aspect = DataAspect())

    hm = heatmap!(ax, 1e-3x, 1e-3y, interior(field, :, :, k);
             rasterize = true, colormap = colormap,
             colorrange = (Tmin, Tmax))
            # )

    Colorbar(fig[1, 2], hm)             # if you want colorbar

    save(filesave * "T_" * fileparam * "_iter$(iteration).png", fig; px_per_unit = 4)
    println("Finished plotting T heatmap")
end

# --- Load regions and compute spectra ---
# Region A
fileparam = "regionA"
output_filename = filehead * "subdomain_" * var * "_" * fileparam * "_iter$(iteration).jld2"
snapshot = load_subdomain_snapshot(output_filename)
T = snapshot[:T]
plot_T_image(T, fileparam)
St_regA = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
println("Freeing large variables...")
T = nothing
snapshot = nothing      # free the variable
GC.gc()                 # force the garbage collector to run immediately

# Region B
fileparam = "regionB"
output_filename = filehead * "subdomain_" * var * "_" * fileparam * "_iter$(iteration).jld2"
snapshot = load_subdomain_snapshot(output_filename)
T = snapshot[:T]
plot_T_image(T, fileparam)
St_regB = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
println("Freeing large variables...")
T = nothing
snapshot = nothing      # free the variable
GC.gc()                 # force the garbage collector to run immediately

# Region C
fileparam = "regionC"
output_filename = filehead * "subdomain_" * var * "_" * fileparam * "_iter$(iteration).jld2"
snapshot = load_subdomain_snapshot(output_filename)
T = snapshot[:T]
plot_T_image(T, fileparam)
St_regC = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
println("Freeing large variables...")
T = nothing
snapshot = nothing      # free the variable
GC.gc()                 # force the garbage collector to run immediately

# --- Plot figure (T) ---
fig = Figure(size = (600, 500))
region_colors  = Makie.wong_colors()[1:3]

axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_T(k)/E_{T_0} (k_{min},\text{day}=7.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                # limits = ((10^-4, 10^0), (1e-12,1e-4)))     # normalized   
                limits = ((10^-4, 10^0), (1e-6,1e2)))       # unnormalized

# ax = Axis(fig[1, 1]; title="Normalized by Region C, z=$(depth) m", axis_kwargs1...)     # normalized
ax = Axis(fig[1, 1]; title="Unnormalized, z=$(depth) m", axis_kwargs1...)               # unnormalized

# global St0 = St_regC.spec[1]        # normalized
global St0 = 1.0                    # unnormalized

# Draw spectra lines for E_T
lines!(ax, St_regA.freq, Real.(St_regA.spec./St0), color = region_colors[1], label = "Region A")
lines!(ax, St_regB.freq, Real.(St_regB.spec./St0), color = region_colors[2], label = "Region B")
lines!(ax, St_regC.freq, Real.(St_regC.spec./St0), color = region_colors[3], label = "Region C")

xlims!(ax, (10^-4, 10^0))
axislegend(ax, labelsize=10, patchsize = (20, 5))

# save(filesave * "spectraT_regionABC_normtest.png", fig)     # normalized
save(filesave * "spectraT_regionABC_test.png", fig)         # unnormalized

println("Finished plotting spectra.")