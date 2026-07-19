using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_subdomain_snapshot, load_snapshots, isotropic_powerspectrum, δ
set_theme!(Theme(fontsize = 12))

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "figures/20260718_nhy_spectra/"

# 1. Define file parameters
# Loop over subdomain files
iteration = 164410
println("Iteration: $(iteration)")

nday = 7.5
println("Plotting iteration $(iteration) on day $(nday)...")
t0 = now()

i = 91
println("--- Subdomain $(i) ---")

# 2. Define the filename of the saved snapshot
fileparam = "subdomain" * string(i)
output_filename = filehead * fileparam * "_iter$(iteration).jld2"

# 3. Load the snapshot
snapshot = load_subdomain_snapshot(output_filename)

T = snapshot[:T]
u = snapshot[:u]
v = snapshot[:v]
w = snapshot[:w]

f = parameters.f 
# ro = compute!(Field(ζ(snapshots, snapshot_number)/f))       # Rossby number: rel vorticity / f
# rd = compute!(Field(δ(snapshots, snapshot_number)/f))       # divergence Rossby number: divergence / f
# can't use Oceananigans ζ and δ computation because the subdomain snapshot is saved as a Dict

println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

# Coordinate arrays
xu, yu, zu = nodes(u)       # u at cell faces in x
xv, yv, zv = nodes(v)       # v at cell faces in y
xw, yw, zw = nodes(w)       # w at cell faces in z
xT, yT, zT = nodes(T)       # T at cell centers

### -------------------------------------------------------------------------
### Figure 1: spectra of E_T, E_u, E_v, E_w

# Define axes for figures
# low k: oscillates over a long spatial distance, "large-scale motions"
# high k: oscillates over a short spatial distance, "small-scale motions"
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_i(k)/E_{T,v}(k_{min},z=-8.4375~m)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((8e-5, 4e-2), (1e-9,1e1)))
axis_kwargs2 = NamedTuple{(:xlabel,:xscale,:yscale,:limits)}(axis_kwargs1)

# Plot figure
fig = Figure(size = (800, 300))
for (i,klev) in enumerate([65,33,1])       # for hy, depths at level = [127,116,98]
    println("Plotting spectra at z = $(zT[klev])m...")

    if i == 1
        # axis labels on leftmost subplot
        ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs1...)
    else
        # axis ticks but no axis labels on middle and rightmost subplots 
        ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs2...)
        hideydecorations!(ax, ticks = false)
    end

    # Compute the auto-spectrum (co-spectrum of field with itself) of T, u, v, w 
    Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev), xu, yu)
    Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev), xv, yv)
    wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
    Sw = isotropic_powerspectrum(wk, wk, xw, yw)
    St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev), xT, yT)
    
    # Data types:
    # interior(u, :, :, klev): SubArray{Float32, 2, Array{Float32, 3}, Tuple{UnitRange{Int64}, UnitRange{Int64}, Int64}, false}
    # wk: Matrix{Float32}  
    # println(typeof(interior(u, :, :, klev)))
    # println(typeof(wk))

    # Saves spectra at lowest k as the reference spectra for later normalization
    if i == 1
        global Sv0,St0 = Sv,St
    end

    # Draw reference k^(-2) and k^(-3) lines
    lines!(ax, Su.freq, 1e-8Su.freq.^-2, linestyle = :dash, color = :black)
    text!(ax, 1e-3, 1e-2; text = L"k^{-2}")
    lines!(ax, Su.freq, 1e-15Su.freq.^-3, linestyle = :dash, color = :gray)
    text!(ax, 10^-3.5, 1e-6; text = L"k^{-3}")
    # Draw spectra lines for E_T, E_u, E_v, E_w
    lines!(ax, St.freq, Real.(St.spec./St0.spec[1]), color = :red, label = L"E_T")
    lines!(ax, Su.freq, Real.(Su.spec./Sv0.spec[1]), color = :blue, label = L"E_u")
    lines!(ax, Sv.freq, Real.(Sv.spec./Sv0.spec[1]), color = :green, label = L"E_v")
    lines!(ax, Sw.freq, Real.(Sw.spec./Sv0.spec[1]), color = :black, label = L"E_w")
    xlims!(ax, (8e-5, 4e-2))

    # Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
    # Anything to the left is motions larger than 10km
    # Anything to the right is motions smaller than 10km 
    # 10km spatial scale is the loose boundary between submesoscale and mesoscale 
    vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

    axislegend(ax, labelsize=10, patchsize = (20, 5))
end
save(filesave * "spectra_" * fileparam * "_iter$(iteration).pdf", fig)
println("Finished plotting spectra, wall time: $((now() - t0).value/1e3) seconds.")

# ### -------------------------------------------------------------------------
# ### Figure 2: spectra of E_{zeta/f}, E_{delta/f}

# # Define axes for figures
# # low k: oscillates over a long spatial distance, "large-scale motions"
# # high k: oscillates over a short spatial distance, "small-scale motions"
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                            # Wavenumber k
#                 ylabel = L"E_i(k)/E_{\zeta/f, \delta/f}(k_{min},z=-3~m)",   # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((8e-5, 4e-2), (1e-1,1e4)))
# axis_kwargs2 = NamedTuple{(:xlabel,:xscale,:yscale,:limits)}(axis_kwargs1)

# # Plot figure
# fig = Figure(size = (800, 300))
# for (i,klev) in enumerate([127, 116, 98])           # depths at level = [127,116,98]
#     println("Plotting spectra at z = $(zT[klev])m...")

#     if i == 1
#         # axis labels on leftmost subplot
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs1...)
#     else
#         # axis ticks but no axis labels on middle and rightmost subplots 
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs2...)
#         hideydecorations!(ax, ticks = false)
#     end

#     # Compute the auto-spectrum (co-spectrum of field with itself) of Rossby number and divergence Rossby number
#     # Rossby number: ratio of inertia to Coriolis force
#     # divergence Rossby number: signals imbalance -> ageostrophy, internal gravity waves, or frontal circulations
#     # convergent regions (delta<0) are where fronts sharpen, divergent regions (delta>0) are where fronts relax
#     # also signifies strong vertical motion
#     So = isotropic_powerspectrum(interior(ro, :, :, klev), interior(ro, :, :, klev), xT, yT)
#     Sd = isotropic_powerspectrum(interior(rd, :, :, klev), interior(rd, :, :, klev), xT, yT)
    
#     # Saves spectra at lowest k as the reference spectra for later normalization
#     if i == 1
#         global So0,Sd0 = So,Sd
#     end

#     # Draw reference k^(-2) and k^(-3) lines
#     lines!(ax, So.freq, 10So.freq.^(1/3), linestyle = :dash, color = :black)
#     text!(ax, 10^-3.5, 0.2; text = L"k^{1/3}")
#     lines!(ax, So.freq, 1e3So.freq.^(2/3), linestyle = :dash, color = :gray)
#     text!(ax, 10^-2, 20; text = L"k^{2/3}")
#     # Draw spectra lines for E_{zeta/f}, E_{delta/f}
#     lines!(ax, So.freq, Real.(So.spec./So0.spec[1]), color = :blue, label = L"E_{\zeta/f}")
#     lines!(ax, Sd.freq, Real.(Sd.spec./Sd0.spec[1]), color = :green, label = L"E_{\delta/f}")
#     xlims!(ax, (8e-5, 4e-2))
    
#     # Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
#     # Anything to the left is motions larger than 10km
#     # Anything to the right is motions smaller than 10km 
#     # 10km spatial scale is the loose boundary between submesoscale and mesoscale 
#     vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

#     axislegend(ax, labelsize=10, patchsize = (20, 5))
# end
# save(filesave * "spectra2_" * fileparams * "_d$(nday).pdf", fig)
# println("Finished plotting spectra, wall time: $((now() - t0).value/1e3) seconds.")
