using LESStudySetup
using CairoMakie
using SixelTerm
using Printf, Dates
using JLD2
using Oceananigans: compute!
using Oceananigans.Grids: xnodes, ynodes, znodes
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_subdomain_snapshot, load_snapshots, isotropic_powerspectrum, δ
set_theme!(Theme(fontsize = 12))

# --- Set file directories ---
filehead = "/orcd/data/abodner/002/shared_datasets/nhyles_output/subdomains_ASR/" 
filesave = "/home/asrobang/orcd/scratch/figures/20260819_regionABC_vis/"


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

# Double check grid spacing
# xu, yu, zu = nodes(snapshots[:u])
# Δx = xu[2] - xu[1]   # spacing between adjacent x-nodes
# Δy = yu[2] - yu[1]
# xT, yT, zT = nodes(snapshots[:T])   # or nodes(snapshots[:w]) for face-centered z
# Δz = zT[2]-zT[1] # diff(zT)                       # vector of spacings between consecutive z-levels


## Computes the isotropic power spectrums of u, v, w, T
function spectrum_uvwT(iteration, nday, fileparam, klev)

    # 1. Define file parameters
    # Loop over subdomain files
    println("Iteration: $(iteration)")
    println("Reading iteration $(iteration) on day $(nday)...")
    t0 = now()
    
    # 2. Define the filename of the saved snapshot
    if contains(fileparam, "region")
        # T
        output_filename = filehead * "subdomain_T_" * fileparam * "_iter$(iteration).jld2"
        snapshot = load_subdomain_snapshot(output_filename)
        T = snapshot[:T]
        xT, yT, zT = nodes(T)       # T at cell centers 
        St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
        println("Freeing large variables...")
        T = nothing
        snapshot = nothing      # free the variable
        GC.gc()                 # force the garbage collector to run immediately

        # u 
        output_filename = filehead * "subdomain_u_" * fileparam * "_iter$(iteration).jld2"
        snapshot = load_subdomain_snapshot(output_filename)
        u = snapshot[:u]
        xu, yu, zu = nodes(u)       # u at cell faces in x
        Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
        println("Freeing large variables...")
        u = nothing
        snapshot = nothing      # free the variable
        GC.gc()                 # force the garbage collector to run immediately

        # v
        output_filename = filehead * "subdomain_v_" * fileparam * "_iter$(iteration).jld2"
        snapshot = load_subdomain_snapshot(output_filename)
        v = snapshot[:v]
        xv, yv, zv = nodes(v)       # v at cell faces in y
        Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
        println("Freeing large variables...")
        v = nothing
        snapshot = nothing      # free the variable
        GC.gc()                 # force the garbage collector to run immediately

        # w
        output_filename = filehead * "subdomain_w_" * fileparam * "_iter$(iteration).jld2"
        snapshot = load_subdomain_snapshot(output_filename)
        w = snapshot[:w]
        xw, yw, zw = nodes(w)       # w at cell faces in z
        wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
        Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
        println("Freeing large variables...")
        w = nothing
        snapshot = nothing      # free the variable
        GC.gc()                 # force the garbage collector to run immediately
    else
        output_filename = filehead * fileparam * "_iter$(iteration).jld2"

        # 3. Load the snapshot
        snapshot = load_subdomain_snapshot(output_filename)

        T = snapshot[:T]
        u = snapshot[:u]
        v = snapshot[:v]
        w = snapshot[:w]

        println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

        # Coordinate arrays
        xu, yu, zu = nodes(u)       # u at cell faces in x
        xv, yv, zv = nodes(v)       # v at cell faces in y
        xw, yw, zw = nodes(w)       # w at cell faces in z
        xT, yT, zT = nodes(T)       # T at cell centers 

        # Compute the auto-spectrum (co-spectrum of field with itself) of T, u, v, w 
        Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev); Δx=dx, Δy=dy)
        Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev); Δx=dx, Δy=dy)
        wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
        Sw = isotropic_powerspectrum(wk, wk; Δx=dx, Δy=dy)
        St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev); Δx=dx, Δy=dy)
    end

    return St, Su, Sv, Sw
end

# ### -------------------------------------------------------------------------
# ### Figure 1: spectra of E_T, E_u, E_v, E_w

# # 1. Define file parameters
# # Loop over subdomain files
# iteration = 164410
# println("Iteration: $(iteration)")

# nday = 7.5
# println("Plotting iteration $(iteration) on day $(nday)...")
# t0 = now()

# i = 91
# println("--- Subdomain $(i) ---")

# # 2. Define the filename of the saved snapshot
# fileparam = "subdomain" * string(i)
# output_filename = filehead * fileparam * "_iter$(iteration).jld2"

# # 3. Load the snapshot
# snapshot = load_subdomain_snapshot(output_filename)

# T = snapshot[:T]
# u = snapshot[:u]
# v = snapshot[:v]
# w = snapshot[:w]

# # ro = compute!(Field(ζ(snapshots, snapshot_number)/f))       # Rossby number: rel vorticity / f
# # rd = compute!(Field(δ(snapshots, snapshot_number)/f))       # divergence Rossby number: divergence / f
# # can't use Oceananigans ζ and δ computation because the subdomain snapshot is saved as a Dict

# println("Loading fields wall time: $((now() - t0).value/1e3) seconds.")

# # Coordinate arrays
# xu, yu, zu = nodes(u)       # u at cell faces in x
# xv, yv, zv = nodes(v)       # v at cell faces in y
# xw, yw, zw = nodes(w)       # w at cell faces in z
# xT, yT, zT = nodes(T)       # T at cell centers

# # Define axes for figures
# # low k: oscillates over a long spatial distance, "large-scale motions"
# # high k: oscillates over a short spatial distance, "small-scale motions"
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
#                 ylabel = L"E_i(k)/E_{T,v}(k_{min},z=-8.4375~m)",     # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((1e-5, 6e-2), (1e-10,1e3)))
# axis_kwargs2 = NamedTuple{(:xlabel,:xscale,:yscale,:limits)}(axis_kwargs1)

# # Plot figure
# fig = Figure(size = (800, 300))
# for (i,klev) in enumerate([65,33,1])       # for hy, depths at level = [127,116,98]
#     println("Plotting spectra at z = $(zT[klev])m...")

#     if i == 1
#         # axis labels on leftmost subplot
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs1...)
#     else
#         # axis ticks but no axis labels on middle and rightmost subplots 
#         ax = Axis(fig[1, i]; title="z=$(zT[klev]) m", axis_kwargs2...)
#         hideydecorations!(ax, ticks = false)
#     end

#     # Compute the auto-spectrum (co-spectrum of field with itself) of T, u, v, w 
#     Su = isotropic_powerspectrum(interior(u, :, :, klev), interior(u, :, :, klev), xu, yu)
#     Sv = isotropic_powerspectrum(interior(v, :, :, klev), interior(v, :, :, klev), xv, yv)
#     wk = (interior(w, :, :, klev)+interior(w, :, :, klev+1))/2      # interpolates between neighboring cells to get depth of cell center
#     Sw = isotropic_powerspectrum(wk, wk, xw, yw)
#     St = isotropic_powerspectrum(interior(T, :, :, klev), interior(T, :, :, klev), xT, yT)
    
#     # Data types:
#     # interior(u, :, :, klev): SubArray{Float32, 2, Array{Float32, 3}, Tuple{UnitRange{Int64}, UnitRange{Int64}, Int64}, false}
#     # wk: Matrix{Float32}  
#     # println(typeof(interior(u, :, :, klev)))
#     # println(typeof(wk))

#     # Saves spectra at lowest k as the reference spectra for later normalization
#     if i == 1
#         global Sv0,St0 = Sv,St
#     end

#     # Draw reference k^(-2) and k^(-3) lines
#     lines!(ax, Su.freq, 1e-8Su.freq.^-2, linestyle = :dash, color = :black)
#     text!(ax, 1e-3, 1e-1; text = L"k^{-2}")
#     lines!(ax, Su.freq, 1e-15Su.freq.^-3, linestyle = :dash, color = :gray)
#     text!(ax, 10^-3.5, 1e-6; text = L"k^{-3}")
#     # Draw spectra lines for E_T, E_u, E_v, E_w
#     lines!(ax, St.freq, Real.(St.spec./St0.spec[1]), color = :red, label = L"E_T")
#     lines!(ax, Su.freq, Real.(Su.spec./Sv0.spec[1]), color = :blue, label = L"E_u")
#     lines!(ax, Sv.freq, Real.(Sv.spec./Sv0.spec[1]), color = :green, label = L"E_v")
#     lines!(ax, Sw.freq, Real.(Sw.spec./Sv0.spec[1]), color = :black, label = L"E_w")
#     xlims!(ax, (1e-5, 6e-2))

#     # Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
#     # Anything to the left is motions larger than 10km
#     # Anything to the right is motions smaller than 10km 
#     # 10km spatial scale is the loose boundary between submesoscale and mesoscale 
#     vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

#     axislegend(ax, labelsize=10, patchsize = (20, 5))
# end
# save(filesave * "spectra_" * fileparam * "_iter$(iteration).pdf", fig)
# println("Finished plotting spectra, wall time: $((now() - t0).value/1e3) seconds.")

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
#     So = isotropic_powerspectrum(interior(ro, :, :, klev), interior(ro, :, :, klev); Δx=dx, Δy=dy)
#     Sd = isotropic_powerspectrum(interior(rd, :, :, klev), interior(rd, :, :, klev); Δx=dx, Δy=dy)
    
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

# ### -------------------------------------------------------------------------
# ### Figure 3: spectra of E_T across days

# klev = 65       # corresponds to depth = -8.4375 m

# subdomain = 97
# println("--- Subdomain $(i) ---")
# fileparam = "subdomain" * string(subdomain)

# St_d05, Su_d05, Sv_d05, Sw_d05 = spectrum_uvwT(22865, 0.5, fileparam, klev)
# St_d15, Su_d15, Sv_d15, Sw_d15 = spectrum_uvwT(42238, 1.5, fileparam, klev)
# St_d25, Su_d25, Sv_d25, Sw_d25 = spectrum_uvwT(62484, 2.5, fileparam, klev)
# St_d35, Su_d35, Sv_d35, Sw_d35 = spectrum_uvwT(82586, 3.5, fileparam, klev)
# St_d45, Su_d45, Sv_d45, Sw_d45 = spectrum_uvwT(103348, 4.5, fileparam, klev)
# St_d55, Su_d55, Sv_d55, Sw_d55 = spectrum_uvwT(123463, 5.5, fileparam, klev)
# St_d65, Su_d65, Sv_d65, Sw_d65 = spectrum_uvwT(143293, 6.5, fileparam, klev)
# St_d75, Su_d75, Sv_d75, Sw_d75 = spectrum_uvwT(164410, 7.5, fileparam, klev)

# # Plot figure (T)
# fig = Figure(size = (600, 500))

# # Define axes for figures
# # low k: oscillates over a long spatial distance, "large-scale motions"
# # high k: oscillates over a short spatial distance, "small-scale motions"
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
#                 ylabel = L"E_T(k)/E_T (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((10^-3.5, 10^0.5), (1e-10,1e2)))

# ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

# colors = [
#     :black,
#     RGBf(0.902, 0.624, 0.000),   # orange
#     RGBf(0.337, 0.706, 0.914),   # sky blue
#     RGBf(0.000, 0.620, 0.451),   # bluish green
#     RGBf(0.941, 0.894, 0.259),   # yellow
#     RGBf(0.000, 0.447, 0.698),   # blue
#     RGBf(0.835, 0.369, 0.000),   # vermillion
#     RGBf(0.800, 0.475, 0.655),   # reddish purple
# ]

# global St0 = St_d75

# # Draw reference k^(-2) and k^(-3) lines
# lines!(ax, St_d05.freq, 1e-12St_d05.freq.^-2, linestyle = :dash, color = :black)
# text!(ax, 1e-3, 1e-5; text = L"k^{-2}")
# lines!(ax, St_d05.freq, 1e-19St_d05.freq.^-3, linestyle = :dash, color = :gray)
# text!(ax, 10^-3.5, 1e-8; text = L"k^{-3}")
# # Draw spectra lines for E_T
# lines!(ax, St_d05.freq, Real.(St_d05.spec./St0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, St_d15.freq, Real.(St_d15.spec./St0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, St_d25.freq, Real.(St_d25.spec./St0.spec[1]), color = colors[3], label = "Day 2.5")
# lines!(ax, St_d35.freq, Real.(St_d35.spec./St0.spec[1]), color = colors[4], label = "Day 3.5")
# lines!(ax, St_d45.freq, Real.(St_d45.spec./St0.spec[1]), color = colors[5], label = "Day 4.5")
# lines!(ax, St_d55.freq, Real.(St_d55.spec./St0.spec[1]), color = colors[6], label = "Day 5.5")
# lines!(ax, St_d65.freq, Real.(St_d65.spec./St0.spec[1]), color = colors[7], label = "Day 6.5")
# lines!(ax, St_d75.freq, Real.(St_d75.spec./St0.spec[1]), color = colors[8], label = "Day 7.5")

# xlims!(ax, (10^-3.5, 10^0.5))

# # Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
# # Anything to the left is motions larger than 10km
# # Anything to the right is motions smaller than 10km 
# # 10km spatial scale is the loose boundary between submesoscale and mesoscale 
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

# axislegend(ax, labelsize=10, patchsize = (20, 5))

# save(filesave * "spectraT_" * fileparam * "_evol.png", fig)
# println("Finished plotting spectra.")

# # Plot figure (u)
# fig = Figure(size = (600, 500))
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
#                 ylabel = L"E_u(k)/E_u (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((10^-3.5, 10^0.5), (1e-10,1e2)))
# ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

# global Su0 = Su_d75

# lines!(ax, Su_d05.freq, 1e-9Su_d05.freq.^-2, linestyle = :dash, color = :black)
# text!(ax, 1e-3, 1e-1; text = L"k^{-2}")
# lines!(ax, Su_d05.freq, 1e-16Su_d05.freq.^-3, linestyle = :dash, color = :gray)
# text!(ax, 10^-3.5, 1e-5; text = L"k^{-3}")
# lines!(ax, Su_d05.freq, Real.(Su_d05.spec./Su0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Su_d15.freq, Real.(Su_d15.spec./Su0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Su_d25.freq, Real.(Su_d25.spec./Su0.spec[1]), color = colors[3], label = "Day 2.5")
# lines!(ax, Su_d35.freq, Real.(Su_d35.spec./Su0.spec[1]), color = colors[4], label = "Day 3.5")
# lines!(ax, Su_d45.freq, Real.(Su_d45.spec./Su0.spec[1]), color = colors[5], label = "Day 4.5")
# lines!(ax, Su_d55.freq, Real.(Su_d55.spec./Su0.spec[1]), color = colors[6], label = "Day 5.5")
# lines!(ax, Su_d65.freq, Real.(Su_d65.spec./Su0.spec[1]), color = colors[7], label = "Day 6.5")
# lines!(ax, Su_d75.freq, Real.(Su_d75.spec./Su0.spec[1]), color = colors[8], label = "Day 7.5")

# xlims!(ax, (10^-3.5, 10^0.5))
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
# axislegend(ax, labelsize=10, patchsize = (20, 5))

# save(filesave * "spectrau_" * fileparam * "_evol.png", fig)
# println("Finished plotting spectra.")

# # Plot figure (v)
# fig = Figure(size = (600, 500))
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
#                 ylabel = L"E_v(k)/E_v (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((10^-3.5, 10^0.5), (1e-10,1e2)))
# ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

# global Sv0 = Sv_d75

# lines!(ax, Su_d05.freq, 1e-9Su_d05.freq.^-2, linestyle = :dash, color = :black)
# text!(ax, 1e-3, 1e-1; text = L"k^{-2}")
# lines!(ax, Su_d05.freq, 1e-16Su_d05.freq.^-3, linestyle = :dash, color = :gray)
# text!(ax, 10^-3.5, 1e-5; text = L"k^{-3}")
# lines!(ax, Sv_d05.freq, Real.(Sv_d05.spec./Sv0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Sv_d15.freq, Real.(Sv_d15.spec./Sv0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Sv_d25.freq, Real.(Sv_d25.spec./Sv0.spec[1]), color = colors[3], label = "Day 2.5")
# lines!(ax, Sv_d35.freq, Real.(Sv_d35.spec./Sv0.spec[1]), color = colors[4], label = "Day 3.5")
# lines!(ax, Sv_d45.freq, Real.(Sv_d45.spec./Sv0.spec[1]), color = colors[5], label = "Day 4.5")
# lines!(ax, Sv_d55.freq, Real.(Sv_d55.spec./Sv0.spec[1]), color = colors[6], label = "Day 5.5")
# lines!(ax, Sv_d65.freq, Real.(Sv_d65.spec./Sv0.spec[1]), color = colors[7], label = "Day 6.5")
# lines!(ax, Sv_d75.freq, Real.(Sv_d75.spec./Sv0.spec[1]), color = colors[8], label = "Day 7.5")

# xlims!(ax, (10^-3.5, 10^0.5))
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
# axislegend(ax, labelsize=10, patchsize = (20, 5))

# save(filesave * "spectrav_" * fileparam * "_evol.png", fig)
# println("Finished plotting spectra.")

# # Plot figure (w)
# fig = Figure(size = (600, 500))
# axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
#                 ylabel = L"E_w(k)/E_w (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
#                 xscale = log10, yscale = log10,
#                 limits = ((10^-3.5, 10^0.5), (1e-10,1e2)))
# ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

# global Sw0 = Sw_d75

# lines!(ax, Su_d05.freq, 1e-9Su_d05.freq.^-2, linestyle = :dash, color = :black)
# text!(ax, 1e-3, 1e-1; text = L"k^{-2}")
# lines!(ax, Su_d05.freq, 1e-16Su_d05.freq.^-3, linestyle = :dash, color = :gray)
# text!(ax, 10^-3.5, 1e-5; text = L"k^{-3}")
# lines!(ax, Sw_d05.freq, Real.(Sw_d05.spec./Sw0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Sw_d15.freq, Real.(Sw_d15.spec./Sw0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Sw_d25.freq, Real.(Sw_d25.spec./Sw0.spec[1]), color = colors[3], label = "Day 2.5")
# lines!(ax, Sw_d35.freq, Real.(Sw_d35.spec./Sw0.spec[1]), color = colors[4], label = "Day 3.5")
# lines!(ax, Sw_d45.freq, Real.(Sw_d45.spec./Sw0.spec[1]), color = colors[5], label = "Day 4.5")
# lines!(ax, Sw_d55.freq, Real.(Sw_d55.spec./Sw0.spec[1]), color = colors[6], label = "Day 5.5")
# lines!(ax, Sw_d65.freq, Real.(Sw_d65.spec./Sw0.spec[1]), color = colors[7], label = "Day 6.5")
# lines!(ax, Sw_d75.freq, Real.(Sw_d75.spec./Sw0.spec[1]), color = colors[8], label = "Day 7.5")

# xlims!(ax, (10^-3.5, 10^0.5))
# vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
# axislegend(ax, labelsize=10, patchsize = (20, 5))

# save(filesave * "spectraw_" * fileparam * "_evol.png", fig)
# println("Finished plotting spectra.")

### -------------------------------------------------------------------------
### Figure 4: spectra of E_T, E_u, E_v, E_w for a single day for region A, B, or C

klev = 65       # corresponds to depth = -8.4375 m

# Define region of interest
region = "B"
println("--- Region $(region) ---")
fileparam = "region" * string(region)
iteration = 164410

# St_d05, Su_d05, Sv_d05, Sw_d05 = spectrum_uvwT(22865, 0.5, fileparam, klev)
# St_d15, Su_d15, Sv_d15, Sw_d15 = spectrum_uvwT(42238, 1.5, fileparam, klev)
# St_d25, Su_d25, Sv_d25, Sw_d25 = spectrum_uvwT(62484, 2.5, fileparam, klev)
St_d35, Su_d35, Sv_d35, Sw_d35 = spectrum_uvwT(82586, 3.5, fileparam, klev)
St_d45, Su_d45, Sv_d45, Sw_d45 = spectrum_uvwT(103348, 4.5, fileparam, klev)
St_d55, Su_d55, Sv_d55, Sw_d55 = spectrum_uvwT(123463, 5.5, fileparam, klev)
St_d65, Su_d65, Sv_d65, Sw_d65 = spectrum_uvwT(143293, 6.5, fileparam, klev)
St_d75, Su_d75, Sv_d75, Sw_d75 = spectrum_uvwT(164410, 7.5, fileparam, klev)

# Save variables
jldsave(filesave * "spectra_" * fileparam * "_iter$(iteration).jld2"; St_d75=St_d75, Su_d75=Su_d75, Sv_d75=Sv_d75, Sw_d75=Sw_d75)

# Plot figure (T)
fig = Figure(size = (600, 500))

# Define axes for figures
# low k: oscillates over a long spatial distance, "large-scale motions"
# high k: oscillates over a short spatial distance, "small-scale motions"
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_T(k)/E_T (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))

ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

colors = [
    :black,
    RGBf(0.902, 0.624, 0.000),   # orange
    RGBf(0.337, 0.706, 0.914),   # sky blue
    RGBf(0.000, 0.620, 0.451),   # bluish green
    RGBf(0.941, 0.894, 0.259),   # yellow
    RGBf(0.000, 0.447, 0.698),   # blue
    RGBf(0.835, 0.369, 0.000),   # vermillion
    RGBf(0.800, 0.475, 0.655),   # reddish purple
]

global St0 = St_d75

# Draw reference k^(-2) and k^(-3) lines
lines!(ax, St_d75.freq, 10^(-10.5).*St_d75.freq.^-2, linestyle = :dash, color = :black)
text!(ax, 1e-3, 1e-4; text = L"k^{-2}")
lines!(ax, St_d75.freq, 10^(-11.5).*St_d75.freq.^(-5/3), linestyle = :dash, color = :gray)
text!(ax, 10^-4, 1e-6; text = L"k^{-5/3}")
# Draw spectra lines for E_T
# lines!(ax, St_d05.freq, Real.(St_d05.spec./St0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, St_d15.freq, Real.(St_d15.spec./St0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, St_d25.freq, Real.(St_d25.spec./St0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, St_d35.freq, Real.(St_d35.spec./St0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, St_d45.freq, Real.(St_d45.spec./St0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, St_d55.freq, Real.(St_d55.spec./St0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, St_d65.freq, Real.(St_d65.spec./St0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, St_d75.freq, Real.(St_d75.spec./St0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-4.5, 10^0.5))

# Vertical line corresponding to wavenumber at wavelength = 10^4 m = 10 km (refers to a 10km spatial scale)
# Anything to the left is motions larger than 10km
# Anything to the right is motions smaller than 10km 
# 10km spatial scale is the loose boundary between submesoscale and mesoscale 
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)

axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "spectraT_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure (u)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_u(k)/E_u (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Su0 = Su_d75

lines!(ax, Su_d75.freq, 1e-7Su_d75.freq.^-2, linestyle = :dash, color = :black)
text!(ax, 1e-3, 1e-3; text = L"k^{-2}")
lines!(ax, Su_d75.freq, 1e-3Su_d75.freq.^(-5/3), linestyle = :dash, color = :gray)
text!(ax, 10^-1, 1e0; text = L"k^{-5/3}")
# lines!(ax, Su_d05.freq, Real.(Su_d05.spec./Su0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Su_d15.freq, Real.(Su_d15.spec./Su0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Su_d25.freq, Real.(Su_d25.spec./Su0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Su_d35.freq, Real.(Su_d35.spec./Su0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Su_d45.freq, Real.(Su_d45.spec./Su0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Su_d55.freq, Real.(Su_d55.spec./Su0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Su_d65.freq, Real.(Su_d65.spec./Su0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Su_d75.freq, Real.(Su_d75.spec./Su0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "spectrau_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure (v)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_v(k)/E_v (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Sv0 = Sv_d75

lines!(ax, Sv_d75.freq, 1e-7Sv_d75.freq.^-2, linestyle = :dash, color = :black)
text!(ax, 1e-3, 1e-3; text = L"k^{-2}")
lines!(ax, Sv_d75.freq, 1e-3Sv_d75.freq.^(-5/3), linestyle = :dash, color = :gray)
text!(ax, 10^-1, 1e0; text = L"k^{-5/3}")
# lines!(ax, Sv_d05.freq, Real.(Sv_d05.spec./Sv0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Sv_d15.freq, Real.(Sv_d15.spec./Sv0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Sv_d25.freq, Real.(Sv_d25.spec./Sv0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Sv_d35.freq, Real.(Sv_d35.spec./Sv0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Sv_d45.freq, Real.(Sv_d45.spec./Sv0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Sv_d55.freq, Real.(Sv_d55.spec./Sv0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Sv_d65.freq, Real.(Sv_d65.spec./Sv0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Sv_d75.freq, Real.(Sv_d75.spec./Sv0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "spectrav_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

# Plot figure (w)
fig = Figure(size = (600, 500))
axis_kwargs1 = (xlabel = "Wavenumber (rad⋅m⁻¹)",                # wavenumber k 
                ylabel = L"E_w(k)/E_w (k_{min},\text{day}=0.5)",     # normalized wrt E at k_min
                xscale = log10, yscale = log10,
                limits = ((10^-4.5, 10^0.5), (1e-17,1e3)))
ax = Axis(fig[1, 1]; title="z=-8.4375 m", axis_kwargs1...)

global Sw0 = Sw_d75

lines!(ax, Sw_d75.freq, 1e-7Sw_d75.freq.^-2, linestyle = :dash, color = :black)
text!(ax, 1e-3, 1e-3; text = L"k^{-2}")
lines!(ax, Sw_d75.freq, 1e-3Sw_d75.freq.^(-5/3), linestyle = :dash, color = :gray)
text!(ax, 10^-1, 1e0; text = L"k^{-5/3}")
# lines!(ax, Sw_d05.freq, Real.(Sw_d05.spec./Sw0.spec[1]), color = colors[1], label = "Day 0.5")
# lines!(ax, Sw_d15.freq, Real.(Sw_d15.spec./Sw0.spec[1]), color = colors[2], label = "Day 1.5")
# lines!(ax, Sw_d25.freq, Real.(Sw_d25.spec./Sw0.spec[1]), color = colors[3], label = "Day 2.5")
lines!(ax, Sw_d35.freq, Real.(Sw_d35.spec./Sw0.spec[1]), color = colors[4], label = "Day 3.5")
lines!(ax, Sw_d45.freq, Real.(Sw_d45.spec./Sw0.spec[1]), color = colors[5], label = "Day 4.5")
lines!(ax, Sw_d55.freq, Real.(Sw_d55.spec./Sw0.spec[1]), color = colors[6], label = "Day 5.5")
lines!(ax, Sw_d65.freq, Real.(Sw_d65.spec./Sw0.spec[1]), color = colors[7], label = "Day 6.5")
lines!(ax, Sw_d75.freq, Real.(Sw_d75.spec./Sw0.spec[1]), color = colors[8], label = "Day 7.5")

xlims!(ax, (10^-4.5, 10^0.5))
vlines!(ax, [2π/10^4]; color = :black, linewidth = 0.5)
axislegend(ax, labelsize=10, patchsize = (20, 5), position = (:left, :bottom))

save(filesave * "spectraw_" * fileparam * "_evol.png", fig)
println("Finished plotting spectra.")

### -------------------------------------------------------------------------