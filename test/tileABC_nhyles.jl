# ===============================================================================
#                    CUSTOM SUBDOMAIN EXTRACTION AND COARSE-GRAINING
# ===============================================================================
#
# This script:
#   1. Extracts checkpoint metadata (simulation time, iteration)
#   2. Extracts a single custom subregion (no halo padding) from the full domain
#   3. Saves the subregion to file
#   4. Loads w and T fields from the saved subregion
#
# ===============================================================================

using Oceananigans
using JLD2
using LESStudySetup
using LESStudySetup.Diagnostics
using LESStudySetup.Diagnostics: load_distributed_checkpoint_subdomain
using LESStudySetup.Diagnostics: load_checkpoint_clock
using LESStudySetup.Diagnostics: save_subdomain_with_halo
using LESStudySetup.Diagnostics: inspect_checkpoint_domain

# ===============================================================================
# SECTION 1: USER CONFIGURATION
# ===============================================================================

# --- Data Paths ---
const CHECKPOINT_DIR = "/orcd/data/abodner/002/shared_datasets/nhyles_output/"
const CHECKPOINT_PREFIX = CHECKPOINT_DIR * "iteration16x/nonhydrostatic_checkpoint_"
const OUTPUT_DIR = CHECKPOINT_DIR * "subdomains_ASR/"

# --- Checkpoint Selection ---
const ITERATION = 164410

# --- Custom Region (core, no halo) ---
const CORE_XLIMS_RAW = (78906.25, 19062.5)
const CORE_YLIMS_RAW = (73906.25, 14062.5)
const Z_LIMITS = (-81.0, 0.0)     # Vertical extent (m): 72 cells at dz=1.125m

# --- Halo ---
const HALO_WIDTH = 0.0            # No halo padding around the core region

# --- Full Domain Size (queried from checkpoint metadata — no field data read) ---
domain_info = inspect_checkpoint_domain(CHECKPOINT_PREFIX, ITERATION)
const DOMAIN_LX = domain_info.Lx_full
const DOMAIN_LY = domain_info.Ly_full

# --- Wraparound limits if first limit > second limit ---
function normalize_wraparound_limits(lims, domain_size)
    lo, hi = lims
    return hi < lo ? (lo, hi + domain_size) : (lo, hi)
end
 
const CORE_XLIMS = normalize_wraparound_limits(CORE_XLIMS_RAW, DOMAIN_LX)
const CORE_YLIMS = normalize_wraparound_limits(CORE_YLIMS_RAW, DOMAIN_LY)

# ===============================================================================
# SECTION 2: EXTRACT CHECKPOINT METADATA
# ===============================================================================

println("\n" * "="^70)
println("STEP 1: Loading Checkpoint Metadata")
println("="^70)

clock_info = load_checkpoint_clock(CHECKPOINT_PREFIX, ITERATION)

println("+---------------------------------------------------------------------+")
println("| Checkpoint Information                                              |")
println("+---------------------------------------------------------------------+")
println("|  Iteration:        $(lpad(clock_info.iteration, 10))                              |")
println("|  Simulation time:  $(lpad(round(clock_info.time_days, digits=3), 10)) days                        |")
println("|                    $(lpad(round(clock_info.time, digits=1), 10)) seconds                     |")
println("+---------------------------------------------------------------------+")

# ===============================================================================
# SECTION 3: DEFINE CUSTOM SUBREGION
# ===============================================================================

println("\n" * "="^70)
println("STEP 2: Defining Custom Subregion")
println("="^70)

# Full region including halo padding, for loading (halo gets cropped after filtering)

full_xlims = (CORE_XLIMS[1] - HALO_WIDTH, CORE_XLIMS[2] + HALO_WIDTH)
full_ylims = (CORE_YLIMS[1] - HALO_WIDTH, CORE_YLIMS[2] + HALO_WIDTH)

println("\nCustom region:")
println("  * x: $(CORE_XLIMS ./ 1e3) km")
println("  * y: $(CORE_YLIMS ./ 1e3) km")
println("  * Size: $((CORE_XLIMS[2]-CORE_XLIMS[1])/1e3) km x $((CORE_YLIMS[2]-CORE_YLIMS[1])/1e3) km")
println("  * Vertical: $(Z_LIMITS) m")
println("  * Halo: none")

# ===============================================================================
# SECTION 4: EXTRACT AND SAVE THE SUBREGION
# ===============================================================================

println("\n" * "="^70)
println("STEP 3: Extracting and Saving Custom Subdomain")
println("="^70)

mkpath(OUTPUT_DIR)
output_file = OUTPUT_DIR * "subdomain_u_regionB_iter$(ITERATION).jld2"

if isfile(output_file)
    println("  Output already exists, skipping: $output_file")
else
    println("  Loading region: x=$(round.(full_xlims ./ 1e3, digits=6))km, y=$(round.(full_ylims ./ 1e3, digits=6))km ...")

    snapshot = load_distributed_checkpoint_subdomain(CHECKPOINT_PREFIX, ITERATION;
        xlims = full_xlims,
        ylims = full_ylims,
        zlims = Z_LIMITS,
        fields = (:u,),
        getEw = false,
        getMLD = 0
    )

    save_subdomain_with_halo(output_file, snapshot;
        core_xlims = CORE_XLIMS,
        core_ylims = CORE_YLIMS,
        halo_width = HALO_WIDTH,
        zlims = Z_LIMITS,
        iteration = ITERATION,
        clock_time = clock_info.time,
        clock_time_days = clock_info.time_days
    )

    println("Done: $output_file")
end

# ===============================================================================
# SECTION 5: LOAD SUBREGION
# ===============================================================================

println("\n" * "="^70)
println("STEP 4: Loading Custom Subdomain")
println("="^70)

input_file = output_file
println("Loading: $input_file")

snapshot = load_subdomain_snapshot(input_file; variables = ("u",))

# Display loaded metadata
grid = snapshot[:grid]
println("\nLoaded subdomain:")
println("  * Grid size: $(grid.Nx) x $(grid.Ny) x $(grid.Nz) cells")
println("  * Resolution: dx=$(grid.Δxᶜᵃᵃ)m, dz=$(grid.Lz / grid.Nz)m")

if haskey(snapshot, :clock_time_days)
    println("  * Simulation time: $(round(snapshot[:clock_time_days], digits=3)) days")
end
if haskey(snapshot, :core_xlims)
    println("  * Core region (valid after filtering):")
    println("      x: $(snapshot[:core_xlims]) m")
    println("      y: $(snapshot[:core_ylims]) m")
end
if haskey(snapshot, :halo_width)
    println("  * Halo width: $(snapshot[:halo_width]) m")
end