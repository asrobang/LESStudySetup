using JLD2

"""
    inspect_checkpoint_domain(filename, iteration)

Reads only the grid metadata from rank 0's checkpoint file — no u/v/w/T field
data is touched — and returns the full domain size (Lx, Ly, Lz), full grid
point counts (Nx, Ny, Nz), per-rank partitioning (Px, Py), and grid spacing
(Δx, Δy, Δz). Cheap: this opens one small JLD2 file and reads a handful of
scalar/array attributes, not the (potentially huge) field datasets.
"""
function inspect_checkpoint_domain(filename, iteration)
    file = jldopen(filename * "0_iteration$(iteration).jld2")

    Px = file["NonhydrostaticModel/grid"].architecture.partition.x
    Py = file["NonhydrostaticModel/grid"].architecture.partition.y
    nx = file["NonhydrostaticModel/grid"].Nx  # points per rank in x
    ny = file["NonhydrostaticModel/grid"].Ny  # points per rank in y
    Nz = file["NonhydrostaticModel/grid"].Nz  # total points in z

    Lx_per_rank = file["NonhydrostaticModel/grid"].Lx
    Ly_per_rank = file["NonhydrostaticModel/grid"].Ly
    Lz_full = file["NonhydrostaticModel/grid"].Lz

    close(file)

    Nx_full = nx * Px
    Ny_full = ny * Py
    Lx_full = Lx_per_rank * Px
    Ly_full = Ly_per_rank * Py

    Δx = Lx_full / Nx_full
    Δy = Ly_full / Ny_full
    Δz = Lz_full / Nz

    info = (; Px, Py, Nx_full, Ny_full, Nz, Lx_full, Ly_full, Lz_full, Δx, Δy, Δz)

    println("Checkpoint domain info ($(filename)0_iteration$(iteration).jld2):")
    println("  * Partitioning: Px=$Px, Py=$Py")
    println("  * Full grid points: Nx=$Nx_full, Ny=$Ny_full, Nz=$Nz")
    println("  * Full domain extent: Lx=$Lx_full m, Ly=$Ly_full m, Lz=$Lz_full m")
    println("  * Grid spacing: Δx=$Δx m, Δy=$Δy m, Δz=$Δz m")

    return info
end