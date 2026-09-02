using DelimitedFiles

export load_gpwv4

struct GPWv4_Data
    ncols::Int64
    nrows::Int64
    xllcorner::Float64
    yllcorner::Float64
    cellsize::Float64
    grid::Matrix{Tuple{NTuple{4, Float64}, Float64}}
end

Base.iterate(data::GPWv4_Data) = iterate(data.grid)
Base.iterate(data::GPWv4_Data, state) = iterate(data.grid, state)

function load_gpwv4(file)
    data = open(file) do f
        ncols = parse(Int64, match(r"ncols *(.*)", readline(f))[1])
        nrows = parse(Int64, match(r"nrows *(.*)", readline(f))[1])
        xllcorner = parse(Float64, match(r"xllcorner *(.*)", readline(f))[1])
        yllcorner = parse(Float64, match(r"yllcorner *(.*)", readline(f))[1])
        cellsize = parse(Float64, match(r"cellsize *(.*)", readline(f))[1])
        NODATA_value = parse(Float64, match(r"NODATA_value *(.*)", readline(f))[1])
        grid = readdlm(f, Float64)
        @assert size(grid) == (nrows, ncols) "Dimension mismatch in file data"


        function convert_cell(index, pop)
            @assert pop >= 0.0
            ϕstart = deg2rad(yllcorner + (index[1]-1) * cellsize)
            ϕstop = deg2rad(yllcorner + index[1] * cellsize)
            λstart = deg2rad(xllcorner + (index[2]-1) * cellsize)
            λstop = deg2rad(xllcorner + index[2] * cellsize)
            return ((ϕstart, ϕstop, λstart, λstop), pop)
        end

        return GPWv4_Data(
            ncols, nrows, xllcorner, yllcorner, cellsize,
            [convert_cell(pair...) for pair in pairs(grid)]
        )
    end
end

function find_pop_cell(pop_data::GPWv4_Data, x)
    ϕ, λ = rad2deg.(x)
    index_ϕ = floor(Int, (ϕ - pop_data.yllcorner) / pop_data.cellsize) + 1
    index_λ = floor(Int, (λ - pop_data.xllcorner) / pop_data.cellsize) + 1
    if 1 <= index_ϕ <= pop_data.nrows
        return pop_data.grid[index_ϕ, index_λ]
    else
        return ((0.0,1.0,0.0,1.0), 0.0)
    end
end

function pop_density(cell)
    ((ϕmin, ϕmax, λmin, λmax), pop) = cell
    cell_area = EARTH_EQUATORIAL_RADIUS^2 * (sin(ϕmax) - sin(ϕmin)) * (λmax - λmin)
    pop_density = pop / cell_area
end