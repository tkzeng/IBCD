using Pkg
Pkg.activate(".")

using CSV, DataFrames, Statistics, DelimitedFiles
using InferCausalGraph
using InferCausalGraph: get_model_params, get_sampling_params, fit_cyclic_model

const MODEL_PARS    = get_model_params(true, 1.0, 0.01)
const SAMPLING_PARS = get_sampling_params(true)

function run_one(file::AbstractString; outdir="output_llcb")
    
    isdir(outdir) || mkpath(outdir)

    expr       = CSV.read(file, DataFrame)
    expr.donor = string.(expr.donor)

    graph  = interventionGraph(expr)
    model  = fit_cyclic_model(graph, false, MODEL_PARS, SAMPLING_PARS)

    postmean = mean(model[2][:])

    name    = splitext(basename(file))[1]
    outfile = joinpath(outdir, "llcb_" * replace(name, "Y_matrix_" => "") * ".csv")
    writedlm(outfile, postmean, ',')

end

if isempty(ARGS)
    println("Usage: julia run_llcb.jl <file1.csv> <file2.csv> ...")
    exit(1)
end

for f in ARGS
    run_one(f)
end
