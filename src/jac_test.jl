#import Pkg; Pkg.add("Hungarian")
#import Pkg; Pkg.add("Rotations")

using StaticArrays, LinearAlgebra, Rotations


include("./measurements.jl")
include("./perception.jl")

x = [-104.32515928707278, -59.09133006297429, 0.0, 0.0, 13.2, 5.7, 5.3]
x_ego = [-91.6639496981265, -75.00125467663771, 2.6455622444987394, 0.7070991651229994, 0.0038137895043522424, -0.003038500205407094, 0.7070975839362454, 2.116217394800072e-18, 5.393271864649958e-13, -1.4061052807163003e-13, -1.9254563454196356e-13, 3.9118422495687925e-15, 1.3222555785935313e-15]

println("x")
println(x)
println("x_ego")
println(x_ego)

difference = 1e-6

z = h(1, x, x_ego)
J = jac_h(x, z)

println("z")
println(z[1])

println("J")
println(J)

for i = 1:7
    e = zeros(7);
    e[i] = difference
    println("e: $(e)")
    dz = (h(1, x+e, x_ego)[1] - z[1]) / difference
    println("difference-column$(i)")
    println(dz- [J[1][i], J[2][i], J[3][i], J[4][i]])

end