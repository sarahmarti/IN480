# Per calcolare i tempi di esecuzione delle funzioni, sia in versione seriale sia parallelizzata, abbiamo scelto di utilizzare 
# la macro @elapsed, al posto della macro @time (o @timev) che restituisce anche la quantità di memoria allocata.
# Dal momento che alla prima chiamata di @elapsed f(args) la funzione f deve essere compilata prima che eseguita, la prima volta
# non deve essere considerata, perché il tempo di esecuzione sarà sicuramente peggiore del normale.

using PyCall

@pyimport larlib as l
@pyimport numpy as n
@pyimport pyplasm as p
@pyimport scipy as s
@pyimport scipy.sparse as ss

include("traduzione.jl")
include("parallele.jl")


# --- csrCreate e csr2DenseMatrix ---

println("\nNumero di processori attivi: ", workers(), "\n")
println("Tempi di esecuzione di csrCreate, csr2DenseMatrix (e figlie)\n")

EV = [[0,1],[0,3],[1,2],[1,3],[1,4],[2,4],[2,5],[3,4],[4,5]]
FV = [[0,1,3,5,6],[1,2,3,4],[3,4,6,7]]

# AB = vcat(EV,EV*2,EV*3,EV*4,EV*5,EV*6,EV*7,EV*8,EV*9,EV*10,EV*11,EV*12,EV*13,EV*14,EV*15,EV*16,EV*17,EV*18,EV*19,EV*20,
# EV*21,EV*22,EV*23,EV*24,EV*25,EV*26,EV*27,EV*28,EV*29,EV*30,EV*31,EV*32,EV*33,EV*34,EV*35,EV*36,EV*37,EV*38,EV*39,EV*40,
# EV*41,EV*42,EV*43,EV*44,EV*45,EV*46,EV*47,EV*48,EV*49,EV*50);


# seriale
println("SERIALE\n")

# prima chiamata -- da non considerare 
csrEV = csrCreate(EV)
csr2DenseMatrix(csrEV)

times1 = @elapsed csrCreate(EV)
println("julia> @timev csrCreate(EV)") 
println(@timev csrCreate(EV))
csrEV = csrCreate(EV)
println()
println("julia> @timev csr2DenseMatrix(csrEV)") 
times2 = @elapsed csr2DenseMatrix(csrEV)
println(@timev csr2DenseMatrix(csrEV))

# parallela con lo stesso numero di processori
println("\nPARALLELA CON LO STESSO NUMERO DI PROCESSORI\n")

# prima chiamata -- da non considerare 
pcsrEV = pcsrCreate(EV)
pcsr2DenseMatrix(pcsrEV)

timep1 = @elapsed pcsrCreate(EV)
println("julia> @timev pcsrCreate(EV)")
println(@timev csrCreate(EV))
pcsrEV = pcsrCreate(EV)
println()
timep2 = @elapsed pcsr2DenseMatrix(pcsrEV)
println("julia> @timev pcsr2DenseMatrix(pcsrEV)")
println(@time pcsr2DenseMatrix(pcsrEV))

# parallela con 15 processori
println("\nPARALLELA CON L'AGGIUNTA DI 15 PROCESSORI\n")
println("Sto aggiungendo 15 processori...")
addprocs(15)
println("Pid dei processori attivi (oltre [1]): ", workers(), "\n")

timepp1 = @elapsed pcsrCreate(EV)
println("julia> @timev pcsrCreate(EV)")
println(@timev csrCreate(EV))
pcsrEV = pcsrCreate(EV)
println()
timepp2 = @elapsed pcsr2DenseMatrix(pcsrEV)
println("julia> @timev pcsr2DenseMatrix(pcsrEV)")
println(@time pcsr2DenseMatrix(pcsrEV))

# tempi in conclusione
println("\nIN CONCLUSIONE\n")

println("Tempo csrCreate SERIALE = ", times1)
println("Tempo csrCreate PARALLELA = ", timep1)
println("Tempo csrCreate PARALLELA CON addprocs(15) = ", timepp1, "\n")

println("Tempo csr2DenseMatrix SERIALE = ", times2)
println("Tempo csr2DenseMatrix PARALLELA = ", timep2)
println("Tempo csr2DenseMatrix PARALLELA CON addprocs(15) = ", timepp2, "\n")

println("Sto rimuovendo 15 processori...")
rmprocs(workers())
println("Numero di processori attivi: ", workers(), "\n")


# --- larFactes, mkSignedEdges, larModelNumbering --- 

println("\nTempi di esecuzione di larFacets, mkSignedEdges, larModelNumbering (e figlie)\n")

V = [[9,0],[13,2],[15,4],[17,8],[14,9],[13,10],[11,11],[9,10],[7,9],[5,9],[3,
8],[0,6],[2,3],[2,1],[5,0],[7,1],[4,2],[12,10],[6,3],[8,3],[3,5],[5,5],[7,6],
[8,5],[10,5],[11,4],[10,2],[13,4],[14,6],[13,7],[11,9],[9,7],[7,7],[4,7],[2,
6],[12,7],[12,5]]

FV = [[0,1,26],[5,6,17],[6,7,17,30],[7,30,31],[7,8,31,32],[24,30,31,35],[3,4,
28],[4,5,17,29,30,35],[4,28,29],[28,29,35,36],[8,9,32,33],[9,10,33],[11,10,
33,34],[11,20,34],[20,33,34],[20,21,32,33],[18,21,22],[21,22,32],[22,23,31,
32],[23,24,31],[11,12,20],[12,16,18,20,21],[18,22,23],[18,19,23],[19,23,24],
[15,19,24,26],[0,15,26],[24,25,26],[24,25,35,36],[2,3,28],[1,2,27,28],[12,13,
16],[13,14,16],[14,15,16,18,19],[1,25,26,27],[25,27,36],[36,27,28]]

VV = []
for i in collect(0:length(V)-1)
	VV = vcat(VV, [[i]])
end 
model = (V, vcat(FV,[collect(0:15)]))

# seriale
println("SERIALE\n")

# prima chiamata -- da non considerare 
_,EV = larFacets(model,2,1)
submodel = mkSignedEdges((V,EV))
larModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)

# seconda chiamata -- OK
times1 = @elapsed larFacets(model,2,1)
println("julia> @timev larFacets(model,2,1)")
println(@timev larFacets(model,2,1))
println()

_,EV = larFacets(model,2,1)

times2 = @elapsed mkSignedEdges((V,EV))
println("julia> @timev mkSignedEdges((V,EV))")
println(@timev mkSignedEdges((V,EV)))
println()

submodel = mkSignedEdges((V,EV))

times3 = @elapsed larModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)
println("julia> @timev larModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)")
println(@timev larModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2))
println()

# parallela con lo stesso numero di processori
println("\nPARALLELA CON LO STESSO NUMERO DI PROCESSORI\n")

# prima chiamata -- da non considerare 
_,EV = plarFacets(model,2,1)
submodel = pmkSignedEdges((V,EV))
plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)

# seconda chiamata -- OK
timep1 = @elapsed plarFacets(model,2,1)
println("julia> @timev plarFacets(model,2,1)")
println(@timev plarFacets(model,2,1))
println()

_,EV = plarFacets(model,2,1)

timep2 = @elapsed pmkSignedEdges((V,EV))
println("julia> @timev pmkSignedEdges((V,EV))")
println(@timev pmkSignedEdges((V,EV)))
println()

submodel = pmkSignedEdges((V,EV))

timep3 = @elapsed plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)
println("julia> @timev plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)")
println(@timev plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2))
println()

# parallela con 15 processori
println("\nPARALLELA CON L'AGGIUNTA DI 15 PROCESSORI\n")
println("Sto aggiungendo 15 processori...")
addprocs(15)
println("Pid dei processori attivi (oltre [1]): ", workers(), "\n")

timepp1 = @elapsed plarFacets(model,2,1)
println("julia> @timev plarFacets(model,2,1)")
println(@timev plarFacets(model,2,1))
println()

_,EV = plarFacets(model,2,1)

timepp2 = @elapsed pmkSignedEdges((V,EV))
println("julia> @timev pmkSignedEdges((V,EV))")
println(@timev pmkSignedEdges((V,EV)))
println()

submodel = pmkSignedEdges((V,EV))

timepp3 = @elapsed plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)
println("julia> @timev plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2)")
println(@timev plarModelNumbering(1,1,1)(V,[VV,EV,FV],submodel,2))

println()

println("Tempo larFacets SERIALE = ", times1)
println("Tempo larFacets PARALLELA = ", timep1)
println("Tempo larFacets PARALLELA CON addprocs(15) = ", timepp1, "\n")

println("Tempo mkSignedEdges SERIALE = ", times2)
println("Tempo mkSignedEdges PARALLELA = ", timep2)
println("Tempo mkSignedEdges PARALLELA CON addprocs(15) = ", timepp2, "\n")

println("Tempo larModelNumbering SERIALE = ", times3)
println("Tempo larModelNumbering PARALLELA = ", timep3)
println("Tempo larModelNumbering PARALLELA CON addprocs(15) = ", timepp3, "\n")

