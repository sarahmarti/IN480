# funzioni del modulo testate:
# csrCreate, csr2DenseMatrix, matrixProduct, csrTranspose 

using PyCall
using DataStructures

@pyimport larlib as l
@pyimport numpy as np
@pyimport pyplasm as p
@pyimport scipy.sparse as ss
@pyimport sys

include("traduzione.jl")
include("largrid.jl")

# input of geometry and topology  
V2 = [[4,10],[8,10],[14,10],[8,7],[14,7],[4,4],[8,4],[14,4]]
EV = [[0,1],[1,2],[3,4],[5,6],[6,7],[0,5],[1,3],[2,4],[3,6],[4,7]]
FV = [[0,1,3,5,6],[1,2,3,4],[3,4,6,7]]

# characteristic matrices
csrFV = csrCreate(FV)
csrEV = csrCreate(EV)
println("FV = ", csr2DenseMatrix(csrFV), "\n")
println("EV = ", csr2DenseMatrix(csrEV), "\n")

# product
csrEF = matrixProduct(csrEV, csrTranspose(csrFV))
println("EF = ", csr2DenseMatrix(csrEF), "\n")

# boundary and coboundary operators
facetLengths = []
for csrCell in csrEV
	facetLengths = vcat(facetLengths, csrCell[:getnnz]())
end
boundary = l.csrBoundaryFilter(csrEF,facetLengths)
coboundary_1 = csrTranspose(boundary)
println("coboundary_1 = ", csr2DenseMatrix(coboundary_1), "\n")

# product operator
mod_2D = V2,FV
V1 = [[0.],[1.],[2.]]
topol_0 = [[0],[1],[2]]
topol_1 = [[0,1],[1,2]]
mod_0D = V1,topol_0
mod_1D = V1,topol_1
V3,CV = larModelProduct([mod_2D,mod_1D])
# mod_3D = V3,CV
larExplodedView(V3,CV)
println("\nk_3 = ", length(CV), "\n")

# 2-skeleton of the 3D product complex
mod_2D_1 = V2,EV
mod_3D_h2 = larModelProduct([mod_2D,mod_0D])
mod_3D_v2 = larModelProduct([mod_2D_1,mod_1D])
_,FV_h = mod_3D_h2
_,FV_v = mod_3D_v2
FV3 = vcat(FV_h, FV_v)
# SK2 = V3,FV3
larExplodedView(V3,FV3)
println("\nk_2 = ", length(FV3), "\n")

# 1-skeleton of the 3D product complex 
list = []
for i in collect(0:length(V2)-1)
	list = vcat(list, [[i]])
end 
mod_2D_0 = V2,list
mod_3D_h1 = larModelProduct([mod_2D_1,mod_0D])
mod_3D_v1 = larModelProduct([mod_2D_0,mod_1D])
_,EV_h = mod_3D_h1
_,EV_v = mod_3D_v1
EV3 = vcat(EV_h, EV_v)
# SK1 = (V3,EV3)
larExplodedView(V3,EV3)
println("\nk_1 = ", length(EV3), "\n")

# boundary and coboundary operators
np.set_printoptions(threshold=sys.maxint)
csrFV3 = csrCreate(FV3)
csrEV3 = csrCreate(EV3)
csrVE3 = csrTranspose(csrEV3)
facetLengths = []
for csrCell in csrEV3
	facetLengths = vcat(facetLengths, csrCell[:getnnz]())
end
# boundary = l.csrBoundaryFilter(csrVE3,facetLengths)
# coboundary_0 = csrTranspose(boundary)
# println("coboundary_0 = ", csr2DenseMatrix(coboundary_0), "\n")

csrEF3 = matrixProduct(csrEV3, csrTranspose(csrFV3))
facetLengths = []
for csrCell in csrFV3
	facetLengths = vcat(facetLengths, csrCell[:getnnz]())
end
boundary = l.csrBoundaryFilter(csrEF3,facetLengths)
coboundary_1 = csrTranspose(boundary)
println("coboundary_1.T = ", csr2DenseMatrix(coboundary_1[:transpose]()), "\n")

csrCV = csrCreate(CV)
csrFC3 = matrixProduct(csrFV3, csrTranspose(csrCV))
facetLengths = []
for csrCell in csrCV
	facetLengths = vcat(facetLengths, csrCell[:getnnz]())
end
boundary = l.csrBoundaryFilter(csrFC3,facetLengths)
coboundary_2 = csrTranspose(boundary)
println("coboundary_2 = ", csr2DenseMatrix(coboundary_2), "\n")

# boundary chain visualisation
boundaryCells_2 = l.boundaryCells(CV,FV3)
larExplodedView(V3, [FV3[k+1] for k in boundaryCells_2])
