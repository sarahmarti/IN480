# NOTA: Louis ha detto che per chiamare una funzione di libreria di python (scipy, numpy, pyplasm)
# si deve prima importare con '@pyimport nomeLibreria as nomeNuovo'
# e poi nel codice chiamare nomeNuovo.metodo
# ESEMPIO: @pyimport scipy as s - e poi - CSRmatrix = s.sparse.csr_matrix(shape)

# NOTA 2: vcat concatena per righe ("along dimension 1"), hcat per colonne ("along dimension 2")

# per costruire VV era: VV = AA(LIST)(range(len(V)))
# for k in collect(0:length(V)-1)
#     VV = cat(1,VV,[[k]])
# end


function triples2mat(triples, shape = "csr")
	n = length(triples)
    data = collect(1:n)
	i = collect(0:n-1)
	j = collect(n:2n-1)
 	ij = [i,j]

    for e in enumerate(triples)
		k = e[1]

		ij[1][k] = triples[k][1]
		ij[2][k] = triples[k][2]
		data[k] = triples[k][3]
    end

	return ss.coo_matrix((data,ij))[:asformat](shape)
end


function brc2Coo(BRCmatrix)
	COOmatrix = vcat([[k-1,col,1] for (k,row) in enumerate(BRCmatrix) for col in row])
	
	return COOmatrix
end


function coo2Csr(COOmatrix)
    CSRmatrix = triples2mat(COOmatrix, "csr")

    return CSRmatrix
end


function csrCreate(BRCmatrix, lengthV = 0, shape = (0,0))
	triples = brc2Coo(BRCmatrix)

	if shape == (0,0)
		CSRmatrix = coo2Csr(triples)
	else
		CSRmatrix = ss.csr_matrix(shape)
       	for (i,j,v) in triples
			CSRmatrix[i,j] = v
		end
	end
	
	return CSRmatrix 
end


function csrGetNumberOfRows(CSRmatrix)
	shape = CSRmatrix[:get_shape]()
	Int = shape[1]

    return Int
end


function csrGetNumberOfColumns(CSRmatrix)
	shape = CSRmatrix[:get_shape]()
	Int = shape[2]

    return Int
end


function csr2DenseMatrix(CSRmatrix)
    nrows = csrGetNumberOfRows(CSRmatrix)
    ncolumns = csrGetNumberOfColumns(CSRmatrix)
	ScipyMat = zeros(Int64,(nrows,ncolumns))
	C = CSRmatrix[:tocoo]()

    for triple in zip(C[:row],C[:col],C[:data])
		ScipyMat[triple[1]+1,triple[2]+1] = triple[3]
	end

    return ScipyMat
end


# dipendenza da STRUCT e i COLORI
function larModelNumbering(scalx=1,scaly=1,scalz=1)

	function larModelNumbering0(V,bases,submodel,numberScaling=1)
		color = [p.ORANGE,p.CYAN,p.GREEN,p.WHITE]
		nums = [collect(0:length(bases[1])-1),collect(0:length(bases[2])-1),collect(0:length(bases[3])-1)]
		hpcs = [submodel]

		for k in collect(0:length(bases)-1)
			cn = l.cellNumbering((V,bases[k+1]),submodel)(nums[k+1],color[k+1],(0.5+0.1*k)*numberScaling)
            hpcs = vcat(hpcs, cn)
		end

        return p.STRUCT(hpcs)
	end

	return larModelNumbering0
end


function setup(model,dim)
	V, cells = model
	csr = csrCreate(cells)
    csrAdjSquareMat = larCellAdjacencies(csr)
	csrAdjSquareMat = csrPredFilter(csrAdjSquareMat,dim)

	return V,cells,csr,csrAdjSquareMat
end


function larFacets(model, dim=3, emptyCellNumber=0)
	V,cells,csr,csrAdjSquareMat = setup(model,dim)
    solidCellNumber = length(cells) - emptyCellNumber
	cellFacets = []

	for i in collect(1:length(cells))
        adjCells = csrAdjSquareMat[i][:tocoo]()
		cell1 = csr[i][:tocoo]()[:col]
		pairs = zip(adjCells[:col],adjCells[:data])

        for pair in pairs
            if (i<pair[1]+1) && (i<solidCellNumber+1)
				cell2 = csr[pair[1]+1][:tocoo]()[:col]
				cell = intersect(cell1,cell2)				
				cellFacets = append!(cellFacets,[sort(cell)])
			end    
		end
	end
	
	# sort and remove duplicates
	t = []
	# crea una lista t di tuple, dopo aver creato l'INSIEME a partire da cellFacets
	for e in Set(cellFacets)
    	t = vcat(t, ntuple(i -> e[i],length(e)))
    end

	# crea un array associando ad ogni elemento di t la sua posizione in ordine crescente 
	sp = sortperm(t)
	# crea una lista ordinata t seguendo le posizioni degli elementi
	t = t[sp]

	list = []
	# trasforma la lista di tuple in una lista di liste
	for elem in t
    	list = vcat(list, [[elem[1],elem[2]]])
    end
    
	return V,list
end


function csrTranspose(CSRmatrix)
	return CSRmatrix[:transpose]()
end


function matrixProduct(CSRm1,CSRm2)
    CSRm = CSRm1 * CSRm2
    return CSRm
end


function larCellAdjacencies(CSRm)
    CSRm = matrixProduct(CSRm,csrTranspose(CSRm))
    return CSRm
end


function check(x,dim)
	return x>=dim
end


function csrPredFilter(CSRm, dim)
	triples = []
	i = []
	j = []
	data = []
	coo = CSRm[:tocoo]()

	for z in zip(coo[:row],coo[:col],coo[:data])
		if check(z[3],dim)
			triples = vcat(triples, [[z[1],z[2],z[3]]])
		end
	end

	for t in triples
    	i = vcat(i,t[1])
        j = vcat(j,t[2])
        data = vcat(data,t[3])
    end

    CSRm = ss.coo_matrix((data,(i,j)),CSRm[:shape])[:tocsr]()

    return CSRm
end


# dipendenza da MKPOL, STRUCT
function mkSignedEdges(model,scalingFactor=1)
	V,EV = model
	assert(length(V[1])==2)
	hpcs = []
	frac = 0.06*scalingFactor

	for e in EV
		# cambio gli indici per farli partire da 1 (siamo in Julia)
		e1 = e[1]+1
		e2 = e[2]+1
		v1 = V[e1]
		v2 = V[e2]
		
		vx = v2[1]-v1[1]
		vy = v2[2]-v1[2]
		nx = -vy
		ny = vx

		v3 = v1 + (0.66*[vx,vy])
		v4 = v1 + ((0.66-frac)*[vx,vy]) + (frac*[nx,ny])
		v5 = v1 + ((0.66-frac)*[vx,vy]) + (-frac*[nx,ny])

		verts = [v1,v2,v3,v4,v5]
		cells = [[1,2],[3,4],[3,5]]

		W = [Any[cell[h] for h=1:length(cell)] for cell in verts]
		CW = [Any[cell[h] for h=1:length(cell)] for cell in cells]

		hpcs = vcat(hpcs, p.MKPOL(PyObject([W,CW,[]])))
	end

	hpc = p.STRUCT(hpcs)
    
    return hpc
end
