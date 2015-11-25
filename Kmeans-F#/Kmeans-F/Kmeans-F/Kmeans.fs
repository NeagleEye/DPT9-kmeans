namespace Kmeans

type InitAssignCluster() =
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster

    member this.AssignCluster (simMat:Matrix<double>, clusterpointer:array<int>) =
        let mutable changed = 0
        let mutable temp_cluster_ID = 0
        let mutable temp_sim = (double 0.0)
        for i in 0 .. col-1 do
            temp_sim <- simMat.[clusterpointer.[i],i]
            temp_cluster_ID <- clusterpointer.[i]
            for j in 0 .. nCluster-1 do
                if j <> clusterpointer.[i] && simMat.[j,i] < temp_sim then
                    temp_sim <- simMat.[j,i]
                    temp_cluster_ID <- j
            if temp_cluster_ID <> clusterpointer.[i] then
                clusterpointer.[i]<- temp_cluster_ID
                simMat.[clusterpointer.[i],i] <- temp_sim
                changed <- changed+1
        changed
        

module WellSeperatedCentroids =
    let mutable minInd = 0 
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster
    let cv = Kmeans.InitParameters.cv
    let mark = Kmeans.InitParameters.mark
    let conceptVector = Kmeans.InitParameters.conceptVector
    let normalCV = Kmeans.InitParameters.normalCV
    let normalVector = Kmeans.Computations.normalVector
    let simMat = Kmeans.InitParameters.simMat
    let mutable min = 0.0 
    let mutable cossum = 0.0
    let clusterpointer = Kmeans.InitParameters.clusterPointer
    cv.[0] <- Kmeans.InitParameters.number
    Kmeans.Computations.ithAddCV cv.[0] conceptVector 0
    mark.[0]<- true
    normalCV.[0] <- normalVector.[cv.[0]]
    Kmeans.Computations.EucDis().EucDis(conceptVector, normalCV.[0], simMat, 0)
    for i in 1 .. nCluster-1 do
        minInd <- 0
        min <- 0.0
        for j in 0 .. col-1 do
            if mark.[j] = false then
                cossum <- 0.0
                for k in 0 .. i do
                    cossum<- cossum+simMat.[k,j]
                if cossum > min then
                    min <- cossum
                    minInd <- j
        cv.[i] <- minInd
        Kmeans.Computations.ithAddCV cv.[i] conceptVector i
        normalCV.[i] <- normalVector.[cv.[i]]
        Kmeans.Computations.EucDis().EucDis(conceptVector, normalCV.[i], simMat, i)
        mark.[cv.[i]] <- true;

    let assign = new InitAssignCluster()
    let changed = assign.AssignCluster(simMat,clusterpointer)
    printfn "%A have changed" changed

module InitKmeans =
    let conceptVector = Kmeans.InitParameters.conceptVector
    let clusterpointer = WellSeperatedCentroids.clusterpointer
    let clustersize = Kmeans.InitParameters.clustersize
    let nCluster = Kmeans.InitParameters.nCluster
    let col = Kmeans.InitParameters.col
    let row = Kmeans.InitParameters.row
    let normalCV = WellSeperatedCentroids.normalCV
    let simMat = WellSeperatedCentroids.simMat
    let clusterQuality = Kmeans.InitParameters.clusterQuality

    for i in 0 .. nCluster-1 do
        for j in 0 .. row-1 do
            conceptVector.[i,j] <- 0.0

    for i in 0 .. col-1 do
        if clusterpointer.[i] >= 0 && clusterpointer.[i] < nCluster then
            Kmeans.Computations.ithAddCV i conceptVector clusterpointer.[i]
        else
            clusterpointer.[i] <- 0
    Kmeans.Computations.Update().ClusterSize(clustersize,clusterpointer)
    for i in 0 .. nCluster-1 do
        Kmeans.Computations.AverageVec conceptVector clustersize.[i] i
    for i in 0 .. nCluster-1 do
        normalCV.[i] <- Kmeans.Computations.Norm2 conceptVector i
    for i in 0 .. nCluster-1 do
        Kmeans.Computations.EucDis().EucDis(conceptVector, normalCV.[i], simMat, i)
    for i in 0 .. col-1 do
        clusterQuality.[clusterpointer.[i]] <- clusterQuality.[clusterpointer.[i]]+simMat.[clusterpointer.[i],i]
    let funval = Kmeans.Computations.Update().Coherence(clusterQuality)
    let result = Kmeans.Computations.Update().Coherence(clusterQuality)

module kMain =
    let mutable newpreResult = 0.0
    let mutable newiter = 0
    let mutable nresult = 0.0
    type KmeansInner() =    
        member this.DoKmeans(conceptVector:Matrix<double>, clusterpointer:array<int>, clustersize:array<int>, nCluster:int, col:int, row:int, normalCV:array<double>, simMat:Matrix<double>, clusterQuality:array<double>, funval:double, result:double, preResult:double, iter:int,oldCV:Matrix<double>,assign:InitAssignCluster,difference:array<double>) = 
            newpreResult <- result
            newiter <- iter+1
            nresult <- 0.0
            if assign.AssignCluster(simMat,clusterpointer) = 0 then
                newiter<-newiter
            else
                Kmeans.Computations.Update().ClusterSize(clustersize,clusterpointer)

                if iter >= 5 then
                    for i in 0 .. nCluster-1 do
                        for j in 0 .. row-1 do
                            oldCV.[i,j] <- conceptVector.[i,j]

                Kmeans.Computations.Update().Centroid(conceptVector,clusterpointer)
                for i in 0 .. nCluster-1 do
                    Kmeans.Computations.AverageVec conceptVector clustersize.[i] i
                for i in 0 .. nCluster-1 do
                    normalCV.[i] <- Kmeans.Computations.Norm2 conceptVector i

                if iter >= 5 then
                    for i in 0 ..nCluster-1 do
                        difference.[i] <- 0.0
                        for j in 0 .. row-1 do
                            difference.[i] <- difference.[i]+((oldCV.[i,j]-conceptVector.[i,j])*(oldCV.[i,j]-conceptVector.[i,j]))
                        
                    if iter > 5 then
                        for i in 0 .. col-1 do
                            simMat.[clusterpointer.[i],i] <- Kmeans.Computations.EucDis().EucDis(conceptVector,i,normalCV.[clusterpointer.[i]],clusterpointer.[i])
                    else
                        for i in 0 .. nCluster-1 do
                            Kmeans.Computations.EucDis().EucDis(conceptVector, normalCV.[i], simMat, i)

                else
                    for i in 0 .. nCluster-1 do
                        Kmeans.Computations.EucDis().EucDis(conceptVector, normalCV.[i], simMat, i)

                for i in 0 .. nCluster-1 do
                    clusterQuality.[i]<- 0.0

                for i in 0 .. col-1 do
                    clusterQuality.[clusterpointer.[i]] <- clusterQuality.[clusterpointer.[i]]+simMat.[clusterpointer.[i],i]
                nresult <- Kmeans.Computations.Update().Coherence(clusterQuality)


module KmeansAlg =
    let conceptVector = InitKmeans.conceptVector
    let clusterpointer = InitKmeans.clusterpointer
    let clustersize = InitKmeans.clustersize
    let nCluster = InitKmeans.nCluster
    let col = InitKmeans.col
    let row = Kmeans.InitParameters.row
    let normalCV = InitKmeans.normalCV
    let simMat = InitKmeans.simMat
    let clusterQuality = InitKmeans.clusterQuality
    let funval = InitKmeans.funval
    let mutable result = InitKmeans.result
    let mutable preResult = 0.0
    let mutable iter = 0
    let oldCV = Kmeans.InitParameters.oldCV
    let mutable assign = WellSeperatedCentroids.assign
    let difference = Kmeans.InitParameters.difference
    printfn "%A" clustersize
    printfn "%A %A %A" result iter preResult
    kMain.KmeansInner().DoKmeans(conceptVector, clusterpointer,clustersize,nCluster,col,row,normalCV,simMat,clusterQuality,funval,result,preResult,iter,oldCV,assign,difference)
    result <- kMain.nresult
    iter <- kMain.newiter
    preResult <- kMain.newpreResult
    printfn "%A %A %A" result iter preResult
    printfn "%A" clustersize
    printfn "%A %A" (preResult - result) (0.001*funval)
    while (preResult - result) > (0.001*funval) do
        kMain.KmeansInner().DoKmeans(conceptVector, clusterpointer,clustersize,nCluster,col,row,normalCV,simMat,clusterQuality,funval,result,preResult,iter,oldCV,assign,difference)
        result <- kMain.nresult
        iter <- kMain.newiter
        preResult <- kMain.newpreResult
        printfn "Is this runned ?"