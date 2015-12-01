namespace Kmeans
//Placeholder for now
module Computations =
    let value = Kmeans.InitParameters.matrix
    let row = Kmeans.InitParameters.row
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster
    let NormalVectorFunc (a:double) (b:double) =
        ( a * a ) + ( b * b )

    let AverageVec (vec:Matrix<double>) (clustersize:int) (cluster:int) =
        for i in 0 .. row-1 do
            vec.[cluster,i] <- vec.[cluster,i]/(double clustersize)

    let Norm2 (vec:Matrix<double>) (cluster:int) =
        let mutable temp = 0.0
        for i in 0 .. row-1 do
            temp <- temp+(vec.[cluster,i]*vec.[cluster,i])
        temp

    let normalVector = [for i in 0 .. col-1 -> (NormalVectorFunc value.[0,i] value.[1,i]) ]

    let ithAddCV (i:int) (CV:Matrix<double>) (k:int) =
        for j in 0 .. (row-1) do
            CV.[k,j] <- CV.[k,j]+value.[j,i] //return value CV

    let mutable result = 0.0
    let mutable resultval = 0.0
    type Update() =
        member this.Centroid (cv:Matrix<double>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                for j in 0 .. row-1 do
                    cv.[i,j] <- 0.0
            for i in 0 .. col-1 do
                ithAddCV i cv clusterpointer.[i]

        member this.ClusterSize(clustersize:array<int>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                clustersize.[i] <- 0
            for i in 0 .. col-1 do
                clustersize.[clusterpointer.[i]] <- clustersize.[clusterpointer.[i]]+1

        member this.Coherence (clusterQuality:array<double>) =
            resultval <- 0.0
            for i in 0 .. nCluster-1 do
                resultval <- resultval+clusterQuality.[i]
            resultval

    type EucDis() =
        member this.EucDis ( x:Matrix<double>,i:int,norm:double, k:int) =
            result <- 0.0
            for j in 0 .. row-1 do
                result <- result + (x.[k,j] * value.[j,i])
            result <- result * -2.0
            result <- result + normalVector.[i] + norm
            result//return value

        member this.EucDis (x:Matrix<double>,norm:double,resultMat:Matrix<double>, j:int) =
            for i in 0 .. col-1 do
                resultMat.[j,i] <- this.EucDis(x,i,norm, j)
        


