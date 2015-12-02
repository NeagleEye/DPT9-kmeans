namespace Kmeans
open System.Threading.Tasks
//Placeholder for now
module Computations =
    let value = Kmeans.InitParameters.matrix
    let row = Kmeans.InitParameters.row
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster
    let NormalVectorFunc (a:float32) (b:float32) =
        ( a * a ) + ( b * b )

    let AverageVec (vec:array<float32>) (clustersize:int) (cluster:int) =
        for i in 0 .. row-1 do
            vec.[cluster*row+i] <- vec.[cluster*row+i]/(float32 clustersize)

    let Norm2 (vec:array<float32>) (cluster:int) =
        let mutable temp = 0.0f
        for i in 0 .. row-1 do
            temp <- temp+(vec.[cluster*row+i]*vec.[cluster*row+i])
        temp

    let normalVector = [|for i in 0 .. col-1 -> (NormalVectorFunc value.[0*col+i] value.[1*col+i]) |]

    let ithAddCV (i:int) (CV:array<float32>) (k:int) =
        for j in 0 .. (row-1) do
            CV.[k*row+j] <- CV.[k*row+j]+value.[j*col+i] //return value CV

    let mutable result = 0.0f
    let mutable resultval = 0.0f
    type Update() =
        member this.Centroid (cv:array<float32>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                for j in 0 .. row-1 do
                    cv.[i*row+j] <- 0.0f
            for i in 0 .. col-1 do
                ithAddCV i cv clusterpointer.[i]

        member this.ClusterSize(clustersize:array<int>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                clustersize.[i] <- 0
            for i in 0 .. col-1 do
                clustersize.[clusterpointer.[i]] <- clustersize.[clusterpointer.[i]]+1

        member this.Coherence (clusterQuality:array<float32>) =
            resultval <- 0.0f
            for i in 0 .. nCluster-1 do
                resultval <- resultval+clusterQuality.[i]
            resultval

    type EucDis() =
        member this.EucDis ( x:array<float32>,i:int,norm:float32, k:int) =
            let mutable result = 0.0f
            for j in 0 .. row-1 do
                result <- result + (x.[k*row+j] * value.[j*col+i])
            result <- result * -2.0f
            result <- result + normalVector.[i] + norm
            result//return value
        
        member this.EucDis ( x:array<float32>,norm:array<float32>, clusterpointer:array<int>, resultMat:array<float32>) =
            Parallel.For(0, col-1, fun i ->
                resultMat.[clusterpointer.[i]*col+i] <- this.EucDis(x,i,norm.[clusterpointer.[i]], clusterpointer.[i])) |> ignore

        member this.EucDis (x:array<float32>,norm:float32,resultMat:array<float32>, j:int) =
            Parallel.For(0, col-1, fun i ->
                resultMat.[j*col+i] <- this.EucDis(x,i,norm, j)) |> ignore
