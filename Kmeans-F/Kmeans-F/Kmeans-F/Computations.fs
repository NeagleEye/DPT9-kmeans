(*   This Code is inspired by Yuqiang Guan and 
     the original source code of the program is released under the GNU Public License (GPL)
     from:
     http://www.dataminingresearch.com/index.php/2010/06/gmeans-clustering-software-compatible-with-gcc-4/
     Copyright (c) 2003, Yuqiang Guan
*)
namespace Kmeans
//Placeholder for now
module Computations =
    let value = Kmeans.InitParameters.matrix
    let row = Kmeans.InitParameters.row
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster
    let NormalVectorFunc (a:double) (b:double) =
        ( a * a ) + ( b * b )
    let NormalVectorFunc2 (a:double) =
        ( a * a )

    let AverageVec (vec:array<double>) (clustersize:int) (cluster:int) =
        for i in 0 .. row-1 do
            vec.[cluster*row+i] <- vec.[cluster*row+i]/(double clustersize)

    let Norm2 (vec:array<double>) (cluster:int) =
        let mutable temp = 0.0
        for i in 0 .. row-1 do
            temp <- temp+(vec.[cluster*row+i]*vec.[cluster*row+i])
        temp

    //let normalVector = [for i in 0 .. col-1 -> (NormalVectorFunc value.[0*col+i] value.[1*col+i]) ]
    let (normalVector: double array) = Array.zeroCreate(col)
    for i in 0 .. col-1 do
        for j in 0 .. row-1 do
            normalVector.[i] <- normalVector.[i]+ NormalVectorFunc2 value.[j*col+i]

    let ithAddCV (i:int) (CV:array<double>) (k:int) =
        for j in 0 .. (row-1) do
            CV.[k*row+j] <- CV.[k*row+j]+value.[j*col+i] //return value CV

    type Update() =
        member this.Centroid (cv:array<double>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                for j in 0 .. row-1 do
                    cv.[i*row+j] <- 0.0
            for i in 0 .. col-1 do
                ithAddCV i cv clusterpointer.[i]

        member this.ClusterSize(clustersize:array<int>, clusterpointer:array<int>) =
            for i in 0 .. nCluster-1 do
                clustersize.[i] <- 0
            for i in 0 .. col-1 do
                clustersize.[clusterpointer.[i]] <- clustersize.[clusterpointer.[i]]+1

        member this.Coherence (clusterQuality:array<double>) =
            let mutable resultval = 0.0
            for i in 0 .. nCluster-1 do
                resultval <- resultval+clusterQuality.[i]
            resultval

    type EucDis() =
        member this.EucDis ( x:array<double>,i:int,norm:double, k:int) =
            let mutable result = 0.0;
            for j in 0 .. row-1 do
                result <- result + (x.[k*row+j] * value.[j*col+i]);
            result <- result * -2.0;
            result <- result + normalVector.[i] + norm;
            result;//return value

        member this.EucDis (x:array<double>,norm:double,resultMat:array<double>, j:int) =
            for i in 0 .. col-1 do
                resultMat.[j*col+i] <- this.EucDis(x,i,norm, j);
        


