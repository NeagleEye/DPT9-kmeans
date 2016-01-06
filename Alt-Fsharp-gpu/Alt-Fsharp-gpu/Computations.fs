(*   This Code is inspired by Yuqiang Guan and 
     the original source code of the program is released under the GNU Public License (GPL)
     from:
     http://www.dataminingresearch.com/index.php/2010/06/gmeans-clustering-software-compatible-with-gcc-4/
     Copyright (c) 2003, Yuqiang Guan
*)
namespace Kmeans
open Brahma.Helpers
open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Microsoft.FSharp.Quotations
open Brahma.FSharp.OpenCL.Extensions


//Placeholder for now
module Computations =
    let mutable usedGPUFunCounter = 0
    //Init GPU
    let platformName = "*"
    
    let localWorkSize = 4  
    let deviceType = DeviceType.Gpu

    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message

    printfn "Using %A" provider 
    //GPU INIT should be finished

    let value = Kmeans.InitParameters.matrix
    let row = Kmeans.InitParameters.row
    let col = Kmeans.InitParameters.col
    let nCluster = Kmeans.InitParameters.nCluster

    let NormalVectorFunc (a:float32) (b:float32) =
        ( a * a ) + ( b * b )
    let NormalVectorFunc2 (a:float32) =
        ( a * a )

    let AverageVec (vec:array<float32>) (clustersize:int) (cluster:int) =
        for i in 0 .. row-1 do
            vec.[cluster*row+i] <- vec.[cluster*row+i]/(float32 clustersize)

    let Norm2 (vec:array<float32>) (cluster:int) =
        let mutable temp = 0.0f
        for i in 0 .. row-1 do
            temp <- temp+(vec.[cluster*row+i]*vec.[cluster*row+i])
        temp

    //let normalVector = [|for i in 0 .. col-1 -> (NormalVectorFunc value.[0*col+i] value.[1*col+i]) |]
    let (normalVector: float32 array) = Array.zeroCreate(col)
    for i in 0 .. col-1 do
        for j in 0 .. row-1 do
            normalVector.[i] <- normalVector.[i]+ NormalVectorFunc2 value.[j*col+i]

    let ithAddCV (i:int) (CV:array<float32>) (k:int) =
        for j in 0 .. (row-1) do
            CV.[k*row+j] <- CV.[k*row+j]+value.[j*col+i] //return value CV

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
            let mutable resultval = 0.0f
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

        member this.GPUEucDis (x:array<float32>,norm:array<float32>,result:array<float32>, clusterpointer:array<int>) =
            //for i in 0 .. col-1 do
            //    simMat.[clusterpointer.[i]*col+i] <- Kmeans.Computations.EucDis().EucDis(conceptVector,i,normalCV.[clusterpointer.[i]],clusterpointer.[i])
            let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head) 
            let command = 
                <@
                    fun (rng:_1D) (result:array<_>) (x:array<_>) (norm:array<_>) (col:int) (value:array<_>) (row:int) (normalVector:array<_>) (clusterPointer:array<int>)->
                        let id = rng.GlobalID0
                        let mutable re = 0.0f
                        re <- re+(x.[clusterPointer.[id]*row+0] * value.[0*col+id])
                        re <- re+(x.[clusterPointer.[id]*row+1] * value.[1*col+id])
                        re <- re * -2.0f
                        re <- re + norm.[clusterPointer.[id]] + normalVector.[id]           
                        result.[clusterPointer.[id]*col+id] <- re
                @>


            let kernel, kernelPrepare, kernelRun = provider.Compile command
            let d = new _1D(col,localWorkSize)    
            kernelPrepare d result x norm col value row normalVector clusterpointer
            let go () =
                let _ = commandQueue.Add(kernelRun())//.Finish()
                usedGPUFunCounter <- usedGPUFunCounter+1 // this is just added because a let cannot only have a let it needs a function
    
            go()

            let _ = commandQueue.Add(result.ToHost provider).Finish()
            //printfn "Result= %A" resultMat
    
            commandQueue.Dispose()
            provider.Dispose()
            provider.CloseAllBuffers()

        member this.EucDis (x:array<float32>,norm:float32,resultMat:array<float32>, j:int) =
            for i in 0 .. col-1 do
                resultMat.[j*col+i] <- this.EucDis(x,i,norm, j)

        
        //custom made for only two rows atm and runs the calculation pretty fast. perhaps a too small calculation.
        member this.GPUEucDis (x:array<float32>,norm:float32,resultMat:array<float32>, j:int) =
            let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head) 
            let command = 
                <@
                    fun (rng:_1D) (result:array<_>) (x:array<_>) (norm:float32) (k:int) (col:int) (value:array<_>) (row:int) (normalVector:array<_>)->
                        let id = rng.GlobalID0
                        let mutable re = 0.0f
                        re <- re+(x.[k*row+0] * value.[0*col+id])
                        re <- re+(x.[k*row+1] * value.[1*col+id])
                        re <- re * -2.0f
                        re <- re + norm + normalVector.[id]           
                        result.[k*col+id] <- re
                @>


            let kernel, kernelPrepare, kernelRun = provider.Compile command
            let d = new _1D(col,localWorkSize)    
            kernelPrepare d resultMat x norm j col value row normalVector
            let go () =
                let _ = commandQueue.Add(kernelRun())//.Finish()
                usedGPUFunCounter <- usedGPUFunCounter+1 // this is just added because a let cannot only have a let it needs a function
    
            go()

            let _ = commandQueue.Add(resultMat.ToHost provider).Finish()
            //printfn "Result= %A" resultMat
    
            commandQueue.Dispose()
            provider.Dispose()
            provider.CloseAllBuffers()


