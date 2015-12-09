(*
    This Program has been written with the idea that CPU and GPU together gives the best result.
    Therefore the Distance calculation is done on the GPU and the Cluster Assign is done on the GPU,
    the rest is done on the CPU and all the initializing.
*)

[<EntryPoint>]
let main argv =

    let mat = Kmeans.InitParameters.matrix
    printfn "%A" mat

    let difference = Kmeans.KmeansAlg.difference
    printfn "%A" difference

    let stopWatch = Kmeans.KmeansAlg.stopWatch
    printfn "Watch has Started"
   (* let simMat2 = Kmeans.InitKmeans.conceptVector
    //now to run the algorithm
    
    let simMat = Kmeans.KmeansAlg.col
    let nresult = Kmeans.kMain.nresult
    let rand = Kmeans.InitParameters.number*)
//    let iter = Kmeans.KmeansAlg.iter
    
//    printfn "Iter: %A" iter
    stopWatch.Stop()
    printfn "%f" stopWatch.Elapsed.TotalMilliseconds
    //printfn "%A" rand
    //let CQ = Kmeans.Computations.commandQueue
    //let pv = Kmeans.Computations.provider
    //CQ.Dispose()
    //pv.Dispose()
    //pv.CloseAllBuffers()
    0 // return an integer exit code
