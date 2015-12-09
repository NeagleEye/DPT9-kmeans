(*
    This Program has been written with the idea that CPU and GPU together gives the best result.
    Therefore the Distance calculation is done on the GPU and the Cluster Assign is done on the GPU,
    the rest is done on the CPU and all the initializing.
*)

[<EntryPoint>]
let main argv =

    let stopWatch = System.Diagnostics.Stopwatch.StartNew()
    printfn "Watch has Started"
    let simMat2 = Kmeans.InitKmeans.conceptVector
    //now to run the algorithm
    
    let simMat = Kmeans.KmeansAlg.col
    let nresult = Kmeans.kMain.nresult
    let rand = Kmeans.InitParameters.number
    let changed = Kmeans.InitKmeans.clusterpointer
    
    printfn "%A" changed
    stopWatch.Stop()
    printfn "%f" stopWatch.Elapsed.TotalSeconds
    //printfn "%A" rand
    //let CQ = Kmeans.Computations.commandQueue
    //let pv = Kmeans.Computations.provider
    //CQ.Dispose()
    //pv.Dispose()
    //pv.CloseAllBuffers()
    0 // return an integer exit code
