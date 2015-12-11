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

    stopWatch.Stop()
    printfn "%f" stopWatch.Elapsed.TotalMilliseconds
    0 // return an integer exit code
