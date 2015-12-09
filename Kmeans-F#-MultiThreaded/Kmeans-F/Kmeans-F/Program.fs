// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
//namespace Kmeans
[<EntryPoint>]
let main argv = 
    let mat = Kmeans.InitParameters.matrix
    printfn "%A" mat
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
    0 // return an integer exit code
