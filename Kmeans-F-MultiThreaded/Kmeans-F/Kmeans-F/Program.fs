// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
//namespace Kmeans
open System.IO
let addLine (line:string) =     
    use wr = StreamWriter("Kmeans_FMCPU.txt", true)
    wr.WriteLine(line)

[<EntryPoint>]
let main argv = 
    let mat = Kmeans.InitParameters.matrix
    printfn "%A" mat

    let difference = Kmeans.KmeansAlg.difference
    printfn "%A" difference

    let stopWatch = Kmeans.KmeansAlg.stopWatch

    stopWatch.Stop()
    printfn "%f" stopWatch.Elapsed.TotalMilliseconds
    addLine(string stopWatch.Elapsed.TotalMilliseconds)
    0 // return an integer exit code
