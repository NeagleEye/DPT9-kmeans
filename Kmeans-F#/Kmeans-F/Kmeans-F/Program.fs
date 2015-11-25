// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
//namespace Kmeans
[<EntryPoint>]
let main argv = 
    let simMat2 = Kmeans.InitKmeans.conceptVector

    //now to run the algorithm
    let simMat = Kmeans.KmeansAlg.col
    let rand = Kmeans.InitParameters.number
    printfn "%A" rand
    0 // return an integer exit code
