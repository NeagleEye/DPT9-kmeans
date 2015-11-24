// Learn more about F# at http://fsharp.net
// See the 'F# Tutorial' project for more help.
//namespace Kmeans
[<EntryPoint>]
let main argv = 
    let matrix = Kmeans.MatrixCreator.matrix
    let col = Kmeans.MatrixCreator.col
    let row = Kmeans.MatrixCreator.row
    printfn "%A" matrix.[col-1,row-1]

    0 // return an integer exit code
