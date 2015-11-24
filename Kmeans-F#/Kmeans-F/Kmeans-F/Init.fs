namespace Kmeans
//module FileReader
[<CompilationRepresentationAttribute(CompilationRepresentationFlags.ModuleSuffix)>]
type ReadFile() =
    let file = new System.IO.StreamReader("AllRandom.mtx")
    
    member this.ReadLine() = file.ReadLine() 

    interface System.IDisposable with
        member this.Dispose() = file.Close()


module MatrixCreator =
    
    let file = new ReadFile()
    let s = file.ReadLine()
    let st = s.Split(' ')
    let col = System.Int32.Parse(st.[1])
    let row = System.Int32.Parse(st.[0])
    let matrix = new Matrix<int>(col,row)
    for i in 0 .. col-1 do
        let line = file.ReadLine()
        let split = line.Split(' ')
        for j in 0 .. row-1 do
            matrix.[i,j] <- System.Int32.Parse(split.[j])
    
module InitParameters =
    let col = MatrixCreator.col
    let row = MatrixCreator.row