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
    let st = file.ReadLine().Split(' ')
    let col = System.Int32.Parse(st.[1])
    let row = System.Int32.Parse(st.[0])
    let matrix = new Matrix<double>(row,col)
    for i in 0 .. col-1 do
        let split = file.ReadLine().Split(' ')
        for j in 0 .. row-1 do
            matrix.[j,i] <- System.Double.Parse(split.[j])
    
module InitParameters =
    let col = MatrixCreator.col
    let row = MatrixCreator.row
    let nCluster = 4
    let matrix = MatrixCreator.matrix
    let simMat = new Matrix<double>(nCluster,col)
    let (normalCV : double array) = Array.zeroCreate (nCluster)
    let (clusterQuality : double array) = Array.zeroCreate (nCluster)
    let (cvNorm : double array) = Array.zeroCreate (nCluster)
    let (difference : double array) = Array.zeroCreate (nCluster)
    let (clustersize : int array) = Array.zeroCreate (nCluster)
    let (cv : int array) = Array.zeroCreate (nCluster)
    let conceptVector = new Matrix<double>(nCluster,row)
    let oldCV = new Matrix<double>(nCluster,row)
    let (clusterPointer : int array) = Array.zeroCreate (col)
    let random = System.Random(0)
    let number = random.Next( 1, col-1 )
    let (mark : bool array) = Array.zeroCreate (col)

    for i in 0 .. col-1 do
        mark.[i]<- false
        for j in 0 .. nCluster-1 do
            simMat.[j,i] <- (double 0.0)

    for j in 0 .. row-1 do
        for i in 0 .. nCluster-1 do
            conceptVector.[i,j] <- (double 0.0)
            oldCV.[i,j] <- (double 0.0)
                
                