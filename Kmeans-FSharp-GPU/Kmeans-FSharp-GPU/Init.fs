namespace Kmeans
//module FileReader
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
    let (matrix : float32 array) = Array.zeroCreate(row*col)

    for i in 0 .. col-1 do
        let split = file.ReadLine().Split(' ')
        for j in 0 .. row-1 do
            let temp = System.Double.Parse(split.[j])
            matrix.[i+j*col] <- (float32 temp)
    
module InitParameters =
    let col = MatrixCreator.col
    let row = MatrixCreator.row
    let nCluster = 4
    let matrix = MatrixCreator.matrix

    let (simMat: float32 array) = Array.zeroCreate(nCluster*col)
    let (normalCV : float32 array) = Array.zeroCreate (nCluster)
    let (clusterQuality : float32 array) = Array.zeroCreate (nCluster)
    let (cvNorm : float32 array) = Array.zeroCreate (nCluster)
    let (difference : float32 array) = Array.zeroCreate (nCluster)
    let (clustersize : int array) = Array.zeroCreate (nCluster)
    let (changedArray : int array) = Array.zeroCreate (col)
    let (cv : int array) = Array.zeroCreate (nCluster)
    let (conceptVector: float32 array) = Array.zeroCreate(nCluster*row)
    let (oldCV : float32 array) = Array.zeroCreate(nCluster*row)
    let (clusterPointer : int array) = Array.zeroCreate (col)
    let random = System.Random(0)
    let number = random.Next( 1, col-1 )
    let (mark : bool array) = Array.zeroCreate (col)
                
                