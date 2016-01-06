(*   This Code is inspired by Yuqiang Guan and 
     the original source code of the program is released under the GNU Public License (GPL)
     from:
     http://www.dataminingresearch.com/index.php/2010/06/gmeans-clustering-software-compatible-with-gcc-4/
     Copyright (c) 2003, Yuqiang Guan
*)
namespace Kmeans
//module FileReader
[<CompilationRepresentationAttribute(CompilationRepresentationFlags.ModuleSuffix)>]
type ReadFile() =
    let file = "AllRandom.mtx"
    let f = new System.IO.StreamReader(file)
    
    member this.ReadLine() = f.ReadLine() 
    member this.SetFile(s) = file = s
    interface System.IDisposable with
        member this.Dispose() = f.Close()


module MatrixCreator =
    
    let file = new ReadFile()
    let st = file.ReadLine().Split(' ')
    let col = System.Int32.Parse(st.[1])
    let row = System.Int32.Parse(st.[0])
    let (matrix : double array) = Array.zeroCreate(row*col)
    for i in 0 .. col-1 do
        let split = file.ReadLine().Split(' ')
        for j in 0 .. row-1 do
            matrix.[i+j*col] <- System.Double.Parse(split.[j])
    
module InitParameters =
    let col = MatrixCreator.col
    let row = MatrixCreator.row
    let nCluster = 9
    let matrix = MatrixCreator.matrix
    let (simMat: double array) = Array.zeroCreate(nCluster*col)
    let (normalCV : double array) = Array.zeroCreate (nCluster)
    let (clusterQuality : double array) = Array.zeroCreate (nCluster)
    let (cvNorm : double array) = Array.zeroCreate (nCluster)
    let (difference : double array) = Array.zeroCreate (nCluster)
    let (clustersize : int array) = Array.zeroCreate (nCluster)
    let (changedArray : int array) = Array.zeroCreate (col)
    let (cv : int array) = Array.zeroCreate (nCluster)
    let (conceptVector: double array) = Array.zeroCreate(nCluster*row)
    let (oldCV : double array) = Array.zeroCreate(nCluster*row)
    let (clusterPointer : int array) = Array.zeroCreate (col)
    let random = System.Random(0)
    let number = random.Next( 1, col-1 )
    let (mark : bool array) = Array.zeroCreate (col)
                
                