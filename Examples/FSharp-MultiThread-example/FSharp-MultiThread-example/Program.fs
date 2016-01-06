open System.Threading.Tasks

[<EntryPoint>]
let main argv =
    let sizeofarray = 5

    let a = Array.zeroCreate(sizeofarray)
    let b = Array.zeroCreate(sizeofarray)
    let c = Array.zeroCreate(sizeofarray)

    Parallel.For(0,sizeofarray, fun i ->
        a.[i]<-i
        b.[i]<-i*10)|> ignore
    
    Parallel.For(0,sizeofarray, fun i ->
        c.[i] <- a.[i] + b.[i])|> ignore
    printfn "%A" c

    0 // return an integer exit code
