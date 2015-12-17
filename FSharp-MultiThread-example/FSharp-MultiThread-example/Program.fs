open System.Threading.Tasks

[<EntryPoint>]
let main argv = 
    printfn "%A" argv

    let a = Array.zeroCreate(20)
    let b = Array.zeroCreate(20)
    let c = Array.zeroCreate(20)

    Parallel.For(0,20, fun i ->
        a.[i]<-i*5
        b.[i]<-i*9)|> ignore
    printfn "%A" c
    Parallel.For(0, 20, fun i ->
        c.[i] <- a.[i] + b.[i])|> ignore
    printfn "%A" c
    0 // return an integer exit code
