open Brahma.Helpers
open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Microsoft.FSharp.Quotations
open Brahma.FSharp.OpenCL.Extensions

[<EntryPoint>]
let main argv = 
    let sizeofarray = 5
    let platformName = "*"
    let localWorkSize = 256 //This is different dependant on Hardware
    let deviceType = DeviceType.Default

    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message

    printfn "Using %A" provider 
    let (a : float32 array) = Array.zeroCreate(sizeofarray) //Float32 is required for Brahma
    let (b : float32 array) = Array.zeroCreate(sizeofarray)
    let (c : float32 array) = Array.zeroCreate(sizeofarray)

    for i in 0 .. sizeofarray-1 do
        a.[i] <- (float32 i)
        b.[i] <- (float32 i)*(float32 10)

    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head) 
    let command = 
        <@
            fun (rng:_1D) (a:array<_>) (b:array<_>) (c:array<_>)->
            let id = rng.GlobalID0
            c.[id]<- a.[id]+b.[id]
        @>
    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d = new _1D(sizeofarray,localWorkSize)    
    kernelPrepare d a b c //Give information to the command.
    let go () =
        let _ = commandQueue.Add(kernelRun())//Run the commnad.
        printfn "%A" c
    go()
    let _ = commandQueue.Add(c.ToHost provider).Finish()//Return array from GPU
    printfn "%A" c
    commandQueue.Dispose()
    provider.Dispose()
    provider.CloseAllBuffers()
    0 // return an integer exit code
