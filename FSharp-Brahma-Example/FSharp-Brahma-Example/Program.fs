open Brahma.Helpers
open OpenCL.Net
open Brahma.OpenCL
open Brahma.FSharp.OpenCL.Core
open Microsoft.FSharp.Quotations
open Brahma.FSharp.OpenCL.Extensions

[<EntryPoint>]
let main argv = 
    let platformName = "*"
    let localWorkSize = 256 //This is different dependant on Hardware
    let deviceType = DeviceType.Default

    let provider =
        try  ComputeProvider.Create(platformName, deviceType)
        with 
        | ex -> failwith ex.Message

    printfn "Using %A" provider 
    let (a : float32 array) = Array.zeroCreate(20) //Float32 is required for Brahma
    let (b : float32 array) = Array.zeroCreate(20)
    let (c : float32 array) = Array.zeroCreate(20)

    for i in 0 .. 19 do
        a.[i] <- (float32 i)*(float32 5)
        b.[i] <- (float32 i)*(float32 9)

    let commandQueue = new CommandQueue(provider, provider.Devices |> Seq.head) 
    let command = 
        <@
            fun (rng:_1D) (a:array<_>) (b:array<_>) (c:array<_>)->
            let id = rng.GlobalID0
            c.[id]<- a.[id]+b.[id]
        @>
    let kernel, kernelPrepare, kernelRun = provider.Compile command
    let d = new _1D(20,localWorkSize)    
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
