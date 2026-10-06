[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 thread_id : SV_DispatchThreadID)
{
    // This entry point intentionally does no work. The example demonstrates
    // the Bevy asset-loading path rather than a render pipeline.
}

