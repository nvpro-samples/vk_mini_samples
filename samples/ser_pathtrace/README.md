# Shading Execution Reorder

![](docs/ser_1.png)

Shading execution reorder is a way to instruct the GPU on how to schedule the tracing of rays. It helps reducing the execution divergence and data divergence.

See this [PDF](https://developer.nvidia.com/sites/default/files/akamai/gameworks/ser-whitepaper.pdf)


## `main()`

To enable SER only a few things have been added. In main(), we have added the extension `VK_NV_ray_tracing_invocation_reorder`

## `RGEN Shader`

The other modification is done in the RGEN shader. We are reordering based on if the ray have hit the environment or not. This is done under the `USE_SER` scope (specialization constant variable).

## Result

As a result when activating SER, we can see that the execution is better distributed.

![](docs/ser_2.png)

